Q retention: one versus two BK fragments (single configuration variable)

Benchmark inputs:
  baseline_8words.ll   One Q BK fragment, Q[:,0:32], retained across BN.
  retained_16words.ll  Two Q BK fragments, Q[:,0:64], retained across BN.
  Use the matching *.launch.json for each file.

Both inputs are exports of the existing frisk-retain-invariant-fragments Pass,
from the same nohint_unrolled.mlir and compiler/options. The ONLY configuration
change is --retain-fragment-words=8 -> --retain-fragment-words=16.
No new compiler Pass, LLVM instruction patch, fragment scheduling experiment,
wait_tail2 transformation, or input-IR structural change is included.
The Pass reports one/two retained operand tiles; generated LLVM and execution
checks confirm the second fragment is actually retained, not just budgeted.

The freshly compiled baseline is BYTE-IDENTICAL to the measured retained_8words:
  SHA256 8923ce4d910b6c2e6ce91fa3345c5f2e702b57d2bef5ef37cd12c726b0efea19
  Previous user median 1.646087 ms, VGPR 228, scratch 0.
The 16-word candidate has not been GPU timed or register-allocated locally.

Unchanged configuration:
  BHSD=(1,32,4096,128), BM=BN=128, BK=32, group8 reverse, K128 swizzle.
  Grid=(32,32,1), block=(256,1,1), static LDS=16384 bytes, dynamic LDS=0.
  Existing fragment prefetch settings, mathematical operations and kernel ABI.
  Additional live Q state: four <4 x half> values = 32 bytes / 8 words per thread.
  Total retained Q state: 8 -> 16 words/thread; allocated VGPR delta is unknown.

Dynamic IR counts per workgroup, N=query_tile+1:
                                 baseline_8words       retained_16words
  Q DMA calls                    2 + 6*N               4 + 4*N
  K/V DMA calls                  16*N                  16*N
  ordinary b64 LDS reads         4 + 140*N             8 + 136*N
  Q FP16 vector scales           4 + 12*N              8 + 8*N
  explicit vmcnt(0) waits        1 + 11*N              2 + 10*N
  workgroup barriers             2 + 16*N              3 + 14*N
  MMAC                          256*N                 256*N
  scheduling hints              8*N                   8*N

These are IR operations, not target DS opcode counts. The compiler regenerates
synchronization for the new lifetimes: one extra wait/barrier in the prologue,
one fewer wait and two fewer barriers per BN iteration. This is a consequence
of retention under the same synchronization settings, not a second experiment.
The prologue uses existing LDS storage; retained fragments live in registers.

Logical global requests over the full grid:
  Q:   404 -> 280 MiB (124 MiB fewer, 30.69% of baseline Q requests).
  QKV: 1460 -> 1336 MiB (8.49% fewer).
  K and V each remain 528 MiB. These are not measured HBM bytes and are not
  predicted timing improvements. Extra VGPR pressure may offset reduced work.

Reproduce from repository root:
  python3 generated_ir/q_two_bk_retained/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/q_two_bk_retained/verify.py
  generate.py --existing verifies already exported inputs without recompiling.
  export_commands.json records actual export commands; manifest.json also lists
  reproduction commands. pass_budget16_probe.* are initial compiler-probe copies,
  not additional benchmark variants. *.opt.ll are local O3 inspection artifacts.

Target test in the existing DTK environment:
  bash generated_ir/q_two_bk_retained/run.sh baseline_8words
  bash generated_ir/q_two_bk_retained/run.sh retained_16words
  bash generated_ir/q_two_bk_retained/run.sh baseline_8words
Each build-<variant>/ saves kernel.hsaco.amdgcn, assembly_counts.json, result.json.
Copy result.json before repeating a variant to preserve every timing round.
Compare program median and final VGPR/SGPR/scratch as well as the PMC counters.

Validation:
  LLVM assembly and O3 verification for both files (generate.log).
  Five actual LLVM CPU execution cases, q=0/1/7/8/31, heads 0/1, two seeds.
  Bitwise equality, reference tolerance, exact per-element Q load multiplicities,
  unchanged ordered K/V and Q[:,64:128] reads, output bounds, delayed DMA readiness,
  and per-byte cross-wave LDS barrier-epoch checks (validation.json/log).
  Removing the second Q prologue wait or its publication barrier must be
  rejected; the negative controls intentionally print FAILED BLOCK diagnostics.
  This is not cycle-accurate GPU validation or a target occupancy prediction.
