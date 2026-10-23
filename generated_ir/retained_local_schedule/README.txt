Small-window wait / fragment scheduling experiments

Benchmark inputs (use the corresponding *.launch.json):
  baseline.ll       Exact copy of invariant_fragment_pass/retained_8words.ll.
                    User measurement: median 1.646087 ms, VGPR 228, scratch 0.
  wait_tail2.ll     Move two final register MMACs of each of the first three
                    QK BK stages after the next Q DMA and before its VM wait.
  fragment_pair.ll  Advance a complete adjacent pair of b64 LDS reads across
                    one preceding MMAC, in 27 surviving read groups.
  combined.ll      Apply both independent transformations.

Suggested order: baseline, wait_tail2, fragment_pair, then combined.
Interleave baseline reruns to distinguish a small speedup from timing drift.
No experiment has a new target-GPU timing or register-allocation result yet.

The baseline is BYTE-IDENTICAL to the latest measured 1.646087 ms IR:
  SHA256 8923ce4d910b6c2e6ce91fa3345c5f2e702b57d2bef5ef37cd12c726b0efea19
Group8 reverse, K128 swizzle, one retained Q BK=32, all layouts and math remain.
BM=BN=128, BK=32, BHSD=(1,32,4096,128), block=(256,1,1), grid=(32,32,1).
Static LDS=16384 bytes; dynamic shared=0. Kernel ABI is unchanged.

wait_tail2:
  Before: ... final two MMACs ... next Q DMA ... wait Q ...
  After:  ... next Q DMA ... final two MMACs ... wait Q ...
  Operands have already been read into registers. Their original LDS reads,
  DMA ordering, fences and all hardware barriers remain in place. The existing
  scheduling-hint pair moves with the two MMACs to bracket the new window.
  Three windows per BN, two MMACs/window. This tests overlap with outstanding
  DMA; it does NOT remove waits or assume a nonzero partial wait counter.
  Holding the old operand registers longer may increase VGPR demand.

fragment_pair:
  Before: MMAC_i; load_next_0; load_next_1; MMAC_i+1; ...
  After:  load_next_0; load_next_1; sched_hint; MMAC_i; sched_hint; ...
  Preserve complete adjacent read groups and their order, allowing backend DS
  combining. One two-read packet is 16 bytes / 4 raw 32-bit words per thread;
  this is NOT a guarantee about final VGPR delta or occupancy. Only groups whose
  two loads survive local LLVM O3 are selected. Duplicate loads eliminated by
  O3 and groups crossing an existing lifetime marker are skipped. No read crosses
  DMA, a hardware barrier/wait, a store, or control flow. V matrix reads are not
  changed in this experiment. The 54 new sched.barrier(0) calls are compiler
  scheduling hints with inaccessible-memory effects, not workgroup barriers.

For N=query_tile+1, every variant has identical dynamic work/traffic counts:
  Q DMA:                  2 + 6*N
  K/V DMA:                16*N
  ordinary b64 LDS reads:  4 + 140*N
  FP16 Q vector scales:   4 + 12*N
  MMAC:                   256*N
  explicit vmcnt(0):      1 + 11*N
  hardware barriers:      2 + 16*N
  Compiler hints:         8*N baseline/wait_tail2; 62*N fragment_pair/combined.

Reproduce from repository root:
  python3 generated_ir/retained_local_schedule/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/retained_local_schedule/verify.py

Target test (uses existing GenHsaco.sh / DTK environment):
  bash generated_ir/retained_local_schedule/run.sh baseline
  bash generated_ir/retained_local_schedule/run.sh wait_tail2
  bash generated_ir/retained_local_schedule/run.sh fragment_pair
  bash generated_ir/retained_local_schedule/run.sh combined
Each variant gets a separate build directory and result.json. Compare program
median plus final VGPR/scratch, and inspect whether the backend preserves the
intended load/MMAC/wait ordering. O3 LLVM text alone cannot prove GPU scheduling.
Files ending .opt.ll are local O3 review artifacts, not the benchmark inputs.

Validation:
  LLVM assembly and default<O3>,verify for all four inputs.
  O3 output still contains all three DMA/MMAC/wait overlap windows and all
  27 advanced read pairs (o3_schedule.json).
  Actual LLVM CPU execution: q tiles 0/1/7/8/31, heads 0/1, two seeds.
  All outputs are bitwise equal; reference atol/rtol=0.01; global and LDS access
  traces are identical; no pending DMA or writes outside the output tile.
  Additional per-byte LDS reader/writer wave masks check cross-wave RAW/WAR/WAW
  conflicts within barrier epochs, with DMA overwrites checked at issue time.
  Removing the initial DMA wait or retained-Q handoff barrier is deliberately
  rejected as a negative control (the corresponding FAILED BLOCK diagnostic in
  validation.log is expected and followed by PASS negative).
  This is not cycle-accurate wave execution or gfx936 hardware verification.

This generator is restricted to the fingerprinted input. No compiler Pass is
changed in this experiment; integrate a winning policy after device results.
