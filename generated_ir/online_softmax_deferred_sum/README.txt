Safe online softmax: defer denominator cross-lane reduction until loop exit

Test IR: kernel.ll
Control IR: baseline.ll (only numeric SSA/block names changed from source)
Source: ../local_reduction_pass/kernel_dtk.ll
Both include the existing mask bypass and four-chain local max/sum reductions.
BHSD=(1,32,4096,128), BM=BN=128, BK=32, group=4, reverse=1.
Launch grid=(32,32,1), block=(256,1,1), K layout BHSD; LDS=16384 bytes.
No compiler pass/defaults are changed by this experiment.

The change
----------
Before, at every key tile t:
  p_t[lane] = thread-local sum(exp2(score - new_max))
  L_t = alpha_t * L_(t-1) + allreduce_sum(p_t)
After, at every key tile t:
  local_L_t[lane] = alpha_t * local_L_(t-1)[lane] + p_t[lane]
After the loop, before output normalization:
  L = allreduce_sum(local_L_final)

The loop-carried denominator starts at zero in both versions. The new version
carries a lane-local partial, not a replicated global denominator. Each row's
four contributing lanes (xor 16 and xor 32) use the same running max and alpha
at every iteration. Therefore, in real arithmetic, summation commutes with
the update: sum(alpha*x + p) = alpha*sum(x) + sum(p).
No extra factor of four is introduced.

This changes floating-point addition and scaling order. It targets the existing
approximate attention contract, not bitwise reproducibility. It is specific to
this checked layout and recurrence; alpha uniformity and exclusive denominator
uses must be re-established before applying it to another kernel.

For two rows per thread:
  denominator shuffles: 4 per iteration -> 4 once after loop
  max shuffles:        4 per iteration -> unchanged
For T=32 iterations, denominator shuffles drop from 128 to 4.
The complete IR path, including 64 output-layout shuffles, drops from 320 to
196 executed shuffle call sites. This is an IR count, not a GPU timing estimate.
At T=1 the dynamic shuffle count is unchanged.

Running max, alpha, score exp2, local four-chain reductions, numerator/PV
accumulation, masks, DMA, barriers, retained Q and output layout stay unchanged.
The four sum shuffles are added at the existing final denominator extractions,
reusing the exit block's xor-lane addresses. They do not use loop-body values
that would fail dominance on a zero-trip path. No fast-math flags are added.

Validation
----------
generate.py is bound to the exact source hash and checks the complete edit.
validate.py verifies both IRs with llvm-as and LLVM24 opt default<O3>,verify.
After O3, there are still four max shuffle sites in the loop and all four
denominator shuffle sites remain in the exit block.

The CPU model executes both actual input LLVM files on eight cases: random
query tiles 0/1/31, positive overflow, negative underflow, changing running max,
32 iterations with increasing max, and uneven lane contributions. It checks
reference accuracy (atol=rtol=0.01), A/B error, alpha equality across xor16/32
partners at every update, dynamic shuffle counts, delayed DMA, output bounds,
finite output, and one-time full Q loading. See numerics.log / validation.json.
CPU checks and local LLVM O3 do not establish target GPU correctness/performance.

GPU A/B (repository root, existing LLVM_DIR/ROCM_BC settings):
  bash generated_ir/online_softmax_deferred_sum/run_gpu.sh
This compiles baseline then deferred at O3, checks outputs and measures each
with warmup=10, repeat=50. Each has its own build directory to avoid overwriting
shared GenHsaco.sh scratch files:
  baseline/kernel.hsaco.amdgcn, baseline/gpu.log, baseline/gpu-result.json
  deferred/kernel.hsaco.amdgcn, deferred/gpu.log, deferred/gpu-result.json
Only test the experiment:
  bash generated_ir/online_softmax_deferred_sum/run_gpu.sh deferred

Reproduce:
  python3 generated_ir/online_softmax_deferred_sum/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/online_softmax_deferred_sum/validate.py
Use kernel.ll for the target toolchain. *.optimized.ll are local LLVM24
inspection artifacts and use that version's printed syntax.
