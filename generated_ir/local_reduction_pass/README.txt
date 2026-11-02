LocalReductionPass validation artifacts

Implementation and usage:
  ../../3rd/deepgengraph/test/local_reduction_README.txt

kernel.ll: standalone pass on the previously validated mask-bypass BN128 IR.
kernel_bn64.ll: standalone pass on the existing mask-bypass BN64 IR.
kernel_max4.ll / kernel_sum4.ll / kernel_max_sum4.ll: operation-selection checks.
kernel_dtk.ll / kernel_bn64_dtk.ll: fresh full-compiler exports, including
  MyTest's LLVM22 text legalization. Use these files for target DTK compilation.
launch.json: BHSD=(1,32,4096,128), grid=(32,32,1), block=(256,1,1).

Standalone LLVM exports use the local LLVM24 printer; they are the inputs to
the CPU execution tests. They are not additional GPU benchmark measurements.
The user validated the earlier experimental IR; checks.json records that this
pass reproduces all three experimental O3 graphs after stripping debug info,
value names and the pass's idempotence marker.

Validation records:
  regression.json/log: 46 generic IR fixtures, 920 native CPU comparisons,
    legality/fallback/idempotence, O3 verification, chains 2/3/4/5/8/16.
  validation-bn128.json, validation-bn64.json: 12 attention CPU cases,
    including overflow/underflow stability, changing max, causal masks,
    delayed DMA and full-Q retention checks.
  compiler/ and compiler-bn128/: six full MyTest exports, off/legal/approx
    at each tile width, including exact commands and tuning result records.
  comparison.json: max-only/sum-only/combined match the experimental O3 IR.
  python-tests.log: 32 tuning/portable-tooling tests.
  mask-bypass-regression.log: existing mask-bypass regression passes.
  build.log: LegacyLLVMExportTest and MyTest build.
  checks.json: consolidated results and full-compiler artifact hashes.

Run all four input options from original source, not already transformed IR:
  --local-reduce=approx --local-reduce-chains=4
  --local-reduce-min-inputs=16 --local-reduce-ops=add,max
Default mode is off. Use legal mode for generic callers that should only use
existing FP reassociation permission. Integer reductions need no FP permission.

Optional target GPU test of the fresh BN128 full-compiler export:
  bash generated_ir/local_reduction_pass/run_gpu.sh
Uses existing GenHsaco.sh/launchKernel.py, O3, warmup 10 and repeat 50.
Build scratch, assembly, gpu.log and gpu-result.json stay in this directory.
GPU correctness, generated ISA and performance of new compiler exports still
need target-machine validation; local LLVM24 O3 equivalence is not a timing test.
