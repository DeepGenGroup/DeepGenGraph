MaskAddBypassPass validation and GPU test artifacts
=================================================

kernel.ll: BN128, produced by the general LLVM pass from
  ../online_softmax_native_acc/kernel.ll
kernel_bn64.ll: corresponding BN64 regression artifact.

BN128 has the same local O3-normalized IR as the previously GPU-tested manual
experiment ../online_softmax_mask_bypass/kernel.ll. Names/debug info are stripped
for this comparison; this does not assert a new gfx936 benchmark measurement.

Validation evidence:
  regression.json/log: 18 legality/fallback fixtures, 3 modes, idempotence,
                       O3 verification, 600 native CPU FP edge comparisons.
  validation-bn128.json, validation-bn64.json: 12 full LLVM attention CPU cases,
                       including max changes, large positive/negative inputs,
                       complete Q retention and delayed DMA checks.
  structure.json: input hashes and comparison with the manual experiment.
  compiler/: full MyTest/TuneAttention exports for BN128 and BN64 with
             off/legal/approx; result.json records pass match counts.
  checks.json: consolidated verification.

Default compiler/tuning mode is off. Enable with:
  --mask-add-bypass=approx                   (MyTest, math=approx required)
  --set mask-add-bypass=approx               (TuneAttention.py)
  --sweep 'mask-add-bypass=off;legal;approx'  (TuneAttention.py)

Standalone transform, from repository root:
  3rd/deepgengraph/build/test/LegacyLLVMExportTest \
    --mask-add-bypass=approx \
    generated_ir/online_softmax_native_acc/kernel.ll > /tmp/mask-bypass.ll

GPU compile and test, from repository root with existing DTK settings:
  bash generated_ir/online_softmax_mask_bypass_pass/run_gpu.sh

Pass design, limitations and reproduction commands:
  ../../3rd/deepgengraph/test/mask_add_bypass_README.txt
