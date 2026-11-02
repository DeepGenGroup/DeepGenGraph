Cross-BN K0 prefetch: candidates for the measured register-pressure regression

Start with k0_sum_anchor.ll and its matching .launch.json.
All candidates retain the measured QK-tail-prefetch4 optimization, 16 KiB LDS,
the original FP32 arithmetic order, and the guarded cross-BN K0 schedule.
The original experiment and benchmark artifacts have not been overwritten.

Measured regression (pmc_test.log and ktest.hsaco (2).amdgcn):
  k_tail_prefetch4 median: 1.6913459897 ms
  cross-BN K0 median:     1.7517440319 ms (+3.571%)
  VGPR: 256; scratch/private segment: 20 bytes; vgpr_spill_count: 4.
  Four static scratch stores in the prologue and four reloads in the loop.
  QK setup has extra vmcnt(0) waits around the scratch reloads.
  All 64 softmax denominator FP32 additions appear in k0.join, after the K0
  prefetch branch. This extends many unreduced FP32 values across earlier PV
  fragments. Scheduling these adds early in LLVM IR alone was insufficient.
  K DMA source address values v208/v209 also live from QK setup to k0.issue.
  The older ktest.hsaco.amdgcn has 227 VGPR and scratch=0 and is not the spill
  artifact. See regression.json for input hashes and extracted evidence.

Variants:
  control.ll
    Byte-identical to the previously measured k_tail_prefetch4 baseline.
  k0_sum_anchor.ll
    Primary diagnostic candidate. Add tied-register, side-effecting empty-asm
    identities after the 64 exact FP32 denominator additions. Their values
    must be computed before the identity at each fragment; the backend cannot
    sink the entire reduction across the K0 branch. The identity instruction
    itself has an empty assembly body; register allocation may still add copies.
    No floating-point reassociation, extra global traffic or hardware barrier.
    K0 overlaps the final PV fragment's 32 MMAC, as in the original experiment.
  k0_sum_anchor_remat.ll
    Additionally recompute next K0 lane/swizzle/LDS addresses at the issue site,
    starting from a fresh tied thread-ID identity. Only the original thread ID,
    next tile, uniform head offset and K resource descriptor cross into the
    issue block; the old per-lane/per-wave K address intermediates do not.
  k0_sum_anchor_remat_after16.ll
    Same changes, but issue after 16 of the final PV fragment's 32 MMAC.
    Trades half of the overlap window for a later issue/control-flow split.

Validation:
  python3 generated_ir/noncausal_cross_bn_k0_lifetimes/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_cross_bn_k0_lifetimes/verify.py

The generator verifies LLVM and O3, unchanged floating-point/MMAC operation
sequence, 64 identity anchors before the K0 guard after O3, address dependency
boundaries, and the 32/16-MMAC overlap windows. The numerical interpreter checks
the delivered IR and O3 candidates, bitwise agreement with control, FP64
reference tolerance, global/LDS access traces and synchronization epochs,
unchanged V scheduling, K0 tile sequence/count, and delayed DMA ownership.
It also rejects removal of the last-tile guard after moving the issue point.
See validation.json for completed tests. These checks do not establish target
register allocation, independent wave timing, or performance. In particular,
these files are candidates; absence of spill is NOT claimed without DTK output.

First GPU comparison:
  bash generated_ir/noncausal_cross_bn_k0_lifetimes/run_gpu.sh control k0_sum_anchor control

All candidates with interleaved controls:
  bash generated_ir/noncausal_cross_bn_k0_lifetimes/run_gpu.sh

Each run uses GenHsaco.sh at O3, saves the input hash/build/HSACO/assembly in an
isolated runs/ directory, and writes resources.json before GPU execution.
The script skips benchmarking variants with scratch/spill or unreadable
resource metadata. Spill-free variants run launchKernel.py --non-causal,
including its correctness comparison, and save result.json/gpu.log.
LLVM_DIR/ROCM_BC/GPU_ARCH use the existing GenHsaco.sh environment settings.
ATTN_WARMUP defaults to 20; ATTN_REPEAT defaults to 100.

For the new assembly, check that denominator additions execute before the
K0 branch, scratch and vgpr_spill_count are zero, and next K0 still issues
before the intended final-PV MMAC. Scratch=0 is a prerequisite for this
comparison, not proof of a performance gain.
