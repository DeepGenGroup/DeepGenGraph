Safe online softmax: four-way thread-local reduction experiment

Artifacts (all retain the previously tested mask bypass):
  baseline/kernel.ll  exact copy of ../online_softmax_mask_bypass/kernel.ll
  max4/kernel.ll      change only the two thread-local max reductions
  sum4/kernel.ll      change only the two thread-local sum reductions
  max_sum4/kernel.ll  apply both changes

Shape BHSD=(1,32,4096,128), BM=BN=128, BK=32; group=4, reverse=1.
Launch grid=(32,32,1), block=(256,1,1), K layout BHSD, LDS=16384 bytes.
Other tuning, including retain=32 words and frag-prefetch=0, is inherited.
These are standalone experimental IR files, not a compiler pass/default change.

Each thread contributes 32 values to each of two softmax rows. Instead of a
single left-to-right chain, each row now uses four independent accumulators:
  s0 = p0 + p4 + p8  + ... + p28
  s1 = p1 + p5 + p9  + ... + p29
  s2 = p2 + p6 + p10 + ... + p30
  s3 = p3 + p7 + p11 + ... + p31
  sum = (s0 + s1) + (s2 + s3)
Max uses the same grouping, retaining the original -inf seed on chain 0.
Original IR dependency depth: sum 31 -> 9; max 32 -> 10 including the seed.
Operation counts are unchanged: 31 adds or 32 maxnum operations per row.
These depths describe the input LLVM graph, not GPU instruction cycles.

Purpose: give instruction scheduling independent work while a previous
reduction result is pending. This may also change packed-add register moves.
It does not reduce exp2 count, cross-thread communication, or matrix work.
All source IR lines outside the selected local reductions are byte-identical:
mask branches, running max/alpha, exp2, P conversion/lifetime fences, shuffle,
DMA, MMAC, retained Q, output normalization, and launch configuration.

Sum changes floating-point addition order; target the existing approximate
attention contract, not bitwise reproducibility. No fast-math flags are added.
Max is intended for the existing finite-score/-inf-mask workload; this is not
a proof for arbitrary signaling NaNs or signed-zero bit patterns.
Four live partial accumulators can increase register pressure. Check VGPR,
scratch and latency on gfx936 before choosing a variant.

Validation completed locally:
  - Hash-bound generation and def-use checks; identical input multiset and
    operation count for each changed reduction, no external interior uses.
  - llvm-as and opt default<O3>,verify using local LLVM 24.
  - CPU execution of actual input LLVM: six cases for each of four artifacts,
    covering random first/diagonal/full-prefix tiles, positive overflow,
    negative underflow, and changing running max. All pass reference tolerance
    atol=rtol=0.01, finite output, delayed DMA, and one-time full Q loading.
  - O3 keeps the four max chains; it SLP-vectorizes sum into vector fadds and
    vector.reduce.fadd. Do not assume the literal input graph survives into
    assembly. Target DTK may optimize differently from local LLVM 24.
See structure.json, checks.json, and each variant's validation.json/numerics.log.
optimized.ll is a local inspection artifact; feed kernel.ll to the target DTK.
GPU code generation, GPU correctness, and timing remain to be tested.

Run on the gfx936 DTK machine from the repository root:
  bash generated_ir/online_softmax_local_reduce/run_gpu.sh

This builds and tests baseline, max4, sum4, max_sum4 in order, at O3, with
warmup=10 and repeat=50. It inherits LLVM_DIR/ROCM_BC/GPU_ARCH settings used by
GenHsaco.sh. Each directory gets kernel.hsaco, kernel.hsaco.amdgcn,
build-gpu.log, gpu.log, and gpu-result.json. Compare medians on the same GPU.
For a shorter comparison, or a repeated baseline at the end:
  bash generated_ir/online_softmax_local_reduce/run_gpu.sh baseline max_sum4 baseline

Reproduce IR (requires the exact original mask-bypass artifact):
  python3 generated_ir/online_softmax_local_reduce/generate.py
Re-run numeric validation for a chosen variant, e.g. sum4:
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    generated_ir/online_softmax_local_reduce/sum4/kernel.ll --bn 128 --group 4 \
    --output generated_ir/online_softmax_local_reduce/sum4/validation.json
