Non-causal safe online softmax

Input: ../../3rd/deepgengraph/test/test_input_spidD_noncasual.mlir
LLVM:  ../test_input_spidD_noncasual.ll
The spelling "noncasual" follows the requested filename; semantics are non-causal.

Derived from test_input_spidD.mlir, with both trilu/add in the high-level
reference and the kernel's causal mask removed. Every query tile visits all
4096 keys (32 BN iterations). The kernel maintains FP32 running max M,
denominator L and output O:
  M' = max(M, rowmax(score))
  alpha = exp2(M - M')
  P = exp2(score - M')
  L' = alpha*L + rowsum(P)
  O' = alpha*O + FP16(P) @ V
The initial state is M=-inf, L=O=0; final output is FP16(O/L).
Q pre-scaling by FP16(log2(e)/sqrt(128)) follows the original kernel.

Configuration follows the best measured safe-softmax configuration documented
in ../../优化方式总结.md and ../online_softmax_deferred_sum/README.txt:
  BM=128, BN=128, BK=32; 256 threads, waves=4,1; layout=3.3
  group4 reverse; K swizzle=8:6; Q/V swizzle off
  retain-fragment-words=32; frag-prefetch=0; global prefetch=1
  async 16-byte DMA, useKPack/asm128; carrier=34816 words/block
  no cross-DK pipeline, DK loop simplification enabled; coalesce-waits=0
  approx math, four-chain local reductions, deferred cross-lane sum
  mask-add-bypass remains enabled in the preset but has no mask to transform.
The old causal timing is not a measurement of this non-causal kernel.

Re-export from the repository root:
  bash generated_ir/spidD_noncasual/reproduce.sh

CPU numerical validation of the actual exported LLVM:
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    generated_ir/test_input_spidD_noncasual.ll --bn 128 --group 4 --non-causal \
    --output generated_ir/spidD_noncasual/validation.json

GPU test, with the existing target LLVM_DIR / ROCM_BC environment:
  bash generated_ir/spidD_noncasual/run_gpu.sh
This builds HSACO at O3 and compares against PyTorch SDPA is_causal=False.
The launcher defaults to causal=True for existing callers; launch.json and
--non-causal explicitly select full attention for this kernel.
Launch ABI: Attn_p2(Q,K,V,O), contiguous FP16 BHSD=(1,32,4096,128), including K;
grid=(32,32,1), block=(256,1,1), dynamic shared memory=0.
GPU artifacts, correctness report and timings are saved in gpu/.

Compiler prerequisite: ReduceOpConversionPattern must preserve the type of
loop-produced scores and insert a Frisk cast at the reduction input. Mutating
the affine loop result's memory space in place disconnects LowerInfo through
unrealized tensor casts. The accompanying compiler fix covers this path.
ConvertOpConversionPattern also preserves its producer's type; the existing
Frisk cast performs the requested element type and storage conversion.

Validation completed locally (result.json / validation.json):
  llvm-as and opt default<O3>,verify passed.
  Seven actual-LLVM CPU cases passed, max absolute error 0.000144574:
  random queries 0/1/31, positive overflow, negative underflow, changing max,
  and query 0 dominated by the final key tile (outside its causal prefix).
  All cases checked finite output, delayed DMA hazards, output write bounds
  and loading each Q element exactly once. An old causal LLVM fails the new
  non-causal reference, as expected (negative-control.log).
  Re-exported causal safe-softmax kernel passed six numerical regressions.
  Four launcher tests passed, including non-causal SDPA and CLI/config behavior.
  Static LDS=16384 bytes; 4 Q tiles retained (32 words/thread); 4 denominator
  shuffle sites moved out of the key loop; four-chain local max/sum applied.
  No GPU test or non-causal performance measurement was run in this environment.

SHA256SUMS paths are relative to the repository root.
run-* directories contain failed intermediate exports kept for diagnostics;
use only ../test_input_spidD_noncasual.ll for GPU testing.
