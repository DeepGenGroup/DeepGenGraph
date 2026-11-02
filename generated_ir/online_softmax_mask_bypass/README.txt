Safe online softmax: fully-visible mask bypass experiment

Test artifact: kernel.ll
Baseline: ../online_softmax_native_acc/kernel.ll (reported median 1.242959 ms)
Shape BHSD=(1,32,4096,128), BM=BN=128, BK=32.
Launch: grid=(32,32,1), block=(256,1,1), K layout BHSD.
Tuning inherited from baseline: group=4, reverse=1, retain=32 words,
frag-prefetch=0, K swizzle=8:6, QK/VO tile=32, LDS=16384 bytes.

This is a standalone LLVM IR experiment, not a compiler-default change.
generate.py requires the exact measured baseline SHA256 and reproduces kernel.ll.
Only the mask merge changes: 16 vector4 score+mask adds move into the masked
branch; the fully-visible branch passes the original QK fragments through PHIs.
The existing complete-tile predicate and diagonal mask are retained.
DMA/MMAC/call sequence, stable max/alpha updates, sum order, O accumulator,
workgroup mapping, output normalization, and LDS allocation are unchanged.
No global fast-math flags are added. Removing +0 can preserve a negative zero;
this experiment targets the existing approximate attention numerical contract.

Local validation:
- llvm-as and opt default<O3>,verify.
- After O3 all 16 vector4 mask fadds remain confined to the masked branch.
- Existing CPU execution of actual LLVM: see numerics.log and validation.json.
- Structure and source/artifact hashes: structure.json.
CPU validation does not establish gfx936 code generation or GPU performance.

On the gfx936 DTK machine, from the repository root (existing LLVM_DIR/ROCM_BC):
  bash generated_ir/online_softmax_mask_bypass/run_gpu.sh

The script compiles at O3 and runs launchKernel.py with warmup=10, repeat=50.
Build intermediates and results stay in this experiment directory:
  kernel.hsaco, kernel.hsaco.amdgcn, build-gpu.log, gpu.log, gpu-result.json
Compare event-timing median with the baseline on the same GPU/test conditions.
Inspect fully-visible path instructions, VGPR/scratch, and correctness.

Reproduce the experimental IR locally:
  python3 generated_ir/online_softmax_mask_bypass/generate.py
