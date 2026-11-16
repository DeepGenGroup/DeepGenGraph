Semantic fragment IR, non-causal attention (2026-11-03)

The block -> fragment boundary now preserves reduce_frag, broadcast_frag,
convert_layout_frag and mask_frag as semantic operations on !frisk.fragment.
Fill produces broadcast fragments plus explicit fragment stores. Elementwise
numeric casts also stay as elementwise_frag. Thread lowering alone expands
reductions, broadcasts, layout exchanges and mask predicates into register
operations and communication. The three early optimization passes remain before
convert-fragment-to-thread; collectives constrain reordering, and pure broadcasts
may participate in invariant-fragment retention.

Inspect:
  00-insert-convert-layout.mlir: exact after insertConvertLayoutOps block IR
  01-fragment.mlir: directly after convert-friskbase-to-fragment
  06-optimized-fragment.mlir: after the early optimizations
  07-thread-bridges.mlir: after convert-fragment-to-thread
  08-thread.mlir: after bridge/address finalization
  kernel.ll: MyTest LLVM output
  kernel.o3.ll: LLVM default<O3> output

Raw fragment boundary: 2 reductions, 11 broadcasts, 4 layout conversions,
5 GEMMs, zero gpu.shuffle operations. Optimized fragment boundary retains
2 reductions, 12 broadcasts, 4 layout conversions and zero gpu.shuffle.
The thread boundary has zero semantic fragment operations and 20 shuffles.
The 32 scalar vector extracts/inserts still visible before the thread boundary
are matrix-read packet unpacking; they do not implement Reduce, Broadcast,
ConvertLayout or Mask. This non-causal case has no Mask; standalone regression
fixtures exercise mask_frag with/without fused addends and uniform specialization.
Reduction nodes retain the entire needed register strip/group, including its
thread-stage fragment widths, to preserve left-to-right FP association across
basic fragments. They are not precomputed partial reductions.

Reproduce from the repository root:
  bash generated_ir/semantic_fragment_noncausal/command.sh \
    > generated_ir/semantic_fragment_noncausal/lowering.log 2>&1
  python3 generated_ir/semantic_fragment_noncausal/save_stages.py
  /data2/xsl/install/bin/opt -S -passes='default<O3>' \
    generated_ir/semantic_fragment_noncausal/kernel.ll \
    -o generated_ir/semantic_fragment_noncausal/kernel.o3.ll
  python3 generated_ir/semantic_fragment_noncausal/validate.py

command.json and command.sh preserve the previous best non-causal configuration:
BHSD 1x32x4096x128, 256 threads, block warps 4x1, useKPack, asm128,
retain 32 words, PV interleave 4, V double buffering, K DMA 64-swizzle,
K LDS prefetch 8, K tail prefetch 4, cross-BN K0, softmax-fragments=pv.
The historical reference is noncausal_cross_bn_k0_lifetimes/k0_sum_anchor.ll
(1.6633449792861938 ms median on the original target run). This is not a
performance measurement of the new output. baseline.ll is the immediately
preceding fragment_level_optimized_noncausal/kernel.ll.

Validation includes standalone serialization/replay of every saved stage,
three early-pass regressions, LLVM verification, strict CPU reduction/broadcast
execution, exact lane/register layout shuffle interpretation, and wave/DMA
numerical simulation of the delivered LLVM (random + future-key-dominates).
Both cases match the preceding output bitwise. The random O3 result also matches
bitwise. Maximum reference absolute errors: 9.917702902600967e-05 (random),
4.617419446795168e-09 (future-key-dominates). DMA completion/ownership checks,
8192 MMACs and cross-BN overlap of 992 MMACs also pass. See numerical_validation.json.

GPU performance and target gfx936 code generation have not been rerun here.
run_gpu.sh uses the same launch shape/configuration when a target DCU is available.
