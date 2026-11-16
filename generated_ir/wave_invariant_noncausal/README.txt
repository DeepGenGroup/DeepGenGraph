Wave-invariant cleanup, non-causal attention (2026-11-03)

kernel.ll / kernel.o3.ll: MyTest output with the unchanged best non-causal
tuning parameters. The passes below are on by default. NOT run on a GPU: no
timing exists for this kernel, and /dev/kfd is unavailable here.

Changes, in pipeline order:
  frisk-simplify-bounded-affine   MLIR pass, affine level, after affine-scalrep.
    Removes K source floordiv/mod using constant affine.for bounds. 3 maps.
    Flag: --simplify-bounded-affine
  frisk-hoist-wave-invariants     MLIR pass, LLVM dialect, after secondLowering.
    lds-bases: readfirstlane chains for DMA LDS bases. Flag: --hoist-lds-bases
      The same flag makes DMA lowering reuse that read, removing a second
      readfirstlane at each DMA.
    lane-shuffles: mbcnt lane ID and bpermute indices. Flag: --hoist-lane-shuffles
      Keeps 2 bpermute indices in VGPRs across the whole loop.
  elideBackedgeBarrier            LLVM pass, right after prefetchNextK.
    Removes the barrier ending each iteration. Flag: --elide-backedge-barrier
    Not an MLIR pass on purpose. On MLIR, V is not double buffered and K0 has not
    moved, so K0's DMA sits between the two barriers and writes the slot V3 reads.
    no_lds_split.ll is that shape: deleting the barrier there is a detected hazard.
    prefetchNextK also anchors on this barrier.

O3 static counts, baseline -> kernel:
  loop integer/address instructions   289 -> 224
  readfirstlane calls                  16 -> 4
  sdiv                                  2 -> 0
  barriers per iteration                9 -> 8   (dynamic: 293 -> 261 per workgroup)
  MMAC 256, DMA 36, s_waitcnt 12: unchanged

Proxy only: llc does not know gfx936. Compiled for gfx90a, which has a different
register file, the loop goes 1653 -> 1525 instructions and vgpr_count 405 -> 396
with no VGPR spills. That is not evidence for register use on gfx936.

Validation (no GPU):
  python3 validate.py          numerical_validation.json/.log
    7 input cases x kernel, kernel.o3: output bitwise equal to baseline, within
    tolerance of the FP64 reference; global trace, ordered LDS values, V schedule
    and K0 schedule unchanged; delayed DMA and barrier-epoch ownership audit.
    3 negative controls are rejected by the audit.
  regression_validation.json   16 scripts: the 14 from async_task_noncausal, plus
    check_wave_invariant_passes.py and check_cross_bn_k0.py.
  Not checked: target codegen, register allocation, independent wave timing.

Reproduce from repository root:
  bash generated_ir/wave_invariant_noncausal/command.sh > generated_ir/wave_invariant_noncausal/lowering.log 2>&1
  /data2/xsl/install/bin/opt -S -passes='default<O3>,verify' generated_ir/wave_invariant_noncausal/kernel.ll -o generated_ir/wave_invariant_noncausal/kernel.o3.ll
  OUT=$PWD/generated_ir/wave_invariant_noncausal/no_lds_split.ll bash generated_ir/wave_invariant_noncausal/command.sh --cross-bn-k0=0 --v-double-buffer=0 --k-dma=128 --k-lds-prefetch=0 --k-tail-prefetch=0 --swizzle-k=off
  python3 generated_ir/wave_invariant_noncausal/validate.py

Suggested GPU order: all four on against all four off, then --hoist-lane-shuffles=0
alone, since it is the only change that adds a value live across the loop.
