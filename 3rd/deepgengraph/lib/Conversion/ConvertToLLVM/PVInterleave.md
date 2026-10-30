# PV accumulator interleaving

`InterleavePVPass` is an opt-in LLVM function pass. `MyTest` runs it at final
export, after accumulator fusion and DMA lowering. `LegacyLLVMExportTest`
provides the same pass on an already exported LLVM module without running
unrelated optimizations.

Both executables accept `--pv-interleave=N`:

| N | Behavior |
|---|---|
| 0 | Disabled (default). |
| 1 | Serial accumulator chains with the same scheduling guards as 2/4; experimental control. |
| 2 | Interleave pairs of accumulator chains. |
| 4 | Interleave four accumulator chains. |

This changes scheduling, not floating-point association or approximation mode.
For four chains A/B/C/D of eight MMACs, mode 4 emits
`A0 B0 C0 D0 A1 B1 C1 D1 ... A7 B7 C7 D7`. Each MMAC retains its original
operands, accumulator SSA edge, attributes and flags. The number describes
independent output accumulator chains, not workgroups, waves or tile sizes.

## Matching and legality

The pass follows the accumulator operand of the exact register-only
`v_mmac_f32_16x16x16_f16` inline assembly. It groups equal-length, independent
chains that consume the same ordered matrix-read packets. A packet must be an
exact low/high half shuffle of a bitcast of
`llvm.hcu.ds.read.m32x16.b16.v4i32.p3`. Neither SSA names, kernel names, the
128x128x32 tile nor the 256-thread configuration is hard-coded.

The current bounded implementation accepts 2–16 chains, 2–32 distinct packets
per chain, and a maximum 2048-instruction region in one basic block. The chain
count must be divisible by the selected width. All packet reads must already
precede the scheduling region. Unsupported formats are left unchanged.

Only MMACs and safe, pure register computations move. Original workgroup
release/acquire fences, `s_barrier`, DMA-address `readfirstlane.i32`, and exact buffer-to-LDS DMA operations stay
in order and at their original MMAC ordinals. No read is hoisted and no reader
handoff is inserted or relocated. Unknown calls, additional memory operations,
nonzero scheduling masks, chain fanout and unresolved cross-chain dependencies
reject the complete candidate region before any mutation.

Every PV MMAC gets `sched.barrier(0)` before and after it, with inaccessible-memory
effects like the existing fragment-prefetch pass. These are compiler scheduling
hints, not added workgroup barriers or waits. Backend wait insertion and
register allocation still determine the final machine schedule and VGPR count.
Mode 1 uses identical guards so their effect can be measured independently.

`frisk.pv_interleave` metadata marks transformed MMACs. Repeated invocation is
idempotent; to compare widths, apply each width to the original input, not to
an already transformed output.

## Reproduction

Build:

```sh
cmake --build 3rd/deepgengraph/build --target MyTest LegacyLLVMExportTest -j 3
python3 3rd/deepgengraph/test/check_pv_interleave.py
```

Standalone, preserving the tested group32/forward/after4 baseline:

```sh
3rd/deepgengraph/build/test/LegacyLLVMExportTest --pv-interleave=4 \
  generated_ir/noncausal_v_handoff_sweep_group32/dma_after4.ll > /tmp/pv.raw.ll
3rd/deepgengraph/build/test/LegacyLLVMExportTest --llvm22-text /tmp/pv.raw.ll > /tmp/pv.ll
```

`--llvm22-text` preserves floating-point bit patterns while converting newer
LLVM textual literal syntax for DTK. The experiment generator records complete
commands, source/binary hashes, pass diagnostics and LLVM/O3 structural checks:

```sh
python3 generated_ir/noncausal_pv_interleave_pass_group32/generate.py
OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_pv_interleave_pass_group32/verify.py
bash generated_ir/noncausal_pv_interleave_pass_group32/run_gpu.sh
```

For fresh MLIR compilation, append `--pv-interleave=4` to a normal `MyTest`
command. This flag only enables interleaving: it does not recreate the separate
K grouped-prefetch or V after4 scheduling experiments. The generated performance
candidates therefore use the existing after4 LLVM baseline. Full MyTest on/off
exports are retained separately under the experiment's `mytest_smoke/` folder.
