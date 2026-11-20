# PV and QK accumulator interleaving in Thread IR

`MyTest --pv-interleave=N` runs `frisk-thread-interleave-pv` after thread
unrolling, address canonicalization and lane-sum rewriting, before async/GPU
lowering. `ThreadInterleavePV.cpp` owns matching, legality and scheduling.
The former LLVM `InterleavePVPass` and `interleavePVAccumulators` are removed;
`LegacyLLVMExportTest` rejects this option and points callers to MyTest.

| N | Behavior |
|---|---|
| 0 | Disabled (default). |
| 1 | Keep serial accumulator chains and add scheduling guards (control). |
| 2 | Interleave pairs of independent accumulator chains. |
| 4 | Interleave groups of four independent accumulator chains. |

For four chains, mode 4 emits `A0 B0 C0 D0 A1 B1 C1 D1 ...`.
Each MMAC retains its operands, accumulator SSA edges and math attributes.
The pass changes neither floating-point association nor approximation mode.
Width describes independent output fragments, not waves or workgroups.

## QK and guard controls

Both entry points use the same chain validation and topological scheduler.
The existing PV defaults are unchanged; QK is opt-in.

| MyTest option | Default | Meaning |
|---|---|---|
| `--qk-interleave=0\|1\|2\|4` | `0` | Schedule ordinary LDS-backed B operands; width 1 is serial with the same preloading policy. |
| `--pv-guard-group=-1\|0..32` | `-1` | PV guard policy, independent of interleave width. |
| `--qk-guard-group=-1\|0..32` | `2` | QK guard policy, independent of interleave width. |
| `--qk-prefetch-words=N` | `32` | Maximum distinct QK B operand footprint, in 32-bit words/thread; zero rejects all QK candidates. |

Guard group `-1` retains the legacy guard before and after every MMA. Zero
emits no guards. A positive N emits one guard **after** every N scheduled MMAs,
including a final partial group; it adds no leading guard. Groups count MMAs
in final scheduled order, not waves, loads or accumulator chains.

For the BM128/BN16/D128 experiment, use:

```sh
--qk-interleave=2 --qk-guard-group=2 --qk-prefetch-words=32
```

This targets `A0 B0 guard A1 B1 guard ... A7 B7 guard`. Guard zero is an
explicit unguarded control, not a promise that the backend preserves the order.
QK is placed after optional paired LDS dedup/prefetch and before softmax fragment
scheduling. Canonical LDS addresses are enabled whenever QK scheduling is enabled.

QK accepts aligned `frisk.lds_load` vector<4xf16> packets through the same
transparent vector views as PV. It may preload late B packets before the first
candidate MMA, preserving load order and each accumulator's arithmetic DAG.
Every distinct packet (including already available packets shared across chains)
costs two budget words. This bounds the B footprint, not total VGPR usage.
The complete scheduling window must contain only those reads, selected MMAs and
speculatable register operations. Stores, other reads, calls, nested control,
DMA, waits, hardware barriers, existing scheduling guards and protected paired
loads reject the window. Address dependencies on an unscheduled MMA also reject
it. No load is speculated across control flow or synchronization.

Both paths honor strictfp/optnone, preserve intermediate-use checks, plan before
mutation, and mark scheduled MMAs for idempotent replay. Rejected-window reasons,
hoisted load counts and emitted guard counts are reported. As with PV, apply a
different configuration to the original input, not previously scheduled output.

## Representation and legality

The pass accepts the exact register-only `v_mmac_f32_16x16x16_f16` template
and its validated padded form, either as `frisk.warp_mma_rr` or as the
`llvm.inline_asm` produced by current thread legalization. The latter has
`has_side_effects` until LLVM export; recognizing the entire instruction,
constraints and operand types establishes the same register-only contract as
`prepareRegisterMMAForLLVM`. Arbitrary assembly and memory clobbers are rejected.

For PV, B operands must be transparent vector views (`bitcast`, `shape_cast`,
`extract`, `from_elements`) of one explicit
`frisk.read_matrix_from_shm` m32x16/b16 packet. No arithmetic, selects or
mixed-packet assembly is accepted. Dynamic extraction indices can remain in
Thread IR: the pass preserves their exact SSA values and never assumes they
are a particular low/high half. This avoids depending on LLVM InstCombine
folding thread-bound affine expressions into constant shuffles.

Chains of equal depth can use different matrix-packet producers, as happens
for different output columns in causal attention. Limits remain 2–16 chains,
2–32 distinct packets per chain, and 2048 operations in one block. Chain count
must be divisible by width; every packet from every chain must precede all
candidate MMACs. No memory access is hoisted. If the combined group fails the
legality checks, groups with identical ordered packets are tried separately
to preserve existing scheduling opportunities.

The pass constructs a complete topological plan before mutation. GPU barriers,
Frisk sync operations, async copies and readiness waits remain in order at
their original MMAC ordinals. Unknown effects, additional reads/writes, nested
control flow, intermediate chain fanout and unresolved cross-chain dependencies
reject the candidate. Only speculatable memory-free register operations may be
pulled in as dependencies. Every accumulator's original K order is preserved.

`frisk.pv_interleave` / `frisk.qk_interleave` mark scheduled MMACs, making replay idempotent. Apply
alternative widths to the original input, not already transformed output.

## Scheduling guards and lowering

The module pass emits typed LLVM-dialect calls to
`llvm.amdgcn.sched.barrier(0)` at the configured group boundaries (by default,
before/after each PV MMAC), with explicit
`memory(inaccessiblemem: readwrite)` effects. A compatible declaration is
created only when a region is actually transformed; a conflicting symbol is
an error. The calls survive standard MLIR/LLVM lowering directly, so export
requires no extra matcher, scheduling pass or metadata reconstruction.
These are compiler scheduling hints, not additional GPU synchronization.
Backend wait insertion, register allocation and physical execution still need
hardware validation.

The production insertion point follows lane-sum analysis so that new guards
cannot hide its structured recurrence proof. Paired LDS, softmax fragment and
tail LDS scheduling now follow it in ThreadIR, before async/GPU lowering; see
`PairedLDS.md` for their order and lowering contracts.

## Reproduction

```sh
cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest LegacyLLVMExportTest -j 3
python3 3rd/deepgengraph/test/check_pv_interleave.py
python3 3rd/deepgengraph/test/check_mma_interleave.py
```

Standalone Thread IR replay:

```sh
3rd/deepgengraph/build/test/ThreadTilingTest --verify-each \
  --frisk-thread-interleave-pv='width=4' before.thread.mlir > after.thread.mlir

3rd/deepgengraph/build/test/ThreadTilingTest --verify-each \
  --frisk-thread-interleave-qk='width=2 guard-group=2 prefetch-words=32' \
  before.thread.mlir > after.thread.mlir
```

`generated_ir/thread_pv_interleave/noncausal/command.sh` reproduces the full
MyTest and O3 export using the tuning arguments and input from
`generated_ir/thread_lds_canonicalization_defer_fix/noncausal/`. The original
artifact remains available for GPU comparisons. Old experiment scripts that
apply `LegacyLLVMExportTest --pv-interleave` to .ll files are historical only.
