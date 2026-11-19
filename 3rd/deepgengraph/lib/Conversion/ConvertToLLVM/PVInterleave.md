# PV accumulator interleaving in Thread IR

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

## Representation and legality

The pass accepts the exact register-only `v_mmac_f32_16x16x16_f16` template
and its validated padded form, either as `frisk.warp_mma_rr` or as the
`llvm.inline_asm` produced by current thread legalization. The latter has
`has_side_effects` until LLVM export; recognizing the entire instruction,
constraints and operand types establishes the same register-only contract as
`prepareRegisterMMAForLLVM`. Arbitrary assembly and memory clobbers are rejected.

B operands must be transparent vector views (`bitcast`, `shape_cast`,
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

`frisk.pv_interleave` marks scheduled MMACs, making replay idempotent. Apply
alternative widths to the original input, not already transformed output.

## Scheduling guards and lowering

The module pass emits typed LLVM-dialect calls to
`llvm.amdgcn.sched.barrier(0)` before/after each MMAC, with explicit
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
```

Standalone Thread IR replay:

```sh
3rd/deepgengraph/build/test/ThreadTilingTest --verify-each \
  --frisk-thread-interleave-pv='width=4' before.thread.mlir > after.thread.mlir
```

`generated_ir/thread_pv_interleave/noncausal/command.sh` reproduces the full
MyTest and O3 export using the tuning arguments and input from
`generated_ir/thread_lds_canonicalization_defer_fix/noncausal/`. The original
artifact remains available for GPU comparisons. Old experiment scripts that
apply `LegacyLLVMExportTest --pv-interleave` to .ll files are historical only.
