# Generic MMA accumulator interleaving in Thread IR

The canonical pass is `frisk-thread-interleave-mma-chains`. It recognizes
independent accumulator SSA chains in ordinary GEMM as well as attention;
chain discovery does not require a matrix-load producer.

```sh
build/test/ThreadTilingTest input.mlir \
  '--frisk-thread-interleave-mma-chains=width=2 guard-group=2 operand-source=any load-motion=none'
```

`operand-source=any|register|lds|matrix` selects available register values or
supported operand packets. `load-motion=none` requires memory packets before
the candidate MMAs. `bounded` can move ordinary LDS packets only within a
proven memory/synchronization-free window; matrix reads must remain available.
`prefetch-words` bounds distinct B-packet footprint when bounded motion is
requested. Packet size comes from its type; this is not a total VGPR limit.

The old PV/QK CLI names are pipeline aliases that map to
`operand-source=matrix load-motion=none` and
`operand-source=lds load-motion=bounded`, respectively. MyTest retains its
existing tuning flags and maps them to the same canonical options. All newly
scheduled MMAs use `frisk.mma_interleave`; old markers are recognized on input.
Both generic passes share a final local schedule-plan validation before motion.

`MyTest --pv-interleave=N` runs `frisk-thread-interleave-pv` after thread
unrolling, address canonicalization and lane-sum rewriting, before async/GPU
lowering. `ThreadInterleaveMMAChains.cpp` owns matching, legality and scheduling.
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

`frisk.mma_interleave` marks scheduled MMACs, making replay idempotent. Apply
alternative widths to the original input, not already transformed output.

## Scheduling guards and lowering

The module pass emits `frisk.sched_barrier` with mask 0 at the configured group
boundaries (by default, before/after each PV MMAC). LDS prefetch and softmax
scheduling use the same operation. Its compiler-private `frisk.schedule`
read/write effect retains it through MLIR CSE/DCE without claiming a program
memory access. Thread schedulers conservatively respect every mask as a boundary.

`frisk-lower-sched-barrier`, also included in production `secondLowering`, emits
`llvm.amdgcn.sched.barrier(mask)` calls with explicit
`memory(inaccessiblemem: readwrite)` effects and preserves origin markers.
It reuses a compatible declaration, rejects conflicting symbols and unsupported
targets, and creates the declaration only when there are barriers to lower.
Existing `frisk.scheduled_mma` anchors retain their LLVM ordering contract.
LLVM-only export optimizations continue to emit the target intrinsic directly.

```mlir
frisk.sched_barrier                      // mask 0: no instruction class crosses
frisk.sched_barrier {mask = 6 : i32}      // allow VALU and SALU to cross
frisk.sched_barrier {mask = 1026 : i32}   // 0x402: allow TRANS and VALU
```

The mask is an **allow** mask, not a blocked-class mask. Supported bits are
0x001 (ALU), 0x002 (VALU), 0x004 (SALU), 0x008 (MFMA/WMMA), 0x010 (VMEM),
0x020 (VMEM read), 0x040 (VMEM write), 0x080 (DS), 0x100 (DS read),
0x200 (DS write), 0x400 (TRANS), and 0x800 (LDSDMA). Bits may be combined;
unknown bits are rejected. The backend interprets classes after instruction
selection; a high-level op or inline asm need not map to a single class.
The mask does not itself pin arbitrary pure computations through LLVM O3.
These are compiler scheduling hints, not additional GPU synchronization or waits.
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
