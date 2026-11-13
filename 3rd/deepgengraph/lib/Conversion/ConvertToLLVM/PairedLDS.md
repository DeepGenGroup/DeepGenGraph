# Paired K/B LDS prefetch

`--k-lds-prefetch=0..8` controls a final LLVM function pass. Zero is the
default and leaves the export unchanged. A nonzero value is the maximum
number of independent MMAC instructions to move a read pair ahead of its
first consumer. `TuneAttention.py` accepts the same key with `--set` or
`--sweep`. `online-vdb-k64-lds2` enables distance 2 on the existing DMA64
configuration; the older `online-vdb-k64` preset remains unchanged.

The pass runs after PV interleaving, V buffering and K DMA/layout conversion.
Enabled runs perform EarlyCSE, InstCombine and EarlyCSE before scheduling,
to eliminate repeated reads and expose common pointer bases. It recognizes
ordinary aligned `<4 x half>` LDS loads used exclusively as the B operand
of supported register MMACs. In attention these are K loads; in another
GEMM they may be B. Matrix V readers and the A operand are not selected.

Within a single basic block, DMA, stores, fences, hardware waits/barriers,
wave operations, existing scheduling hints and unknown operations split
the movable regions. Loads never move across these boundaries. Pairing
requires the same pointer base and an encodable constant byte difference:
normal read2 offsets are preferred, followed by read2st64. Ties use the
first MMAC consumer. Neither SSA names, old assembly register numbers,
fixed tile sizes, nor global argument positions select the pairs.

All selected pairs in a region are first closed by hoisting the later
read and its pure address dependencies. Complete pairs are then hoisted
before an independent MMAC, without delaying any load. Before committing,
the pass checks SSA order and that no scheduling hint splits any selected
pair. It brackets each anchor with `llvm.amdgcn.sched.barrier(0)`; these
are compiler scheduling hints, not workgroup synchronization. It preserves
MMAC order, floating-point flags, DMA and hardware synchronization order,
and shared-memory capacity. Unsupported movements are skipped and reported.

The deterministic address-based pairs can differ from the historical
assembly-derived pairing table. The optimization implements paired prefetch
and protection, not a promise of bit-identical target assembly. Target
code generation still decides read2 selection, waits, registers and speed.

The function attribute `frisk.k_lds_prefetch` records the applied distance.
Repeated application with the same distance is a no-op. Conflicting explicit
reapplication fails instead of silently stacking scheduling hints.

Tests: `python 3rd/deepgengraph/test/check_paired_lds.py` and
`python -m unittest script.test_tune_attention`. Attention IR and numerical
validation are saved under `generated_ir/noncausal_k_lds_prefetch_pass/`.

## Completion across local reductions

`--k-tail-prefetch=0..8` enables `PrefetchPairedLDSTailPass`, independently of
the original option. The default is zero. It runs after the original K LDS
prefetch and softmax-fragment scheduling in both `MyTest` and
`LegacyLLVMExportTest`; `TuneAttention.py` accepts it in `--set` and `--sweep`
and records `k_tail_prefetch_reports` in the IR facts. For example, retain
`--k-lds-prefetch=8` and add `--k-tail-prefetch=4`.

The name refers to the motivating QK tail, not a hard-coded loop position.
The pass examines every basic block and finds unprotected aligned b64
MMAC B loads in regions containing local FP max/min or addition. It allows
`extractelement`, `insertelement`, `shufflevector`, `extractvalue`,
`insertvalue`, ordinary `fadd`, and unconstrained `maxnum`, `minnum`,
`maximum`, `minimum` intrinsics to remain in a region. These operations do
not move or change their operands, flags or evaluation order. Only whole
read pairs and their pure address dependencies are hoisted, using the same
address-based matching and SSA checks as the original pass. No kernel name,
SSA name, tensor shape, fixed LDS capacity, 128-byte spacing, or particular
softmax layout is required. Supported MMAC operand types and aligned LDS
load types still follow the b64 register-MMAC matcher described above.

Unlike the original entry point, completion does not run CSE/InstCombine
or rebuild an already measured IR. Existing `frisk.k_lds_pair` and
`frisk.k_tail_pair` loads are boundaries, as are all existing scheduling
hints, other memory accesses, DMA, fences, waits, hardware barriers, wave
operations, constrained FP and unknown calls, even unknown readnone calls.
Regions without a recognized local reduction are not scheduled. `strictfp`
and `optnone` functions are skipped. The pass adds no hardware synchronization
and does not move instructions across basic blocks.

New pairs use `frisk.k_tail_pair`; new scheduling guards and the applied
function distance use `frisk.k_tail_prefetch`. Successful repeated application
with the same distance is a no-op; a conflicting distance is rejected by the
module entry point. To compare distances, start from the original input.
Skipped functions are not marked. Diagnostics report successful regions,
pairs, moved pairs, anchors and the number of examined reduction regions.
The first pair may still require an immediate wait: it has no earlier
independent MMAC inside its region.

The measured artifact had `1.707992 ms` before completion and `1.691346 ms`
with distance 4 (about 0.97% lower); distances 2 and 8 were also faster in
the supplied `pmc_test.log`. These are artifact measurements, not a claim
that the transformation improves every kernel. The C++ pass reproduces the
2/4/8 experiment's complete instruction sequence, operands and flags on
that baseline, ignoring metadata and attribute numbering after LLVM22 text
legalization. The pass versions still need target GPU measurements.

Tests: `python3 3rd/deepgengraph/test/check_paired_lds_tail.py`, the original
paired-LDS and softmax-fragment regressions, and
`python3 -m unittest script.test_tune_attention`. Pass-exported IR,
structural comparisons, CPU numerical/LDS-epoch validation, GPU test scripts
and the full `MyTest` smoke path are in
`generated_ir/noncausal_qk_tail_prefetch_pass/`.

## Cross-iteration K0 prefetch with sum lifetime constraints

`--cross-bn-k0=0|1` enables `PrefetchNextKPass` (default 0). It runs last,
after V double buffering, K DMA layout, paired LDS prefetch, softmax fragment
scheduling and QK-tail completion in `MyTest` and `LegacyLLVMExportTest`.
`TuneAttention.py` forwards the option through `--set`/`--sweep` and records
`cross_bn_k0_reports`. The tested combination keeps `--k-tail-prefetch=4`
and adds `--cross-bn-k0=1`.

The pass implements the measured `k0_sum_anchor.ll` strategy, not the optional
address-rematerialization or delayed-issue experiments. The supplied log
records 1.663345 ms for that strategy versus 1.691346 ms for QK-tail-only.
Earlier cross-iteration prefetch without sum constraints used 256 VGPR and
four spilled values: all 64 denominator FP32 additions were sunk into the
block following the new prefetch guard. The pass therefore combines the
DMA movement with tied-register empty-asm identities after the exact scalar
FP32 additions in loop-carried sum recurrences. The identities are convergent,
nounwind, willreturn and memory(none), with side effects preventing deletion
or sinking of the complete reduction across the guard. FP operands, flags,
order, MMAC order, LDS size, waits and hardware barriers are preserved.

Matching is independent of SSA/kernel names, head count, fixed tile sizes,
packet counts, the numeric loop bound and the absolute LDS slot displacement.
The supported export is an AMDGPU wave64 kernel with a fixed 1D workgroup,
one static LDS allocation, and a single two-block loop (condition/PHIs plus
compute/latch). The loop starts at zero and advances by one. Its invariant
bound must be available in the entry block and proven nonnegative by LLVM
known-bits analysis. Both `IV < bound` (with a proven nonzero bound) and
`IV <= bound` are supported, including inverted exit tests and swapped
comparison operands. This covers causal `IV <= queryTile` without adding
one to the bound or speculatively reading a future tile. It is a final-export
pass before O3 loop canonicalization; already-folded one-block loops,
potentially zero-trip or unproven bounds, nested loops, nonuniform control, strictfp/optnone functions
and unsupported memory/address expressions are skipped with diagnostics.

The first exact buffer_load_dwordx2/x4-to-LDS group is K0. The pass evaluates
the restricted LDS address DAG for every workgroup lane, including swizzles
and readfirstlane, using the shared LDS address evaluator. It rejects
overlapping packet writes, unknown addresses and any overlap with final-PV
reads. It requires the original K0 ready wait/publication, final-PV ready
wait/publication, final reader handoff and a released prologue LDS region.
Unknown calls, memory effects, volatile/atomic reads and unexpected writes
are rejected before mutation. The new DMA source is the original address DAG
with the induction value replaced by its successor, not an assumed tile-byte
offset. Prologue priming uses zero; future issue reuses the loop's unsigned
comparison (`next < bound` or `next <= bound`).
No extra iteration, global load, workgroup barrier or LDS allocation is added.

The sum constraints are discovered by walking loop-carried SSA values back
through arithmetic/aggregate wrappers, stopping at loop PHIs and MMAC/exp/max
calls. They do not depend on temporary names or local-reduction metadata.
Successful application is marked `frisk.cross_bn_k0="sum-anchor-v1"`; repeated
application is a no-op. New constraints carry `frisk.cross_bn_sum_anchor`.
Failed matching does not mark or partially rewrite the function.

Tests: `python3 3rd/deepgengraph/test/check_cross_bn_k0.py`, the V-double-buffer,
softmax-fragment, paired-LDS and tail regressions, and the tuning-script tests.
Pass-exported IR, comparison against the measured instruction graph (ignoring
SSA names, debug metadata and attribute numbering), CPU numerical/delayed-DMA
checks, a complete MyTest smoke export and GPU scripts are under
`generated_ir/noncausal_cross_bn_k0_pass/`. Target register allocation and
performance still require DTK/GPU validation; the GPU script checks scratch
and spill before benchmarking.
