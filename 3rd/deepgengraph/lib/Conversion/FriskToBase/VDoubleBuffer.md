# Shared-memory execution planning

`MyTest --v-double-buffer=1` enables logical stream planning in
`frisk-auto-pipeline=double-buffer=true`, followed by physical validation and
materialization in `lower-frisk-fragments=double-buffer=true`. The independent
`frisk-thread-double-buffer-v` pass has been removed. LLVM does not choose
buffer slots or copy issue times.

## Planning boundary

When either buffering or cross-iteration planning is enabled, AutoPipeline
first fully unrolls marked inner affine loops (at most 64 trips, cumulative
65536 extra operations). Loops owned by a hint or explicit schedule are left
to that owner. The outer compute loop remains rolled. This exposes the QK
opening copy before planning, while preserving its accumulator recurrence.
Copies from different source storage are analyzed separately; the current
two-slot contract selects one eligible tail stream per block. Other copies
remain subject to dependency/alias-checked scheduling, not a new global search.

The early layout phase determines packing, XOR composition, DMA width,
participants and source/destination maps. The execution phase runs after these
maps are materialized and bounded fragment loops are exposed, **before** the
logical allocations become views of an LDS pool.

`AutoPipeline` identifies straight-line full-tile global-to-shared copies with
direct GEMM B consumers, common source storage and matching tile shapes. It
records `frisk.shared_pipeline_plan = array<i64: stage, count, slot, issueAfter>`
on each copy. Stage IDs are block-local, slots alternate 0/1, and `issueAfter`
names the preceding stage's exact task readiness (`-1` for the first issue,
which retains ordinary alias-checked hoisting across preceding compute).
This is a provisional logical schedule: serial stream order and
independent logical allocations remain intact until physical validation. Thus
budget or packet rejection retains a valid serial fallback. Generic DMA/wait
scheduling preserves interior copies and every exact readiness operation. It
may hoist the unanchored first copy, with its pure address dependencies, across
independent compute; it still cannot cross another DMA, wait, barrier, unknown
effect or aliasing mutation. Freezing this first launch until immediately before
its wait would lose the QK/softmax overlap window even with valid V ping-pong.

`SharedMemoryPlan::execution` consumes that schedule and owns:

- Matrix-copy stages, their exact completion tasks and all matrix readers.
- Validation of the declared stage order, logical slot IDs and issue anchors;
  it never discovers or schedules an unmarked V stream.
- Pure source-address dependencies needed at each issue anchor.
- Lifetimes projected onto the proposed issue order.
- Physical pool offsets and collective storage-handoff barriers.
- An input snapshot that must still match before any materialization.

Planning is read-only. The allocator sees each later stage's first write at the
operation immediately after the preceding readiness event. Thus adjacent
stages interfere; stages two positions apart may reuse storage. The ordinary
allocator resolves interference with all other live buffers, including Q/K.
The initial policy accepts only two alternating slots within the unbuffered
plan's LDS budget. Otherwise it retains ordinary scheduling and allocation.

`materializeExecution()` validates the snapshot before mutation. It creates
only the planned pool/views/barriers and moves copies to the planned anchors.
It does not search for spare storage or re-run layout/scheduling decisions.
Static destination views dominate the relocated copies. The
`frisk.shared_pipeline` attributes identify materialized copy/wait/read stages
and prevent subsequent async scheduling from changing their order.

## Synchronization and eligibility

The next copy starts **after** current readiness: this backend's readiness
operation waits for DMA completion and synchronizes the block. It also releases
old readers of the alternate slot. Each stage retains its own readiness event,
and loop-backedge reader protection remains in place until the final ThreadIR
barrier-elision check.

Candidates require 2–32 stages with distinct reusable logical allocations,
matching sizes and a common global source. Physical addresses must prove full,
aligned wave transfers at the planned 4/8/16-byte width. All aliases/accesses
of a stage buffer must belong to its copy, completion and matrix readers.
Per-lane matrix footprints and bounded inner affine loops are checked.
Uniform control flow, exact local completion, available pure source-address
expressions and final reader synchronization are required. Unknown effects,
additional readers/writers, explicit intervening synchronization, incompatible
existing schedules and insufficient pool capacity reject the candidate.

The `Shared memory pipeline:` diagnostic prints stages, slots and LDS usage.
The historical `V double buffer:` schema is also emitted for tuning scripts;
its `removed-barriers` field now counts serial handoff barriers avoided by the
planned overlap, rather than barriers inserted and subsequently deleted.
An enabled option does not by itself establish that a pipeline was selected.

## Validation

```sh
cmake --build 3rd/deepgengraph/build \
  --target MyTest ThreadTilingTest SharedExecutionPlanTest -j4
python3 3rd/deepgengraph/test/unittest/check_v_double_buffer.py
python3 3rd/deepgengraph/test/unittest/check_shm_liveness.py
python3 generated_ir/shared_memory_plan_pipeline/generate.py
```

The early planner regression is
`python3 3rd/deepgengraph/test/unittest/check_auto_pipeline_v_stream.py`.
The execution regression includes a read-only/stale-plan C++ test and an async
ownership model over the emitted pool offsets, task tokens, waits and readers.
The model repeats the stream across a loop backedge and distinguishes source
slice contents. It does not simulate the GPU ISA, concurrent wave timing or
throughput. Full attention LLVM verification/address comparisons and target
GPU numerical/performance measurements remain separate checks.

Physical fixtures explicitly supply the early-plan contract; missing, malformed,
incomplete, reordered or incompatible contracts reject the entire candidate.

The integration artifacts are in `generated_ir/shared_memory_plan_pipeline/`.
For both causal and noncausal attention, O3 LLVM is identical to
`shared_memory_plan_wave_bases` after substituting artifact directory paths:
four stages, two 8192-byte slots, 16384-byte LDS. The per-thread address check
also matches all 32 matrix and 80 ordinary LDS reads across 256 threads.
`verification/summary.json` records both input hashes and validation logs.
Target code generation and GPU performance have not been measured here.
On the target machine, run `bash generated_ir/shared_memory_plan_pipeline/run_gpu.sh`
with the usual toolchain environment; set `ATTENTION_KIND=causal` for causal
attention. This compares the before/after kernels in both run orders.

## Cross-iteration first-copy prefetch

`MyTest --cross-bn-k0=1` enables `cross-iteration=true` in both AutoPipeline
and lower-frisk-fragments. AutoPipeline selects the opening induction-dependent
matrix-B copy and the final matrix stream's readiness in an outer affine/scf
loop. It records unit attributes `frisk.cross_iteration_plan` on the loop,
`frisk.cross_iteration_issue` on the opening DMA and
`frisk.cross_iteration_anchor` on the tail DMA. These are provisional: the
original task SSA and serial ordering remain valid until physical validation.
The tail issue/readiness is protected during intervening scheduling.

The execution planner consumes these identities, recovers readiness from each
DMA's exact task, and validates circular lifetimes together with V buffering
and physical allocation. It no longer discovers unmarked cross-iteration
candidates. A rejected candidate retains the ordinary schedule; MyTest does
not retry the transformation in LLVM. Physical fixtures must carry the early
contract, just as V buffering fixtures do.

The projected lifetime has two parts: the loop prefix through the current
buffer's last reader, and the suffix from the proposed issue point through the
backedge. The allocator may reuse the middle but must keep the final readers
disjoint from the prefetched buffer. Handoff planning protects both transitions
inside the iteration. The initial policy preserves the existing LDS budget and
any already selected alternating stage offsets.

Materialization guards the prologue against zero trips, carries the exact task
token through the loop, and guards the next issue with the original upper bound
and step. The inactive path yields `async_task_completed`. An explicit exit
drain retains a locally checkable completion path, including zero trips. The
prologue has a reader handoff before issuing the first copy. Unknown memory
effects, global writes in the loop, nonuniform bounds, unavailable source
addresses, unsupported DMA and overlapping final readers reject the candidate.

Only scheduling protection remains in LLVM on the main export path:
`anchorNextKSums` anchors the scalar FP32 recurrence additions before register
lifetime optimization. The standalone legacy LLVM exporter retains the old
transformation for compatibility and regression comparisons. The optional
`frisk-thread-elide-backedge-barrier` pass runs after the final ThreadIR LDS
schedule, before `frisk-lower-async-copy`. MyTest's existing
`--elide-backedge-barrier=0|1` switch controls it (default: enabled). There is no
LLVM backedge-barrier pass or legacy-exporter elision option.

The pass requires a block-uniform affine/scf loop marked by the shared execution
plan. It removes only a trailing `gpu.barrier` when the next iteration starts
with readiness for a loop-carried task, and the loop exit drains that exact task
result. Only region-free, pure operations and register-only GPU shuffles may
separate the barriers. Memory accesses, DMA, unknown effects and control regions
reject removal. Readiness waits and their publication barriers are preserved,
including the exit drain for zero trips. Run this after the backedge insertion
and memory scheduling passes; scheduling memory across the proven interval or
re-running unconditional backedge insertion invalidates the placement contract.

The regression checks negative hazards and task mismatches, preserves waits
through lowering, and runs the cyclic DMA ownership model on affine/scf loops
with zero, one, multiple and dynamic trips:

```sh
python3 3rd/deepgengraph/test/unittest/check_thread_backedge_barrier.py
```

Additional validation:

```sh
python3 3rd/deepgengraph/test/unittest/check_shared_cross_iteration.py
python3 3rd/deepgengraph/test/unittest/check_cross_bn_k0.py
```
