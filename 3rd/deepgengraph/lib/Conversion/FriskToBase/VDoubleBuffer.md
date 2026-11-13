# Thread IR V double buffering in existing LDS

`frisk-thread-double-buffer-v` is an opt-in MLIR function pass. `MyTest
--v-double-buffer=1` runs it after `LowerFriskFragments`, buffer hoisting,
canonicalization and CSE, before Thread IR legalization and DMA lowering.
The LLVM implementation and export hook have been removed. The legacy LLVM
export test rejects this option with a migration diagnostic.

This first migration intentionally runs after physical pooling: the pass
borrows an adjacent slice from the existing allocation and never creates or
resizes an allocation. Thus the original storage plan's capacity is preserved
without relying on a second allocator run to recover the old memory usage.
Ordinary async scheduling does not run again on this pooled storage.

## Transformation

Identify 2–32 V slices from exact `async_copy_gm_to_shm` task identities,
`wait_buffer_ready`, and `read_matrix_from_shm` consumers. Alternate even and
odd slices between the original and spare slots. Move each next issue, along
with only its pure descriptor/index dependencies, immediately after the
current ready operation. Remove the intervening reader handoff barrier.
Keep every ready operation and the final reader barrier. Register computations
and all accumulator edges stay in their original order.

The ready operation publishes the current DMA and releases old readers of
the other slot. The next issue must stay AFTER it: the current backend uses
wait-all. The final reader barrier protects reuse on the outer-loop backedge.
The `frisk.v_double_buffer` read attribute makes the pass idempotent.

## Checks before mutation

- One static, aligned, identity-layout byte LDS pool; fixed 1D wave64 workgroup
  with 64–1024 threads. Static slot sizes and displacement come from descriptors.
- Full-wave 16-byte direct DMA, bounded constant rounds, source/destination
  alignment, the existing exact source-map/packet legality analysis, uniform
  descriptors, and identical underlying global source for the V stream.
  Staged copies, synchronous fallback and unknown destination aliases reject.
- Exact per-lane matrix addresses, including bounded static inner affine loops.
  Unknown coordinates and reads outside the copied slice reject. Descriptor
  extents must also fit, so relocated typed views stay inside the allocation.
- Block-uniform control flow, one outer compute loop, exact local task waits,
  reader/handoff/issue order and a final reader barrier. Explicit pipelines
  and fragment lookahead schedules are not rescheduled.
- No unknown effects or other shared users in the changed region. Earlier
  same-iteration readers of the spare must follow a completed overwrite in
  that iteration. Prologue readers must precede the compute loop; shared
  readers after V or in unrelated regions reject. Pending unrelated DMA
  cannot remain in flight into the changed region.
- Every relocated dependency must already dominate the new issue point or
  be a same-block pure, speculatable operation. Never clone a load, another
  DMA, an MMA or a convergent operation to make its address available.

Unknown aliases and effects fail closed, with no partial rewrite. Log records
retain the existing `V double buffer:` format for `TuneAttention.py`, including
slice count, slot bytes, displacement, removed barriers and unchanged LDS bytes.
An enabled option is not evidence of a successful match.

## Validation and reproduction

```sh
cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest LegacyLLVMExportTest -j3
python3 3rd/deepgengraph/test/check_v_double_buffer.py
python3 generated_ir/thread_v_double_buffer/generate.py
OPENBLAS_NUM_THREADS=1 python3 generated_ir/thread_v_double_buffer/verify.py
```

The generated directory preserves both user-selected parameter families,
input snapshots, commands, raw and O3 LLVM, launch sidecars, pass reports and
validation records. GPU numerical/performance validation is separate from
LLVM verification and the CPU delayed-DMA model.
