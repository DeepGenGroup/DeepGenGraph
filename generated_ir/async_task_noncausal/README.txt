Async task tokens, non-causal attention (2026-11-03)

async_copy_gm_to_shm now returns !frisk.async_task (C++ friskAsyncTaskType).
wait_buffer_ready consumes that task, independent of destination buffer aliases
and rotation. Both pipeline schedulers carry tasks through SSA iter_args/yields.
Inactive prefetch branches return async_task_completed. Storage planning counts
completion as a lifetime use even though waits no longer reference the buffer.

Fragment-level lifetime/scheduling update (2026-11-03):
  load_frag/store_frag now expose effects on their exact memory operands.
  Stage 06 runs conservative async scheduling and logical SHM analysis; all
  12 allocations are understood, and no physical pool is created here.
  Task completion analysis supports affine/SCF carriers, branch-local tasks
  and multi-slot FIFOs with proven head consumption and exit drains. It rejects
  conditional-only completion, missing drains, discarded tasks and unknown
  escapes. Cross-iteration tasks occupy the whole loop through their drains.
  After thread finalization and packing, lifetimes are recomputed. The physical
  plan still reduces the 12 allocations from 98304 bytes to a 16384-byte pool.
  06-logical-liveness.log and 08-physical-liveness.log contain both analyses.
  fragment_liveness_validation.json records 10 regression scripts for this
  update. numerical_validation.json records the regenerated kernel's results.

Inspect:
  01-fragment.mlir: semantic fragment IR with copy/wait task tokens
  06-optimized-fragment.mlir: early fragment optimizations and conservative async scheduling
  07-thread-bridges.mlir / 08-thread.mlir: thread lowering
  kernel.ll: MyTest output with the unchanged best non-causal parameters
  kernel.o3.ll: LLVM default<O3> output
  pipeline-example.mlir: 3-stage task FIFO carried by affine.for iter_args
  regression_validation.json: 14 passed regression scripts and their logs
  numerical_validation.json: actual LLVM wave/DMA numerical-model results

Reproduce from repository root:
  bash generated_ir/async_task_noncausal/command.sh > generated_ir/async_task_noncausal/lowering.log 2>&1
  python3 generated_ir/async_task_noncausal/save_stages.py
  /data2/xsl/install/bin/opt -S -passes='default<O3>' generated_ir/async_task_noncausal/kernel.ll -o generated_ir/async_task_noncausal/kernel.o3.ll
  python3 generated_ir/async_task_noncausal/validate.py

command.json preserves BHSD 1x32x4096x128, 256 threads, warps 4x1,
useKPack, asm128, retain 32 words, PV interleave 4, V double buffering,
K DMA 64-swizzle, K LDS prefetch 8, K tail prefetch 4, cross-BN K0,
softmax-fragments=pv. baseline.ll is semantic_fragment_noncausal/kernel.ll.
The historical measured reference is 1.6633449792861938 ms; no GPU timing was
performed for this change. /dev/kfd is unavailable in this environment.

The gfx936 backend still lowers readiness to wait.asyncmark 0 plus a block
barrier. Task SSA enables precise compiler dependencies; it does not introduce
selective hardware waits. Direct DMA tasks can cross iterations. Staged repack
still requires a local matched wait or the existing legal fallback. Function
token ABI and unsupported token consumers receive explicit diagnostics.
