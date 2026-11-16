#ifndef DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_ASYNCDMA_H
#define DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_ASYNCDMA_H

namespace mlir::frisk {
// Upstream global.load.async.lds does not accept 8B. Preserve a valid,
// convergent side-effecting call until buffer-asm export emits dwordx2.
// Arguments: source pointer, wave LDS base, bytes per lane, offset, aux.
// This is an emission bridge, never a runtime dependency in exported kernels.
inline constexpr char PlannedAsyncDMA[] = "__frisk_async_load_lds";
}

#endif
