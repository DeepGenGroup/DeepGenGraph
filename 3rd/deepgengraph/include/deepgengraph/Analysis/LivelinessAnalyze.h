#ifndef DEEPGENGRAPH_ANALYSIS_LIVELINESSANALYZE_H
#define DEEPGENGRAPH_ANALYSIS_LIVELINESSANALYZE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mlir::frisk {

/// Storage liveness, not SSA descriptor liveness. Positions refer to operations
/// in the function entry block. A nested access covers its entire enclosing
/// top-level operation, including all loop iterations and both conditional arms.
struct ShmLiveInterval {
  memref::AllocOp allocation;
  SmallVector<Value> aliases;
  uint64_t sizeBytes = 0;
  uint64_t alignment = 1;
  unsigned firstUse = 0;
  unsigned lastUse = 0; // Inclusive; equal endpoints interfere.
  bool reusable = false;
  std::string skipReason;
};

struct ShmLivenessResult {
  SmallVector<Operation *> programOrder;
  SmallVector<ShmLiveInterval> buffers;

  /// Unsupported buffers always interfere. Indices address buffers above.
  bool interferes(unsigned lhs, unsigned rhs) const;
};

struct ShmPlacement {
  unsigned bufferIndex;
  uint64_t offsetBytes;
};

struct ShmReusePlan {
  SmallVector<ShmPlacement> placements;
  /// Totals cover eligible allocations only; skipped buffers are unchanged.
  uint64_t originalBytes = 0;
  uint64_t pooledBytes = 0;
  uint64_t alignment = 1;
};

struct ShmReuseStats {
  uint64_t originalBytes = 0;
  uint64_t pooledBytes = 0;
  unsigned reusedAllocations = 0;
  unsigned insertedBarriers = 0;
};

/// Call after pipeline scheduling, LowerInfo inference and thread tiling. Handles
/// static identity-layout memref.alloc in address space 3; follows views, selects,
/// if results and for iter_args. Unknown/escaping/async uses are excluded. Multi-
/// block functions and accesses inside unsupported regions are also excluded.
/// For full coverage, run after FinalizeThreadTiling: outstanding Frisk bridges
/// defer physical accesses and are intentionally treated as unsupported uses.
/// Analysis/plan contain IR handles and must be recomputed after modifying IR.
ShmLivenessResult analyzeShmLiveness(func::FuncOp kernel);

/// Deterministic size-first packing; interfering buffers never overlap in bytes.
ShmReusePlan planShmReuse(const ShmLivenessResult &liveness);

void dumpShmLiveness(const ShmLivenessResult &liveness, llvm::raw_ostream &os);

/// Recompute analysis, pack into memref<Nxi8, 3>, replace allocations with typed
/// memref.view and insert entry-block gpu.barrier at storage handoffs as needed.
/// Requires a function executed collectively by a thread block (thread_num attr).
/// Applies only if bytes decrease. Returns true iff IR changed. Existing pool
/// allocations are excluded, making repeated calls safe. No barrier is inserted
/// inside a conditional or a loop. Skipped allocations remain untouched.
bool reuseSharedMemory(func::FuncOp kernel, ShmReuseStats *stats = nullptr);

} // namespace mlir::frisk

#endif // DEEPGENGRAPH_ANALYSIS_LIVELINESSANALYZE_H
