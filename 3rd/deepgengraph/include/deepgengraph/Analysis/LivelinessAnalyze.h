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
/// in the function entry block for diagnostics. Interference additionally
/// resolves accesses inside collective loops when storage is iteration-local.
struct ShmLiveInterval {
  // allocation           这块 LDS 的 alloc。
  // aliases              指向它的所有 SSA 值（视图、select、循环 carrier 等）。
  // accesses             所有真实访问，含 DMA token 的使用者。
  // iterationLocalLoops  内容不跨回边的循环，由内到外。
  // sizeBytes/alignment  字节大小和对齐。
  // firstUse/lastUse     入口块中的首尾位置。
  // asyncAcrossIterations  DMA 跨迭代（FIFO 流水），不能算迭代局部。
  // reusable             是否参与复用。false 时 skipReason 给出原因。
  memref::AllocOp allocation;
  SmallVector<Value> aliases;
  SmallVector<Operation *, 0> accesses;
  SmallVector<Operation *, 0> iterationLocalLoops;
  uint64_t sizeBytes = 0;
  uint64_t alignment = 1;
  unsigned firstUse = 0;
  unsigned lastUse = 0; // Inclusive; equal endpoints interfere.
  bool asyncAcrossIterations = false;
  bool reusable = false;
  std::string skipReason;
};

struct ShmLivenessResult {
  // programOrder：入口块的 op 顺序，firstUse/lastUse 是它的下标。
  // buffers：所有 LDS alloc 的分析结果。
  SmallVector<Operation *> programOrder;
  SmallVector<ShmLiveInterval> buffers;

  /// Unsupported buffers always interfere. Indices address buffers above.
  bool interferes(unsigned lhs, unsigned rhs) const;
};

struct ShmPlacement {
  // bufferIndex：ShmLivenessResult::buffers 的下标。offsetBytes：在 pool 内的字节偏移。
  unsigned bufferIndex;
  uint64_t offsetBytes;
};

struct ShmReusePlan {
  // placements：每块 buffer 的放置结果。
  // originalBytes：不复用时的总字节。pooledBytes：复用后的 pool 大小。alignment：pool 的对齐。
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

/// Supports logical fragment accesses as well as finalized physical accesses.
/// Call again after scheduling, layout lowering and packing before physical reuse.
/// Handles
/// static identity-layout memref.alloc in address space 3; follows views, selects,
/// if results and for iter_args. Async task completion follows structured SSA
/// carriers, including loop backedges and exit drains. Unknown/escaping uses or
/// tasks without a guaranteed completion are excluded. Multi-
/// block functions and accesses inside unsupported regions are also excluded.
/// For full physical coverage, run after FinalizeThreadTiling: outstanding Frisk bridges
/// defer physical accesses and are intentionally treated as unsupported uses.
/// Analysis/plan contain IR handles and must be recomputed after modifying IR.
ShmLivenessResult analyzeShmLiveness(func::FuncOp kernel);

/// Deterministic size-first packing; interfering buffers never overlap in bytes.
ShmReusePlan planShmReuse(const ShmLivenessResult &liveness);

void dumpShmLiveness(const ShmLivenessResult &liveness, llvm::raw_ostream &os);

/// Protect shared-buffer reads from the next iteration's writes. Run after
/// physical accesses are exposed and before pooling. Unlike storage handoffs,
/// this also applies when a loop repeatedly uses a single allocation. Inserts
/// barriers only in provably block-uniform loops; returns the number inserted.
unsigned synchronizeSharedMemoryLoopBackedges(func::FuncOp kernel);

/// True only in the function entry or provably block-uniform enclosing loops.
/// Does not accept conditional regions or infer uniformity from a barrier.
bool isInBlockUniformControlFlow(Operation *op);

/// Recompute analysis, pack into memref<Nxi8, 3>, replace allocations with typed
/// memref.view and insert gpu.barrier at storage handoffs as needed.
/// Requires a function executed collectively by a thread block (thread_num attr).
/// Applies only if bytes decrease. Returns true iff IR changed. Existing pool
/// allocations are excluded, making repeated calls safe. Loop-body barriers
/// require provably block-uniform loop bounds and collective enclosing control
/// flow. Conditional regions remain atomic. Typed fragments and unfinalized
/// thread-tile bridges defer pooling, so an early pool cannot lock out later
/// scratch allocations or fix storage before physical scheduling.
bool reuseSharedMemory(func::FuncOp kernel, ShmReuseStats *stats = nullptr);

} // namespace mlir::frisk

#endif // DEEPGENGRAPH_ANALYSIS_LIVELINESSANALYZE_H
