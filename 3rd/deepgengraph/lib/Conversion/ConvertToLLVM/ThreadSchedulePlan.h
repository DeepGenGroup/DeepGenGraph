#ifndef DEEPGENGRAPH_THREAD_SCHEDULE_PLAN_H
#define DEEPGENGRAPH_THREAD_SCHEDULE_PLAN_H
#include "ThreadScheduleUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::frisk::thread_schedule {
inline constexpr StringLiteral MMAInterleaveMarker = "frisk.mma_interleave";
inline constexpr StringLiteral LDSPairMarker = "frisk.lds_pair";
inline bool protectedLDSPair(Operation *op) {
  return op->hasAttr(LDSPairMarker) || op->hasAttr("frisk.k_lds_pair") ||
         op->hasAttr("frisk.k_tail_pair");
}
// Clients prove effects, readiness, participation and budgets. This final
// check covers membership and SSA availability in a contiguous window.
struct LocalSchedulePlan {
  ArrayRef<Operation *> original;
  ArrayRef<Operation *> order;
  bool validate() const {
    if (original.empty() || order.size() != original.size()) return false;
    Block *block = original.front()->getBlock();
    if (!original.back()->getNextNode()) return false;
    llvm::SmallPtrSet<Operation *, 32> members;
    for (Operation *op : original)
      if (op->getBlock() != block || !members.insert(op).second) return false;
    Operation *next = original.front();
    for (Operation *op : original) {
      if (op != next) return false;
      next = next->getNextNode();
    }
    DenseMap<Operation *, unsigned> positions;
    for (auto [n, op] : llvm::enumerate(order))
      if (!members.contains(op) || !positions.try_emplace(op, n).second) return false;
    for (Operation *op : order)
      for (Value value : op->getOperands()) {
        Operation *def = value.getDefiningOp();
        if (!def || def->getBlock() != block) continue;
        auto position = positions.find(def);
        if (position != positions.end()) {
          if (position->second >= positions[op]) return false;
        } else if (!def->isBeforeInBlock(original.front())) return false;
      }
    return preservesScheduleBarriers(original, order);
  }
  void apply() const {
    Operation *end = original.back()->getNextNode();
    for (Operation *op : order) op->moveBefore(end);
  }
};
} // namespace mlir::frisk::thread_schedule
#endif
