// Remove redundant reader handoffs after all ThreadIR memory scheduling.
// SharedMemoryPlan has already proved slot ownership for next-iteration DMA.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "ThreadScheduleUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace mlir::frisk {
namespace {
constexpr StringLiteral CrossMarker = "frisk.cross_bn_k0";

Block *loopBody(Operation *op) {
  if (auto loop = dyn_cast<affine::AffineForOp>(op)) return loop.getBody();
  if (auto loop = dyn_cast<scf::ForOp>(op)) return loop.getBody();
  return nullptr;
}

// No regions, calls, memory accesses or task issue/synchronization operations may
// hide between the removed barrier and either replacement synchronization.
// gpu.shuffle is a register-only wave collective used by lane-sum deferral.
bool registerOnly(Operation *op) {
  return !isa<CallOpInterface>(op) &&
         (thread_schedule::pure(op) || isa<gpu::ShuffleOp>(op));
}

struct ThreadElideBackedgeBarrierPass
    : PassWrapper<ThreadElideBackedgeBarrierPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadElideBackedgeBarrierPass)
  StringRef getArgument() const final { return "frisk-thread-elide-backedge-barrier"; }
  StringRef getDescription() const final {
    return "Remove reader barriers covered by cross-iteration readiness and exit drain";
  }
  void runOnOperation() final {
    auto function = getOperation();
    auto marker = function->getAttrOfType<StringAttr>(CrossMarker);
    if (function.isExternal() || !llvm::hasSingleElement(function.getBody()) ||
        !marker || marker.getValue() != "shared-plan-v1" ||
        thread_schedule::protectedFunction(function)) return;
    for (Operation &loop : function.getBody().front()) {
      Block *body = loopBody(&loop);
      if (!body || !loop.hasAttr(CrossMarker)) continue;
      auto skip = [&](StringRef why) {
        llvm::errs() << "Thread elide backedge barrier: skipped, function="
                     << function.getSymName() << ": " << why << '\n';
      };
      if (!isInBlockUniformControlFlow(body->getTerminator())) {
        skip("requires a block-uniform loop"); continue;
      }

      Operation *tail = body->getTerminator()->getPrevNode();
      while (tail && registerOnly(tail)) tail = tail->getPrevNode();
      if (!tail || !isa<gpu::BarrierOp>(tail)) {
        skip("loop does not end in a reader barrier"); continue;
      }

      Operation *first = &body->front();
      while (first != body->getTerminator() && registerOnly(first))
        first = first->getNextNode();
      auto head = dyn_cast<WaitBufferReadyOp>(first);
      if (!head) { skip("LDS, DMA or control before opening readiness"); continue; }
      auto task = dyn_cast<BlockArgument>(head.getToken());
      if (!task || task.getOwner() != body || task.getArgNumber() == 0 ||
          task.getArgNumber() > loop.getNumResults()) {
        skip("opening readiness must consume a loop-carried task"); continue;
      }

      // In particular, zero trips and the last iteration must retain the
      // planner's explicit drain. Never infer that a completed token makes its
      // block barrier redundant: it also releases readers before later users.
      Operation *after = loop.getNextNode();
      while (after && registerOnly(after)) after = after->getNextNode();
      auto drain = dyn_cast_or_null<WaitBufferReadyOp>(after);
      if (!drain || drain.getToken() != loop.getResult(task.getArgNumber() - 1)) {
        skip("requires the matching exit drain before memory effects"); continue;
      }
      tail->erase();
      llvm::errs() << "Thread elide backedge barrier: removed 1 per iteration, function="
                   << function.getSymName() << '\n';
    }
  }
};
} // namespace

std::unique_ptr<Pass> createThreadElideBackedgeBarrierPass() {
  return std::make_unique<ThreadElideBackedgeBarrierPass>();
}
} // namespace mlir::frisk
