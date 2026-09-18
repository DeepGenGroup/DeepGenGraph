#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_IRDEEPOPTIMIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

// Only unroll indices entirely determined by constants and small static IVs.
// Lane-dependent addresses and the outer K/V-block loop stay dynamic.
static bool collectIndexLoops(Value value,
                              llvm::SmallPtrSetImpl<Operation *> &loops) {
  if (matchPattern(value, m_Constant()))
    return true;
  if (auto loop = affine::getForInductionVarOwner(value)) {
    auto count = affine::getConstantTripCount(loop);
    if (!count || *count == 0 || *count > 64)
      return false;
    loops.insert(loop);
    return true;
  }
  auto *op = value.getDefiningOp();
  if (!op || !isa<affine::AffineApplyOp, arith::AddIOp, arith::SubIOp,
                  arith::MulIOp, arith::DivSIOp, arith::DivUIOp,
                  arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
                  arith::IndexCastUIOp>(op))
    return false;
  for (Value operand : op->getOperands())
    if (!collectIndexLoops(operand, loops))
      return false;
  return true;
}

class VectorOpLoopUnrollPass
    : public impl::IRDeepOptimizeBase<VectorOpLoopUnrollPass> {
  void runOnOperation() override {
    auto kernel = getOperation();
    RewritePatternSet patterns(&getContext());
    for (auto op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    // Bound cumulative code growth, not just individual loop trip counts.
    constexpr uint64_t maxExtraOperations = 65536;
    uint64_t extraOperations = 0;
    while (true) {
      if (failed(applyPatternsGreedily(kernel, frozen))) {
        signalPassFailure();
        return;
      }
      llvm::SmallPtrSet<Operation *, 16> candidates;
      auto collect = [&](auto op) {
        for (OpFoldResult position : op.getMixedPosition()) {
          auto index = dyn_cast<Value>(position);
          if (!index)
            continue;
          llvm::SmallPtrSet<Operation *, 8> dependencies;
          if (collectIndexLoops(index, dependencies))
            candidates.insert(dependencies.begin(), dependencies.end());
        }
      };
      kernel.walk([&](vector::InsertOp op) { collect(op); });
      kernel.walk([&](vector::ExtractOp op) { collect(op); });
      affine::AffineForOp selected;
      uint64_t selectedCost = 0;
      kernel.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) {
        if (selected || !candidates.contains(loop))
          return;
        uint64_t bodySize = 0;
        loop.getBody()->walk([&](Operation *) { ++bodySize; });
        auto count = affine::getConstantTripCount(loop);
        uint64_t cost = bodySize * (*count - 1);
        if (cost > maxExtraOperations - extraOperations)
          return;
        selected = loop;
        selectedCost = cost;
      });
      if (!selected)
        break;
      if (failed(affine::loopUnrollFull(selected))) {
        selected.emitError("failed to staticize register-vector indices");
        signalPassFailure();
        return;
      }
      extraOperations += selectedCost;
      // Recollect after every rewrite: unrolling invalidates nested op handles.
    }
  }
};
} // namespace

std::unique_ptr<Pass> createIRDeepOptimizePass() {
  return std::make_unique<VectorOpLoopUnrollPass>();
}

std::unique_ptr<Pass> createVectorOpLoopUnrollPass() {
  return createIRDeepOptimizePass();
}
} // namespace mlir::frisk
