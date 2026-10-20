// Specialize small static loops before shared storage planning. Constant induction
// variables expose rotating-buffer selectors, prefetch guards and addresses
// to canonicalization without making any assumptions about tensor geometry.
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "deepgengraph/Conversion/VectorCanonicalization.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"

namespace mlir::pipeline {
#define GEN_PASS_DEF_SIMPLIFYLOOPCONTROL
#include "deepgengraph/Pipeline/Passes.h.inc"
} // namespace mlir::pipeline

using namespace mlir;

namespace {
// Use wider arithmetic for signed bounds and reject a wrapping final
// increment. Unroll helpers use the target index width.
std::optional<uint64_t> tripCount(Operation *op) {
  int64_t lb, ub, step;
  unsigned width = 64;
  if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
    if (!loop.hasConstantBounds())
      return std::nullopt;
    lb = loop.getConstantLowerBound();
    ub = loop.getConstantUpperBound();
    step = loop.getStepAsInt();
  } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
    APInt l, u, s;
    if (loop.getUnsignedCmp() ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&l)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&u)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&s)) ||
        l.getBitWidth() > 64 || u.getBitWidth() > 64 || s.getBitWidth() > 64)
      return std::nullopt;
    lb = l.getSExtValue();
    ub = u.getSExtValue();
    step = s.getSExtValue();
    if (auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType()))
      width = integer.getWidth();
  } else {
    return std::nullopt;
  }
  if (step <= 0)
    return std::nullopt;
  if (ub <= lb)
    return 0;
  auto wide = [](int64_t v) { return APInt(64, uint64_t(v)).sext(128); };
  APInt distance = wide(ub) - wide(lb), stride = wide(step);
  APInt count = (distance + stride - 1).udiv(stride);
  APInt final = wide(lb) + count * stride;
  if (!final.isSignedIntN(width) || !count.isIntN(64))
    return std::nullopt;
  return count.getZExtValue();
}

struct SimplifyLoopControl
    : pipeline::impl::SimplifyLoopControlBase<SimplifyLoopControl> {
  using SimplifyLoopControlBase::SimplifyLoopControlBase;

  LogicalResult simplify() {
    RewritePatternSet patterns(&getContext());
    auto relevant = [](StringRef name) {
      return name == "arith" || name == "affine" || name == "scf" ||
             name == "memref" || name == "vector";
    };
    for (Dialect *dialect : getContext().getLoadedDialects())
      if (relevant(dialect->getNamespace()))
        dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : getContext().getRegisteredOperations())
      if (relevant(op.getDialectNamespace()))
        op.getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns),
                                  frisk::unsafeVectorCanonicalizations());
    return applyPatternsGreedily(getOperation(), frozen);
  }

  void runOnOperation() final {
    if (!maxTripCount || !maxExpandedOps) {
      getOperation().emitError("loop simplification budgets must be positive");
      return signalPassFailure();
    }
    if (onlyMarked) {
      bool candidate = false;
      getOperation().walk([&](Operation *op) {
        candidate |= isa<affine::AffineForOp, scf::ForOp>(op) &&
            (op->hasAttr("pipeline.scheduled") ||
             op->hasAttr("frisk.simplify_loop"));
      });
      if (!candidate)
        return;
    }
    // Fold constant bounds before considering loops. This also makes a second
    // application stable after selector/guard folding.
    if (failed(simplify()))
      return signalPassFailure();
    SmallVector<Operation *> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isa<affine::AffineForOp, scf::ForOp>(op) &&
          (!onlyMarked || op->hasAttr("pipeline.scheduled") ||
           op->hasAttr("frisk.simplify_loop")))
        loops.push_back(op);
    });
    uint64_t remaining = maxExpandedOps;
    unsigned unrolled = 0;
    for (Operation *op : loops) {
      auto count = tripCount(op);
      if (!count || *count > maxTripCount)
        continue;
      uint64_t bodySize = 0;
      op->walk([&](Operation *) { ++bodySize; });
      // Charge the complete expanded body, not only the extra copies. This
      // bounds growth for nested loops as well, without overflow.
      if (*count && bodySize > remaining / *count)
        continue;
      remaining -= bodySize * *count;
      if (!*count) {
        ValueRange inits = isa<scf::ForOp>(op)
            ? cast<scf::ForOp>(op).getInitArgs()
            : cast<affine::AffineForOp>(op).getInits();
        op->replaceAllUsesWith(inits);
        op->erase();
      } else {
        auto result = isa<scf::ForOp>(op)
            ? loopUnrollFull(cast<scf::ForOp>(op))
            : affine::loopUnrollFull(cast<affine::AffineForOp>(op));
        if (failed(result)) {
          op->emitError("failed to unroll a proven constant small loop");
          return signalPassFailure();
        }
      }
      ++unrolled;
    }
    if (failed(simplify()))
      return signalPassFailure();
    if (unrolled)
      llvm::errs() << "SimplifyLoopControl: " << unrolled
                   << " small loops specialized; selectors/guards/addresses folded\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pipeline::createSimplifyLoopControlPass() {
  return std::make_unique<SimplifyLoopControl>();
}
