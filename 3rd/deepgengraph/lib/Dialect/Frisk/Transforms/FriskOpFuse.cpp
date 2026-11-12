#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_FRISKOPFUSE
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {
// Schedule arrays index the operations in a block. Do not invalidate an
// authored schedule or rewrite already tiled operations with this block pass.
static bool hasSchedule(Operation *op) {
  for (; op; op = op->getParentOp())
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().strref().starts_with("pipeline."))
        return true;
  return false;
}

static bool isZeroTile(Value value) {
  if (value.getDefiningOp<ZeroOp>())
    return true;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseFPElementsAttr>(constant.getValue())
                        : DenseFPElementsAttr{};
  return dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero() &&
         !dense.getSplatValue<APFloat>().isNegative();
}

static bool sameTile(Type lhs, Type rhs) {
  auto a = dyn_cast<ShapedType>(lhs), b = dyn_cast<ShapedType>(rhs);
  if (!a || !b || !a.hasStaticShape() || !b.hasStaticShape() ||
      a.getRank() != 2 || b.getRank() != 2 || a.getShape() != b.getShape() ||
      a.getElementType() != b.getElementType() ||
      !isa<FloatType>(a.getElementType()))
    return false;
  for (Type type : {lhs, rhs}) {
    if (!isa<MemRefType, VectorType>(type)) return false;
    if (auto vector = dyn_cast<VectorType>(type); vector && vector.isScalable())
      return false;
  }
  return true;
}

// Delay A/B reads to the add, where the seed is available. This supports a
// late alpha*old seed without hoisting its memory reads. Any intervening write,
// synchronization, region, or unknown effect prevents moving the GEMM.
static bool canDelayAcross(Operation *op) {
  if (op->getNumRegions() || hasSchedule(op) ||
      isa<gpu::BarrierOp, SyncThreadsInBlockOp, WaitBufferReadyOp, GemmWaitOp>(op))
    return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return isMemoryEffectFree(op);
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect());
  });
}

struct FuseGemmAdd : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp add, PatternRewriter &rewriter) const override {
    if (add->hasAttr("tiled") || hasSchedule(add))
      return failure();
    for (unsigned side = 0; side < 2; ++side) {
      auto gemm = add->getOperand(side).getDefiningOp<GemmOp>();
      Value seed = add->getOperand(1 - side);
      if (!gemm || !gemm->hasOneUse() || gemm->hasAttr("tiled") ||
          hasSchedule(gemm) || gemm->getBlock() != add->getBlock() ||
          !gemm->isBeforeInBlock(add) || !isZeroTile(gemm.getAcc()) ||
          gemm.getC().getType() != add.getType() ||
          !sameTile(seed.getType(), add.getType()))
        continue;
      bool movable = true;
      for (Operation *op = gemm->getNextNode(); op != add; op = op->getNextNode())
        if (!canDelayAcross(op)) {
          movable = false;
          break;
        }
      if (!movable)
        continue;

      // Legacy loop carriers use block-shaped vectors while GEMM results use
      // memref-shaped logical tiles. This same-element cast changes only the
      // representation, never the shape, precision, or accumulator layout.
      rewriter.setInsertionPoint(add);
      if (seed.getType() != add.getType())
        seed = rewriter.create<CastOp>(add.getLoc(), add.getType(), seed);
      auto fused = cast<GemmOp>(rewriter.clone(*gemm));
      fused.getAccMutable().assign(seed);
      // clone preserves transpose and any instruction/layout annotations.
      fused->setLoc(add.getLoc());
      Operation *zero = gemm.getAcc().getDefiningOp();
      rewriter.replaceOp(add, fused.getC());
      rewriter.eraseOp(gemm);
      if (zero->use_empty() && !hasSchedule(zero))
        rewriter.eraseOp(zero);
      return success();
    }
    return failure();
  }
};

class FriskOpFusePass : public impl::FriskOpFuseBase<FriskOpFusePass> {
  void runOnOperation() override {
    auto function = getOperation();
    if (!allowReassociation || function->hasAttr("llvm.strictfp") ||
        function->hasAttr("strictfp"))
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseGemmAdd>(&getContext());
    // No unrelated folding/DCE: scheduled blocks must remain byte-for-byte
    // stable, and only the matched add, GEMM and dead zero are rewritten.
    walkAndApplyPatterns(function, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> createFriskOpFusePass() {
  return std::make_unique<FriskOpFusePass>();
}
} // namespace mlir::frisk
