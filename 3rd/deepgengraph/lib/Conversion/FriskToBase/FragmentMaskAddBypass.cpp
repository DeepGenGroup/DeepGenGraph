// Specialize fused causal masks while scores are still typed fragment SSA.
#include "CausalMaskLowering.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Conversion/MaskAddBypass.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"

namespace mlir::frisk {
namespace {
class FragmentMaskAddBypassPass
    : public PassWrapper<FragmentMaskAddBypassPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FragmentMaskAddBypassPass)
  FragmentMaskAddBypassPass() = default;
  FragmentMaskAddBypassPass(const FragmentMaskAddBypassPass &other)
      : PassWrapper(other) {}
  Option<std::string> mode{*this, "mode", llvm::cl::init("approx"),
      llvm::cl::desc("off or approx; permits signed-zero differences in mask additions")};
  StringRef getArgument() const final { return "frisk-bypass-fragment-mask-adds"; }
  StringRef getDescription() const final {
    return "Bypass whole-tile zero-mask additions in typed FragmentIR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FriskDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto function = getOperation();
    if (mode != "off" && mode != "approx") {
      function.emitError("mask-add-bypass must be off or approx");
      return signalPassFailure();
    }
    if (mode == "off" || mask_add_bypass::isProtected(function)) return;
    SmallVector<MaskFragOp> masks;
    function.walk([&](MaskFragOp mask) { masks.push_back(mask); });
    unsigned groups = 0;
    int64_t additions = 0;
    for (auto mask : masks) {
      // Respect uniform-masks and the earlier fusion/effect checks. The score
      // is already available here; do not move its computation or any loads.
      if (!mask.getUniform() || !mask.getAddend()) continue;
      auto triangle = causal_mask::match(&mask.getRegion().front());
      if (!triangle) continue;
      auto constant = triangle->clear.getDefiningOp<arith::ConstantOp>();
      auto clear = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
      if (!clear || !clear.getValue().isZero() || clear.getValue().isNegative()) continue;
      OpBuilder b(mask);
      auto loc = mask.getLoc();
      Value complete = causal_mask::buildClearTileCondition(
          b, loc, cast<MemRefType>(mask.getBlockType()), mask.getStarts(), triangle->diagonal);
      if (!complete) continue;
      auto branch = b.create<scf::IfOp>(loc, TypeRange{mask.getType()}, complete, true);
      branch->setAttr("frisk.uniform_mask", b.getUnitAttr());
      branch->setAttr("frisk.mask_add_bypass", b.getUnitAttr());
      mask.getResult().replaceAllUsesWith(branch.getResult(0));
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      b.create<scf::YieldOp>(loc, mask.getAddend());
      mask->moveBefore(&branch.getElseRegion().front(), branch.getElseRegion().front().end());
      // The outer branch already covers the clear tile. Avoid regenerating
      // the same branch during thread lowering or when reapplying this pass.
      mask.setUniform(false);
      b.setInsertionPointAfter(mask);
      b.create<scf::YieldOp>(loc, mask.getResult());
      ++groups;
      additions += mask.getType().getRegisterType().getNumElements();
    }
    llvm::errs() << "MaskAddBypassFragment: " << function.getSymName() << ": "
                 << groups << " groups, " << additions
                 << " additions bypassed (approx)\n";
  }
};
} // namespace
std::unique_ptr<Pass> createFragmentMaskAddBypassPass() {
  return std::make_unique<FragmentMaskAddBypassPass>();
}
} // namespace mlir::frisk
