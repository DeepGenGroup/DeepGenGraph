// Share FP32 reciprocals while register values are still thread-level SSA.
// This is an explicitly approximate transform, enabled only by approx/mmac-recip.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_APPROXIMATETHREADMATH
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

static bool isProtected(func::FuncOp function) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none"})
    if (function->hasAttr(name)) return true;
  if (auto attrs = function->getAttrOfType<ArrayAttr>("passthrough"))
    for (Attribute attr : attrs) {
      if (auto pair = dyn_cast<ArrayAttr>(attr))
        if (!pair.empty()) attr = pair[0];
      if (auto name = dyn_cast<StringAttr>(attr))
        if (name.getValue() == "strictfp" || name.getValue() == "optnone")
          return true;
    }
  return false;
}

class ApproximateThreadMathPass
    : public impl::ApproximateThreadMathBase<ApproximateThreadMathPass> {
  void runOnOperation() override {
    auto function = getOperation();
    if (isProtected(function)) return;
    unsigned divisions = 0, sharedReciprocals = 0;
    function.walk([&](Block *block) {
      // Each block owns its cache: never speculate across a branch or loop.
      llvm::DenseMap<Value, Value> reciprocals;
      for (Operation &operation : llvm::make_early_inc_range(*block)) {
        auto divide = dyn_cast<arith::DivFOp>(operation);
        if (!divide) continue;
        Type type = divide.getType();
        auto vector = dyn_cast<VectorType>(type);
        if (!type.isF32() && !(vector && vector.getElementType().isF32()))
          continue;
        Value denominator = divide.getRhs();
        if (matchPattern(denominator, m_Constant())) continue;
        OpBuilder builder(divide);
        auto oneAttr = builder.getOneAttr(type);
        Attribute numerator;
        bool isOne = matchPattern(divide.getLhs(), m_Constant(&numerator)) &&
                     numerator == oneAttr;
        // Reuse only plain arcp reciprocals: extra nnan/ninf assumptions on
        // an existing result must not propagate to unrelated divisions.
        // Recognizing our own output also makes the pass idempotent.
        if (isOne && divide.getFastmath() == arith::FastMathFlags::arcp) {
          reciprocals.try_emplace(denominator, divide.getResult());
          continue;
        }
        Value &reciprocal = reciprocals[denominator];
        if (!reciprocal) {
          auto one = builder.create<arith::ConstantOp>(divide.getLoc(), oneAttr);
          reciprocal = builder.create<arith::DivFOp>(
              divide.getLoc(), one, denominator, arith::FastMathFlags::arcp);
          ++sharedReciprocals;
        }
        // Match the old LLVM transform: only the reciprocal has arcp; the
        // multiplication does not gain reassoc/nnan/ninf/contract assumptions.
        Value product = builder.create<arith::MulFOp>(
            divide.getLoc(), divide.getLhs(), reciprocal);
        divide.replaceAllUsesWith(product);
        divide.erase();
        ++divisions;
      }
    });
    llvm::errs() << "Approximate thread math: " << divisions
                 << " reciprocal divisions, " << sharedReciprocals
                 << " shared reciprocals\n";
  }
};
} // namespace

std::unique_ptr<Pass> createApproximateThreadMathPass() {
  return std::make_unique<ApproximateThreadMathPass>();
}
} // namespace mlir::frisk
