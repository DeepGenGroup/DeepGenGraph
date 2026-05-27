#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKCANONICALIZE

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

class FriskCanonicalizePass : public impl::FriskCanonicalizeBase<FriskCanonicalizePass> {
  void runOnOperation() override {
    auto kernelOp = getOperation();
    std::vector<Value> toBeOptimizedVals;
    kernelOp->walk([&](affine::AffineLoadOp op){
      auto memTy = op.getMemref().getType();
      int64_t len = 1;
      for(auto s : memTy.getShape()){
        len *= s;
      }
      if(len == 1){
        toBeOptimizedVals.push_back(op.getMemref());
      }
    });
    OpBuilder builder(kernelOp->getContext());
    for(auto v : toBeOptimizedVals){
      v.getDefiningOp<typename OpTy>()
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFriskCanonicalizePass() {
  return std::make_unique<FriskCanonicalizePass>();
}

} // namespace mlir::frisk
