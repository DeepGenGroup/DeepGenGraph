#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_IRDEEPOPTIMIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;

class VectorOpLoopUnrollPass : public impl::IRDeepOptimizeBase<VectorOpLoopUnrollPass> {
public:
  void runOnOperation(){
  
    auto kernel = getOperation();
    if(!kernel->hasAttr(THREAD_NUM)){
      return;
    }
    mlir::DenseSet<affine::AffineForOp> forOps;
    kernel->walk([&](mlir::vector::InsertOp insertOp){
      auto positions = insertOp.getMixedPosition();
      for(auto pos : positions){
        // 静态索引，例如这里的 0
        if (auto attr = dyn_cast<Attribute>(pos)) {
          auto intAttr = cast<IntegerAttr>(attr);
          llvm::outs() << "static index: " << intAttr.getInt() << "\n";
          continue;
        }
        // 动态索引，例如这里的 %arg4
        Value index = cast<Value>(pos);

        if (auto forOp = affine::getForInductionVarOwner(index)) {
          // llvm::outs() << "index is affine.for IV\n";
          forOps.insert(forOp);
        }
      }
    });
    kernel->walk([&](mlir::vector::ExtractOp extractOp){
      auto positions = extractOp.getMixedPosition();
      for(auto pos : positions){
        // 静态索引，例如这里的 0
        if (auto attr = dyn_cast<Attribute>(pos)) {
          auto intAttr = cast<IntegerAttr>(attr);
          llvm::outs() << "static index: " << intAttr.getInt() << "\n";
          continue;
        }
        // 动态索引，例如这里的 %arg4
        Value index = cast<Value>(pos);

        if (auto forOp = affine::getForInductionVarOwner(index)) {
          // llvm::outs() << "index is affine.for IV\n";
          forOps.insert(forOp);
        }
      }
    });
    
    for(auto op : forOps){
      affine::loopUnrollFull(op);
    }
  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createVectorOpLoopUnrollPass() {
  return std::make_unique<VectorOpLoopUnrollPass>();
}

}  // end namespace frisk 
