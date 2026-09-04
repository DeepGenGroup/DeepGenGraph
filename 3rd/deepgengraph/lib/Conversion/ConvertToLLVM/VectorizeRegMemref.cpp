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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_REGMEMREFVECTORIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;

// memref.alloc 分配shm 改为定义 global shm var
struct RegFillSematicConversion : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if(!op->hasAttr(REG_FILL_SEMATIC)){
      return failure();
    }
    mlir::Value oldRegBuffer;
    op->walk([&](mlir::affine::AffineStoreOp store){
      oldRegBuffer = store.getMemref();
    });
    auto memTy = mlir::cast<MemRefType>(oldRegBuffer.getType());
    assert(memTy.getMemorySpaceAsInt() == int(friskMs::Local));

    auto vecType = VectorType::get(memTy.getShape(), memTy.getElementType());
    auto fillval = op->getAttr(REG_FILL_SEMATIC);
    float fill = 0.0f;
    if(auto ty = mlir::dyn_cast<IntegerAttr>(fillval)){
      fill = ty.getInt();
    }
    else if(auto ty = mlir::dyn_cast<FloatAttr>(fillval)){
      fill = ty.getValue().convertToFloat();
    }
    std::vector<Attribute> val = {fillval};
    auto valAttr = DenseElementsAttr::get(vecType, val);
    auto vectorInit = rewriter.create<arith::ConstantOp>(op->getLoc(), vecType, valAttr);
    // 追踪所有 oldRegBuffer 的使用者
    mlir::DenseMap<mlir::Operation*, mlir::Operation*> oldNewOpMapper;
    for(auto user : oldRegBuffer.getUsers()){
      rewriter.setInsertionPointAfter(user);
      if(auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(user)){
        // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> position);
        mlir::ValueRange vr = loadOp.getIndices();
        auto indices = getAsOpFoldResult(vr);
        std::vector<int64_t> staticshape (indices.size(), ShapedType::kDynamic );
        auto attr = rewriter.getDenseI64ArrayAttr(staticshape);
        auto newOp = rewriter.create<vector::ExtractOp>(loadOp->getLoc(), loadOp.getType(), vectorInit.getResult(), vr, attr );
        oldNewOpMapper.insert({loadOp, newOp});
      }
      else if(auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user)){

      }
    }

    for(auto [old, newop] : oldNewOpMapper) {
      rewriter.replaceOp(old, newop);
    }
    rewriter.eraseOp(oldRegBuffer.getDefiningOp()) ;
    // rewriter.replaceAllUsesWith(oldRegBuffer, newVecInit.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};


class RegMemrefVectorizePass : public impl::RegMemrefVectorizeBase<RegMemrefVectorizePass> {
public:
  void runOnOperation(){
    auto kernel = getOperation();
    if(!kernel->hasAttr(THREAD_NUM)){
      return;
    }
    // --------- step 2 ： AllocOp创建shm 改为分配 global shm var 并 get_global
    MLIRContext *context = kernel->getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      LLVM::LLVMDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect>();
    
    target.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp op){
      return !op->hasAttr(REG_FILL_SEMATIC);
    });
    RewritePatternSet patterns(context);
    patterns.add<RegFillSematicConversion>(context);
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createRegMemrefVectorizePass() {
  return std::make_unique<RegMemrefVectorizePass>();
}

}  // end namespace frisk 
