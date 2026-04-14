#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "deepgengraph/Analysis/ThreadAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_DEEPGENGRAPHTOFRISK
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp>{
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override 
  {
    auto gridAttr = op->getAttr("grid");
    auto loc = op->getLoc();
    auto name = op.getName();
    auto oldFuncType = op.getFunctionType();
    auto converter = getTypeConverter();
    llvm::SmallVector<Type> newInputs;
    llvm::SmallVector<Type> newOutputs;
    for(auto ty : oldFuncType.getInputs()){
      newInputs.push_back(converter->convertType(ty));
    }
    auto newFuncType = rewriter.getFunctionType(newInputs, newOutputs);
    TypeConverter::SignatureConversion sc {oldFuncType.getNumInputs()};
    for(int i=0;i<oldFuncType.getNumInputs();++i){
      sc.addInputs(i, converter->convertType(oldFuncType.getInput(i)));
    }
    rewriter.convertRegionTypes(&op->getRegion(0), *converter, &sc);
    rewriter.applySignatureConversion(&op.getFunctionBody().front(), sc);
    auto newKernelOp = rewriter.create<frisk::KernelOp>(loc, op.getName(), newFuncType);
    newKernelOp->setAttr("grid", gridAttr);
    rewriter.inlineRegionBefore(op->getRegion(0), newKernelOp.getRegion(), newKernelOp.getRegion().end());

    auto terminator = newKernelOp->getRegion(0).front().getTerminator();
    rewriter.setInsertionPoint(terminator);
    auto newRet = rewriter.create<frisk::EndOp>(loc);
    rewriter.replaceOp(terminator, newRet);

    // add parallel op
    rewriter.setInsertionPointToStart(&newKernelOp->getRegion(0).front());
    auto ranges = mlir::cast<DenseI64ArrayAttr>(gridAttr).asArrayRef();

    auto parallelOp = rewriter.create<frisk::ParallelOp>(loc, ranges, 128);
    auto parallelEntry = parallelOp.addEntryBlock();
    Block::iterator iter = parallelEntry->begin();
    auto nextOp = parallelOp->getNextNode();
    
    while(nextOp != nullptr && !mlir::isa<frisk::EndOp>(nextOp)){
      auto _next = nextOp->getNextNode();
      rewriter.moveOpBefore(nextOp, parallelEntry , parallelEntry->end());
      nextOp = _next;
    }
    auto innerEndOp = parallelEntry->getOps<frisk::EndOp>().begin();
    rewriter.moveOpBefore(*innerEndOp, parallelEntry, parallelEntry->end());
    rewriter.replaceOp(op, newKernelOp);
    return success();
  }
};

class ConvertDeepgengraphToFrisk : public impl::DeepgengraphToFriskBase<ConvertDeepgengraphToFrisk>
{
  public:
  void runOnOperation() override {
    auto ctx = getOperation()->getContext();
    TypeConverter tc;
    Operation* op = getOperation();
    tc.addConversion([](mlir::Type normalTy){ return normalTy; });
    tc.addConversion([](mlir::TensorType tensorTy){
      // tensor -> memref
      auto memref = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
      return memref;
    });
    tc.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(
        loc, resultType, inputs).getResult(0);
    });
    tc.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(
        loc, resultType, inputs).getResult(0);
    });

    ConversionTarget target {*ctx};
    target.addLegalDialect<
      FriskDialect,
      memref::MemRefDialect,
      func::FuncDialect,
      deepgengraph::DeepgengraphDialect,
      deepgengraph::triton::DeepgengraphTritonDialect
    >();
    target.addIllegalOp<deepgengraph::KernelOp>();
    RewritePatternSet ps {ctx};
    ps.add<KernelOpConversionPattern>(tc, ctx);
    applyPartialConversion(op, target, std::move(ps));
  }
};

std::unique_ptr<Pass> createConvertDeepgenGraphToFriskPass(){
  return std::make_unique<ConvertDeepgengraphToFrisk>();
}

} // namespace mlir::deepgengraph
