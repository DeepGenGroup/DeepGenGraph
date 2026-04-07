#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "deepgengraph/Analysis/ThreadAnalysis.h"

namespace mlir::threadimp {

#define GEN_PASS_DEF_CONVERTBLOCKCALCULATEOPTOTHREADLEVELIMP
#include "deepgengraph/Dialect/ThreadImp/Transforms/Passes.h.inc"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"


static TypeConverter* GetThreadImpTypeConverter(){
  static TypeConverter converter;
  // ptr & block_ptr -> threadimp::pointer
  converter.addConversion([](Type normalTy){return normalTy;});
  converter.addConversion([](deepgengraph::triton::PointerType ty){
    auto eleTy = ty.getPointeeType().getElementType();
    auto newTy = threadimp::PointerType::get(eleTy, threadimp::MemSpace::GM);
    return newTy;
  });
  converter.addConversion([](deepgengraph::triton::BlockPointerType ty){
    auto eleTy = ty.getPointeeType().getElementType();
    auto newTy = threadimp::PointerType::get(eleTy, MemSpace::SHM);
    return newTy;
  });
  // materialization：类型还没转换时，插入临时 cast
  converter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
  return builder.create<UnrealizedConversionCastOp>(
      loc, resultType, inputs).getResult(0);
  });
  converter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
  return builder.create<UnrealizedConversionCastOp>(
      loc, resultType, inputs).getResult(0);
  });

  return &converter;
}

// 假设你的 Dialect namespace 是 deepgengraph_triton 和 deepgengraph
struct BroadcastableBinaryOpConversionPattern : 
  public mlir::OpInterfaceConversionPattern<deepgengraph::BroadcastableBinaryOpInterface> {
  using OpInterfaceConversionPattern<deepgengraph::BroadcastableBinaryOpInterface>::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(deepgengraph::BroadcastableBinaryOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override  {
    auto threads = analyze::BlockThreads();
    auto lTensorType = mlir::cast<TensorType>(op.getLhs().getType());
    auto rTensorType = mlir::cast<TensorType>(op.getRhs().getType());
    auto copyInLhs = rewriter.create<threadimp::CopyShmToReg>(op->getLoc(), op.getLhs().getType(), op.getLhs(), threads);
    auto copyInRhs = rewriter.create<threadimp::CopyShmToReg>(op->getLoc(), op.getRhs().getType(), op.getRhs(), threads);
    OpKind kind;
    if(mlir::isa<deepgengraph::AddOp>(op)){
      kind = OpKind::add;
    }
    else if(mlir::isa<deepgengraph::SubOp>(op)){
      kind = OpKind::sub;
    }
    else if(mlir::isa<deepgengraph::MulOp>(op)){
      kind = OpKind::mul;
    }
    else if(mlir::isa<deepgengraph::DivOp>(op)){
      kind = OpKind::div;
    }
    auto newop = rewriter.create<ThreadElementwiseBinaryOp>(op->getLoc(), copyInLhs.getResult(), copyInRhs.getResult(), kind );
    auto copyOut = rewriter.create<threadimp::CopyRegToShm>(op->getLoc(), op.getResult().getType(), newop.getResult(), threads);
    rewriter.replaceOp(op, copyOut);
    return mlir::success();
  }
};


class ConvertBlockOpToThreadLevelImp : public impl::ConvertBlockCalculateOpToThreadLevelImpBase<ConvertBlockOpToThreadLevelImp> {

  void runOnOperation() override {
    // 获取当前的顶层 Operation（通常是 ModuleOp 或 FuncOp）
    mlir::Operation *op = getOperation();
    mlir::MLIRContext *context = &getContext();

    // 初始化 Pattern 集合
    mlir::RewritePatternSet patterns(context);

    // ==========================================
    // 核心步骤：将你的 Interface Pattern 注册进来
    // ==========================================
    patterns.add<BroadcastableBinaryOpConversionPattern>(context);

    // 你也可以在这里混合注册其他的普通 OpRewritePattern
    // patterns.add<SomeOtherSpecificOpPattern>(context);

    // 使用 Greedy Pattern Rewrite 引擎应用这些 Patterns
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
std::unique_ptr<Pass> createConvertBlockCalcOpToThreadImpPass(){
  return std::make_unique<ConvertBlockOpToThreadLevelImp>(); 
}

} // namespace mlir::deepgengraph
