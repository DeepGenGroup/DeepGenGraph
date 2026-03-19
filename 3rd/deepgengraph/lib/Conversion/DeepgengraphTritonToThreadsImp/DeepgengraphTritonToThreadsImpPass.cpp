#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "dbg.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "deepgengraph/Analysis/ThreadAnalysis.h"

namespace mlir::deepgengraph {
#define GEN_PASS_DEF_CONVERTDEEPGENGRAPHTRITONTOTHREADLEVELIMP
#include "deepgengraph/Conversion/DeepgengraphTritonToThreadImp/Passes.h.inc"

static int32_t blockThreads = 128 ;





// struct BinaryOpConvertToThreadImp : public OpInterfaceConversionPattern<deepgengraph::BroadcastableBinaryOpInterface> {
//   using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

//   LogicalResult matchAndRewrite(
//     BroadcastableBinaryOpInterface op,
//     ArrayRef<Value> operands,
//     ConversionPatternRewriter& rewriter) const override 
//   {
//     auto loc = op->getLoc();
//     llvm::outs() << "------------Interface : enter " << op << "\n"; llvm::outs().flush();
//     auto lhsBShape = op.getLhsBroadcastedShape();
//     auto rhsBShape = op.getRhsBroadcastedShape();
//     auto retShape = op.getExpectedResultShape();

//     auto lhsElementType = mlir::cast<mlir::TensorType>(op.getLhs().getType()).getElementType();
//     auto rhsElementType = mlir::cast<mlir::TensorType>(op.getRhs().getType()).getElementType();
//     auto retElementType = mlir::cast<mlir::TensorType>(op.getResult().getType()).getElementType();

//     auto lhsMemrefType = mlir::MemRefType::get(lhsBShape, lhsElementType);
//     auto rhsMemrefType = mlir::MemRefType::get(rhsBShape, rhsElementType);
//     auto retMemrefType = mlir::MemRefType::get(retShape, retElementType);

//     auto allocLhs = rewriter.create<mlir::memref::AllocOp>(loc, lhsMemrefType );
//     auto allocRhs = rewriter.create<mlir::memref::AllocOp>(loc, rhsMemrefType );
//     auto allocRet = rewriter.create<mlir::memref::AllocOp>(loc, retMemrefType );

//     allocRet->moveAfter(op);

//     auto th = getBlockDims(op->getName().getStringRef());
//     op->setAttr("thread.x", rewriter.getI32IntegerAttr(th.tx_count));
//     op->setAttr("thread.y", rewriter.getI32IntegerAttr(th.ty_count));
//     return success();
//   }

// };




class ConvertDeepgengraphTritonToThreadImp : public impl::ConvertDeepgengraphTritonToThreadLevelImpBase<ConvertDeepgengraphTritonToThreadImp>
{
public:
  
  // 分析deviceKernelOp中的 deengengraphOp。根据其调用链，分配对应的 sharedmem buffer。
  void memoryUseAnalysis(mlir::deepgengraph::triton::DeviceKernelOp kernelOp) {
    // auto threadCountAttr = mlir::cast<mlir::IntegerAttr>(kernelOp->getAttr("blockThreads"));
    // int64_t blockThreads = threadCountAttr.getInt();
    kernelOp->walk([&](mlir::Operation* op){
      auto opDialect = op->getDialect();
      if(mlir::isa<mlir::deepgengraph::DeepgengraphDialect>(opDialect)){
        if(mlir::isa<mlir::deepgengraph::BroadcastableBinaryOpInterface>(op)){
          auto binaryop = mlir::cast<mlir::deepgengraph::BroadcastableBinaryOpInterface>(op);
          auto lshape = binaryop.getLhsBroadcastedShape();
          auto rshape = binaryop.getRhsBroadcastedShape();

        }
      }
    });
  }

  void runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp k = getOperation();
    ConversionTarget target(*context);

    // clang-format off
    target.addLegalDialect<
      deepgengraph::DeepgengraphDialect,
      deepgengraph::triton::DeepgengraphTritonDialect,
      tensor::TensorDialect,
      linalg::LinalgDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      mlir::math::MathDialect,
      mlir::func::FuncDialect,
      mlir::memref::MemRefDialect,
      scf::SCFDialect>();
    
    // target.addIllegalDialect<DeepgengraphDialect>();
    RewritePatternSet patterns0(context);
    // patterns0.add<
    //   BinaryOpConvertToThreadImp
    //   >(context);
    k->walk([&](mlir::deepgengraph::triton::DeviceKernelOp op){
      memoryUseAnalysis(op);
    });
    if (failed(applyPartialConversion(k, target, std::move(patterns0)))){
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertDeepgengraphTritonToThreadImpPass() {
  return std::make_unique<ConvertDeepgengraphTritonToThreadImp>();
}

} // namespace mlir::deepgengraph