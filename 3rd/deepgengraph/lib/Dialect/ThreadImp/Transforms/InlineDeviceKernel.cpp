#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
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

namespace mlir::threadimp {

#define GEN_PASS_DEF_DEEPGENGRAPHINLINEDEVICEKERNEL
#include "deepgengraph/Dialect/ThreadImp/Transforms/Passes.h.inc"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

// 假设你的 Dialect namespace 是 deepgengraph_triton 和 deepgengraph
struct InlineDeviceKernelPattern : public mlir::OpRewritePattern<deepgengraph::triton::DeviceKernelOp> {
  using OpRewritePattern<deepgengraph::triton::DeviceKernelOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(deepgengraph::triton::DeviceKernelOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    
    op->getParentOp()->setAttr("grid", op.getGridAttr());
    // 1. 获取 device_kernel 内部的 Region 和唯一的 Block
    mlir::Region &region = op.getRegion();
    if (region.empty()) {
      return mlir::failure();
    }
    mlir::Block &bodyBlock = region.front();

    std::vector<deepgengraph::triton::DeviceYieldOp> invalidOps;
    op->walk([&](deepgengraph::triton::DeviceYieldOp yieldOp){
      invalidOps.push_back(yieldOp);
    });
    for(auto e : invalidOps){
      rewriter.eraseOp(e);
    }

    // 2. 准备替换 Block Arguments 的 Values
    llvm::SmallVector<mlir::Value> replacements;
    mlir::Location loc = op.getLoc();

    // 2a. 替换 Grid 坐标参数 (%arg3, %arg4, %arg5)
    // 【注意】因为失去了 device_kernel，你需要生成新的 Op 来获取 grid 坐标。
    // 这里假设你的 Dialect 中有类似 deepgengraph::BlockIdOp 的操作。
    // 如果你是要生成 scf.parallel 循环，你需要在这里构建循环并获取 ivs。
    mlir::Value gridX = rewriter.create<threadimp::GetBlockIdOp>(loc, /*dim=*/0); 
    mlir::Value gridY = rewriter.create<threadimp::GetBlockIdOp>(loc, /*dim=*/1);
    mlir::Value gridZ = rewriter.create<threadimp::GetBlockIdOp>(loc, /*dim=*/2);
    replacements.push_back(gridX);
    replacements.push_back(gridY);
    replacements.push_back(gridZ);

    // 2b. 替换指针参数 (%arg6, %arg7, %arg8, %arg9)
    // 直接将 device_kernel 接收的 args ([%0, %1, %2, %3]) 传入
    for (mlir::Value arg : op.getArgs()) {
      replacements.push_back(arg);
    }

    // 校验参数数量是否匹配
    if (replacements.size() != bodyBlock.getNumArguments()) {
      return op->emitError("Block arguments size does not match replacements size.");
    }

    // 3. 将 bodyBlock 展开到 device_kernel 所在的位置
    // 这一步会自动将内部用到 %arg3~%arg9 的地方替换为我们上面准备好的 replacements
    rewriter.inlineBlockBefore(&bodyBlock, op, replacements);

    // 4. 删除原有的 device_kernel 操作

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ModifyKernelSignaturePattern : public mlir::OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern<deepgengraph::KernelOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    deepgengraph::KernelOp op,
    OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const override 
  {
    using namespace deepgengraph::triton;
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> arg_attrs = {});
    auto oldType = op.getFunctionType();
    std::vector<Type> inputs = oldType.getInputs();
    std::vector<Type> outputs = {};
    std::vector<Type> oldOutputTypes = oldType.getResults();
    for(auto ty : oldType.getResults()){
      inputs.push_back(ty);
    }
    auto newFuncType = rewriter.getFunctionType(inputs, outputs);
    TypeConverter::SignatureConversion conversion(oldType.getNumInputs());
    for(auto i = 0;i < oldType.getNumInputs(); ++i){
      conversion.addInputs(i, oldType.getInputs()[i]);
    }
    auto outIdx = oldType.getNumResults();
    for(auto ty : oldType.getResults()){
      conversion.addInputs(ty);
    }
    rewriter.applySignatureConversion(&op.getBody().front(), conversion);
    rewriter.modifyOpInPlace(op, [&](){
      op.setFunctionType(newFuncType);
    });
    std::vector<int32_t> outIdxAray;
    for(int i=0;i<outIdx;++i){
      outIdxAray.push_back(i + oldType.getNumInputs());
    }
    op->setAttr("out_idx", rewriter.getI32ArrayAttr(outIdxAray));
    
    auto terminator = op.getBody().front().getTerminator();
    if(auto returnOp = mlir::dyn_cast_or_null<deepgengraph::ReturnOp>(terminator)){
      rewriter.setInsertionPoint(returnOp);
      // 创建一个新的没有操作数的 ReturnOp
      rewriter.create<deepgengraph::ReturnOp>(returnOp.getLoc()); 
      // 删掉旧的 ReturnOp
      rewriter.eraseOp(returnOp);
    }
    // replace empty_ptr ops with ptr_of
    std::vector<EmptyPointerOp> ops;
    op->walk([&](EmptyPointerOp emptyOp){
      ops.push_back(emptyOp);
    });
    
    for(auto oldEmptyOp : ops){
      for(int i=0;i<oldOutputTypes.size();++i){
        if(oldEmptyOp.getTensorType() == oldOutputTypes[i]){
          // remark as invalid type
          oldOutputTypes[i] = mlir::IndexType::get(op->getContext());
          rewriter.setInsertionPoint(oldEmptyOp);
          auto newop = rewriter.create<deepgengraph::triton::PointerOfOp>(oldEmptyOp->getLoc(), op.getFunctionBody().getArgument(i + oldType.getNumInputs()) );
          rewriter.replaceOp(oldEmptyOp, newop);
        }
      }
    }
    return success();
  }
};


// 假设你的 Dialect namespace 是 deepgengraph_triton 和 deepgengraph
struct RemoveEmptyPtrOpPattern : public mlir::OpRewritePattern<deepgengraph::triton::EmptyPointerOp> {
  using OpRewritePattern<deepgengraph::triton::EmptyPointerOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
    deepgengraph::triton::EmptyPointerOp op,
    mlir::PatternRewriter &rewriter) const override {
    
    return success();
  }
};


class InlineDeviceKernel : public impl::DeepgengraphInlineDevicekernelBase<InlineDeviceKernel> {
public:
  void runOnOperation() override {
    mlir::RewritePatternSet ps1(&getContext());
    mlir::RewritePatternSet ps2(&getContext());
    ps1.add<InlineDeviceKernelPattern>(&getContext());\
    TypeConverter tc;
    ConversionTarget target(getContext());
    ps2.add<ModifyKernelSignaturePattern >(tc, &getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(ps1)))) {
      signalPassFailure();
    }
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
    target.addDynamicallyLegalOp<deepgengraph::KernelOp>(
      [](deepgengraph::KernelOp op){
        return op.getFunctionType().getNumResults() == 0;
      });
    // 注意，applyPatternsAndFoldGreedily 不可以用于 OpConversionPattern 
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(ps2)))) {
      // signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createInlineDevicekernelOpPass(){
  return std::make_unique<InlineDeviceKernel>(); 
}

} // namespace mlir::deepgengraph
