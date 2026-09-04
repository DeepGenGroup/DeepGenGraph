#include <map>
#include <string>

#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_CONVERTFRISKTOBASE
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

// struct KernelOpConversion : public OpConversionPattern<KernelOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//     FunctionType funcType = mlir::dyn_cast<FunctionType>(kernelOp.getFunctionType());
//     ArrayRef<Type> inputTypes = funcType.getInputs();
//     // new func
//     func::FuncOp funcOp = rewriter.create<func::FuncOp>(kernelOp.getLoc(), kernelOp.getSymName(), funcType);
//     if (auto threadNumAttr = kernelOp->getAttr("thread_num")) {
//       funcOp->setAttr("thread_num", threadNumAttr);
//     }
//     auto& region = funcOp->getRegion(0);
//     region.emplaceBlock();
//     auto& body = funcOp.front();
//     SmallVector<Location> locs(inputTypes.size(), kernelOp.getLoc());
//     body.addArguments(inputTypes, locs);

//     auto& oldBlock = kernelOp->getRegion(0).front();
//     auto& newBlock = funcOp->getRegion(0).front();
//     // replace all uses with
//     for (unsigned i=0; i<oldBlock.getNumArguments(); ++i) {
//         Value oldArg = oldBlock.getArgument(i);
//         Value newArg = newBlock.getArgument(i);
//         rewriter.replaceAllUsesWith(oldArg,newArg);
//     }
//     // move operation from origin kernelOp
//     newBlock.getOperations().splice(newBlock.getOperations().begin(), oldBlock.getOperations());
//     // llvm::outs() <<  << "\n";
//     rewriter.eraseOp(&(newBlock.back()));
//     // add returnop
//     rewriter.setInsertionPointToEnd(&body);
//     rewriter.create<func::ReturnOp>(funcOp.getLoc());
//     // remove origin kernelOp
//     rewriter.eraseOp(kernelOp);
//     return success();
//   }
// };

struct KernelOpConversion : public OpConversionPattern<KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 1. 获取函数签名类型
    FunctionType funcType = mlir::dyn_cast<FunctionType>(kernelOp.getFunctionType());
    if (!funcType) return failure();

    // 2. 使用 rewriter 创建新的 func::FuncOp
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(kernelOp.getLoc(), kernelOp.getSymName(), funcType);

    // 3. 转移所需的属性
    if (auto threadNumAttr = kernelOp->getAttr("thread_num")) {
      funcOp->setAttr("thread_num", threadNumAttr);
    }

    // 4. 将原 kernelOp 的 Region 移动（内联）到新的 funcOp 中
    // 这一步由 rewriter 接管，会自动保留原有的 BlockArguments 及其所有 Use 关系，
    // 替代了原本手动 create block -> add argument -> replaceAllUsesWith -> splice operations 的繁琐步骤。
    rewriter.inlineRegionBefore(kernelOp.getRegion(), funcOp.getRegion(), funcOp.getRegion().end());

    // 5. 替换原有的 Terminator
    // 获取刚刚移动过来的 Block 及其最后一个操作（原 terminator）
    Block &body = funcOp.getRegion().front();
    Operation *terminator = body.getTerminator();
    
    // 将插入点设置在原 terminator 处，并使用 rewriter 将其替换为 func::ReturnOp
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(terminator);

    // 6. 使用 rewriter 安全删除原 kernelOp
    rewriter.eraseOp(kernelOp);

    return success();
  }
};

// struct ParallelOpConversion : public OpConversionPattern<ParallelOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(ParallelOp parallelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//     constexpr gpu::Dimension dims[] = {gpu::Dimension::z, gpu::Dimension::y, gpu::Dimension::x};
//     SmallVector<Value, 4> bids;
//     // create gpu blockidx
//     auto grid = parallelOp.getGrid();
//     rewriter.setInsertionPoint(parallelOp);
//     for (unsigned i=0; i<grid.size(); i++) {
//       auto bidOp = rewriter.create<gpu::BlockIdOp>(parallelOp.getLoc(), dims[i]);
//       bidOp->setAttr("range", rewriter.getI32IntegerAttr(grid[i]));
//       bids.push_back(bidOp);
//     }
//     // create gpu threadIdx
//     auto tidOp = rewriter.create<gpu::ThreadIdOp>(parallelOp.getLoc(), gpu::Dimension::x);
//     tidOp->setAttr("range", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
//     // set kernelOp
//     Operation *op = parallelOp->getParentOp();
//     rewriter.modifyOpInPlace(op, [&](){
//       op->setAttr("thread_num", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
//     });
//     // collect
//     auto& block = parallelOp->getRegion(0).front();
//     SmallVector<Operation*> opsToMove;
//     for (auto &op : block.getOperations()) {
//       if (!op.hasTrait<OpTrait::IsTerminator>()) {
//         opsToMove.push_back(&op);
//       }
//     }
//     // move
//     Operation *pos = parallelOp.getOperation();
//     for (Operation *op : opsToMove) {
//       // op->moveAfter(pos);
//       rewriter.moveOpAfter(op, pos);
//       pos = op;
//     }
//     // replace uses
//     for (unsigned i=0; i<block.getNumArguments(); ++i) {
//       Value oldArg = block.getArgument(i);
//       rewriter.replaceAllUsesWith(oldArg, bids[i]);
//     }
//     rewriter.eraseOp(parallelOp);
//     return success();
//   }
// };

struct ParallelOpConversion : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ParallelOp parallelOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = parallelOp.getLoc();
    constexpr gpu::Dimension dims[] = {gpu::Dimension::z, gpu::Dimension::y, gpu::Dimension::x};

    // 1. 生成 gpu blockIdx
    auto grid = parallelOp.getGrid();
    SmallVector<Value, 4> bids;
    for (unsigned i = 0; i < grid.size(); i++) {
      auto bidOp = rewriter.create<gpu::BlockIdOp>(loc, dims[i]);
      // 对于新创建的 Op，直接 setAttr 是安全的，因为它们还未对其他 Pass 可见
      bidOp->setAttr("range", rewriter.getIndexAttr(grid[i]));
      bids.push_back(bidOp);
    }

    // 2. 生成 gpu threadIdx
    auto tidOp = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    tidOp->setAttr("range", rewriter.getIndexAttr(parallelOp.getThreadNum()));

    // 3. 规范化修改 Parent Op（原代码这里用得很标准）
    Operation *parentOp = parallelOp->getParentOp();
    rewriter.modifyOpInPlace(parentOp, [&]() {
      parentOp->setAttr("thread_num", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
    });

    // 4. 提取要处理的 Block
    Block &block = parallelOp.getRegion().front();

    // 准备 Block Arguments 的替代值
    SmallVector<Value, 4> replArgs;
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      replArgs.push_back(bids[i]);
    }

    // 5. 规范化的 Inline 操作
    // 步骤 A: 首先删除原 block 内的 terminator，防止它被错误地移入 Parent Block 导致 IR 结构破坏
    rewriter.eraseOp(block.getTerminator());

    // 步骤 B: 使用 rewriter 的内联方法，这会自动移动所有剩余的 Ops 并替换原 Block arguments
    // 该方法受 Dialect Conversion 框架的完全监控并支持回滚
    rewriter.inlineBlockBefore(&block, parallelOp, replArgs);

    // 6. 删除原 Op
    rewriter.eraseOp(parallelOp);

    return success();
  }
};

struct ForOpConversion : public OpConversionPattern<ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ForOp dforOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    uint64_t lb = dforOp.getLower();
    uint64_t ub = dforOp.getUpper();
    uint64_t step = dforOp.getStep();
    Value div = dforOp.getInductionVar();
    Value aiv;
    auto aforOp = rewriter.create<affine::AffineForOp>(dforOp.getLoc(), lb, ub, step, mlir::ValueRange({}), 
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        aiv = iv;
      });
    // move
    aforOp.getBody()->getOperations().splice(aforOp.getBody()->getOperations().begin(), dforOp.getBody()->getOperations());
    rewriter.eraseOp(&(aforOp.getBody()->back()));
    rewriter.setInsertionPointToEnd(aforOp.getBody());
    rewriter.create<affine::AffineYieldOp>(aforOp.getLoc());
    // replace
    div.replaceAllUsesWith(aiv);
    rewriter.eraseOp(dforOp);
    return success();
  }
};

class ConvertFriskToBase : public impl::ConvertFriskToBaseBase<ConvertFriskToBase> {
public:
  
  void runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    ConversionTarget target(*context);

    // clang-format off
    target.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect>();

    target.addIllegalOp<KernelOp>();
    target.addIllegalOp<ParallelOp>();
    target.addIllegalOp<ForOp>();
    RewritePatternSet patterns(context);
    patterns.add<KernelOpConversion>(context);
    patterns.add<ParallelOpConversion>(context);
    patterns.add<ForOpConversion>(context);
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertFriskToBasePass() {
  return std::make_unique<ConvertFriskToBase>();
}

}
