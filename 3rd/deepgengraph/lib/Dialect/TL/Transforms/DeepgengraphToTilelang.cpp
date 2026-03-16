#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::deepgengraph {

#define GEN_PASS_DEF_DEEPGENGRAPHTOTILELANG
#include "deepgengraph/Dialect/TL/Transforms/Passes.h.inc"

} // namespace mlir::deepgengraph

namespace mlir::deepgengraph{

// ------- rewrite patterns

// deepgengraph.kernel -> func.func
// deepgengraph.return -> func.return
// device_kernel -> tilelang.with_kernel

class TritonTypeConverter : public TypeConverter {
  public:
  TritonTypeConverter(){
    addConversion([](triton::PointerType type){
      return tilelang::PointerType::get(type.getPointeeType(), tilelang::MemSpace::GM);
    });
    addConversion([](triton::BlockPointerType type){
      return tilelang::BlockPointerType::get(type.getPointeeType(), tilelang::MemSpace::GM);
    });
  }
};

struct KernelOpRewriteRule : public OpRewritePattern<deepgengraph::KernelOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(deepgengraph::KernelOp op, mlir::PatternRewriter &rewriter) const override {
    auto tc = std::make_shared<TritonTypeConverter>();
    auto funcType = op.getFunctionType();
    auto funcOp = rewriter.create<func::FuncOp>(op->getLoc(), op.getSymName(), funcType);
    funcOp.getBody().takeBody(op.getBody());
    rewriter.eraseOp(op);

      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange srcs);
    funcOp->walk([&](deepgengraph::ReturnOp oldReturnOp){
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(oldReturnOp);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(oldReturnOp, oldReturnOp.getSrcs());
    });
    
    triton::DeviceKernelOp device_kernel_op = nullptr;
    funcOp.walk([&](triton::DeviceKernelOp device_kernel){
      device_kernel_op = device_kernel;
    });
    if(device_kernel_op != nullptr){
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(device_kernel_op);
      std::vector<Value> args;
      for(const auto & e : device_kernel_op.getArgs()){
        args.push_back(e);
      }
      std::vector<int64_t> grid;
      grid.assign(device_kernel_op.getGrid().begin(),device_kernel_op.getGrid().end()) ;
      auto withKernelOp = rewriter.create<tilelang::WithKernelOp>(device_kernel_op->getLoc(), args, grid);
      withKernelOp.getScopedBody().takeBody(device_kernel_op->getRegion(0));
      rewriter.eraseOp(device_kernel_op);
      withKernelOp->walk([&](triton::DeviceYieldOp yield){
        rewriter.eraseOp(yield);
      });
    }
    return llvm::success();
  }
};

struct PointerOpRewriteRule : public OpRewritePattern< triton::PointerOfOp > {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite( triton::PointerOfOp op, mlir::PatternRewriter &rewriter) const override {
    auto tc = std::make_shared<TritonTypeConverter>();

    return llvm::success();
  }
};



/**
 * @brief Conversion Pass. 将 deepgengraph.kernel 转为tilelang 的语义表达。
 * input ir : deepgengraph IR & deepgengraph_triton IR
 * output ir : tilelang IR
 */
class ConversionPass : public mlir::deepgengraph::impl::DeepgengraphToTilelangBase<ConversionPass> {

public:
  void runOnOperation() override {
    using namespace mlir;
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto mod = getOperation();
    auto tar = ConversionTarget(*context);
    tar.addLegalDialect<
      func::FuncDialect,
      deepgengraph::DeepgengraphDialect,
      deepgengraph::triton::DeepgengraphTritonDialect,
      tilelang::TilelangDialect
    >();
    tar.addIllegalOp<deepgengraph::KernelOp>();
    patterns.add<KernelOpRewriteRule>(context);
    // patterns.add<DeviceKernelOpRewriteRule>(context);
    
    auto ret = applyPartialConversion(mod, tar , std::move(patterns));
    if (ret.failed()) {
      signalPassFailure();
    }
  }
};


std::unique_ptr<mlir::Pass> createConvertDeepgenGraphToTilelangPass(){ return std::make_unique<ConversionPass>(); }

}  // namespace mlir::deepgengraph
