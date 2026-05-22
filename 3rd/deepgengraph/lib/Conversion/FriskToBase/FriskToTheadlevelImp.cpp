#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
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

std::map<func::FuncOp, LowerInfoAnalysis*> s_map_kernel_lowerInfo {};
using friskMs = frisk::attr::MemorySpace;

// frisk.gemm(%7, %10) to %13 {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x128xf16, 3>, memref<128x128xf32>
struct GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto kernel = op->getParentOfType<func::FuncOp>();
    auto tid_x = kernel.getOps<gpu::ThreadIdOp>();

    auto lia = s_map_kernel_lowerInfo.at(kernel);
    auto infoA = lia->getInfo(adaptor.getA());
    auto infoB = lia->getInfo(adaptor.getB());
    auto infoC = lia->getInfo(adaptor.getC());

    // A B from shm, C is local. 
    // %localA = frisk.alloc_buffer(Local, tA.shape()) 
    // frisk.copy(%shmA, %localA)
    // 使用localA计算GEMM

    auto insertBfferOps = [&](const LowerInfo& li, mlir::Value val){
      auto shmBuffer = li.buffer;
      auto localShape = li.thread_widths;
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> shape, Type elementType, int64_t alignment, int64_t memorySpace);
      auto eleType = mlir::dyn_cast<MemRefType>(shmBuffer.getType()).getElementType();
      mlir::Value localBuffer{};
      {
        RewriterBase::InsertionGuard ig{rewriter};
        rewriter.setInsertionPointAfter(shmBuffer.getDefiningOp());
        localBuffer = rewriter.create<frisk::AllocBufferOp>(shmBuffer.getLoc(), localShape, eleType, 16, int(friskMs::Local));
      }
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange map_operands, ::mlir::AffineMapAttr offset_map);

      auto offsetMap = li.getAffineMap();

    };

  }
};


// 在frisk改写为base表达后（去掉了parallel，引入了tx） 进一步切分其他op到thread上
class ConvertFriskBaseToThreadIR : public impl::ConvertFriskToBaseBase<ConvertFriskBaseToThreadIR> {
public:
  
  void runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    
    // 寻找所有kernel 执行lowerInfo推断
    mod->walk([&](func::FuncOp funcOp){
      if(funcOp->hasAttr("thread_num")){
        LowerInfoAnalysis* lia = new LowerInfoAnalysis{funcOp};
        s_map_kernel_lowerInfo.insert(std::make_pair(funcOp, lia));
      }
    });
    for(auto [kernel,ana] : s_map_kernel_lowerInfo){
      ana->run();
    }

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

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadIR() {
  return std::make_unique<ConvertFriskBaseToThreadIR>();
}

}
