#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;
static DenseMap<Value, LowerInfo> s_info;

static std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs,
    const std::vector<const char*> &labels) {
  std::vector<mlir::affine::AffineForOp> loops;
  loops.reserve(upperBounds.size());
  outIvs.reserve(upperBounds.size());

  // 确保标签数量与循环层数一致（可选的安全检查）
  size_t numLoops = upperBounds.size();
  
  for (size_t i = 0; i < numLoops; ++i) {
    // 1. 定义下界、上界和步长 (下界默认为 0，步长默认为 1)
    int64_t lowerBound = 0;
    int64_t step = 1;
    auto ub = upperBounds[i];
    // // ---- 情况 1：ub == 0 (循环不执行) ----
    // if (ub <= 0) {
    //   // 放入一个空值/虚值占位，防止外部按索引访问 outIvs 时越界
    //   outIvs.push_back(mlir::Value());
    //   // 插入点保持不变，后续的内层循环会直接平铺在当前层（虽然逻辑上内层也不会被执行）
    //   continue; 
    // }

    // // ---- 情况 2：ub == 1 (循环只执行一次，退化为常数 0) ----
    // if (ub == 1) {
    //   // 创建一个常数 0 作为当前层的伪迭代变量 (IV)
    //   mlir::Value constantZero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    //   outIvs.push_back(constantZero);
      
    //   // 注意：这里不需要 builder.setInsertionPointToStart(...)
    //   // 因为没有生成新的 block，接下来的内层循环直接依附在当前 block 中
    //   continue;
    // }
    // if(ub >= 2){
    if(ub >= 0){
      // 2. 创建当前层的 AffineForOp
      auto forOp = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBounds[i], step);
      // 3. 如果提供了对应的 label，则为其添加 StringAttr 属性
      if (i < labels.size() && labels[i] != nullptr) {
        forOp->setAttr("iterLabel", builder.getStringAttr(labels[i]));
      }
      mlir::Value iv = forOp.getInductionVar();
      // 4. 收集当前循环的迭代变量 (Induction Variable) 和 Op 本身
      outIvs.push_back(iv);
      loops.push_back(forOp);
      // 5. 将 builder 的插入点移动到当前循环体的末尾（yield 之前），以便下一层循环嵌套在内部
      builder.setInsertionPointToStart(forOp.getBody());
    }
  }

  return loops;
}


// frisk.gemm(%7, %10) to %13 {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x128xf16, 3>, memref<128x128xf32>
class GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // llvm::outs() << "\n --op \n"; op.dump();
    // llvm::outs() << "\n --opParent \n"; op->getParentOp()->dump();
    // llvm::outs() << "\n --opParentParent \n"; op->getParentOp()->getParentOp()->dump();
    llvm::outs() << "-- enter GemmOpConversion \n";llvm::outs().flush();
    auto kernel = getOuterMostOpWithName(op, func::FuncOp::getOperationName().data());
    assert(kernel->hasAttr("thread_num"));
    auto funcOp = mlir::cast<func::FuncOp>(kernel);
    gpu::ThreadIdOp tidx = nullptr;
    funcOp->walk([&](mlir::gpu::ThreadIdOp tidOp){
      if(tidx == nullptr && tidOp.getDimension() == gpu::Dimension::x){
        tidx = tidOp;
      }
    });
    assert(tidx != nullptr);
    // funcOp->dump();
    // 3. 在 Rewrite 过程中，推荐安全地获取缓存，防止野指针崩溃
    llvm::outs() << "-- start getinfo \n";llvm::outs().flush();
    auto infoA = s_info.at(op.getA());
    auto infoB = s_info.at(op.getB());
    auto infoC = s_info.at(op.getC());
    
    /** 
    A B from shm, C is local. 
    %localA, %localB, %localC = frisk.alloc_buffer(Local,) 
    frisk.copy(%shmA, %localA), frisk.copy(%shmB, %localB)
    frisk.fill(%localC, 0)
    for(i,j,k){
      %a = affine.load %localA[i,k] 
      %b = affine.load %localB[k,j]
      %c = affine.load %localC[i,j]
      %c += %a * %b
      affine.store(%c, %localC[i,j])
    }
    frisk.copy(%localC, %shmC)
    */

    auto addCopyFromShmToRegInNestedForOp = [&](LowerInfo& li, mlir::Value val, bool fromShmToLocal){
      RewriterBase::InsertionGuard ig{rewriter};
      auto mapExprs = li.getAffineMap();
      auto shmBuffer = li.buffer;
      auto localShape = li.get_thread_total_widths();
      auto eleType = mlir::dyn_cast<MemRefType>(shmBuffer.getType()).getElementType();
      mlir::Value localReg{};
      {
        // 分配local buffer
        RewriterBase::InsertionGuard ig{rewriter};
        rewriter.setInsertionPointAfter(shmBuffer.getDefiningOp());
        localReg = rewriter.create<frisk::AllocBufferOp>(shmBuffer.getLoc(), localShape, eleType, 16, int(friskMs::Local));
      }
      std::vector<Value> mapOperands {tidx};
      auto indiceMap = AffineMap::get(li.get_dimcount(), 0, mapExprs, op->getContext());
      std::vector<Value> ivs;
      // 嵌套for loop，设置插入点到最内侧loop
      auto loops = createNestedAffineFor(rewriter, op->getLoc(), li.getItervarUbs(), ivs, li.getIterVarLabels());
      for(auto v : ivs){
        if(v != nullptr){
          mapOperands.push_back(v);
        }
      }
      // copy数据
      Value &src = fromShmToLocal ? shmBuffer : localReg;
      Value &dst = fromShmToLocal ? localReg : shmBuffer;
      // auto copyData = rewriter.create<frisk::CopyOp>(op->getLoc(), src, dst);
      auto copyData = rewriter.create<frisk::CopyOp>(op->getLoc(), src, dst,  mapOperands, indiceMap);
      return copyData;
    };
    llvm::outs() << "-- start copyInA \n";llvm::outs().flush();
    auto copyInA = addCopyFromShmToRegInNestedForOp(infoA, op.getA(), true);
    llvm::outs() << "-- start copyInB \n";llvm::outs().flush();
    auto copyInB = addCopyFromShmToRegInNestedForOp(infoB, op.getB(), true);
    auto localA = copyInA.getDst();
    auto localB = copyInB.getDst();
    {
      int m = localA.getType().getShape()[0];
      int k = localA.getType().getShape()[1];
      int n = localB.getType().getShape()[1];
      std::vector<int> gemmUbs = {m,n,k};
      std::vector<mlir::Value> outIvs;
      std::vector<const char*> looplabels = {"m","n","k"};
      RewriterBase::InsertionGuard ig{rewriter};
      createNestedAffineFor(rewriter, op->getLoc(), gemmUbs, outIvs,looplabels);
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ValueRange mapOperands);
      auto dimM = mlir::getAffineDimExpr(0, op->getContext());
      auto dimN = mlir::getAffineDimExpr(1, op->getContext());
      auto dimK = mlir::getAffineDimExpr(2, op->getContext());
      llvm::SmallVector<AffineExpr,3> _mapA = {dimM,dimK};
      llvm::SmallVector<AffineExpr,3> _mapB = {dimK,dimN};

      auto affineMapA = AffineMap::get(3,0,_mapA, op->getContext());
      auto affineMapB = AffineMap::get(3,0,_mapB, op->getContext());
      std::vector<Value> vr;
      for(auto iv : outIvs){
        if(iv != nullptr){
          vr.push_back(iv);
        }
      }
      auto a = rewriter.create<affine::AffineLoadOp>(op->getLoc(), localA, affineMapA, vr); 
      auto b = rewriter.create<affine::AffineLoadOp>(op->getLoc(), localB, affineMapB, vr); 
      auto ab = rewriter.create<arith::MulFOp>(op->getLoc(), a,b);
      
    }
    llvm::outs() << "-- start copyOutC \n";llvm::outs().flush();
    auto copyOutC = addCopyFromShmToRegInNestedForOp(infoC, op.getC(), false);
    rewriter.eraseOp(op);
    llvm::outs() << "-- exit gemmopconversion \n";llvm::outs().flush();
    return success();
  }
};

/**
  frisk.block (%arg5, %arg6) to (128, 128) {
    %c0_2 = arith.constant 0 : index
    %24 = affine.load %8[%arg5, %arg6] : memref<128x128xf32>
    %25 = affine.load %18[%arg5, %arg6] : memref<128x128xf32>
    %26 = arith.addf %24, %25 : f32
    affine.store %26, %19[%arg5, %arg6] : memref<128x128xf32>
  } 
 */
class BlockOpConversion : public OpConversionPattern<frisk::BlockOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "-- enter BlockopConv \n";llvm::outs().flush();
    auto kernel = getOuterMostOp<func::FuncOp>(op);
    assert(kernel->hasAttr("thread_num"));
    auto funcOp = mlir::cast<func::FuncOp>(kernel);
    gpu::ThreadIdOp tidx = nullptr;
    funcOp->walk([&](mlir::gpu::ThreadIdOp tidOp){
      if(tidx == nullptr && tidOp.getDimension() == gpu::Dimension::x){
        tidx = tidOp;
      }
    });
    // 如果没找到线程 ID，直接返回匹配失败
    assert(tidx != nullptr);
    std::vector<affine::AffineStoreOp> storeOps {};
    std::vector<affine::AffineLoadOp> loadOps {}; 
    op->walk([&](Operation* childOp){
      if(mlir::isa<affine::AffineLoadOp>(childOp)){
        loadOps.push_back(mlir::cast<affine::AffineLoadOp>(childOp));
      }
      if(mlir::isa<affine::AffineStoreOp>(childOp)){
        storeOps.push_back(mlir::cast<affine::AffineStoreOp>(childOp));
      }
    });
    assert(!storeOps.empty());
    std::vector<AllocBufferOp> threadLocalForStoreOps;
    std::vector<AllocBufferOp> threadLocalForLoadOps;
    // 根据 load 和 storeOps，创建其 thread级别的 localBuffer
    {
      RewriterBase::InsertionGuard ig{rewriter};
      auto outMostFor = getOuterMostOp<affine::AffineForOp>(op);
      if(outMostFor != nullptr){
        rewriter.setInsertionPoint(outMostFor);
      }
      for(auto storeOp : storeOps){
        auto info = s_info.at(storeOp.getMemref()); 
        info.show();
        auto threadLocal = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), info.get_thread_total_widths(), storeOp.getMemref().getType().getElementType(), 16, int(friskMs::Local));
        threadLocalForStoreOps.push_back(threadLocal);
      }
      for(auto loadOp : loadOps){
        auto info = s_info.at(loadOp.getMemref());
        info.show();
        auto threadLocal = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), info.get_thread_total_widths(), loadOp.getMemref().getType().getElementType(), 16, int(friskMs::Local));
        threadLocalForLoadOps.push_back(threadLocal);
      }
    }
    
    return success();
  }
};



// 在frisk改写为base表达后（去掉了parallel，引入了tx） 进一步切分其他op到thread上
class ConvertFriskBaseToThreadLevelIR : public impl::ConvertFriskBaseToThreadLevelIRBase<ConvertFriskBaseToThreadLevelIR> {
public:
  
  void runOnOperation(){
    MLIRContext *context = &getContext();
    auto kernel = getOperation();
    if(!kernel->hasAttr("thread_num")){
      return;
    }
    s_info = LowerInfoAnalysis::run(kernel);
    llvm::outs() << "-- lowerinfo analyze done\n";llvm::outs().flush();
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

    target.addIllegalOp<KernelOp,ParallelOp,ForOp,
      GemmOp, BlockOp
    >();
    RewritePatternSet patterns(context);
    patterns.add<
      GemmOpConversion, BlockOpConversion
    >(context);
    llvm::outs() << "-- lowerinfo partialconversion\n";llvm::outs().flush();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))){
      return signalPassFailure();
    }
    llvm::outs() << "-- exit Pass\n";llvm::outs().flush();
  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass() {
  return std::make_unique<ConvertFriskBaseToThreadLevelIR>();
}

}  // end namespace frisk 
