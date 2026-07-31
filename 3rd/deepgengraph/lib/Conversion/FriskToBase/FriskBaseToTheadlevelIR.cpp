#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
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

  return loops;
}

struct WgmmaMNKLoopInfo {
  int mLoopNum;
  int nLoopNum;
  int kLoopNum;
  int kLoopStep;

};

// 计算：k轴循环次数，Y和X方向迭代次数（=blockRepeat）
static WgmmaMNKLoopInfo GetWgmmaInfo(const LowerInfo& C, int bk){
  WgmmaMNKLoopInfo info {};
  auto ctype = mlir::cast<MemRefType>(C.buffer.getType());
  auto mma_k =  WgmmaConfig::mma_k_bytes * 8 / ctype.getElementTypeBitWidth();
  info.mLoopNum = C.get_block_repeat()[0];
  info.nLoopNum = C.get_block_repeat()[1];
  info.kLoopNum = bk / mma_k;
  info.kLoopStep = mma_k;
  return info;
}

// frisk.copy 转换为 cudacore的copy 或 tma copy
class CopyOpConversion : public OpConversionPattern<frisk::CopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto srcMemref = mlir::cast<MemRefType>(adaptor.getSrc().getType());
    auto dstMemref = mlir::cast<MemRefType>(adaptor.getDst().getType());
    if(srcMemref.getElementType() != dstMemref.getElementType()){
      // src dst的元素类型不同。lower为 arith.extf 或 truncf
    }
    else{
      // tma
      if(op->hasAttr("dev")){
        auto attr = op->getAttrOfType<frisk::DevKindAttr>("dev");
        if(attr.getValue() == attr::DevKind::TMA){

        }
        else{
          assert(false);
        }
      }
      else{
        // cudacore copy

      }
    }
    return success();
  }
};

// frisk.gemm(%7, %10) to %13 {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x128xf16, 3>, memref<128x128xf32>
/**

%acc = alloc_buffer(local, infoC.get_thread_total_widths())
tma_wait %smA
tma_wait %smB
for(i=0;i < BM; i+= infoA.blockWidths[0]){
  for(int j=0;j < BN; j+= infoB.blockWidths[1]){
    for(k=0;k< BK; k+=mma_k) {
      wgmma(%smA[i,k], %smB[k,j], %acc)
    }
  }
}

 */
class GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
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

    auto infoA = s_info.at(op.getA());
    auto infoB = s_info.at(op.getB());
    auto infoC = s_info.at(op.getC());
    // infoA.show("A");
    // infoB.show("B");
    // infoC.show("C");
    
    assert(infoA.get_block_repeat()[1] == infoB.get_block_repeat()[0]);  // k轴上的 for循环次数. A 列迭代数 == B 行迭代数
    
    auto typeA = mlir::cast<MemRefType>(adaptor.getA().getType());
    auto typeB = mlir::cast<MemRefType>(adaptor.getB().getType());
    bool is_ss = true;
    if(typeA.getMemorySpaceAsInt() == int(friskMs::Local)){
      is_ss = false;
    }
    assert(typeB.getMemorySpaceAsInt() == int(friskMs::Shared));

    // mma_k 和乘数有关
    auto mma_k =  WgmmaConfig::mma_k_bytes * 8 / mlir::cast<MemRefType>(infoB.buffer.getType()).getElementTypeBitWidth();

    auto shapeSmA = typeA.getShape();
    auto shapeSmB = typeB.getShape();
    auto BM = shapeSmA[0];
    auto BK = shapeSmA[1];
    auto BN = shapeSmB[1];

    mlir::Value localAcc {};
    {
      RewriterBase::InsertionGuard ig{rewriter};
      auto outerMostFor = getOuterMostOp<affine::AffineForOp>(op);
      rewriter.setInsertionPoint(outerMostFor);
      auto shape = infoC.get_thread_total_widths();
      auto eTy = mlir::cast<MemRefType>(infoC.buffer.getType()).getElementType();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment = IntegerAttr());
      auto memTy = MemRefType::get(shape,eTy, AffineMap{}, int(friskMs::Local));
      localAcc = rewriter.create<memref::AllocaOp>(op->getLoc(), memTy);
      // localAcc = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), shape, eTy, 16, int64_t(friskMs::Local));
    }
    
    auto ctx = op->getContext();

    if(is_ss){
      auto ivM = getAffineDimExpr(0, ctx);
      auto ivN = getAffineDimExpr(1, ctx);
      auto ivK = getAffineDimExpr(2, ctx);
      RewriterBase::InsertionGuard ig{rewriter};
      auto loopBK = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, BK, mma_k);
      rewriter.setInsertionPointToStart(loopBK.getBody());
      
      auto loopBM = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, BM , infoA.get_block_widths()[0]);
      rewriter.setInsertionPointToStart(loopBM.getBody());
      
      auto loopBN = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, BN, infoB.get_block_widths()[1]);
      rewriter.setInsertionPointToStart(loopBN.getBody());
      
      SmallVector<AffineExpr,2> indiceA = { ivM, ivK };
      SmallVector<AffineExpr,2> indiceB = {ivK, ivN};
      SmallVector<mlir::Value> iterVars { loopBM.getInductionVar(), loopBN.getInductionVar(), loopBK.getInductionVar() };
    
      auto mapA = AffineMap::get(3, 0, indiceA, ctx);
      auto mapB = AffineMap::get(3, 0, indiceB, ctx);
      // now insertion point is inside innermost forOp
      std::vector<int64_t> mnk = {WgmmaConfig::mma_m, infoB.get_block_widths()[1] ,mma_k};
      rewriter.create<WgMmaAsyncSSOp>(op->getLoc(), adaptor.getA(), adaptor.getB(), localAcc, mapA, iterVars, mapB, iterVars, mnk);
      // copy local to smC (不必要)
    }
    else{
      auto ivN = getAffineDimExpr(0, ctx);
      auto ivK = getAffineDimExpr(1, ctx);
      RewriterBase::InsertionGuard ig{rewriter}; 
      auto loopBK = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, BK, mma_k);
      rewriter.setInsertionPointToStart(loopBK.getBody());
      
      auto loopBN = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, BN, infoB.get_block_widths()[1]);
      rewriter.setInsertionPointToStart(loopBN.getBody());
      
      SmallVector<AffineExpr,2> indiceB = {ivK, ivN};
      SmallVector<mlir::Value> iterVars { loopBN.getInductionVar(), loopBK.getInductionVar() };
    
      auto mapB = AffineMap::get(2, 0, indiceB, ctx);
      // now insertion point is inside innermost forOp
      std::vector<int64_t> mnk = {WgmmaConfig::mma_m, infoB.get_block_widths()[1] ,mma_k};
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value localA, ::mlir::Value smB, ::mlir::Value localAcc, ::mlir::AffineMap smBMap, ::mlir::ValueRange smBMapOperands, ::llvm::ArrayRef<int64_t> mnk);
      rewriter.create<WgMmaAsyncLSOp>(op->getLoc(), infoA.buffer, adaptor.getB(), localAcc, mapB, iterVars, mnk);
      // copy local to smC (不必要)
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/**
  frisk.block (%arg5, %arg6) to (128, 128) {
    %c0_2 = arith.constant 0 : index
    %24 = affine.load %8[%arg5, %arg6] : memref<128x128xf32, 3>
    %25 = affine.load %18[%arg5, %arg6] : memref<128x128xf32>
    %26 = arith.addf %24, %25 : f32
    affine.store %26, %19[%arg5, %arg6] : memref<128x128xf32, 3>
  }
    ->  
  for (%itwx, %itwy) to (TWX, TWY) {
    %c0_2 = arith.constant 0 : index
    %24 = affine.load %8[%arg5, %arg6] : memref<128x128xf32, 3>
    %25 = affine.load %18[%arg5, %arg6] : memref<128x128xf32>
    %26 = arith.addf %24, %25 : f32
    affine.store %26, %19[%arg5, %arg6] : memref<128x128xf32, 3>
  } 
  转化为线程级别的实现：
  1.loadOp 的 srcMem，如果为block级别大小的local，将其切成thread-level-size local
  （ 新建 alloc_local_buffer, 之后用新的replace旧的的allUses。最后删除旧的 ）
    若为 shm，则不用新建 alloc_local_buffer
  2.storeOp dstMem 同理
  3.blockOp的 blockArgs, 改变映射范围为 thread-level-size，创建两重forOp。
  4.blockOp内的Op，搬运到 nestedFor里。改变 blockArg映射
  5.删除 blockOp
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
    assert(tidx != nullptr);
    // 寻找内部所有 load store ops
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
    std::array<int64_t, 2> threadLevelSize;
    // 收集所有需要替换的 local AllocBufferOp，去重
    llvm::DenseMap<AllocBufferOp, std::array<int64_t, 2>> allocLocalsToReplace;

    for (auto loadOp : loadOps) {
      auto srcValue = loadOp.getMemref();
      // if (srcValue.getType().getMemorySpaceAsInt() == int(friskMs::Local)) {
        auto srcDefOp = srcValue.getDefiningOp<AllocBufferOp>();
        if (srcDefOp != nullptr && !allocLocalsToReplace.count(srcDefOp)) {
          allocLocalsToReplace[srcDefOp] = s_info.at(srcValue).get_thread_total_widths();
          auto i = s_info.at(srcValue);
          i.show("block_load");
        }
      // }
    }

    for (auto storeOp : storeOps) {
      auto dstVal = storeOp.getMemref();
      // if (dstVal.getType().getMemorySpaceAsInt() == int(friskMs::Local)) {
        auto srcDefOp = dstVal.getDefiningOp<AllocBufferOp>();
        if (srcDefOp != nullptr && !allocLocalsToReplace.count(srcDefOp)) {
          allocLocalsToReplace[srcDefOp] = s_info.at(dstVal).get_thread_total_widths();
          auto i = s_info.at(dstVal);
          i.show("block_store");
        }
      // }
    }
    IRMapping mapper;
    // 统一替换，每个 AllocBufferOp 只处理一次
    for (auto &[srcDefOp, sz] : allocLocalsToReplace) {
      rewriter.setInsertionPoint(srcDefOp);
      auto ty = MemRefType::get(sz, srcDefOp.getElementType(), AffineMap{}, srcDefOp.getMemorySpace());
      if(srcDefOp.getMemorySpace() == int(friskMs::Local)){
        auto newAlloc = rewriter.create<memref::AllocaOp>(srcDefOp->getLoc(), ty);
        mapper.map(srcDefOp->getResult(0), newAlloc->getResult(0));
      }
      else{
        mapper.map(srcDefOp->getResult(0), srcDefOp->getResult(0));
      }
      // 同步更新 threadLevelSize，供后续 createNestedAffineFor 使用
      threadLevelSize = sz;
    }
    rewriter.setInsertionPoint(op);
    // frisk.blocOp 根据newbuffer的size，生成 nestedFor
    std::vector<mlir::Value> newIvs {};
    std::vector<const char* > labels {};
    std::vector<int> thread_level_sz { threadLevelSize.begin(), threadLevelSize.end() };
    for(auto _ : thread_level_sz){
      labels.push_back(nullptr);
    }
    createNestedAffineFor(rewriter, op->getLoc(), thread_level_sz, newIvs, labels);
    
    for(auto [oldIndex, newIter] : llvm::zip(op.getBody()->getArguments(), newIvs)){
      mapper.map(oldIndex,newIter);
    }
    // 根据映射规则，将frisk.blockOp 内的全部op 搬运到 nestedFOr的最内层 （createNestedAffineFor 之后，insertionPoint已经在最内侧了，不用动）
    // 将 blockOp body 内的所有 op 按序 clone 到当前 insertionPoint（nestedFor 最内层）
    // 跳过 block terminator（frisk.yield 或类似）
    Block *body = op.getBody();
    for (auto &childOp : body->without_terminator()) {
      rewriter.clone(childOp, mapper);
    }
    // blockOp body 外部（blockOp 之后）可能还有对旧 alloc 的 use（如 copy-out 等）
    // 用 replaceAllUsesExcept 只替换 blockOp 外部的 use
    for (auto &[srcDefOp, sz] : allocLocalsToReplace) {
      auto temp = mapper.lookupOrNull(srcDefOp->getResult(0));
      if(temp == nullptr){
        continue;
      }
      auto newAlloc = temp.getDefiningOp<memref::AllocaOp>();
      // 只替换 blockOp 之外还残留的 use
      if(newAlloc != nullptr){
        srcDefOp->getResult(0).replaceAllUsesExcept(
            newAlloc->getResult(0),
            SmallPtrSet<Operation *, 1>{op});
        rewriter.eraseOp(srcDefOp);
      }
    }
    // 删除原 blockOp（连同其 body 一起消除）
    rewriter.eraseOp(op);

    return success();
  }
};


// GemmOp的 ABC判断是否满足硬件要求。如果 memspaceA B 不满足，则加上对应buffer alloc
struct GemmOperatorInsertBuffer : public mlir::OpRewritePattern<frisk::GemmOp> {
  using mlir::OpRewritePattern<frisk::GemmOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(frisk::GemmOp op, mlir::PatternRewriter &rewriter) const override {
    auto hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000 , op->getContext());
    auto memTypeA = mlir::cast<MemRefType>(op.getMatrixA().getType());
    auto memspaceA = memTypeA.getMemorySpaceAsInt();
    auto memTypeB = mlir::cast<MemRefType>(op.getMatrixB().getType());
    auto memspaceB = memTypeB.getMemorySpaceAsInt();
    mlir::Value bufferA = nullptr;
    mlir::Value bufferB = nullptr;
    
    auto p = LowerInfoAnalysis::getGemmProblem(op);
    auto mma =  LowerInfoAnalysis::selectGemmInst(p, hw);
    assert(mma != nullptr);
    if(memspaceA != (int)mma->desc_a.memspace){
      bufferA = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),  memTypeA.getShape(), memTypeA.getElementType(), 16, (int)mma->desc_a.memspace);
      auto copyToA = rewriter.create<frisk::CopyOp>(op->getLoc(), op.getMatrixA(), bufferA);
    }
    if(memspaceB != (int)mma->desc_b.memspace){
      bufferB = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),  memTypeB.getShape(), memTypeB.getElementType(), 16, (int)mma->desc_b.memspace);
      auto copyToB = rewriter.create<frisk::CopyOp>(op->getLoc(), op.getMatrixB(), bufferB);
    }
    if(bufferA != nullptr || bufferB != nullptr){
      auto bufA = bufferA == nullptr ? op.getMatrixA() : bufferA;
      auto bufB = bufferB == nullptr ? op.getMatrixB() : bufferB;
      auto newGemm = rewriter.create<frisk::GemmOp>(op->getLoc(),  bufA, bufB, op.getMatrixC(), op.getTransA(), op.getTransB());
      rewriter.replaceOp(op, newGemm);
      return success();
    }
    else{
      return failure();
    }
  }
};


// 在frisk改写为base表达后（去掉了parallel，引入了tx） 进一步切分其他op到thread上
class ConvertFriskBaseToThreadLevelIR : public impl::ConvertFriskBaseToThreadLevelIRBase<ConvertFriskBaseToThreadLevelIR> {
public:
  
  void runOnOperation(){
    MLIRContext *context = &getContext();
    auto kernel = getOperation();
    OpBuilder builder{context};
    if(!kernel->hasAttr("thread_num")){
      return;
    }
    // -------- step 1 : 根据硬件信息，在block级别语义IR上视情况插入buffer，使得指令符合要求（如gemm的 abc memspace需求）
    RewritePatternSet ps0(context);
    ps0.add<
      GemmOperatorInsertBuffer
    >(context);
    
    // llvm::SmallVector<frisk::GemmOp> gemms {};
    // kernel->walk([&](frisk::GemmOp gemm){
    //   gemms.push_back(gemm);
    // });

    // PatternRewriter rewriter{context};
    // for(auto gemm : gemms){
    //   GemmOperatorInsertBuffer pattern{context};
    //   if(failed(pattern.matchAndRewrite(gemm, rewriter))){
    //     llvm::outs() << "rewrite failed\n";llvm::outs().flush();
    //   }
    // }

    GreedyRewriteConfig cfg;
    cfg.strictMode = GreedyRewriteStrictness::ExistingOps;
    cfg.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;

    if(failed(applyPatternsGreedily(kernel, std::move(ps0), cfg))) {
      llvm::errs() << "gemmOp buffer memspace 适配失败 !\n";
    }
    llvm::outs() << "after GemmOperatorInsertBuffer\n"; llvm::outs().flush();
    kernel->dump();
    // -------- step 2 ：进行 layoutInfer 得到block级别IR上，每个buffer的 访问模式。
    s_info = LowerInfoAnalysis::run(kernel);
    llvm::outs() << "-- lowerinfo analyze done\n";llvm::outs().flush();
    
    auto warpLayout = s_info.begin()->getSecond().get_warp_layout();
    auto blockLayout = s_info.begin()->getSecond().get_block_layout();

    kernel->setAttr("warp_layout", DenseI64ArrayAttr::get(context, warpLayout));
    kernel->setAttr("block_layout", DenseI64ArrayAttr::get(context, blockLayout));

    // 标记 tensorcore和tma设备
    kernel->walk([&](Operation* childOp){
      if(mlir::isa<frisk::GemmOp>(childOp)){
        childOp->setAttr("dev", frisk::DevKindAttr::get(context, ::mlir::frisk::attr::DevKind::TCore));
      }
      else if(mlir::isa<frisk::CopyOp>(childOp)){
        auto concreteOp = mlir::cast<frisk::CopyOp>(childOp);
        auto srcTy = concreteOp.getSrc().getType();
        if(srcTy.getMemorySpaceAsInt() == int(friskMs::Global)){
          childOp->setAttr("dev", frisk::DevKindAttr::get(context, ::mlir::frisk::attr::DevKind::TMA));
        }
      }
    });

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
      BlockOp, GemmOp
    >();
    RewritePatternSet patterns(context);
    patterns.add<
      BlockOpConversion, GemmOpConversion
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
