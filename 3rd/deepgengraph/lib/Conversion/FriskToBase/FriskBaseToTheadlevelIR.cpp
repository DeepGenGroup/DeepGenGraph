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
#include "mlir/IR/BuiltinOps.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;
static LowerInfoMap* s_info { nullptr};
static HWSpecification* s_hw {nullptr};

static DenseMap<mlir::Value, mlir::Value> s_buffer_replace;

static LowerInfo getLowerInfoOrDie(Value buffer, Operation *op) {
  LowerInfo *info = s_info->getLowerInfo(buffer, op);
  // 注意 ： 运行到这里时，lowerInfo必须全部推断完毕
  // if (info == nullptr) {
  //   info = s_info->getLastInfferedInfo(buffer, op);
  // }
  assert(info != nullptr && "LowerInfo not found");
  return *info;
}

static std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs) {
  std::vector<mlir::affine::AffineForOp> loops;
  loops.reserve(upperBounds.size());

  // 确保标签数量与循环层数一致（可选的安全检查）
  size_t numLoops = upperBounds.size();
  
  for (size_t i = 0; i < numLoops; ++i) {
    // 1. 定义下界、上界和步长 (下界默认为 0，步长默认为 1)
    int64_t lowerBound = 0;
    int64_t step = 1;
    auto ub = upperBounds[i];
    // 2. 创建当前层的 AffineForOp
    auto forOp = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBounds[i], step);
    mlir::Value iv = forOp.getInductionVar();
    // 4. 收集当前循环的迭代变量 (Induction Variable) 和 Op 本身
    outIvs.push_back(iv);
    loops.push_back(forOp);
    // 5. 将 builder 的插入点移动到当前循环体的末尾（yield 之前），以便下一层循环嵌套在内部
    builder.setInsertionPointToStart(forOp.getBody());
  }

  return loops;
}

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
    // get lowerInfo
    auto infoA = getLowerInfoOrDie(op.getA(), op.getOperation());
    auto infoB = getLowerInfoOrDie(op.getB(), op.getOperation());
    auto infoC = getLowerInfoOrDie(op.getC(), op.getOperation());
    infoA.show("A");
    infoB.show("B");
    infoC.show("C");
    
    assert(infoA.get_block_repeat()[1] == infoB.get_block_repeat()[0]);  // k轴上的 for循环次数. A 列迭代数 == B 行迭代数
    assert(infoA.mmaInst->name == infoB.mmaInst->name);
    auto typeA = mlir::cast<MemRefType>(adaptor.getA().getType());
    auto typeB = mlir::cast<MemRefType>(adaptor.getB().getType());
    // 创建thread-level 的buffer。并全局查找其是否已经完成alloc了。若没有，注册之；否则使用已注册的thread-level buffer
    auto threadLevelBufferCreate = [&](LowerInfo info, bool needInitZero, coordXY_t shape, bool registReplace )-> Value{
      // acc：需要保存当前线程负责的bufferC所有数据。结果不能跨循环覆盖
      // 否则只需要保留单次 wmma 的所需数据即可。可跨循环复用
      auto memrefTy = mlir::cast<MemRefType>(info.buffer.getType());
      auto ety = memrefTy.getElementType();

      auto it = s_buffer_replace.find(info.buffer);
      if(it == s_buffer_replace.end()){
        auto newAlloc = rewriter.create<AllocBufferOp>(op->getLoc(), shape, ety, 1,int(friskMs::Local));
        if(registReplace){
          s_buffer_replace[info.buffer] = newAlloc;
        }
        if(needInitZero){
          // acc 需要初始化为0
          rewriter.create<frisk::FillOp>(op->getLoc(), info.buffer, rewriter.getFloatAttr(ety, 0.0));
        }
        return newAlloc;
      }
      else{
        if(registReplace){
          return it->second;  
        }
        else{
          auto newAlloc = rewriter.create<AllocBufferOp>(op->getLoc(), shape, ety, 1, int(friskMs::Local));
          if (needInitZero) {
            // acc 需要初始化为0
            rewriter.create<frisk::FillOp>(op->getLoc(), info.buffer, rewriter.getFloatAttr(ety, 0.0));
          }
          return newAlloc;
        }
      }
    };
    
    if(s_hw->getKind() == HW_KIND_DCU){
      auto newA = threadLevelBufferCreate(infoA, false, infoA.get_thread_widths() , true);
      auto newB = threadLevelBufferCreate(infoB, false, infoB.get_thread_widths() , true);
      auto newC = threadLevelBufferCreate(infoC, false, infoC.get_thread_total_widths(), true);
      auto instC = threadLevelBufferCreate(infoC, true, infoC.get_thread_widths(), false);

      auto [br0, br1] = infoC.get_block_repeat();
      int twA0 = infoA.get_thread_widths()[0];
      int twA1 = infoA.get_thread_widths()[1];
      int wrA0 = infoA.get_warp_repeat()[0];
      int wrA1 = infoA.get_warp_repeat()[1];
      
      int twB0 = infoB.get_thread_widths()[0];
      int twB1 = infoB.get_thread_widths()[1];
      int wrB0 = infoB.get_warp_repeat()[0];
      int wrB1 = infoB.get_warp_repeat()[1];
      

      int kloopCount = infoA.get_block_repeat()[1];

      std::vector<int> mn_loops = {int(br0), int(br1)};
      std::vector<int> k_loops = {kloopCount};
      std::vector<Value> ivs_block {};  // itervar mnk

      // 指令在bufferC上的循环(m,n)
      createNestedAffineFor(rewriter, op->getLoc(), mn_loops, ivs_block);
      // insPoint 位于 mn loop内
      {
        RewriterBase::InsertionGuard ig{rewriter};
        createNestedAffineFor(rewriter, op->getLoc(), k_loops, ivs_block);
        // insPoint 位于 kloop 内 
        // 单次指令所需数据的构建 A
        if (mlir::cast<MemRefType>(infoA.buffer.getType()).getMemorySpaceAsInt() == (int)friskMs::Shared) {
          RewriterBase::InsertionGuard _temp{rewriter};
          std::vector<int> ubs = {wrA0, wrA1, twA0, twA1};
          std::vector<mlir::Value> instIvs;
          createNestedAffineFor(rewriter, op->getLoc(), ubs, instIvs);
          std::vector<Value> mapOperands{
              tidx,         ivs_block[0], instIvs[0], instIvs[2],
              ivs_block[1], instIvs[1],   instIvs[3]}; // 0:tidx, 1:iv_bx, iv_wx , iv_tx ,iv_by, iv_wy, iv_ty
          auto map = mlir::AffineMap::get(infoA.get_dimcount(), 0, infoA.getAffineMap(), rewriter.getContext());
          rewriter.create<frisk::CopyOp>(op->getLoc(), infoA.buffer, newA, mapOperands, map);
        }
        // 单次指令所需数据的构建 B
        if (mlir::cast<MemRefType>(infoB.buffer.getType()).getMemorySpaceAsInt() == (int)friskMs::Shared) {
          RewriterBase::InsertionGuard _temp{rewriter};
          std::vector<int> ubs = {wrB0, wrB1, twB0, twB1};
          std::vector<mlir::Value> instIvs;
          createNestedAffineFor(rewriter, op->getLoc(), ubs, instIvs);
          std::vector<Value> mapOperands{
              tidx,         ivs_block[0], instIvs[0], instIvs[2],
              ivs_block[1], instIvs[1],   instIvs[3]}; // 0:tidx, 1:iv_bx, iv_wx , iv_tx ,iv_by, iv_wy, iv_ty
          auto map = mlir::AffineMap::get(infoB.get_dimcount(), 0, infoB.getAffineMap(), rewriter.getContext());
          rewriter.create<frisk::CopyOp>(op->getLoc(), infoB.buffer, newB, mapOperands, map);
        }
        // AB copy ok. 计算wmma（ instC 具有累加语义）
        rewriter.create<frisk::WarpMmaOp>(op->getLoc(), newA, newB, instC);
      }
      // kloop ends. 需要将instC累加结果写回 newC with loopMN
      auto [wrC0, wrC1] = infoC.get_warp_repeat();
      auto [twC0, twC1] = infoC.get_thread_widths();
      std::vector<int> loopUbs = {(int)wrC0,(int)wrC1,(int)twC0,(int)twC1};
      std::vector<mlir::Value> ivs{};
      // 构建循环写回: 点对点拷贝(或许能用 affine.load/store 拷贝元素。 但后期需向量化优化)
      createNestedAffineFor(rewriter, op->getLoc(), loopUbs, ivs);
      auto _br0 = rewriter.getAffineDimExpr(0);
      auto _br1 = rewriter.getAffineDimExpr(1);
      auto _wr0 = rewriter.getAffineDimExpr(2);
      auto _wr1 = rewriter.getAffineDimExpr(3);
      auto _tr0 = rewriter.getAffineDimExpr(4);
      auto _tr1 = rewriter.getAffineDimExpr(5);
      auto _tid = rewriter.getAffineDimExpr(6);
      
      std::vector<AffineExpr> instCToNewC = {
        _br0 * _wr0 * twC0 + _tr0,  
        _br1 * _wr1 * twC1 + _tr1  
      };
      auto instCToNewCMap = AffineMap::get(7,0,instCToNewC, rewriter.getContext());
      std::vector<Value> mapOper = {ivs_block[0], ivs_block[1]  ,ivs[0],ivs[1], ivs[2], ivs[3], tidx  };
      rewriter.create<frisk::CopyOp>(op->getLoc() ,instC, newC, mapOper ,instCToNewCMap);
      // newC 写回完成
      // 后续使用newC 做其他op的计算（之前已经过Layout推定）
      // rewriter.replaceAllUsesWith(infoC.buffer, newC);  // notes 这里不进行buffer的替换。因后续op的convert依赖于buffer的loweInfo。而新buffer无LowerInfo，故会报错
      // 解决方案：全局记录block-level buffer的replace。如果有，直接使用。若没有，自己新建
      rewriter.eraseOp(op);
    }
    else if(s_hw->getKind() == HW_KIND_NVIDIA){
      // ...
    }
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
          auto i = getLowerInfoOrDie(srcValue, op.getOperation());
          allocLocalsToReplace[srcDefOp] = i.get_thread_total_widths();
          i.show("block_load");
        }
      // }
    }

    for (auto storeOp : storeOps) {
      auto dstVal = storeOp.getMemref();
      // if (dstVal.getType().getMemorySpaceAsInt() == int(friskMs::Local)) {
        auto srcDefOp = dstVal.getDefiningOp<AllocBufferOp>();
        if (srcDefOp != nullptr && !allocLocalsToReplace.count(srcDefOp)) {
          auto i = getLowerInfoOrDie(dstVal, op.getOperation());
          allocLocalsToReplace[srcDefOp] = i.get_thread_total_widths();
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
        auto it = s_buffer_replace.find(srcDefOp);
        mlir::Value replaceVal = nullptr;
        if(it != s_buffer_replace.end()){
          replaceVal = it->second;
        }
        else{
          auto newAlloc = rewriter.create<memref::AllocaOp>(srcDefOp->getLoc(), ty);
          replaceVal = newAlloc->getResult(0);
          s_buffer_replace[srcDefOp] = newAlloc;  
        }
        mapper.map(srcDefOp->getResult(0), replaceVal);
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

// 用于重写 frisk.copy 进行数据类型转换的情况。frisk.copy <3x3xf32> to <3x3xf16>
class CopyConvertOpRewrite : public OpConversionPattern<frisk::CopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 匹配模式检查
    auto srcTy = mlir::cast<MemRefType>(op.getSrcMemRef().getType());
    auto dstTy = mlir::cast<MemRefType>(op.getDstMemRef().getType());
    bool needConvert = false;
    if(srcTy.getShape() == dstTy.getShape() && srcTy.getElementType() != dstTy.getElementType()){
      needConvert = true;
    }
    if(!needConvert){
      return failure();
    }

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

    auto srcInfo = s_info->getLowerInfo(op.getSrcMemRef(), op);
    auto dstInfo = s_info->getLowerInfo(op.getDstMemRef(), op);
    auto [tw0,tw1] = srcInfo->get_thread_total_widths();
    std::vector<int> ubs = {int(tw0), int(tw1)};
    std::vector<mlir::Value> outIvs;
    auto itSrc = s_buffer_replace.find(srcInfo->buffer);
    if(itSrc == s_buffer_replace.end()){
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> shape, Type elementType, int64_t alignment, int64_t memorySpace);
      std::vector<int64_t> shape = {tw0,tw1};
      auto newBuffer = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), shape, srcTy.getElementType(),1, int(friskMs::Local));
      s_buffer_replace[srcInfo->buffer] = newBuffer;
    }
    
    auto itDst = s_buffer_replace.find(dstInfo->buffer);
    if(itDst == s_buffer_replace.end()){
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> shape, Type elementType, int64_t alignment, int64_t memorySpace);
      std::vector<int64_t> shape = {tw0,tw1};
      auto newBuffer = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), shape, dstTy.getElementType(),1, int(friskMs::Local));
      s_buffer_replace[dstInfo->buffer] = newBuffer;
    }
    
    createNestedAffineFor(rewriter, op->getLoc(), ubs, outIvs);
    auto srcValue = rewriter.create<affine::AffineLoadOp>(op->getLoc(), s_buffer_replace[srcInfo->buffer], outIvs);
    mlir::Value converted{};
    if(srcTy.getElementType().getIntOrFloatBitWidth() < dstTy.getElementType().getIntOrFloatBitWidth()){
      converted = rewriter.create<arith::ExtFOp>(op->getLoc(), dstTy.getElementType(), srcValue.getResult() );
    }
    else{
      converted = rewriter.create<arith::TruncFOp>(op->getLoc(), dstTy.getElementType(), srcValue.getResult() );
    }
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), converted, s_buffer_replace[dstInfo->buffer], outIvs);
    rewriter.eraseOp(op);
    return success();
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
    if(s_hw == nullptr){
      s_hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, context);
    }

    // -------- step 1 ：进行 layoutInfer 得到block级别IR上，每个buffer的 访问模式。
    s_info = LowerInfoAnalysis::run(kernel);
    llvm::outs() << "-- lowerinfo analyze done\n";llvm::outs().flush();
    
    auto warpLayout = s_info->begin()->getSecond().get_warp_layout();
    auto blockLayout = s_info->begin()->getSecond().get_block_layout();

    kernel->setAttr("warp_layout", DenseI64ArrayAttr::get(context, warpLayout));
    kernel->setAttr("block_layout", DenseI64ArrayAttr::get(context, blockLayout));

    ConversionTarget target(*context);
  
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
    target.addDynamicallyLegalOp<frisk::CopyOp>([](frisk::CopyOp op){
      auto srctype = mlir::cast<MemRefType>(op.getSrcMemRef().getType());
      auto dsttype = mlir::cast<MemRefType>(op.getDstMemRef().getType());
      return !(srctype.getElementType() != dsttype.getElementType() && 
        srctype.getShape() == dsttype.getShape());
    });
    RewritePatternSet patterns(context);
    patterns.add<
      BlockOpConversion, GemmOpConversion, CopyConvertOpRewrite
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
