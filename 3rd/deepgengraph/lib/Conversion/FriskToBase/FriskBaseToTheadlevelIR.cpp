#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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

static bool isLocalMemref(Value buffer) {
  auto ty = mlir::cast<MemRefType>(buffer.getType());
  auto memorySpace = ty.getMemorySpaceAsInt();
  return memorySpace == int(friskMs::Local) || memorySpace == 5;
}

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


static gpu::ThreadIdOp findThreadIdxOp(mlir::Operation* currOp){
  auto kernel = getOuterMostOpWithName(currOp, func::FuncOp::getOperationName().data());
  assert(kernel->hasAttr("thread_num"));
  auto funcOp = mlir::cast<func::FuncOp>(kernel);
  gpu::ThreadIdOp tidx = nullptr;
  funcOp->walk([&](mlir::gpu::ThreadIdOp tidOp){
    if(tidx == nullptr && tidOp.getDimension() == gpu::Dimension::x){
      tidx = tidOp;
    }
  });
  assert(tidx != nullptr);
  return tidx;
}

static SmallVector<Value, 2> buildMappedAccessIndices(OpBuilder &builder,
                                                      Location loc,
                                                      LowerInfo &info,
                                                      Value tidx,
                                                      ArrayRef<Value> tileIvs,
                                                      unsigned rank) {
  std::vector<Value> mapOperands;
  mapOperands.push_back(tidx);

  for (int i = 0; i < 2; ++i) {
    Value iv = i < static_cast<int>(tileIvs.size())
                   ? tileIvs[i]
                   : builder.create<arith::ConstantIndexOp>(loc, 0).getResult();
    auto warpRepeat = info.get_warp_repeat()[i];
    auto threadWidth = info.get_thread_widths()[i];
    auto repeatWidth = warpRepeat * threadWidth;
    auto d0 = builder.getAffineDimExpr(0);
    auto brMap = AffineMap::get(1, 0, d0.floorDiv(repeatWidth), builder.getContext());
    auto wrMap = AffineMap::get(1, 0, (d0 % repeatWidth).floorDiv(threadWidth), builder.getContext());
    auto trMap = AffineMap::get(1, 0, d0 % threadWidth, builder.getContext());
    mapOperands.push_back(builder.create<affine::AffineApplyOp>(loc, brMap, iv));
    mapOperands.push_back(builder.create<affine::AffineApplyOp>(loc, wrMap, iv));
    mapOperands.push_back(builder.create<affine::AffineApplyOp>(loc, trMap, iv));
  }

  SmallVector<Value, 2> indices;
  auto affineExprs = info.getAffineMap();
  for (unsigned i = 0; i < rank && i < affineExprs.size(); ++i) {
    auto oneResultMap =
        AffineMap::get(info.get_dimcount(), 0, affineExprs[i], builder.getContext());
    indices.push_back(builder.create<affine::AffineApplyOp>(loc, oneResultMap, mapOperands));
  }
  return indices;
}

static SmallVector<Value, 2> makeLocalIndices(ArrayRef<Value> ivs, unsigned rank) {
  SmallVector<Value, 2> indices;
  for (unsigned i = 0; i < rank && i < ivs.size(); ++i) {
    indices.push_back(ivs[i]);
  }
  return indices;
}

class GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto tidx = findThreadIdxOp(op);
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
    auto isLocalBuffer = [](Value buffer) {
      auto ty = mlir::cast<MemRefType>(buffer.getType());
      auto memorySpace = ty.getMemorySpaceAsInt();
      return memorySpace == int(friskMs::Local) || memorySpace == 5;
    };
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
          rewriter.create<frisk::FillOp>(op->getLoc(), newAlloc, rewriter.getFloatAttr(ety, 0.0));
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
            rewriter.create<frisk::FillOp>(op->getLoc(), newAlloc, rewriter.getFloatAttr(ety, 0.0));
          }
          return newAlloc;
        }
      }
    };
    
    if(s_hw->getKind() == HW_KIND_DCU){
      auto newA = threadLevelBufferCreate(infoA, false, infoA.get_thread_widths() , isLocalBuffer(infoA.buffer));
      auto newB = threadLevelBufferCreate(infoB, false, infoB.get_thread_widths() , isLocalBuffer(infoB.buffer));
      auto newC = threadLevelBufferCreate(infoC, false, infoC.get_thread_total_widths(), true);
      auto instC = threadLevelBufferCreate(infoC, false, infoC.get_thread_widths(), false);

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
      auto instCTy = mlir::cast<MemRefType>(instC.getType());
      rewriter.create<frisk::FillOp>(op->getLoc(), instC,
                                     rewriter.getFloatAttr(instCTy.getElementType(), 0.0));
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
        _br0 * (wrC0 * twC0) + _wr0 * twC0 + _tr0,
        _br1 * (wrC1 * twC1) + _wr1 * twC1 + _tr1
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
 * @brief 应该区分 lowerinfo和 thread hold buffer size
 lowerInfo决定了该buffer应如何访问（tid+ thread持有元素数量 + br/wr/tr for循环）
 thread-buffer-sz 决定了单个线程持有多少元素
 * 
 */

class ReduceOpConversion : public OpConversionPattern<frisk::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(frisk::ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tidx = findThreadIdxOp(op);
    auto getReduceLowerInfo = [&](Value buffer) -> FailureOr<LowerInfo> {
      if (auto *info = s_info->getLowerInfo(buffer, op.getOperation())) {
        return *info;
      }
      if (auto *info = s_info->getNearestInferedInfo(buffer, op.getOperation(), true)) {
        return *info;
      }
      if (auto *info = s_info->getNearestInferedInfo(buffer, op.getOperation(), false)) {
        return *info;
      }
      return failure();
    };
    // 获取src dst的LowerInfo
    auto srcInfoOr = getReduceLowerInfo(op.getSrc());
    auto dstInfoOr = getReduceLowerInfo(op.getDst());
    if (failed(srcInfoOr) || failed(dstInfoOr)) {
      return op.emitOpError("LowerInfo not found for reduce operands");
    }
    auto srcInfo = *srcInfoOr;
    auto dstInfo = *dstInfoOr;
    srcInfo.buffer = op.getSrc();
    dstInfo.buffer = op.getDst();
    srcInfo.show("reduce_src");
    dstInfo.show("reduce_dst");
    // 获取src 的 memrefType
    auto srcTy = mlir::cast<MemRefType>(adaptor.getSrc().getType());
    auto dstTy = mlir::cast<MemRefType>(adaptor.getDst().getType());
    auto elemTy = srcTy.getElementType();
    if (elemTy != dstTy.getElementType()) {
      return op.emitOpError("source and destination element types must match");
    }
    if (!mlir::isa<FloatType>(elemTy)) {
      return op.emitOpError("thread-level reduce currently supports floating-point memrefs");
    }
    // 获取reduce的规约轴长度
    int64_t reduceDim = op.getDim();
    int64_t reduceExtent = srcTy.getDimSize(reduceDim);
    if (ShapedType::isDynamic(reduceExtent) || reduceExtent <= 0) {
      return op.emitOpError("thread-level reduce requires a positive static reduce extent");
    }
    
    // 单个线程持有的数据量
    auto [srcTw0, srcTw1] = srcInfo.get_thread_widths();
    auto [srcWr0, srcWr1] = srcInfo.get_warp_repeat();

    auto getOrCreateLocalReplacement = [&](LowerInfo &info, MemRefType originalTy,
                                           ArrayRef<int64_t> shape,
                                           bool forRead) -> FailureOr<Value> {
      auto it = s_buffer_replace.find(info.buffer);
      if (it != s_buffer_replace.end()) {
        return it->second;
      }
      if (forRead) {
        return failure();
      }
      auto newTy =
          MemRefType::get(shape, originalTy.getElementType(), AffineMap{},
                          originalTy.getMemorySpace());
      auto newBuffer = rewriter.create<memref::AllocaOp>(op->getLoc(), newTy);
      s_buffer_replace[info.buffer] = newBuffer.getResult();
      return newBuffer.getResult();
    };

    std::array<int64_t, 2> dstThreadShape2D = dstInfo.get_thread_total_widths();
    SmallVector<int64_t, 2> dstLoopShape;
    for (unsigned i = 0; i < dstTy.getRank(); ++i) {
      dstLoopShape.push_back(dstThreadShape2D[i]);
    }
    if (dstLoopShape.empty()) {
      return op.emitOpError("rank-0 reduce destination is not supported");
    }

    Value srcBuffer = op.getSrc();
    Value dstBuffer = op.getDst();
    bool srcIsLocal = isLocalMemref(srcInfo.buffer);
    bool dstIsLocal = isLocalMemref(dstInfo.buffer);
    if (srcIsLocal) {
      auto replacement = getOrCreateLocalReplacement(srcInfo, srcTy, {}, true);
      if (succeeded(replacement)) {
        srcBuffer = *replacement;
        srcTy = mlir::cast<MemRefType>(srcBuffer.getType());
        reduceExtent = srcTy.getDimSize(reduceDim);
      }
    }
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, int32_t offset, int32_t width, ShuffleMode mode);
    // rewriter.create<gpu::ShuffleOp>(op->getLoc(), )
    if (dstIsLocal) {
      auto replacement = getOrCreateLocalReplacement(dstInfo, dstTy, dstLoopShape, false);
      if (failed(replacement)) {
        return failure();
      }
      dstBuffer = *replacement;
      dstTy = mlir::cast<MemRefType>(dstBuffer.getType());
    }

    auto makeIdentity = [&]() -> FailureOr<Value> {
      double identity = 0.0;
      auto kind = op.getKind();
      if (kind == "add") {
        identity = 0.0;
      } else if (kind == "mul") {
        identity = 1.0;
      } else if (kind == "min") {
        identity = std::numeric_limits<double>::infinity();
      } else if (kind == "max") {
        identity = -std::numeric_limits<double>::infinity();
      } else {
        return failure();
      }
      auto attr = rewriter.getFloatAttr(elemTy, identity);
      return rewriter.create<arith::ConstantOp>(op->getLoc(), attr).getResult();
    };

    auto combine = [&](Value lhs, Value rhs) -> FailureOr<Value> {
      auto kind = op.getKind();
      if (kind == "add") {
        return rewriter.create<arith::AddFOp>(op->getLoc(), lhs, rhs).getResult();
      }
      if (kind == "mul") {
        return rewriter.create<arith::MulFOp>(op->getLoc(), lhs, rhs).getResult();
      }
      if (kind == "min") {
        return rewriter.create<arith::MinNumFOp>(op->getLoc(), lhs, rhs).getResult();
      }
      if (kind == "max") {
        return rewriter.create<arith::MaxNumFOp>(op->getLoc(), lhs, rhs).getResult();
      }
      return failure();
    };

    std::vector<int> loopUbs;
    for (int64_t ub : dstLoopShape) {
      loopUbs.push_back(static_cast<int>(ub));
    }
    std::vector<Value> dstTileIvs;
    createNestedAffineFor(rewriter, op->getLoc(), loopUbs, dstTileIvs);

    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto accTy = MemRefType::get({1}, elemTy);
    auto acc = rewriter.create<memref::AllocaOp>(op->getLoc(), accTy);
    auto identity = makeIdentity();
    if (failed(identity)) {
      return op.emitOpError("unsupported reduce kind");
    }
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), *identity, acc,
                                           ValueRange{zeroIdx});

    auto dstIndices =
        dstIsLocal ? makeLocalIndices(dstTileIvs, dstTy.getRank())
                   : buildMappedAccessIndices(rewriter, op->getLoc(), dstInfo,
                                              tidx, dstTileIvs, dstTy.getRank());

    std::vector<Value> redIvs;
    auto redLoops = createNestedAffineFor(rewriter, op->getLoc(),
                                          {static_cast<int>(reduceExtent)}, redIvs);

    SmallVector<Value, 2> srcIndices;
    if (srcIsLocal && srcBuffer != op.getSrc()) {
      if (srcTy.getRank() == dstTy.getRank()) {
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          srcIndices.push_back(i == static_cast<unsigned>(reduceDim) ? redIvs[0]
                                                                     : dstTileIvs[i]);
        }
      } else {
        unsigned dstPos = 0;
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          if (i == static_cast<unsigned>(reduceDim)) {
            srcIndices.push_back(redIvs[0]);
          } else {
            srcIndices.push_back(dstTileIvs[dstPos++]);
          }
        }
      }
    } else {
      if (srcTy.getRank() == dstTy.getRank()) {
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          srcIndices.push_back(i == static_cast<unsigned>(reduceDim) ? redIvs[0]
                                                                     : dstIndices[i]);
        }
      } else {
        unsigned dstPos = 0;
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          if (i == static_cast<unsigned>(reduceDim)) {
            srcIndices.push_back(redIvs[0]);
          } else {
            srcIndices.push_back(dstIndices[dstPos++]);
          }
        }
      }
    }

    auto current =
        rewriter.create<affine::AffineLoadOp>(op->getLoc(), acc, ValueRange{zeroIdx});
    auto srcValue =
        rewriter.create<affine::AffineLoadOp>(op->getLoc(), srcBuffer, srcIndices);
    auto next = combine(current.getResult(), srcValue.getResult());
    if (failed(next)) {
      return op.emitOpError("unsupported reduce kind");
    }
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), *next, acc,
                                           ValueRange{zeroIdx});

    rewriter.setInsertionPointAfter(redLoops.back());
    auto result =
        rewriter.create<affine::AffineLoadOp>(op->getLoc(), acc, ValueRange{zeroIdx});
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), result.getResult(), dstBuffer,
                                           dstIndices);
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
    struct AccessedBufferInfo {
      Value buffer;
      std::optional<LowerInfo> lowerInfo;
    };

    std::array<int64_t, 2> threadLevelSize = {0, 0};
    llvm::DenseMap<AllocBufferOp, std::array<int64_t, 2>> allocLocalsToReplace;
    std::vector<AccessedBufferInfo> bufferInfos;

    auto get2DShape = [](Value buffer) -> std::array<int64_t, 2> {
      auto ty = cast<MemRefType>(buffer.getType());
      auto shape = ty.getShape();
      if (shape.size() == 1) {
        return {shape[0], 1};
      }
      return {shape[0], shape[1]};
    };

    auto setThreadLevelSizeIfUnset = [&](std::array<int64_t, 2> sz) {
      if (threadLevelSize[0] == 0 && threadLevelSize[1] == 0) {
        threadLevelSize = sz;
      }
    };

    auto isThreadLocalTile = [](Value buffer) {
      if (auto alloc = buffer.getDefiningOp<AllocBufferOp>()) {
        return alloc.getMemorySpace() == int(friskMs::Local);
      }
      auto ty = cast<MemRefType>(buffer.getType());
      auto memorySpace = ty.getMemorySpaceAsInt();
      return memorySpace == int(friskMs::Local) || memorySpace == 5;
    };

    auto findBufferInfo = [&](Value buffer) -> AccessedBufferInfo * {
      for (auto &info : bufferInfos) {
        if (info.buffer == buffer) {
          return &info;
        }
      }
      return nullptr;
    };

    auto recordBufferInfo = [&](Value buffer, const char *label) {
      if (!isa<MemRefType>(buffer.getType()) || findBufferInfo(buffer) != nullptr) {
        return;
      }

      AccessedBufferInfo recordedInfo{buffer, std::nullopt};
      if (auto *lowerInfo = s_info->getLowerInfo(buffer, op.getOperation())) {
        recordedInfo.lowerInfo = *lowerInfo;
        setThreadLevelSizeIfUnset(lowerInfo->get_thread_total_widths());
        lowerInfo->show(label);
      } else {
        // 前面的 block conversion 可能已经把原 frisk.alloc_buffer 替换为
        // memref.alloca 形式的 thread tile。这个新 value 已经是 lowered 后
        // 的形态，不会出现在 LowerInfoMap 中，因此直接用当前 memref shape
        // 作为循环 shape。
        setThreadLevelSizeIfUnset(get2DShape(buffer));
      }
      bufferInfos.push_back(recordedInfo);
    };
    
    // 对每个affine.load 检查其memref，记录buffer Info。
    // 追踪来源 srcDefOp， 标记 allocLocalsToReplace[srcDefOp] = lowerInfo 的thread总计算量
    for (auto loadOp : loadOps) {
      auto srcValue = loadOp.getMemref();
      recordBufferInfo(srcValue, "block_load");
      auto srcDefOp = srcValue.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(srcValue);
      if (srcDefOp != nullptr && info != nullptr && info->lowerInfo && isThreadLocalTile(srcValue) &&
          !allocLocalsToReplace.count(srcDefOp)) {
        allocLocalsToReplace[srcDefOp] = info->lowerInfo->get_thread_total_widths();
      }
    }

    for (auto storeOp : storeOps) {
      auto dstVal = storeOp.getMemref();
      recordBufferInfo(dstVal, "block_store");
      auto dstDefOp = dstVal.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(dstVal);
      if (dstDefOp != nullptr && info != nullptr && info->lowerInfo && isThreadLocalTile(dstVal) &&
          !allocLocalsToReplace.count(dstDefOp)) {
        allocLocalsToReplace[dstDefOp] = info->lowerInfo->get_thread_total_widths();
      }
    }

    assert(threadLevelSize[0] > 0 && threadLevelSize[1] > 0 && "thread-level block size not inferred");
    IRMapping mapper;
    IRMapping localMapper;
    // local/register block buffer 会物化成每个线程自己的 tile；shared buffer
    // 仍然保持 block-sized，后面通过 LowerInfo map 生成真实 block 坐标。
    for (auto &[srcDefOp, sz] : allocLocalsToReplace) {
      rewriter.setInsertionPoint(srcDefOp);
      auto ty = MemRefType::get(sz, srcDefOp.getElementType(), AffineMap{}, srcDefOp.getMemorySpace());
      auto it = s_buffer_replace.find(srcDefOp);
      mlir::Value replaceVal = nullptr;
      if (it != s_buffer_replace.end()) {
        replaceVal = it->second;
      } else {
        auto newAlloc = rewriter.create<memref::AllocaOp>(srcDefOp->getLoc(), ty);
        replaceVal = newAlloc->getResult(0);
        s_buffer_replace[srcDefOp] = newAlloc;
      }
      mapper.map(srcDefOp->getResult(0), replaceVal);
      localMapper.map(srcDefOp->getResult(0), replaceVal);
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

    // newIvs 是 thread-level local tile 坐标，例如 4x32。local/register
    // buffer 用它直接访问；shared/block buffer 必须恢复成原 128x128 block
    // 内的真实坐标，否则就会出现 4x32 循环写 128x128 memref 的错配。
    std::vector<Value> blockIvs;
    LowerInfo *blockIndexInfo = nullptr;
    for (auto &info : bufferInfos) {
      if (info.lowerInfo && !isThreadLocalTile(info.buffer)) {
        blockIndexInfo = &*info.lowerInfo;
        break;
      }
    }
    if (blockIndexInfo != nullptr && newIvs.size() == 2) {
      std::vector<Value> mapOperands;
      mapOperands.push_back(tidx);
      for (int i = 0; i < 2; ++i) {
        auto warpRepeat = blockIndexInfo->get_warp_repeat()[i];
        auto threadWidth = blockIndexInfo->get_thread_widths()[i];
        auto repeatWidth = warpRepeat * threadWidth;
        auto d0 = rewriter.getAffineDimExpr(0);
        auto brMap = AffineMap::get(1, 0, d0.floorDiv(repeatWidth), rewriter.getContext());
        auto wrMap = AffineMap::get(1, 0, (d0 % repeatWidth).floorDiv(threadWidth), rewriter.getContext());
        auto trMap = AffineMap::get(1, 0, d0 % threadWidth, rewriter.getContext());
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), brMap, newIvs[i]));
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), wrMap, newIvs[i]));
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), trMap, newIvs[i]));
      }

      // LowerInfo::getAffineMap() 的 operand 顺序是
      // [tid, blockX, warpX, threadX, blockY, warpY, threadY]。这里先把
      // 压平后的 thread tile iv 拆回这几层，再得到 shared/block 坐标。
      auto affineExprs = blockIndexInfo->getAffineMap();
      for (auto expr : affineExprs) {
        auto oneResultMap = AffineMap::get(blockIndexInfo->get_dimcount(), 0, expr, rewriter.getContext());
        blockIvs.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), oneResultMap, mapOperands));
      }
    } else {
      blockIvs = newIvs;
    }

    for(auto [oldIndex, newIter] : llvm::zip(op.getBody()->getArguments(), blockIvs)){
      mapper.map(oldIndex,newIter);
    }
    for(auto [oldIndex, newIter] : llvm::zip(op.getBody()->getArguments(), newIvs)){
      localMapper.map(oldIndex,newIter);
    }
    // 根据映射规则，将frisk.blockOp 内的全部op 搬运到 nestedFOr的最内层 （createNestedAffineFor 之后，insertionPoint已经在最内侧了，不用动）
    // 将 blockOp body 内的所有 op 按序 clone 到当前 insertionPoint（nestedFor 最内层）
    // 跳过 block terminator（frisk.yield 或类似）
    Block *body = op.getBody();
    for (auto &childOp : body->without_terminator()) {
      IRMapping *activeMapper = &mapper;
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(childOp)) {
        if (isThreadLocalTile(loadOp.getMemref())) {
          activeMapper = &localMapper;
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(childOp)) {
        if (isThreadLocalTile(storeOp.getMemref())) {
          activeMapper = &localMapper;
        }
      }
      auto *cloned = rewriter.clone(childOp, *activeMapper);
      // 后续普通 arith/math op 使用 mapper，后续 local store 使用 localMapper。
      // 因此不管当前 op 用哪套下标克隆，都把 result 同步给两套映射。
      for (auto [oldResult, newResult] : llvm::zip(childOp.getResults(), cloned->getResults())) {
        if (!mapper.lookupOrNull(oldResult)) {
          mapper.map(oldResult, newResult);
        }
        if (!localMapper.lookupOrNull(oldResult)) {
          localMapper.map(oldResult, newResult);
        }
      }
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
    assert(srcInfo != nullptr && dstInfo != nullptr && "copy-convert LowerInfo not found");
    auto [tw0,tw1] = srcInfo->get_thread_total_widths();
    auto [dstTw0, dstTw1] = dstInfo->get_thread_total_widths();
    assert(tw0 == dstTw0 && tw1 == dstTw1 &&
           "copy-convert expects source and destination thread tiles to match");
    std::vector<int> ubs = {int(tw0), int(tw1)};
    std::vector<mlir::Value> outIvs;

    auto isThreadLocalTile = [](Value buffer) {
      auto ty = cast<MemRefType>(buffer.getType());
      auto memorySpace = ty.getMemorySpaceAsInt();
      return memorySpace == int(friskMs::Local) || memorySpace == 5;
    };

    auto getOrCreateLocalReplacement = [&](LowerInfo *info, Type elementType,
                                           bool forRead) -> FailureOr<Value> {
      auto it = s_buffer_replace.find(info->buffer);
      if (it != s_buffer_replace.end()) {
        return it->second;
      }
      if (forRead) {
        return failure();
      }
      std::vector<int64_t> shape = {tw0, tw1};
      auto newBuffer = rewriter.create<frisk::AllocBufferOp>(
          op->getLoc(), shape, elementType, 1, int(friskMs::Local));
      s_buffer_replace[info->buffer] = newBuffer;
      return newBuffer.getResult();
    };

    createNestedAffineFor(rewriter, op->getLoc(), ubs, outIvs);

    auto getAccessIndices = [&](LowerInfo *info,
                                bool useLocalTile) -> SmallVector<Value, 2> {
      if (useLocalTile) {
        return SmallVector<Value, 2>{outIvs.begin(), outIvs.end()};
      }

      std::vector<Value> mapOperands;
      mapOperands.push_back(tidx);
      for (int i = 0; i < 2; ++i) {
        auto warpRepeat = info->get_warp_repeat()[i];
        auto threadWidth = info->get_thread_widths()[i];
        auto repeatWidth = warpRepeat * threadWidth;
        auto d0 = rewriter.getAffineDimExpr(0);
        auto brMap = AffineMap::get(1, 0, d0.floorDiv(repeatWidth), rewriter.getContext());
        auto wrMap = AffineMap::get(1, 0, (d0 % repeatWidth).floorDiv(threadWidth), rewriter.getContext());
        auto trMap = AffineMap::get(1, 0, d0 % threadWidth, rewriter.getContext());
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), brMap, outIvs[i]));
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), wrMap, outIvs[i]));
        mapOperands.push_back(rewriter.create<affine::AffineApplyOp>(op->getLoc(), trMap, outIvs[i]));
      }

      SmallVector<Value, 2> indices;
      auto affineExprs = info->getAffineMap();
      for (auto expr : affineExprs) {
        auto oneResultMap =
            AffineMap::get(info->get_dimcount(), 0, expr, rewriter.getContext());
        indices.push_back(
            rewriter.create<affine::AffineApplyOp>(op->getLoc(), oneResultMap, mapOperands));
      }
      return indices;
    };

    bool srcIsLocal = isThreadLocalTile(srcInfo->buffer);
    bool dstIsLocal = isThreadLocalTile(dstInfo->buffer);
    auto srcBuffer = srcInfo->buffer;
    auto dstBuffer = dstInfo->buffer;
    if (srcIsLocal) {
      auto replacement = getOrCreateLocalReplacement(srcInfo, srcTy.getElementType(), true);
      if (failed(replacement)) {
        return op.emitOpError("local source has no thread-level replacement for dtype conversion");
      }
      srcBuffer = *replacement;
    }
    if (dstIsLocal) {
      auto replacement = getOrCreateLocalReplacement(dstInfo, dstTy.getElementType(), false);
      if (failed(replacement)) {
        return failure();
      }
      dstBuffer = *replacement;
    }

    auto srcValue = rewriter.create<affine::AffineLoadOp>(
        op->getLoc(), srcBuffer, getAccessIndices(srcInfo, srcIsLocal));
    mlir::Value converted{};
    if(srcTy.getElementType().getIntOrFloatBitWidth() < dstTy.getElementType().getIntOrFloatBitWidth()){
      converted = rewriter.create<arith::ExtFOp>(op->getLoc(), dstTy.getElementType(), srcValue.getResult() );
    }
    else{
      converted = rewriter.create<arith::TruncFOp>(op->getLoc(), dstTy.getElementType(), srcValue.getResult() );
    }
    rewriter.create<affine::AffineStoreOp>(
        op->getLoc(), converted, dstBuffer, getAccessIndices(dstInfo, dstIsLocal));
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
      BlockOp, GemmOp, ReduceOp
    >();
    target.addDynamicallyLegalOp<frisk::CopyOp>([](frisk::CopyOp op){
      auto srctype = mlir::cast<MemRefType>(op.getSrcMemRef().getType());
      auto dsttype = mlir::cast<MemRefType>(op.getDstMemRef().getType());
      return !(srctype.getElementType() != dsttype.getElementType() && 
        srctype.getShape() == dsttype.getShape());
    });
    RewritePatternSet patterns(context);
    patterns.add<
      BlockOpConversion, GemmOpConversion, ReduceOpConversion, CopyConvertOpRewrite
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
