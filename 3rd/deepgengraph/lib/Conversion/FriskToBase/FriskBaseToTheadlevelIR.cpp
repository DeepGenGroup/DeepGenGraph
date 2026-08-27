#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR

#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;
static LowerInfoMap* s_info { nullptr};
static HWSpecification* s_hw {nullptr};

static DenseMap<mlir::Value, mlir::Value> s_buffer_replace;
static DenseMap<frisk::ConvertLayoutOp , std::pair<LowerInfo*, LowerInfo*>> convertLayoutInfo;

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

static void insertConvertLayoutOps(LowerInfoMap &infoMap ) {
  SmallVector<std::pair<Value, Operation *>, 8> inserted;

  for (auto &entry : infoMap) {
    LowerInfo &toInfo = entry.second;
    if (toInfo.convertFrom == nullptr || toInfo.buffer == nullptr ||
        toInfo.op == nullptr) {
      continue;
    }

    auto key = std::make_pair(toInfo.buffer, toInfo.op);
    if (llvm::any_of(inserted, [&](const auto &insertedKey) {
          return insertedKey == key;
        })) {
      continue;
    }
    inserted.push_back(key);

    OpBuilder builder(toInfo.op);
    auto convertLayout = builder.create<frisk::ConvertLayoutOp>(
        toInfo.op->getLoc(), toInfo.buffer,
        builder.getStringAttr("lowerinfo.convert"));
    convertLayoutInfo.insert({convertLayout, {toInfo.convertFrom, &toInfo}});
  }
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

static SmallVector<Value, 2> makeLocalIndices(ArrayRef<Value> ivs, unsigned rank) {
  SmallVector<Value, 2> indices;
  for (unsigned i = 0; i < rank && i < ivs.size(); ++i) {
    indices.push_back(ivs[i]);
  }
  return indices;
}

static Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static Value createSingleDimAffineApply(OpBuilder &builder, Location loc,
                                        AffineExpr expr, Value operand) {
  auto map = AffineMap::get(1, 0, expr, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, operand);
}

static Value modBy(OpBuilder &builder, Location loc, Value operand,
                   int64_t divisor) {
  assert(divisor > 0 && "affine modulo divisor must be positive");
  if (divisor == 1) {
    return createIndexConstant(builder, loc, 0);
  }
  auto d0 = builder.getAffineDimExpr(0);
  return createSingleDimAffineApply(builder, loc, d0 % divisor, operand);
}

static Value floorDivBy(OpBuilder &builder, Location loc, Value operand,
                        int64_t divisor) {
  assert(divisor > 0 && "affine floordiv divisor must be positive");
  if (divisor == 1) {
    return operand;
  }
  auto d0 = builder.getAffineDimExpr(0);
  return createSingleDimAffineApply(builder, loc, d0.floorDiv(divisor), operand);
}

static Value flattenXY(OpBuilder &builder, Location loc, ArrayRef<Value> xy,
                       coordXY_t order, coordXY_t layout) {
  assert(xy.size() == 2 && "expected two coordinates");
  if (flat_size(layout) == 1) {
    return createIndexConstant(builder, loc, 0);
  }
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  AffineExpr dims[] = {d0, d1};
  AffineExpr flat = dims[order[0]] + dims[order[1]] * layout[order[0]];
  auto map = AffineMap::get(2, 0, flat, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, xy);
}

static SmallVector<Value, 7>
buildLowerInfoMapOperands(OpBuilder &builder, Location loc, LowerInfo &info,
                          Value tidx, Value br0, Value br1, Value iu0,
                          Value iu1, Value wr0, Value wr1, Value reg0,
                          Value reg1) {
  SmallVector<Value, 7> operands;
  operands.push_back(tidx);
  operands.push_back(br0);
  operands.push_back(br1);
  operands.push_back(iu0);
  operands.push_back(iu1);
  SmallVector<Value, 2> wrXY{wr0, wr1};
  SmallVector<Value, 2> regXY{reg0, reg1};
  operands.push_back(flattenXY(builder, loc, wrXY,
                               info.base_layout.warp_repeat_order,
                               info.get_warp_repeat()));
  operands.push_back(flattenXY(builder, loc, regXY,
                               info.base_layout.thread_creg_order,
                               info.get_thread_widths()));
  return operands;
}

static SmallVector<Value, 2> applyLowerInfoMap(OpBuilder &builder, Location loc,
                                               LowerInfo &info,
                                               ArrayRef<Value> mapOperands,
                                               unsigned rank) {
  SmallVector<Value, 2> indices;
  auto map = info.getAffineMap();
  for (unsigned i = 0; i < rank && i < map.getNumResults(); ++i) {
    if (info.ignoreDim >= 0 && static_cast<unsigned>(info.ignoreDim) == i) {
      indices.push_back(createIndexConstant(builder, loc, 0));
      continue;
    }
    auto oneResultMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), map.getResult(i),
                       builder.getContext());
    indices.push_back(
        builder.create<affine::AffineApplyOp>(loc, oneResultMap, mapOperands));
  }
  return indices;
}

static SmallVector<Value, 2>
buildMappedAccessIndices(OpBuilder &builder, Location loc, LowerInfo &info,
                         Value tidx, ArrayRef<Value> tileIvs, unsigned rank) {
  Value zero = createIndexConstant(builder, loc, 0);
  SmallVector<Value, 2> brs;
  SmallVector<Value, 2> ius;
  SmallVector<Value, 2> wrs;
  SmallVector<Value, 2> regs;

  for (int i = 0; i < 2; ++i) {
    Value iv = i < static_cast<int>(tileIvs.size()) ? tileIvs[i] : zero;
    int64_t threadWidth = info.get_thread_widths()[i];
    int64_t warpRepeat = info.get_warp_repeat()[i];
    int64_t instUnroll = info.warpInstUnroll[i];
    int64_t repeatWidth = warpRepeat * threadWidth;
    int64_t unrollWidth = instUnroll * repeatWidth;

    brs.push_back(floorDivBy(builder, loc, iv, unrollWidth));
    ius.push_back(modBy(builder, loc,
                        floorDivBy(builder, loc, iv, repeatWidth),
                        instUnroll));
    Value withinInst = modBy(builder, loc, iv, repeatWidth);
    wrs.push_back(floorDivBy(builder, loc, withinInst, threadWidth));
    regs.push_back(modBy(builder, loc, withinInst, threadWidth));
  }

  auto operands = buildLowerInfoMapOperands(builder, loc, info, tidx, brs[0],
                                            brs[1], ius[0], ius[1], wrs[0],
                                            wrs[1], regs[0], regs[1]);
  return applyLowerInfoMap(builder, loc, info, operands, rank);
}

static AffineMap buildThreadTileOffsetMap(OpBuilder &builder, LowerInfo &info) {
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);
  auto d3 = builder.getAffineDimExpr(3);
  auto d4 = builder.getAffineDimExpr(4);
  auto d5 = builder.getAffineDimExpr(5);

  auto [wr0, wr1] = UnflattenIndexToXY(
      d4, info.base_layout.warp_repeat_order, info.get_warp_repeat());
  auto [reg0, reg1] = UnflattenIndexToXY(
      d5, info.base_layout.thread_creg_order, info.get_thread_widths());

  std::array<AffineExpr, 2> indices;
  indices[0] =
      ((d0 * info.warpInstUnroll[0] + d2) * info.get_warp_repeat()[0] + wr0) *
          info.get_thread_widths()[0] +
      reg0;
  indices[1] =
      ((d1 * info.warpInstUnroll[1] + d3) * info.get_warp_repeat()[1] + wr1) *
          info.get_thread_widths()[1] +
      reg1;
  return AffineMap::get(6, 0, indices, builder.getContext());
}

// gemmOp的 Layout是直接推定的。不会存在问题。直接按照 wmma的相关要求进行变换即可
// 从 AB读数据 - 构建wmma IJ循环 - 构建K循环，计算单个wmmaInst区域 - 结果写回C - wmmaInst区域 在IJ滑动。覆盖完整buffer
class GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto tidx = findThreadIdxOp(op);
    // get lowerInfo
    auto infoA = getLowerInfoOrDie(op.getA(), op.getOperation());
    auto infoB = getLowerInfoOrDie(op.getB(), op.getOperation());
    auto infoC = getLowerInfoOrDie(op.getC(), op.getOperation());
    llvm::errs() << "[gemm-lower] A br=[" << infoA.get_block_repeat()[0] << ","
                 << infoA.get_block_repeat()[1] << "] buffer=" << op.getA() << "\n";
    llvm::errs() << "[gemm-lower] B br=[" << infoB.get_block_repeat()[0] << ","
                 << infoB.get_block_repeat()[1] << "] buffer=" << op.getB() << "\n";
    llvm::errs() << "[gemm-lower] C br=[" << infoC.get_block_repeat()[0] << ","
                 << infoC.get_block_repeat()[1] << "] buffer=" << op.getC() << "\n";
    
    assert(infoA.get_block_repeat()[1] * infoA.warpInstUnroll[1] == infoB.get_block_repeat()[0] * infoB.warpInstUnroll[0] );  // k轴上的 for循环次数. A 列迭代数 == B 行迭代数
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
      // 若AB为local，将其直接替换为local buffer；否则，添加 copyfrom shm to reg 的逻辑。返回这个reg buffer
      auto newA = threadLevelBufferCreate(infoA, false, infoA.get_thread_widths() , isLocalBuffer(infoA.buffer));
      auto newB = threadLevelBufferCreate(infoB, false, infoB.get_thread_widths() , isLocalBuffer(infoB.buffer));
      // C: 创建reg级别buffer 注册到 s_buffer_replace 中，存放最终结果； instWMMA 的acc 临时用，不用注册到全局列表里
      auto newC = threadLevelBufferCreate(infoC, false, infoC.get_thread_own_data_size(), true);
      auto instCShape = infoC.get_thread_widths() * infoC.get_warp_repeat();  // 单个指令需要的 instC shape
      auto instC = threadLevelBufferCreate(infoC, false, instCShape, false);
      auto instName = op->getAttrOfType<StringAttr>("inst_name");
      auto [br0, br1] = infoC.get_block_repeat();
      auto [wiu0, wiu1] = infoC.warpInstUnroll;
      int wrA0 = infoA.get_warp_repeat()[0];
      int wrA1 = infoA.get_warp_repeat()[1];
      int wrB0 = infoB.get_warp_repeat()[0];
      int wrB1 = infoB.get_warp_repeat()[1];
      int kloopCount = infoA.get_block_repeat()[1];
      int kWarpInstUnroll = infoA.warpInstUnroll[1];

      std::vector<int> mn_wiu_loops = {int(br0), int(br1), int(wiu0), int(wiu1)};
      std::vector<int> k_loops = {kloopCount, kWarpInstUnroll};
      std::vector<Value> ivs_block {};  // itervar mnk

      // 指令在bufferC上的循环(m,n)
      std::vector<const char*> label = {"br0","br1","wiu0","wiu1"};
      createNestedAffineFor(rewriter, op->getLoc(), mn_wiu_loops, ivs_block, label);
      // insPoint 位于 mn loop内
      auto instCTy = mlir::cast<MemRefType>(instC.getType());
      rewriter.create<frisk::FillOp>(op->getLoc(), instC,
                                     rewriter.getFloatAttr(instCTy.getElementType(), 0.0));
      std::vector<affine::AffineForOp> kLoopOps;
      {
        RewriterBase::InsertionGuard ig{rewriter};
        std::vector<const char*> _label = {"kBlock", "kWarpInstUnroll"};
        kLoopOps = createNestedAffineFor(rewriter, op->getLoc(), k_loops, ivs_block, _label);
        // 创建kloop。目前 insPoint 位于 kWarpInstUnroll 内
        Value zero = createIndexConstant(rewriter, op->getLoc(), 0);
        auto d0 = rewriter.getAffineDimExpr(0);
        auto d1 = rewriter.getAffineDimExpr(1);
        auto kLinearMap =
            AffineMap::get(2, 0, d0 * kWarpInstUnroll + d1,
                           rewriter.getContext());
        Value kLinear = rewriter.create<affine::AffineApplyOp>(
            op->getLoc(), kLinearMap, ValueRange{ivs_block[4], ivs_block[5]});
        Value bKBr = floorDivBy(rewriter, op->getLoc(), kLinear,
                                infoB.warpInstUnroll[0]);
        Value bKIu = modBy(rewriter, op->getLoc(), kLinear,
                           infoB.warpInstUnroll[0]);
        auto instUnrollIvOrZero = [&](LowerInfo &info, int dim,
                                      Value candidate) -> Value {
          return info.warpInstUnroll[dim] > 1 ? candidate : zero;
        };
        // 单次指令所需数据的构建 A
        // 注意：对于wmma/wgmma op，其指令内部的layout由 creg & warp_repeat 描述
        if (mlir::cast<MemRefType>(infoA.buffer.getType()).getMemorySpaceAsInt() == (int)friskMs::Shared) {
          RewriterBase::InsertionGuard _temp{rewriter};
          std::vector<int> ubs = {wrA0, wrA1};
          std::vector<mlir::Value> wrIvs;
          std::vector<const char*> _labels = {"wrA0", "wrA1"};
          createNestedAffineFor(rewriter, op->getLoc(), ubs, wrIvs, _labels);
          auto mapOperands = buildLowerInfoMapOperands(
              rewriter, op->getLoc(), infoA, tidx,
              /*br0=*/ivs_block[0], /*br1=*/ivs_block[4],
              /*iu0=*/instUnrollIvOrZero(infoA, 0, ivs_block[2]),
              /*iu1=*/instUnrollIvOrZero(infoA, 1, ivs_block[5]),
              /*wr0=*/wrIvs[0], /*wr1=*/wrIvs[1],
              /*reg0=*/zero, /*reg1=*/zero);
          auto map = infoA.getAffineMap();
          rewriter.create<frisk::CopyOp>(op->getLoc(), infoA.buffer, newA,
                                         mapOperands, map);
        }
        // 单次指令所需数据的构建 B
        if (mlir::cast<MemRefType>(infoB.buffer.getType()).getMemorySpaceAsInt() == (int)friskMs::Shared) {
          RewriterBase::InsertionGuard _temp{rewriter};
          std::vector<int> ubs = {wrB0, wrB1};
          std::vector<mlir::Value> wrIvs;
          std::vector<const char*> _labels = {"wrB0", "wrB1"};
          createNestedAffineFor(rewriter, op->getLoc(), ubs, wrIvs, _labels);
          auto mapOperands = buildLowerInfoMapOperands(
              rewriter, op->getLoc(), infoB, tidx,
              /*br0=*/bKBr, /*br1=*/ivs_block[1],
              /*iu0=*/instUnrollIvOrZero(infoB, 0, bKIu),
              /*iu1=*/instUnrollIvOrZero(infoB, 1, ivs_block[3]),
              /*wr0=*/wrIvs[0], /*wr1=*/wrIvs[1],
              /*reg0=*/zero, /*reg1=*/zero);
          auto map = infoB.getAffineMap();
          rewriter.create<frisk::CopyOp>(op->getLoc(), infoB.buffer, newB,
                                         mapOperands, map);
        }
        // AB copy ok. 计算wmma（ instC 具有累加语义）
        auto wmma = rewriter.create<frisk::WarpMmaOp>(op->getLoc(), newA, newB, instC);
        rewriter.modifyOpInPlace(wmma, [&](){
          wmma->setAttr("inst_name", instName);
          wmma->setAttr("inst_constraints", op->getAttr("inst_constraints"));
        });
      }
      // kloop ends. 需要将instC累加结果写回 newC with loopMN
      if (!kLoopOps.empty()) {
        rewriter.setInsertionPointAfter(kLoopOps.front());
      }
      Value zero = createIndexConstant(rewriter, op->getLoc(), 0);
      std::vector<Value> cIvs;
      createNestedAffineFor(rewriter, op->getLoc(),
                            {static_cast<int>(flat_size(infoC.get_warp_repeat()))},
                            cIvs);
      SmallVector<Value, 6> mapOperands{
          ivs_block[0], ivs_block[1], ivs_block[2], ivs_block[3], cIvs[0],
          zero};
      auto instCToNewCMap = buildThreadTileOffsetMap(rewriter, infoC);
      rewriter.create<frisk::CopyOp>(op->getLoc(), instC, newC, mapOperands,
                                     instCToNewCMap);
      
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
    if (reduceDim < 0 || reduceDim >= srcTy.getRank()) {
      return op.emitOpError("invalid reduce dimension");
    }
    if (reduceDim >= 2) {
      return op.emitOpError("thread-level reduce currently supports 2D "
                            "LowerInfo only");
    }
    int64_t reduceExtent = srcTy.getDimSize(reduceDim);
    if (ShapedType::isDynamic(reduceExtent) || reduceExtent <= 0) {
      return op.emitOpError("thread-level reduce requires a positive static reduce extent");
    }
    
    // 单个线程持有的数据量
    auto [srcTw0, srcTw1] = srcInfo.get_thread_own_data_size();
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

    std::array<int64_t, 2> dstThreadShape2D = dstInfo.get_thread_own_data_size();
    if (dstInfo.ignoreDim >= 0 &&
        static_cast<unsigned>(dstInfo.ignoreDim) < dstThreadShape2D.size()) {
      dstThreadShape2D[dstInfo.ignoreDim] = 1;
    }
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

    int64_t localReduceExtent = reduceExtent;
    if (!srcIsLocal || srcBuffer == op.getSrc()) {
      localReduceExtent = srcInfo.get_thread_own_data_size()[reduceDim];
    }
    if (ShapedType::isDynamic(localReduceExtent) || localReduceExtent <= 0) {
      return op.emitOpError("thread-level reduce requires each thread to own "
                            "a positive static reduce extent");
    }
    if (localReduceExtent > std::numeric_limits<int>::max()) {
      return op.emitOpError("thread-level reduce extent is too large");
    }

    std::vector<Value> redIvs;
    auto redLoops = createNestedAffineFor(rewriter, op->getLoc(),
                                          {static_cast<int>(localReduceExtent)},
                                          redIvs);

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
      SmallVector<Value, 2> srcTileIvs;
      if (srcTy.getRank() == dstTy.getRank()) {
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          srcTileIvs.push_back(i == static_cast<unsigned>(reduceDim)
                                   ? redIvs[0]
                                   : dstTileIvs[i]);
        }
      } else {
        unsigned dstPos = 0;
        for (unsigned i = 0; i < srcTy.getRank(); ++i) {
          if (i == static_cast<unsigned>(reduceDim)) {
            srcTileIvs.push_back(redIvs[0]);
          } else {
            srcTileIvs.push_back(dstTileIvs[dstPos++]);
          }
        }
      }
      srcIndices = buildMappedAccessIndices(rewriter, op->getLoc(), srcInfo,
                                            tidx, srcTileIvs, srcTy.getRank());
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
    auto localResult =
        rewriter.create<affine::AffineLoadOp>(op->getLoc(), acc, ValueRange{zeroIdx});

    auto isPowerOfTwo = [](int64_t value) {
      return value > 0 && (value & (value - 1)) == 0;
    };
    auto warpLayout = srcInfo.get_warp_layout();
    auto warpLayoutOrder = srcInfo.base_layout.warp_layout_order;
    auto blockLayout = srcInfo.get_block_layout();
    if (blockLayout[reduceDim] != 1) {
      return op.emitOpError("shuffle reduce only supports reduce dimension "
                            "within a single warp layout");
    }
    int64_t reduceLaneExtent = warpLayout[reduceDim];
    if (!isPowerOfTwo(reduceLaneExtent)) {
      return op.emitOpError("shuffle reduce requires power-of-two lane extent "
                            "along the reduce dimension");
    }
    int64_t reduceLaneStride = 1;
    if (warpLayoutOrder[0] == reduceDim) {
      reduceLaneStride = 1;
    } else if (warpLayoutOrder[1] == reduceDim) {
      reduceLaneStride = warpLayout[warpLayoutOrder[0]];
    } else {
      return op.emitOpError("reduce dimension is not present in warp layout order");
    }

    Value reduced = localResult.getResult();
    for (int64_t laneOffset = 1; laneOffset < reduceLaneExtent;
         laneOffset <<= 1) {
      auto shuffled = rewriter.create<gpu::ShuffleOp>(
          op->getLoc(), reduced,
          static_cast<int32_t>(laneOffset * reduceLaneStride),
          static_cast<int32_t>(srcInfo.warp_threads), gpu::ShuffleMode::XOR);
      auto combined = combine(reduced, shuffled.getShuffleResult());
      if (failed(combined)) {
        return op.emitOpError("unsupported reduce kind");
      }
      reduced = *combined;
    }

    Value laneId = modBy(rewriter, op->getLoc(), tidx, srcInfo.warp_threads);
    Value reduceLaneCoord =
        modBy(rewriter, op->getLoc(),
              floorDivBy(rewriter, op->getLoc(), laneId, reduceLaneStride),
              reduceLaneExtent);
    auto isReduceLeader = rewriter.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::eq, reduceLaneCoord, zeroIdx);
    auto ifOp =
        rewriter.create<scf::IfOp>(op->getLoc(), isReduceLeader, false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), reduced, dstBuffer,
                                           dstIndices);
    rewriter.setInsertionPointAfter(ifOp);
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
    std::array<int64_t, 2> blockRepeats = {0, 0};
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
    
    // thread层面，每个buffer 当前tid持有的元素不一定一样。
    auto collectMaxThreadlevelSz = [&](std::array<int64_t, 2> sz) {
      threadLevelSize[0] = std::max(threadLevelSize[0], sz[0]);
      threadLevelSize[1] = std::max(threadLevelSize[1], sz[1]);
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

    auto recordBufferInfo = [&](Value buffer, bool isOutBuffer , const char *label) {
      if (!isa<MemRefType>(buffer.getType()) || findBufferInfo(buffer) != nullptr) {
        return;
      }

      AccessedBufferInfo recordedInfo{buffer, std::nullopt};
      if (auto *lowerInfo = s_info->getLowerInfo(buffer, op.getOperation())) {
        recordedInfo.lowerInfo = *lowerInfo;
        collectMaxThreadlevelSz(lowerInfo->get_thread_own_data_size());
        lowerInfo->show(label);
      } else {
        // 前面的 block conversion 可能已经把原 frisk.alloc_buffer 替换为
        // memref.alloca 形式的 thread tile。这个新 value 已经是 lowered 后
        // 的形态，不会出现在 LowerInfoMap 中，因此直接用当前 memref shape
        // 作为循环 shape。
        collectMaxThreadlevelSz(get2DShape(buffer));
      }
      bufferInfos.push_back(recordedInfo);
    };
    
    // 对每个affine.load 检查其memref，记录buffer Info。
    // 追踪来源 srcDefOp， 标记 allocLocalsToReplace[srcDefOp] = lowerInfo 的thread总计算量
    for (auto loadOp : loadOps) {
      auto srcValue = loadOp.getMemref();
      recordBufferInfo(srcValue,false, "block_load");
      auto srcDefOp = srcValue.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(srcValue);
      if (srcDefOp != nullptr && info != nullptr && info->lowerInfo && isThreadLocalTile(srcValue) &&
          !allocLocalsToReplace.count(srcDefOp)) {
        allocLocalsToReplace[srcDefOp] = info->lowerInfo->get_thread_own_data_size();
      }
    }

    for (auto storeOp : storeOps) {
      auto dstVal = storeOp.getMemref();
      recordBufferInfo(dstVal, true,"block_store");
      auto dstDefOp = dstVal.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(dstVal);
      if (dstDefOp != nullptr && info != nullptr && info->lowerInfo && isThreadLocalTile(dstVal) &&
          !allocLocalsToReplace.count(dstDefOp)) {
        allocLocalsToReplace[dstDefOp] = info->lowerInfo->get_thread_own_data_size();
      }
    }

    // 统计该blockOp应该以多少 blockRepeat 为准。 每个子op的 blockRepeat可能不同
    for(auto e : bufferInfos){
      if(e.lowerInfo){
        auto br = e.lowerInfo->get_block_repeat();
        if (e.lowerInfo->ignoreDim >= 0 &&
            static_cast<unsigned>(e.lowerInfo->ignoreDim) < br.size()) {
          br[e.lowerInfo->ignoreDim] = 1;
        }
        blockRepeats[0] = std::max(blockRepeats[0], br[0]);
        blockRepeats[1] = std::max(blockRepeats[1], br[1]);
      }
    }

    assert(threadLevelSize[0] > 0 && threadLevelSize[1] > 0 && "thread-level block size not inferred");
    assert(blockRepeats[0] > 0 && blockRepeats[1] > 0 && "br not inferred");
    IRMapping mapper;
    IRMapping localMapper;
    // local/register block buffer 会物化成每个线程自己的 tile；shared buffer
    // 仍然保持 block-sized，后面通过 LowerInfo map 生成真实 block 坐标。
    // 对每个loadOp/storeOp，寻找thread 级别buffer是否已有注册。没有则创建+注册
    for (auto &[srcDefOp, sz] : allocLocalsToReplace) {
      rewriter.setInsertionPoint(srcDefOp);
      auto oldBuffer = srcDefOp->getResult(0);
      auto it = s_buffer_replace.find(oldBuffer);
      mlir::Value replaceVal = nullptr;
      if (it != s_buffer_replace.end()) {
        replaceVal = it->second;
      }
      else {
        auto ty = MemRefType::get(sz, srcDefOp.getElementType(), AffineMap{}, srcDefOp.getMemorySpace());
        auto newAlloc = rewriter.create<memref::AllocaOp>(srcDefOp->getLoc(), ty);
        replaceVal = newAlloc->getResult(0);
        s_buffer_replace[oldBuffer] = replaceVal;
      }
      mapper.map(oldBuffer, replaceVal);
      localMapper.map(oldBuffer, replaceVal);
    }
    rewriter.setInsertionPoint(op);
    // frisk.blocOp 根据newbuffer的size，生成 nestedFor
    std::vector<mlir::Value> threadOwnDataIvs {};
    std::vector<mlir::Value> blockRepeatIvs {};
    std::vector<const char* > labels {};
    std::vector<int> thread_level_sz { threadLevelSize.begin(), threadLevelSize.end() };
    std::vector<int> blockop_br { blockRepeats.begin(), blockRepeats.end() };

    // 创建 thread_own_data_sz 循环
    createNestedAffineFor(rewriter, op->getLoc(), thread_level_sz, threadOwnDataIvs);
    // block_repeat 循环
    createNestedAffineFor(rewriter, op->getLoc(), blockop_br, blockRepeatIvs);
    // TODO : 缺少 iwu 循环 和 warp_repeat 循环

    auto constIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(op->getLoc(), v);
    };

    auto affineApply1 = [&](AffineExpr expr, Value operand) -> Value {
      auto map = AffineMap::get(1, 0, expr, rewriter.getContext());
      return rewriter.create<affine::AffineApplyOp>(op->getLoc(), map, operand);
    };

    auto modBy = [&](Value operand, int64_t divisor) -> Value {
      assert(divisor > 0 && "affine modulo divisor must be positive");
      if (divisor == 1) {
        return constIndex(0);
      }
      auto d0 = rewriter.getAffineDimExpr(0);
      return affineApply1(d0 % divisor, operand);
    };

    auto floorDivBy = [&](Value operand, int64_t divisor) -> Value {
      assert(divisor > 0 && "affine floordiv divisor must be positive");
      if (divisor == 1) {
        return operand;
      }
      auto d0 = rewriter.getAffineDimExpr(0);
      return affineApply1(d0.floorDiv(divisor), operand);
    };

    auto getBufferInfo = [&](Value buffer) -> AccessedBufferInfo * {
      if (auto mapped = mapper.lookupOrNull(buffer)) {
        for (auto &info : bufferInfos) {
          if (info.buffer == mapped) {
            return &info;
          }
        }
      }
      return findBufferInfo(buffer);
    };

    auto getLinearIvForDim = [&](unsigned dim) -> Value {
      if (dim < threadOwnDataIvs.size()) {
        return threadOwnDataIvs[dim];
      }
      return constIndex(0);
    };

    auto getRepeatIvForDim = [&](unsigned dim) -> Value {
      if (dim < blockRepeatIvs.size()) {
        return blockRepeatIvs[dim];
      }
      return constIndex(0);
    };

    auto buildAccessCoords = [&](Value originalBuffer,
                                 unsigned rank) -> SmallVector<Value, 2> {
      SmallVector<Value, 2> coords;
      auto *accessInfo = getBufferInfo(originalBuffer);
      if (accessInfo == nullptr || !accessInfo->lowerInfo) {
        for (unsigned i = 0; i < rank; ++i) {
          coords.push_back(getLinearIvForDim(i));
        }
        return coords;
      }

      LowerInfo &info = *accessInfo->lowerInfo;
      bool useLocalTile = isThreadLocalTile(originalBuffer);
      if (useLocalTile) {
        for (unsigned i = 0; i < rank; ++i) {
          Value linearIv = getLinearIvForDim(i);
          int64_t ownDataSize = i < 2 ? info.get_thread_own_data_size()[i] : 1;
          coords.push_back(modBy(linearIv, ownDataSize));
        }
        return coords;
      }

      SmallVector<Value, 2> brs;
      SmallVector<Value, 2> ius;
      SmallVector<Value, 2> wrs;
      SmallVector<Value, 2> regs;
      for (int i = 0; i < 2; ++i) {
        int64_t warpRepeat = info.get_warp_repeat()[i];
        int64_t threadWidth = info.get_thread_widths()[i];
        int64_t repeatWidth = warpRepeat * threadWidth;
        int64_t instUnroll = info.warpInstUnroll[i];
        int64_t unrollWidth = repeatWidth * instUnroll;
        int64_t ownBlockRepeat = info.get_block_repeat()[i];
        Value linearIv = getLinearIvForDim(i);
        Value repeatIv = getRepeatIvForDim(i);
        Value br = modBy(repeatIv, ownBlockRepeat);
        Value linearInTile = modBy(linearIv, unrollWidth);

        brs.push_back(br);
        ius.push_back(modBy(floorDivBy(linearInTile, repeatWidth), instUnroll));
        Value withinInst = modBy(linearInTile, repeatWidth);
        wrs.push_back(floorDivBy(withinInst, threadWidth));
        regs.push_back(modBy(withinInst, threadWidth));
      }

      auto mapOperands = buildLowerInfoMapOperands(
          rewriter, op->getLoc(), info, tidx, brs[0], brs[1], ius[0], ius[1],
          wrs[0], wrs[1], regs[0], regs[1]);
      return applyLowerInfoMap(rewriter, op->getLoc(), info, mapOperands, rank);
    };

    std::function<Value(Value, ArrayRef<Value>)> remapValueForAccess =
        [&](Value operand, ArrayRef<Value> accessCoords) -> Value {
      Block *body = op.getBody();
      if (auto blockArg = dyn_cast<BlockArgument>(operand);
          blockArg && blockArg.getOwner() == body &&
          blockArg.getArgNumber() < accessCoords.size()) {
        return accessCoords[blockArg.getArgNumber()];
      }
      if (auto applyOp = operand.getDefiningOp<affine::AffineApplyOp>()) {
        SmallVector<Value, 4> operands;
        operands.reserve(applyOp.getMapOperands().size());
        for (Value mapOperand : applyOp.getMapOperands()) {
          operands.push_back(remapValueForAccess(mapOperand, accessCoords));
        }
        return rewriter.create<affine::AffineApplyOp>(
            applyOp.getLoc(), applyOp.getAffineMap(), operands);
      }
      return mapper.lookupOrDefault(operand);
    };

    auto remapAffineOperands = [&](ValueRange oldOperands,
                                   ArrayRef<Value> accessCoords) {
      SmallVector<Value, 4> newOperands;
      newOperands.reserve(oldOperands.size());
      for (Value operand : oldOperands) {
        newOperands.push_back(remapValueForAccess(operand, accessCoords));
      }
      return newOperands;
    };

    // 根据映射规则，将frisk.blockOp 内的全部op 搬运到 nestedFor 的最内层。
    // load/store 需要按各自 memref 的 LowerInfo 重建访问下标；其他 op 只
    // 需要普通 SSA value 映射。
    Block *body = op.getBody();
    Value defaultIndexBuffer;
    for (auto storeOp : storeOps) {
      Value buffer = storeOp.getMemref();
      if (findBufferInfo(buffer) != nullptr && !isThreadLocalTile(buffer)) {
        defaultIndexBuffer = buffer;
        break;
      }
    }
    if (!defaultIndexBuffer) {
      for (auto &info : bufferInfos) {
        if (info.lowerInfo && !isThreadLocalTile(info.buffer)) {
          defaultIndexBuffer = info.buffer;
          break;
        }
      }
    }
    if (!defaultIndexBuffer) {
      for (auto &info : bufferInfos) {
        if (info.lowerInfo) {
          defaultIndexBuffer = info.buffer;
          break;
        }
      }
    }

    SmallVector<Value, 2> defaultBlockCoords;
    if (defaultIndexBuffer) {
      defaultBlockCoords =
          buildAccessCoords(defaultIndexBuffer, body->getNumArguments());
    } else {
      for (unsigned i = 0; i < body->getNumArguments(); ++i) {
        defaultBlockCoords.push_back(getLinearIvForDim(i));
      }
    }
    for (auto [oldIndex, newIter] :
         llvm::zip(body->getArguments(), defaultBlockCoords)) {
      mapper.map(oldIndex, newIter);
      localMapper.map(oldIndex, newIter);
    }

    for (auto &childOp : body->without_terminator()) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(childOp)) {
        Value memref = mapper.lookupOrDefault(loadOp.getMemref());
        auto accessCoords =
            buildAccessCoords(loadOp.getMemref(), loadOp.getAffineMap().getNumResults());
        auto mapOperands = remapAffineOperands(loadOp.getMapOperands(), accessCoords);
        auto newLoad = rewriter.create<affine::AffineLoadOp>(
            loadOp.getLoc(), memref, loadOp.getAffineMap(), mapOperands);
        mapper.map(loadOp.getResult(), newLoad.getResult());
        localMapper.map(loadOp.getResult(), newLoad.getResult());
        continue;
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(childOp)) {
        Value memref = mapper.lookupOrDefault(storeOp.getMemref());
        Value valueToStore = mapper.lookupOrDefault(storeOp.getValueToStore());
        auto accessCoords =
            buildAccessCoords(storeOp.getMemref(), storeOp.getAffineMap().getNumResults());
        auto mapOperands = remapAffineOperands(storeOp.getMapOperands(), accessCoords);
        rewriter.create<affine::AffineStoreOp>(
            storeOp.getLoc(), valueToStore, memref, storeOp.getAffineMap(), mapOperands);
        continue;
      }
      auto *cloned = rewriter.clone(childOp, mapper);
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
      Value oldBuffer = srcDefOp->getResult(0);
      auto temp = mapper.lookupOrNull(oldBuffer);
      if(temp == nullptr){
        continue;
      }
      auto newAlloc = temp.getDefiningOp<memref::AllocaOp>();
      // 只替换 blockOp 之外还残留的 use
      if(newAlloc != nullptr){
        oldBuffer.replaceAllUsesExcept(
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

// convertLayoutOp -> sync op
class ConvertLayoutOpConversion : public OpConversionPattern<frisk::ConvertLayoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(frisk::ConvertLayoutOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    auto it = convertLayoutInfo.find(op);
    if(it != convertLayoutInfo.end()){
      auto [from, to] = it->getSecond();
      // 根据from 构建for循环，用 frisk.copy 将线程持有的数据拷贝到 from.buffer(shm)
      // 根据to 构建for循环 用 frisk.copy 将 from.buffer(shm) 的数据拷贝到线程持有的buffer
      assert(from->buffer == to->buffer);
      auto memTy = mlir::cast<MemRefType>(from->buffer.getType());
      assert(memTy.getMemorySpaceAsInt() == (int)friskMs::Shared);
      auto sync = rewriter.create<frisk::SyncThreadsInBlockOp>(op->getLoc(), from->buffer);
      rewriter.replaceOp(op, sync);
    }
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
    auto [tw0,tw1] = srcInfo->get_thread_widths();
    auto [dstTw0, dstTw1] = dstInfo->get_thread_widths();
    assert(tw0 == dstTw0 && tw1 == dstTw1 &&
           "copy-convert expects source and destination thread tiles to match");

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
      auto [own0, own1] = info->get_thread_own_data_size();
      std::vector<int64_t> shape = {own0, own1};
      auto newBuffer = rewriter.create<frisk::AllocBufferOp>(
          op->getLoc(), shape, elementType, 1, int(friskMs::Local));
      s_buffer_replace[info->buffer] = newBuffer;
      return newBuffer.getResult();
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

    auto loc = op->getLoc();
    auto srcTile = rewriter.create<frisk::AllocBufferOp>(
        loc, std::vector<int64_t>{tw0, tw1}, srcTy.getElementType(), 1,
        int(friskMs::Local));
    auto dstTile = rewriter.create<frisk::AllocBufferOp>(
        loc, std::vector<int64_t>{tw0, tw1}, dstTy.getElementType(), 1,
        int(friskMs::Local));

    auto effectiveDimValue = [](LowerInfo *info, int dim, int64_t value) {
      return info->ignoreDim == dim ? int64_t{1} : value;
    };

    auto srcTileCounts = srcInfo->get_block_repeat() *
                         srcInfo->warpInstUnroll *
                         srcInfo->get_warp_repeat();
    auto dstTileCounts = dstInfo->get_block_repeat() *
                         dstInfo->warpInstUnroll *
                         dstInfo->get_warp_repeat();
    for (int dim = 0; dim < 2; ++dim) {
      srcTileCounts[dim] = effectiveDimValue(srcInfo, dim, srcTileCounts[dim]);
      dstTileCounts[dim] = effectiveDimValue(dstInfo, dim, dstTileCounts[dim]);
      if (srcTileCounts[dim] != dstTileCounts[dim]) {
        return op.emitOpError("copy-convert source/destination tile counts do not match");
      }
    }

    auto srcBr = srcInfo->get_block_repeat();
    auto srcWr = srcInfo->get_warp_repeat();
    std::vector<int> outerUbs = {
        int(effectiveDimValue(srcInfo, 0, srcBr[0])),
        int(effectiveDimValue(srcInfo, 1, srcBr[1])),
        int(effectiveDimValue(srcInfo, 0, srcInfo->warpInstUnroll[0])),
        int(effectiveDimValue(srcInfo, 1, srcInfo->warpInstUnroll[1])),
        int(effectiveDimValue(srcInfo, 0, srcWr[0])),
        int(effectiveDimValue(srcInfo, 1, srcWr[1]))};
    std::vector<const char*> outerLabels = {
        "br0", "br1", "iwu0", "iwu1", "wr0", "wr1"};
    std::vector<Value> outerIvs;
    createNestedAffineFor(rewriter, loc, outerUbs, outerIvs, outerLabels);

    Value zero = createIndexConstant(rewriter, loc, 0);
    auto buildTileBase = [&](int dim) -> Value {
      auto d0 = rewriter.getAffineDimExpr(0);
      auto d1 = rewriter.getAffineDimExpr(1);
      auto d2 = rewriter.getAffineDimExpr(2);
      auto expr = ((d0 * srcInfo->warpInstUnroll[dim] + d1) *
                       srcInfo->get_warp_repeat()[dim] +
                   d2) *
                  srcInfo->get_thread_widths()[dim];
      auto map = AffineMap::get(3, 0, expr, rewriter.getContext());
      return rewriter.create<affine::AffineApplyOp>(
          loc, map,
          ValueRange{outerIvs[dim], outerIvs[dim + 2], outerIvs[dim + 4]});
    };
    SmallVector<Value, 2> tileBaseIvs{buildTileBase(0), buildTileBase(1)};

    struct CopyOffset {
      SmallVector<Value> operands;
      AffineMap map;
    };

    auto buildCopyOffset = [&](LowerInfo *info, bool useLocalTile) -> CopyOffset {
      SmallVector<Value, 2> brs;
      SmallVector<Value, 2> ius;
      SmallVector<Value, 2> wrs;
      for (int i = 0; i < 2; ++i) {
        Value iv = tileBaseIvs[i];
        int64_t threadWidth = info->get_thread_widths()[i];
        int64_t warpRepeat = info->get_warp_repeat()[i];
        int64_t instUnroll = info->warpInstUnroll[i];
        int64_t repeatWidth = warpRepeat * threadWidth;
        int64_t unrollWidth = instUnroll * repeatWidth;

        brs.push_back(floorDivBy(rewriter, loc, iv, unrollWidth));
        ius.push_back(modBy(rewriter, loc,
                            floorDivBy(rewriter, loc, iv, repeatWidth),
                            instUnroll));
        Value withinInst = modBy(rewriter, loc, iv, repeatWidth);
        wrs.push_back(floorDivBy(rewriter, loc, withinInst, threadWidth));
      }

      if (useLocalTile) {
        SmallVector<Value> wrXY{wrs[0], wrs[1]};
        SmallVector<Value> operands{brs[0], brs[1], ius[0], ius[1],
                                    flattenXY(rewriter, loc, wrXY,
                                              info->base_layout.warp_repeat_order,
                                              info->get_warp_repeat()),
                                    zero};
        return {operands, buildThreadTileOffsetMap(rewriter, *info)};
      }

      auto operands = buildLowerInfoMapOperands(
          rewriter, loc, *info, tidx, brs[0], brs[1], ius[0], ius[1],
          wrs[0], wrs[1], zero, zero);
      return {SmallVector<Value>{operands.begin(), operands.end()},
              info->getAffineMap()};
    };

    auto srcOffset = buildCopyOffset(srcInfo, srcIsLocal);
    rewriter.create<frisk::CopyOp>(loc, srcBuffer, srcTile.getResult(),
                                   srcOffset.operands, srcOffset.map);

    std::vector<Value> regIvs;
    std::vector<const char*> regLabels = {"tw0", "tw1"};
    auto regLoops = createNestedAffineFor(
        rewriter, loc, std::vector<int>{int(tw0), int(tw1)}, regIvs,
        regLabels);

    auto srcValue = rewriter.create<affine::AffineLoadOp>(
        loc, srcTile.getResult(), ValueRange{regIvs[0], regIvs[1]});
    mlir::Value converted{};
    if(srcTy.getElementType().getIntOrFloatBitWidth() < dstTy.getElementType().getIntOrFloatBitWidth()){
      converted = rewriter.create<arith::ExtFOp>(loc, dstTy.getElementType(), srcValue.getResult() );
    }
    else{
      converted = rewriter.create<arith::TruncFOp>(loc, dstTy.getElementType(), srcValue.getResult() );
    }
    rewriter.create<affine::AffineStoreOp>(
        loc, converted, dstTile.getResult(), ValueRange{regIvs[0], regIvs[1]});

    if (!regLoops.empty()) {
      rewriter.setInsertionPointAfter(regLoops.front());
    }
    auto dstOffset = buildCopyOffset(dstInfo, dstIsLocal);
    rewriter.create<frisk::CopyOp>(loc, dstTile.getResult(), dstBuffer,
                                   dstOffset.operands, dstOffset.map);
    rewriter.eraseOp(op);
    return success();
  }
};

// 规则：
// 将frisk.copy 转换为对应的mlir基础op (点对点 或者 vector_copy)
// 判断 src dst 的shape
// srcSHape 更大，则indexmap作用于src。 dst更大时 map作用于 dst
// 当src和dst其中一个来自 buffer_view, 那么另一个必然为同 shape。此时map取 buffer_view 的map
// src dst 一样shape时，直接点对点copy
class CopyOpRewrite : public OpConversionPattern<frisk::CopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto srcMem = op.getSrcMemRef();
    auto dstMem = op.getDstMemRef();
    auto srcMemType = mlir::cast<MemRefType>(srcMem.getType());
    auto dstMemType = mlir::cast<MemRefType>(dstMem.getType());
    if (srcMemType.getElementType() != dstMemType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "copy-to-base only handles same element type copies");
    }

    struct BufferInfo {
      Value realBuffer;
      MemRefType realType;
      bool fromView = false;
      AffineMap viewMap;
      SmallVector<Value, 4> viewOperands;
    };

    auto resolveBuffer = [](Value buffer) -> BufferInfo {
      BufferInfo info;
      info.realBuffer = buffer;
      info.realType = cast<MemRefType>(buffer.getType());

      if (auto viewOp = buffer.getDefiningOp<frisk::BufferViewOp>()) {
        info.fromView = true;
        info.realBuffer = viewOp.getSource();
        info.realType = cast<MemRefType>(info.realBuffer.getType());
        info.viewMap = viewOp.getIndexMap();
        info.viewOperands.assign(viewOp.getIndices().begin(),
                                 viewOp.getIndices().end());
      }
      return info;
    };

    auto srcInfo = resolveBuffer(srcMem);
    auto dstInfo = resolveBuffer(dstMem);

    auto productOfShape = [&](ArrayRef<int64_t> shape) -> FailureOr<int64_t> {
      int64_t size = 1;
      for (int64_t dim : shape) {
        if (dim < 0) {
          (void)rewriter.notifyMatchFailure(
              op, "copy-to-base expects static memref shapes");
          return failure();
        }
        if (dim != 0 && size > std::numeric_limits<int64_t>::max() / dim) {
          (void)rewriter.notifyMatchFailure(op, "memref shape is too large");
          return failure();
        }
        size *= dim;
      }
      return size;
    };

    // 比较两个buffer的shape，
    auto compareShapeSize =
        [&](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) -> FailureOr<int> {
      auto lhsSize = productOfShape(lhs);
      if (failed(lhsSize)) {
        return failure();
      }
      auto rhsSize = productOfShape(rhs);
      if (failed(rhsSize)) {
        return failure();
      }
      if (*lhsSize == *rhsSize) {
        return 0;
      }
      return *lhsSize > *rhsSize ? 1 : -1;
    };

    SmallVector<int64_t, 4> copyShape;
    bool hasMappedSide = false;
    bool isMapForSrc = true;
    AffineMap indexMap;
    SmallVector<Value, 4> indexMapOperands;

    if (srcInfo.fromView || dstInfo.fromView) {
      if (srcInfo.fromView && dstInfo.fromView) {
        return rewriter.notifyMatchFailure(
            op, "copy-to-base does not support copy between two buffer_view ops");
      }
      if (srcMemType.getShape() != dstMemType.getShape()) {
        return rewriter.notifyMatchFailure(
            op, "buffer_view copy expects the other buffer to have the same shape");
      }
      hasMappedSide = true;
      isMapForSrc = srcInfo.fromView;
      indexMap = srcInfo.fromView ? srcInfo.viewMap : dstInfo.viewMap;
      indexMapOperands =
          srcInfo.fromView ? srcInfo.viewOperands : dstInfo.viewOperands;
      copyShape.assign(srcMemType.getShape().begin(), srcMemType.getShape().end());
    } else if (srcMemType.getShape() == dstMemType.getShape()) {
      copyShape.assign(srcMemType.getShape().begin(), srcMemType.getShape().end());
    } else {
      auto shapeCompare =
          compareShapeSize(srcMemType.getShape(), dstMemType.getShape());
      if (failed(shapeCompare)) {
        return failure();
      }
      if (*shapeCompare == 0) {
        return rewriter.notifyMatchFailure(
            op, "copy-to-base cannot infer map side for different shapes with the same element count");
      }
      hasMappedSide = true;
      isMapForSrc = *shapeCompare > 0;
      indexMap = op.getOffsetMap();
      indexMapOperands.assign(op.getMapOperands().begin(),
                              op.getMapOperands().end());
      auto copyShapeRef =
          isMapForSrc ? dstMemType.getShape() : srcMemType.getShape();
      copyShape.assign(copyShapeRef.begin(), copyShapeRef.end());
    }

    std::vector<int> loopUpperBounds;
    loopUpperBounds.reserve(copyShape.size());
    for (int64_t dim : copyShape) {
      if (dim < 0 || dim > std::numeric_limits<int>::max()) {
        return rewriter.notifyMatchFailure(
            op, "copy-to-base expects static int-sized loop bounds");
      }
      loopUpperBounds.push_back(static_cast<int>(dim));
    }

    std::vector<Value> copyIvs;
    auto loops =
        createNestedAffineFor(rewriter, op->getLoc(), loopUpperBounds, copyIvs);

    auto buildDirectIndices = [&](unsigned rank) -> FailureOr<SmallVector<Value>> {
      if (rank != copyIvs.size()) {
        (void)rewriter.notifyMatchFailure(
            op, "direct copy side rank must match copy iteration rank");
        return failure();
      }
      return SmallVector<Value>{copyIvs.begin(), copyIvs.end()};
    };

    auto addIndexValues = [&](Value lhs, Value rhs) -> Value {
      auto d0 = rewriter.getAffineDimExpr(0);
      auto d1 = rewriter.getAffineDimExpr(1);
      auto map = AffineMap::get(2, 0, d0 + d1, rewriter.getContext());
      return rewriter.create<affine::AffineApplyOp>(
          op->getLoc(), map, ValueRange{lhs, rhs});
    };

    auto buildMappedIndices =
        [&](MemRefType realType) -> FailureOr<SmallVector<Value>> {
      unsigned rank = realType.getRank();
      if (indexMap.getNumResults() != rank) {
        (void)rewriter.notifyMatchFailure(
            op, "index map result count must match mapped buffer rank");
        return failure();
      }
      if (indexMap.getNumInputs() != indexMapOperands.size()) {
        (void)rewriter.notifyMatchFailure(
            op, "index map operand count does not match map input count");
        return failure();
      }
      if (copyIvs.size() > rank) {
        (void)rewriter.notifyMatchFailure(
            op, "copy iteration rank is larger than mapped buffer rank");
        return failure();
      }

      SmallVector<Value> indices;
      indices.reserve(rank);
      unsigned loopStart = rank - copyIvs.size();
      for (unsigned i = 0; i < rank; ++i) {
        auto oneResultMap =
            AffineMap::get(indexMap.getNumDims(), indexMap.getNumSymbols(),
                           indexMap.getResult(i), rewriter.getContext());
        Value index = rewriter.create<affine::AffineApplyOp>(
            op->getLoc(), oneResultMap, indexMapOperands);
        if (i >= loopStart) {
          index = addIndexValues(index, copyIvs[i - loopStart]);
        }
        indices.push_back(index);
      }
      return indices;
    };

    FailureOr<SmallVector<Value>> srcIndices;
    FailureOr<SmallVector<Value>> dstIndices;
    if (!hasMappedSide) {
      srcIndices = buildDirectIndices(srcInfo.realType.getRank());
      dstIndices = buildDirectIndices(dstInfo.realType.getRank());
    } else if (isMapForSrc) {
      srcIndices = buildMappedIndices(srcInfo.realType);
      dstIndices = buildDirectIndices(dstInfo.realType.getRank());
    } else {
      srcIndices = buildDirectIndices(srcInfo.realType.getRank());
      dstIndices = buildMappedIndices(dstInfo.realType);
    }
    if (failed(srcIndices) || failed(dstIndices)) {
      return failure();
    }

    auto value = rewriter.create<affine::AffineLoadOp>(
        op->getLoc(), srcInfo.realBuffer, *srcIndices);
    rewriter.create<affine::AffineStoreOp>(
        op->getLoc(), value.getResult(), dstInfo.realBuffer, *dstIndices);

    if (!loops.empty()) {
      rewriter.setInsertionPointAfter(loops.front());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// fillop -> 点对点赋值
class FillOpRewrite : public OpConversionPattern<frisk::FillOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::FillOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 获取 value，构建for循环。将value 赋值给 memref的每个点
    auto memref = adaptor.getMemref();
    auto memrefType = mlir::cast<MemRefType>(memref.getType());
    auto loc = op->getLoc();

    std::vector<int> loopUpperBounds;
    loopUpperBounds.reserve(memrefType.getRank());
    SmallVector<int64_t> fillShape(memrefType.getShape());
    if (s_info && isLocalMemref(op.getMemref()) && memrefType.getRank() == 2) {
      LowerInfo *lowerInfo = s_info->getLowerInfo(op.getMemref(), op.getOperation());
      if (!lowerInfo) {
        for (auto &entry : *s_info) {
          if (entry.second.buffer == op.getMemref()) {
            lowerInfo = &entry.second;
            break;
          }
        }
      }
      if (lowerInfo) {
        auto threadOwnData = lowerInfo->get_thread_own_data_size();
        fillShape[0] = threadOwnData[0];
        fillShape[1] = threadOwnData[1];
      }
    }
    for (int64_t dim : fillShape) {
      if (dim < 0 || dim > std::numeric_limits<int>::max()) {
        return rewriter.notifyMatchFailure(
            op, "fill-to-base expects static int-sized memref shape");
      }
      loopUpperBounds.push_back(static_cast<int>(dim));
    }

    auto valueAttr = dyn_cast<TypedAttr>(op.getValueAttr());
    if (!valueAttr) {
      return rewriter.notifyMatchFailure(op, "fill value must be typed");
    }
    auto val = rewriter.create<arith::ConstantOp>(
        loc, memrefType.getElementType(), valueAttr);
    std::vector<Value> ivs;
    auto loops = createNestedAffineFor(rewriter, loc, loopUpperBounds, ivs);
    rewriter.create<affine::AffineStoreOp>(loc, val.getResult(), memref,
                                           ValueRange{ivs});
    if (!loops.empty()) {
      rewriter.setInsertionPointAfter(loops.front());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AllocBufferOpConversion : public OpConversionPattern<frisk::AllocBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AllocBufferOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto memtype =  mlir::cast<MemRefType>(op.getResult().getType());
    if(memtype.getMemorySpaceAsInt() == (int)friskMs::Shared){
      auto newOp = rewriter.create<memref::AllocOp>(op->getLoc(), memtype, op.getAlignmentAttr() );
      rewriter.replaceOp(op, newOp);
    }
    else if(memtype.getMemorySpaceAsInt() == (int)friskMs::Local){
      auto newOp = rewriter.create<memref::AllocaOp>(op->getLoc(), memtype, op.getAlignmentAttr() );
      rewriter.replaceOp(op, newOp);
    }
    return llvm::success();
  }
};

static std::optional<int64_t> getStaticElementCount(MemRefType type) {
  if (!type.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim < 0 || (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim)) {
      return std::nullopt;
    }
    count *= dim;
  }
  return count;
}

static SmallVector<int64_t> getContiguousStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= std::max<int64_t>(shape[i], 1);
  }
  return strides;
}

static void applyBufferReuse(func::FuncOp kernel,
                             const LivelinessAnalyzer &liveliness) {
  struct ReuseGroup {
    SmallVector<memref::AllocOp, 4> allocs;
    MemRefType firstType;
    Type elementType;
    Attribute memorySpace;
    Block *parentBlock = nullptr;
    int64_t maxElements = 0;
    uint64_t maxAlignment = 0;
    bool allSameType = true;
    bool valid = true;
  };

  std::map<unsigned, ReuseGroup> groups;
  kernel.walk([&](memref::AllocOp alloc) {
    Value buffer = alloc.getMemref();
    auto colorIt = liveliness.rootShmColors.find(buffer);
    if (colorIt == liveliness.rootShmColors.end()) {
      return;
    }

    auto type = cast<MemRefType>(buffer.getType());
    if (type.getMemorySpaceAsInt() != int(friskMs::Shared)) {
      return;
    }
    auto elementCount = getStaticElementCount(type);
    if (!elementCount) {
      return;
    }
    if (llvm::any_of(alloc->getUsers(), [](Operation *user) {
          return isa<memref::DeallocOp>(user);
        })) {
      return;
    }

    auto &group = groups[colorIt->second];
    if (group.allocs.empty()) {
      group.firstType = type;
      group.elementType = type.getElementType();
      group.memorySpace = type.getMemorySpace();
      group.parentBlock = alloc->getBlock();
    } else {
      group.allSameType = group.allSameType && type == group.firstType;
      if (type.getElementType() != group.elementType ||
          type.getMemorySpace() != group.memorySpace ||
          alloc->getBlock() != group.parentBlock) {
        group.valid = false;
      }
    }

    group.maxElements = std::max(group.maxElements, *elementCount);
    if (auto alignment = alloc.getAlignment()) {
      group.maxAlignment = std::max(group.maxAlignment, *alignment);
    }
    group.allocs.push_back(alloc);
  });

  OpBuilder builder(kernel.getContext());
  unsigned reusedBuffers = 0;
  for (auto &[color, group] : groups) {
    if (!group.valid || group.allocs.size() < 2) {
      continue;
    }

    auto firstAlloc = group.allocs.front();
    builder.setInsertionPoint(firstAlloc);
    IntegerAttr alignmentAttr;
    if (group.maxAlignment > 0) {
      alignmentAttr = builder.getI64IntegerAttr(group.maxAlignment);
    }

    MemRefType backingType = group.allSameType
                                 ? group.firstType
                                 : MemRefType::get({group.maxElements},
                                                   group.elementType,
                                                   AffineMap{},
                                                   group.memorySpace);
    auto backing =
        builder.create<memref::AllocOp>(firstAlloc.getLoc(), backingType,
                                        alignmentAttr);
    backing->setAttr("shm_reuse_color", builder.getI64IntegerAttr(color));
    backing->setAttr("shm_reuse_group_size",
                     builder.getI64IntegerAttr(group.allocs.size()));

    SmallVector<Operation *> erased;
    erased.reserve(group.allocs.size());
    for (auto alloc : group.allocs) {
      Value replacement = backing.getMemref();
      auto originalType = cast<MemRefType>(alloc.getMemref().getType());
      if (!group.allSameType) {
        builder.setInsertionPoint(alloc);
        auto shape = originalType.getShape();
        auto strides = getContiguousStrides(shape);
        replacement = builder
                          .create<memref::ReinterpretCastOp>(
                              alloc.getLoc(), originalType, backing.getMemref(),
                              /*offset=*/0, shape, strides)
                          .getResult();
      }
      alloc.getMemref().replaceAllUsesWith(replacement);
      erased.push_back(alloc);
      ++reusedBuffers;
    }

    for (Operation *op : erased) {
      op->erase();
    }
  }

  llvm::outs() << "[applyBufferReuse] reused shared buffers: " << reusedBuffers
               << "\n";
  llvm::outs().flush();

}

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
    llvm::outs() << "\n-------------- lowerinfo analyze done\n";llvm::outs().flush();
    s_info->print();
    llvm::outs() << "\n-------------- lowerinfo print done!\n";llvm::outs().flush();

    auto warpLayout = s_info->begin()->getSecond().get_warp_layout();
    auto blockLayout = s_info->begin()->getSecond().get_block_layout();
    auto blockLayoutOrder = s_info->begin()->getSecond().get_block_layout_order();

    kernel->setAttr("warp_layout", DenseI64ArrayAttr::get(context, warpLayout));
    kernel->setAttr("block_layout", DenseI64ArrayAttr::get(context, blockLayout));
    kernel->setAttr("block_layout_order", DenseI64ArrayAttr::get(context, blockLayoutOrder));
    
   

    // 根据 layout推定结果，插入 convertLAyoutOp
    insertConvertLayoutOps(*s_info);

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
      BlockOp, GemmOp, ReduceOp, ConvertLayoutOp, CopyOp, FillOp
    >();

    RewritePatternSet patterns(context);
    patterns.add<
      BlockOpConversion, GemmOpConversion, ReduceOpConversion, CopyConvertOpRewrite, CopyOpRewrite, FillOpRewrite, 
      ConvertLayoutOpConversion
    >(context);

    llvm::outs() << "-- lowerinfo partialconversion\n";llvm::outs().flush();
    applyPartialConversion(kernel, target, std::move(patterns));
    
    // -------- step2 : 替换 allocbuffer -> memref.alloc / alloca
    ConversionTarget t2(*context);
    t2.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect>();
    t2.addIllegalOp<KernelOp,ParallelOp,ForOp,
      BlockOp, GemmOp, ReduceOp, ConvertLayoutOp, CopyOp, FillOp, AllocBufferOp
    >();
    RewritePatternSet ps2(context);
    ps2.add<
      AllocBufferOpConversion
    >(context);
    if (failed(applyFullConversion(kernel, t2, std::move(ps2)))){
      return signalPassFailure();
    }
    llvm::outs() << "-- convert to thread level IR done!\n";llvm::outs().flush();
    // -------- step 3 生命周期分析。buffer 复用优化
    LivelinessAnalyzer liveliness;
    liveliness.run(kernel);
    applyBufferReuse(kernel, liveliness);

  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass() {
  return std::make_unique<ConvertFriskBaseToThreadLevelIR>();
}

}  // end namespace frisk 
