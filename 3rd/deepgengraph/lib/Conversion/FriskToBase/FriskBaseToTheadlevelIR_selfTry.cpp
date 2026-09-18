//===----------------------------------------------------------------------===//
// Frisk block-level -> thread-level lowering
//
// 阅读入口：文件末尾 ConvertFriskBaseToThreadLevelIR::runOnOperation。
// 阶段顺序：布局推导 -> Block/GEMM/Mask -> SSA Fill -> Elementwise -> Reduce
//           -> Copy/Fill/ConvertLayout -> AllocBuffer -> 清理。
// 每阶段的合法性配置与 Pattern 注册放在同一个 lower* 方法中。
//
// 两种坐标不要混用：
//   block 坐标：shared/global buffer 的真实坐标，包含 tid 对布局的影响。
//   thread 坐标：一个线程持有的 local memref/vector 内部坐标。
// LowerInfo 描述访问映射；thread_own_data_size 描述线程实际持有的形状。
// 形状相同不代表布局相同，跨线程重排必须经过 ConvertLayout 的 shared scratch。
//
// 代码组织：布局查询/坐标工具；vector 循环生成；计算 Pattern；Block；
//           布局转换与 Copy；Fill/Alloc；Pass 驱动。
// 大型改写使用具名函数或单次调用对象，lambda 仅保留在 MLIR 回调和短谓词中。
// 更详细的 Pattern 对照与扩展约定见同目录 FriskBaseToThreadLevelIR.md。
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cassert>
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
#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR

#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

using friskMs = frisk::attr::MemorySpace;
// 保留现有分析层的共享状态。每个 kernel 开始时重置替换表；这些静态表及
// LowerInfoAnalysis 当前按串行执行使用，不能仅凭 Pattern 的 const
// 方法推断线程安全。
static LowerInfoMap *s_info{nullptr};
static HWSpecification *s_hw{nullptr};

// 原 block buffer/SSA 值 -> thread memref 或 vector。vector 替换前须检查
// dominance， 否则循环外的使用点可能错误引用只在循环体内定义的值。
static DenseMap<mlir::Value, mlir::Value> s_buffer_replace;
static DenseMap<frisk::ConvertLayoutOp, std::pair<LowerInfo, LowerInfo>>
    convertLayoutInfo;

static Value getVectorReplacement(Value value, Operation *user = nullptr);
static Value getVectorValue(Value value);
static FailureOr<Value> extractThreadTile(Value fullVector, VectorType tileTy,
                                          LowerInfo &info, OpBuilder &rewriter,
                                          Location loc, Operation *anchorOp);

static bool isLocalMemref(Value buffer) {
  auto ty = mlir::cast<MemRefType>(buffer.getType());
  auto memorySpace = ty.getMemorySpaceAsInt();
  return memorySpace == int(friskMs::Local) || memorySpace == 5;
}

static bool isGlobalMemref(Value buffer) {
  auto ty = mlir::cast<MemRefType>(buffer.getType());
  return ty.getMemorySpaceAsInt() == int(friskMs::Global);
}

static bool isSharedMemref(Value buffer) {
  auto ty = mlir::cast<MemRefType>(buffer.getType());
  return ty.getMemorySpaceAsInt() == int(friskMs::Shared);
}

static FailureOr<int64_t> getThreadNum(Operation *op,
                                       PatternRewriter &rewriter) {
  auto kernel =
      getOuterMostOpWithName(op, func::FuncOp::getOperationName().data());
  if (kernel == nullptr || !kernel->hasAttr("thread_num")) {
    (void)rewriter.notifyMatchFailure(op, "missing thread_num attribute");
    return failure();
  }
  auto threadNumAttr = dyn_cast<IntegerAttr>(kernel->getAttr("thread_num"));
  if (!threadNumAttr || threadNumAttr.getInt() <= 0) {
    (void)rewriter.notifyMatchFailure(op, "invalid thread_num attribute");
    return failure();
  }
  return threadNumAttr.getInt();
}

static LowerInfo getLowerInfoOrDie(Value buffer, Operation *op) {
  LowerInfo *info = s_info->getLowerInfo(buffer, op);
  // 仅供要求分析已完成的路径使用；允许布局缺失的 Pattern 应使用
  // findLowerInfoForValue。
  assert(info != nullptr && "LowerInfo not found");
  return *info;
}

/// 按 consumer、其他 user、分析表中的 buffer
/// 顺序查找，优先保留当前使用点的布局。
static LowerInfo *findDirectLowerInfo(Value value, Operation *consumerOp) {
  if (!s_info)
    return nullptr;
  if (consumerOp) {
    if (auto *info = s_info->getLowerInfo(value, consumerOp))
      return info;
  }
  for (Operation *user : value.getUsers()) {
    if (auto *info = s_info->getLowerInfo(value, user))
      return info;
  }
  for (auto &entry : *s_info) {
    if (entry.second.buffer == value)
      return &entry.second;
  }
  return nullptr;
}

/// conversion materialization 可能包了一层 cast；只解开单输入
/// cast，避免猜测多值对应关系。
static LowerInfo *findLowerInfoForValue(Value value, Operation *consumerOp) {
  if (auto *info = findDirectLowerInfo(value, consumerOp))
    return info;
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      castOp && castOp.getInputs().size() == 1)
    return findDirectLowerInfo(castOp.getInputs()[0], consumerOp);
  return nullptr;
}

static std::optional<LowerInfo> copyLowerInfoForValue(Value value,
                                                      Operation *consumerOp) {
  if (auto *info = findLowerInfoForValue(value, consumerOp))
    return *info;
  return std::nullopt;
}

static LowerInfo *findLowerInfoForMaterialization(Value input, Type resultType,
                                                  Operation *consumerOp) {
  if (auto *info = findLowerInfoForValue(input, consumerOp)) {
    return info;
  }
  if (s_info == nullptr || consumerOp == nullptr) {
    return nullptr;
  }

  for (OpOperand &operand : consumerOp->getOpOperands()) {
    Value candidate = operand.get();
    if (candidate == input || candidate.getType() != resultType) {
      continue;
    }
    if (auto *info = findLowerInfoForValue(candidate, consumerOp)) {
      return info;
    }
  }
  return nullptr;
}

static void registerMaterializedLowerInfo(UnrealizedConversionCastOp castOp,
                                          LowerInfo *lowerInfo,
                                          Operation *consumerOp) {
  if (s_info == nullptr || lowerInfo == nullptr || consumerOp == nullptr) {
    return;
  }
  LowerInfo castInfo = *lowerInfo;
  castInfo.buffer = castOp.getResult(0);
  s_info->addLowerInfo(consumerOp, castInfo);
}

static void registerConvertedValueLowerInfo(Value converted,
                                            const LowerInfo &lowerInfo,
                                            Operation *producerOp) {
  if (s_info == nullptr || converted == nullptr || producerOp == nullptr) {
    return;
  }
  LowerInfo convertedInfo = lowerInfo;
  convertedInfo.buffer = converted;
  s_info->addLowerInfo(producerOp, convertedInfo);
}

/// 在布局发生变化的 consumer 前插入一次 ConvertLayout，并保存新旧布局。
/// 先登记 LowerInfo，再替换 consumer 的 operand，避免改写后丢失使用点信息。
static void insertConvertLayoutOps(LowerInfoMap &infoMap) {
  SmallVector<std::pair<Value, Operation *>, 8> inserted;

  for (auto &entry : infoMap) {
    LowerInfo &toInfo = entry.second;
    if (toInfo.convertFrom == nullptr || toInfo.buffer == nullptr ||
        toInfo.op == nullptr) {
      continue;
    }

    auto key = std::make_pair(toInfo.buffer, toInfo.op);
    if (llvm::is_contained(inserted, key)) {
      continue;
    }
    inserted.push_back(key);

    OpBuilder builder(toInfo.op);
    auto convertLayoutOp = builder.create<frisk::ConvertLayoutOp>(
        toInfo.op->getLoc(), toInfo.buffer.getType(), toInfo.buffer,
        builder.getStringAttr("lowerinfo.convert"));
    LowerInfo fromInfo = *toInfo.convertFrom;
    LowerInfo newInfo = toInfo;
    newInfo.buffer = convertLayoutOp->getResult(0);
    s_info->updateLowerInfoForLayoutConvertOp(convertLayoutOp, toInfo);
    convertLayoutInfo.insert({convertLayoutOp, {fromInfo, newInfo}});
    toInfo.op->replaceUsesOfWith(toInfo.buffer, convertLayoutOp->getResult(0));
  }
}

static gpu::ThreadIdOp findThreadIdxOp(mlir::Operation *currOp) {
  auto kernel =
      getOuterMostOpWithName(currOp, func::FuncOp::getOperationName().data());
  assert(kernel->hasAttr("thread_num"));
  auto funcOp = mlir::cast<func::FuncOp>(kernel);
  gpu::ThreadIdOp tidx = nullptr;
  funcOp->walk([&](mlir::gpu::ThreadIdOp tidOp) {
    if (tidx == nullptr && tidOp.getDimension() == gpu::Dimension::x) {
      tidx = tidOp;
    }
  });
  assert(tidx != nullptr);
  return tidx;
}

static SmallVector<Value, 2> makeLocalIndices(ArrayRef<Value> ivs,
                                              unsigned rank) {
  SmallVector<Value, 2> indices;
  for (unsigned i = 0; i < rank && i < ivs.size(); ++i) {
    indices.push_back(ivs[i]);
  }
  return indices;
}

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
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
  return createSingleDimAffineApply(builder, loc, d0.floorDiv(divisor),
                                    operand);
}

static SmallVector<Value>
buildContiguousIndicesFromLinear(OpBuilder &builder, Location loc,
                                 Value linearIndex, ArrayRef<int64_t> shape) {
  SmallVector<Value> indices;
  indices.reserve(shape.size());
  int64_t stride = 1;
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= std::max<int64_t>(shape[i], 1);
  }
  for (unsigned i = 0; i < shape.size(); ++i) {
    int64_t dim = shape[i];
    if (dim == 1) {
      indices.push_back(createIndexConstant(builder, loc, 0));
      continue;
    }
    Value index = floorDivBy(builder, loc, linearIndex, strides[i]);
    if (dim > 1) {
      index = modBy(builder, loc, index, dim);
    }
    indices.push_back(index);
  }
  return indices;
}

static Value addIndexValues(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, d0 + d1, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, ValueRange{lhs, rhs});
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

/// LowerInfo map 固定接收七个参数：
/// [tid, br0, br1, iu0, iu1, flattened_warp_repeat, flattened_register]。
/// wr/reg 的展平顺序由 layout 指定，不能假定行优先。
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
    auto oneResultMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                       map.getResult(i), builder.getContext());
    indices.push_back(
        builder.create<affine::AffineApplyOp>(loc, oneResultMap, mapOperands));
  }
  return indices;
}

/// 将线程 tile 下标拆为 br/iu/wr/reg，再应用含 tid 的布局 map 得到 block 坐标。
/// 每维满足 iv = (((br * instUnroll + iu) * warpRepeat + wr) * threadWidth +
/// reg)。
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
    ius.push_back(modBy(builder, loc, floorDivBy(builder, loc, iv, repeatWidth),
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

/// 上述分解的线程内逆映射，不包含 tid；用于将一个 MMA 片段放回完整 thread
/// vector。
static AffineMap buildThreadTileOffsetMap(OpBuilder &builder, LowerInfo &info) {
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);
  auto d3 = builder.getAffineDimExpr(3);
  auto d4 = builder.getAffineDimExpr(4);
  auto d5 = builder.getAffineDimExpr(5);

  auto [wr0, wr1] = UnflattenIndexToXY(d4, info.base_layout.warp_repeat_order,
                                       info.get_warp_repeat());
  auto [reg0, reg1] = UnflattenIndexToXY(d5, info.base_layout.thread_creg_order,
                                         info.get_thread_widths());

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

/// 为 vector 的每个元素生成循环，并用 iter_args/yield 传递逐步更新的 vector。
/// 单位维度直接使用 0；返回后插入点位于最外层循环之后。
/// BodyEmitter::emit 只负责单个元素，不负责循环、yield 或插入点恢复。
struct VectorTileLoopOptions {
  StringRef marker;
  StringRef labelPrefix;
  Value unitIndex; // 可复用调用方已有的 0，保留 GEMM 的常量放置位置。
};

class VectorTileLoopNest {
public:
  VectorTileLoopNest(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
                     VectorTileLoopOptions options = {})
      : builder(builder), loc(loc), shape(shape), options(options) {}

  template <typename BodyEmitter>
  Value emit(Value initial, BodyEmitter &body, unsigned dim = 0) {
    if (dim == shape.size())
      return body.emit(indices, initial);
    if (shape[dim] == 1) {
      indices.push_back(options.unitIndex
                            ? options.unitIndex
                            : createIndexConstant(builder, loc, 0));
      Value result = emit(initial, body, dim + 1);
      indices.pop_back();
      return result;
    }
    auto loop = builder.create<affine::AffineForOp>(loc, 0, shape[dim], 1,
                                                    ValueRange{initial});
    if (!options.marker.empty())
      loop->setAttr(options.marker, builder.getBoolAttr(true));
    if (!options.labelPrefix.empty())
      loop->setAttr("iterLabel", builder.getStringAttr(
                                     Twine(options.labelPrefix) + Twine(dim)));
    builder.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    Value result = emit(loop.getRegionIterArgs()[0], body, dim + 1);
    builder.setInsertionPointToEnd(loop.getBody());
    if (result)
      builder.create<affine::AffineYieldOp>(loc, result);
    indices.pop_back();
    builder.setInsertionPointAfter(loop);
    return result ? loop.getResult(0) : Value{};
  }

private:
  OpBuilder &builder;
  Location loc;
  ArrayRef<int64_t> shape;
  VectorTileLoopOptions options;
  SmallVector<Value, 2> indices;
};

static Value getGemmUnrollIndex(LowerInfo &info, int dim, Value candidate,
                                Operation *op, OpBuilder &rewriter) {
  Value zero = createIndexConstant(rewriter, op->getLoc(), 0);
  return info.warpInstUnroll[dim] > 1 ? candidate : zero;
}

static FailureOr<Value> requireGemmLocalVector(Value adapted, StringRef name,
                                               Operation *op,
                                               PatternRewriter &rewriter) {
  if (mlir::isa<VectorType>(adapted.getType())) {
    return adapted;
  }
  return rewriter.notifyMatchFailure(
      op, Twine("local ") + name + " operand was not materialized as a vector");
}

static void
noteDeadCastCandidate(Value value,
                      SmallVectorImpl<Operation *> &materializedCastsToErase) {
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    Operation *castOperation = castOp.getOperation();
    if (!llvm::is_contained(materializedCastsToErase, castOperation)) {
      materializedCastsToErase.push_back(castOperation);
    }
  }
}

static FailureOr<Value> materializeGemmThreadTile(Value vector,
                                                  VectorType vecTy,
                                                  LowerInfo &info,
                                                  StringRef name, Operation *op,
                                                  PatternRewriter &rewriter) {
  auto vectorTy = mlir::dyn_cast<VectorType>(vector.getType());
  if (!vectorTy) {
    return failure();
  }
  if (vectorTy == vecTy) {
    return vector;
  }
  auto tile =
      extractThreadTile(vector, vecTy, info, rewriter, op->getLoc(), op);
  if (succeeded(tile)) {
    return *tile;
  }
  return rewriter.notifyMatchFailure(
      op, Twine("vector-backed ") + name +
              " operand does not match the GEMM thread tile type");
}

static Value copyGemmOperandToRegisters(LowerInfo &info, Value src,
                                        VectorType vecTy, Value br0, Value br1,
                                        Value iu0, Value iu1, Value tidx,
                                        Value zero, Operation *op,
                                        OpBuilder &rewriter) {
  auto mapOperands = buildLowerInfoMapOperands(
      rewriter, op->getLoc(), info, tidx, br0, br1, iu0, iu1,
      /*wr0=*/zero, /*wr1=*/zero, /*reg0=*/zero, /*reg1=*/zero);
  auto copy = rewriter.create<frisk::CopyToRegOp>(
      op->getLoc(), vecTy, src, mapOperands, info.getAffineMap());
  // A vector-backed producer owns the whole thread tile. Preserve the GEMM
  // fragment offset separately from the shared-memory (block) coordinates.
  SmallVector<AffineExpr, 2> offsets;
  for (unsigned i = 0; i < 2; ++i) {
    offsets.push_back(
        (rewriter.getAffineDimExpr(1 + i) * info.warpInstUnroll[i] +
         rewriter.getAffineDimExpr(3 + i)) *
        (info.get_warp_repeat()[i] * info.get_thread_widths()[i]));
  }
  copy->setAttr("gemm_thread_offset", AffineMapAttr::get(
      AffineMap::get(7, 0, offsets, rewriter.getContext())));
  return copy.getResult();
}

/// 匹配：已有 A/B/C LowerInfo 和 MMA 指令属性的 GEMM，当前实现 DCU 路径。
/// 改写：编译期枚举 MN 片段，K 循环读 A/B 并发射 WarpMmaRR，再静态插入片段。
/// A/B 的 K 迭代数必须一致；累加器保留原实现的零初始化语义。
class GemmOpConversion : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tidx = findThreadIdxOp(op);
    // get lowerInfo
    auto infoA = getLowerInfoOrDie(op.getA(), op.getOperation());
    auto infoB = getLowerInfoOrDie(op.getB(), op.getOperation());
    auto infoC = getLowerInfoOrDie(op.getC(), op.getOperation());
    llvm::errs() << "[gemm-lower] A br=[" << infoA.get_block_repeat()[0] << ","
                 << infoA.get_block_repeat()[1] << "] buffer=" << op.getA()
                 << "\n";
    llvm::errs() << "[gemm-lower] B br=[" << infoB.get_block_repeat()[0] << ","
                 << infoB.get_block_repeat()[1] << "] buffer=" << op.getB()
                 << "\n";
    llvm::errs() << "[gemm-lower] C br=[" << infoC.get_block_repeat()[0] << ","
                 << infoC.get_block_repeat()[1] << "] buffer=" << op.getC()
                 << "\n";
    auto ka = infoA.get_block_repeat()[1] * infoA.warpInstUnroll[1];
    auto kb = infoB.get_block_repeat()[0] * infoB.warpInstUnroll[0];
    assert(ka == kb); // k轴上的 for循环次数. A 列迭代数 == B 行迭代数
    assert(infoA.mmaInst->asm_str == infoB.mmaInst->asm_str);
    auto typeA = mlir::cast<MemRefType>(op.getA().getType());
    auto typeB = mlir::cast<MemRefType>(op.getB().getType());
    auto typeC = mlir::cast<MemRefType>(op.getC().getType());

    if (s_hw->getKind() == HW_KIND_DCU) {
      auto instName = op->getAttrOfType<StringAttr>("inst_name");
      auto [br0, br1] = infoC.get_block_repeat();
      auto [wiu0, wiu1] = infoC.warpInstUnroll;
      int kloopCount = infoA.get_block_repeat()[1];
      int kWarpInstUnroll = infoA.warpInstUnroll[1];

      auto regCShape = infoC.get_thread_own_data_size();
      auto regC_singleInstShape =
          infoC.get_thread_widths() * infoC.get_warp_repeat();
      auto regCTy = VectorType::get(regCShape, typeC.getElementType());
      auto tempVecTy =
          VectorType::get(regC_singleInstShape, typeC.getElementType());
      auto regCInit = rewriter.create<arith::ConstantOp>(
          op->getLoc(), regCTy,
          mlir::cast<TypedAttr>(rewriter.getZeroAttr(regCTy)));

      // A fragment contains thread_creg * warp_repeat elements. Its position
      // in the packed thread tile is static even though its LDS address depends
      // on the lane. Enumerate MN here, retaining the K reduction as a loop.
      int mnLoopCount = br0 * br1 * wiu0 * wiu1;
      int kLoopCount = kloopCount * kWarpInstUnroll;
      Value regC = regCInit;
      SmallVector<Operation *, 2> materializedCastsToErase;
      for (int mn = 0; mn < mnLoopCount; ++mn) {
        int tmp = mn;
        int iu1 = tmp % wiu1;
        tmp /= wiu1;
        int iu0 = tmp % wiu0;
        tmp /= wiu0;
        int repeat1 = tmp % br1;
        int repeat0 = tmp / br1;
        Value cWiu0 = createIndexConstant(rewriter, op->getLoc(), iu0);
        Value cWiu1 = createIndexConstant(rewriter, op->getLoc(), iu1);
        Value cBr0 = createIndexConstant(rewriter, op->getLoc(), repeat0);
        Value cBr1 = createIndexConstant(rewriter, op->getLoc(), repeat1);

        auto tempVecInit = rewriter.create<arith::ConstantOp>(
            op->getLoc(), tempVecTy,
            mlir::cast<TypedAttr>(rewriter.getZeroAttr(tempVecTy)));
        auto kFor = rewriter.create<affine::AffineForOp>(
            op->getLoc(), 0, kLoopCount, 1, ValueRange{tempVecInit.getResult()});
        kFor->setAttr("iterLabel", rewriter.getStringAttr("k"));
        rewriter.setInsertionPointToStart(kFor.getBody());

        Value kLinear = kFor.getInductionVar();
        Value kRegC = kFor.getRegionIterArgs()[0];
        Value aKBr = floorDivBy(rewriter, op->getLoc(), kLinear, kWarpInstUnroll);
        Value aKIu = modBy(rewriter, op->getLoc(), kLinear, kWarpInstUnroll);
        Value bKBr =
            floorDivBy(rewriter, op->getLoc(), kLinear, infoB.warpInstUnroll[0]);
        Value bKIu =
            modBy(rewriter, op->getLoc(), kLinear, infoB.warpInstUnroll[0]);

        Value zero = createIndexConstant(rewriter, op->getLoc(), 0);

        // 2. 优先使用已有 vector；shared 通过 CopyToReg 读取；local 要求已物化。
        Value thBufferA;
        auto vecTyA =
            VectorType::get(infoA.get_thread_widths(), typeA.getElementType());
        if (Value aVector = getVectorValue(adaptor.getA())) {
          auto materializedA = materializeGemmThreadTile(aVector, vecTyA, infoA,
                                                         "A", op, rewriter);
          if (failed(materializedA)) {
            return failure();
          }
          thBufferA = *materializedA;
          noteDeadCastCandidate(op.getA(), materializedCastsToErase);
          noteDeadCastCandidate(adaptor.getA(), materializedCastsToErase);
        } else if (isSharedMemref(op.getA())) {
          thBufferA = copyGemmOperandToRegisters(
              infoA, adaptor.getA(), vecTyA,
              /*br0=*/cBr0, /*br1=*/aKBr,
              /*iu0=*/getGemmUnrollIndex(infoA, 0, cWiu0, op, rewriter),
              /*iu1=*/getGemmUnrollIndex(infoA, 1, aKIu, op, rewriter), tidx,
              zero, op, rewriter);
        } else {
          auto localA = requireGemmLocalVector(adaptor.getA(), "A", op, rewriter);
          if (failed(localA)) {
            return failure();
          }
          thBufferA = *localA;
        }

        Value thBufferB;
        auto vecTyB =
            VectorType::get(infoB.get_thread_widths(), typeB.getElementType());
        if (Value bVector = getVectorValue(adaptor.getB())) {
          auto materializedB = materializeGemmThreadTile(bVector, vecTyB, infoB,
                                                         "B", op, rewriter);
          if (failed(materializedB)) {
            return failure();
          }
          thBufferB = *materializedB;
          noteDeadCastCandidate(op.getB(), materializedCastsToErase);
          noteDeadCastCandidate(adaptor.getB(), materializedCastsToErase);
        } else if (isSharedMemref(op.getB())) {
          thBufferB = copyGemmOperandToRegisters(
              infoB, adaptor.getB(), vecTyB,
              /*br0=*/bKBr, /*br1=*/cBr1,
              /*iu0=*/getGemmUnrollIndex(infoB, 0, bKIu, op, rewriter),
              /*iu1=*/getGemmUnrollIndex(infoB, 1, cWiu1, op, rewriter), tidx,
              zero, op, rewriter);
        } else {
          auto localB = requireGemmLocalVector(adaptor.getB(), "B", op, rewriter);
          if (failed(localB)) {
            return failure();
          }
          thBufferB = *localB;
        }

        // 3. K 循环累加单个 MMA 片段，退出 K 后再写入完整 C thread vector。
        auto wmma = rewriter.create<frisk::WarpMmaRROp>(
            op->getLoc(), tempVecTy, thBufferA, thBufferB, kRegC);
        rewriter.modifyOpInPlace(wmma, [&]() {
          wmma->setAttr("inst_name", instName);
          wmma->setAttr("inst_constraints", op->getAttr("inst_constraints"));
        });

        rewriter.create<affine::AffineYieldOp>(op->getLoc(), wmma.getResult());

        rewriter.setInsertionPointAfter(kFor);

        SmallVector<int64_t, 2> offsets{
            (repeat0 * wiu0 + iu0) * regC_singleInstShape[0],
            (repeat1 * wiu1 + iu1) * regC_singleInstShape[1]};
        auto insert = rewriter.create<vector::InsertStridedSliceOp>(
            op->getLoc(), kFor.getResult(0), regC, offsets,
            SmallVector<int64_t, 2>{1, 1});
        // Consumed by the post-thread-lowering accumulator fusion pass.
        insert->setAttr("frisk.mma_fragment", rewriter.getUnitAttr());
        regC = insert.getResult();
      }

      registerConvertedValueLowerInfo(regC, infoC, op.getOperation());
      rewriter.replaceOp(op, regC);
      for (Operation *castOperation : materializedCastsToErase) {
        if (llvm::all_of(castOperation->getResults(),
                         [](Value result) { return result.use_empty(); })) {
          rewriter.eraseOp(castOperation);
        }
      }
    } else if (s_hw->getKind() == HW_KIND_NVIDIA) {
      // ...
    }
    return success();
  }
};

/// 把 mask region 的参数替换为 starts + block 坐标，克隆标量计算并写入 thread
/// tile。
struct MaskTileElement {
  ConversionPatternRewriter &rewriter;
  Location loc;
  LowerInfo *resultInfo;
  Value tidx;
  MemRefType resultTy;
  frisk::MaskOp::Adaptor adaptor;
  Block *body;
  frisk::MaskYieldOp yieldOp;
  Value emit(ArrayRef<Value> tileIvs, Value initVector) {
    auto accessIndices = buildMappedAccessIndices(
        rewriter, loc, *resultInfo, tidx, tileIvs, resultTy.getRank());
    SmallVector<Value, 2> regionArgs;
    regionArgs.reserve(accessIndices.size());
    for (auto [start, accessIndex] :
         llvm::zip(adaptor.getStarts(), accessIndices)) {
      regionArgs.push_back(addIndexValues(rewriter, loc, start, accessIndex));
    }

    IRMapping mapper;
    for (auto [oldArg, newArg] : llvm::zip(body->getArguments(), regionArgs)) {
      mapper.map(oldArg, newArg);
    }
    for (auto &childOp : body->without_terminator()) {
      auto *cloned = rewriter.clone(childOp, mapper);
      for (auto [oldResult, newResult] :
           llvm::zip(childOp.getResults(), cloned->getResults())) {
        mapper.map(oldResult, newResult);
      }
    }

    Value scalar = mapper.lookupOrDefault(yieldOp->getOperand(0));
    SmallVector<OpFoldResult, 2> position;
    position.reserve(tileIvs.size());
    for (Value iv : tileIvs) {
      position.push_back(iv);
    }
    return rewriter.create<vector::InsertOp>(loc, scalar, initVector, position)
        .getResult();
  }
};

/// 匹配：rank <= 2、带 LowerInfo 的 memref mask，region yield 一个标量。
/// 改写：对线程持有的每个元素执行 region，结果以 vector SSA 值替换原 memref。
class MaskOpConversion : public OpConversionPattern<frisk::MaskOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(frisk::MaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto resultTy = mlir::dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "mask result must be a memref");
    }
    if (resultTy.getRank() > 2) {
      return rewriter.notifyMatchFailure(
          op, "mask vector lowering currently supports rank <= 2");
    }

    LowerInfo *resultInfo =
        s_info->getLowerInfo(op.getResult(), op.getOperation());
    if (resultInfo == nullptr) {
      for (auto user : op->getUsers()) {
        resultInfo = s_info->getLowerInfo(op.getResult(), user);
        if (resultInfo) {
          break;
        }
      }
    }
    if (resultInfo == nullptr) {
      for (auto &entry : *s_info) {
        if (entry.second.buffer == op.getResult()) {
          resultInfo = &entry.second;
          break;
        }
      }
    }
    if (resultInfo == nullptr) {
      return rewriter.notifyMatchFailure(op, "LowerInfo not found for mask");
    }

    auto threadOwnData = resultInfo->get_thread_own_data_size();
    SmallVector<int64_t, 2> vectorShape;
    vectorShape.reserve(resultTy.getRank());
    for (int64_t i = 0; i < resultTy.getRank(); ++i) {
      if (threadOwnData[i] <= 0) {
        return rewriter.notifyMatchFailure(
            op, "mask vector lowering expects positive thread_own_data_sz");
      }
      vectorShape.push_back(threadOwnData[i]);
    }

    auto vecTy = VectorType::get(vectorShape, resultTy.getElementType());
    Value resultVector =
        rewriter
            .create<arith::ConstantOp>(
                loc, vecTy, mlir::cast<TypedAttr>(rewriter.getZeroAttr(vecTy)))
            .getResult();

    Block *body = &op.getRegion().front();
    auto yieldOp = dyn_cast<frisk::MaskYieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp->getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(
          op, "mask vector lowering expects one mask_yield operand");
    }
    if (body->getNumArguments() != static_cast<unsigned>(resultTy.getRank())) {
      return rewriter.notifyMatchFailure(
          op, "mask region argument count must match result rank");
    }

    auto tidx = findThreadIdxOp(op);

    MaskTileElement element{rewriter, loc,     resultInfo, tidx,
                            resultTy, adaptor, body,       yieldOp};
    resultVector =
        VectorTileLoopNest(rewriter, loc, vectorShape, {{}, "mask_tw", {}})
            .emit(resultVector, element);
    registerConvertedValueLowerInfo(resultVector, *resultInfo,
                                    op.getOperation());
    rewriter.replaceOp(op, resultVector);
    return success();
  }
};

static VectorType
getVecTypeByLowerInfoWithThreadOwnData(const LowerInfo &info) {
  auto shape = info.get_thread_own_data_size();
  Type elementType;
  if (auto memTy = mlir::dyn_cast<MemRefType>(info.buffer.getType())) {
    elementType = memTy.getElementType();
  } else if (auto vecTy = mlir::dyn_cast<VectorType>(info.buffer.getType())) {
    elementType = vecTy.getElementType();
  } else if (auto shapedTy =
                 mlir::dyn_cast<ShapedType>(info.buffer.getType())) {
    elementType = shapedTy.getElementType();
  }
  assert(elementType && "LowerInfo buffer must carry an element type");
  return VectorType::get(shape, elementType);
}

static VectorType getConvertedBufferType(mlir::Value adaptedValue,
                                         const LowerInfo &info) {
  auto vec = mlir::dyn_cast<VectorType>(adaptedValue.getType());
  if (vec) {
    return vec;
  }
  auto mem = mlir::dyn_cast<MemRefType>(adaptedValue.getType());
  if (mem) {
    if (mem.getMemorySpaceAsInt() == int(friskMs::Local) ||
        mem.getMemorySpaceAsInt() == 5) {
      return getVecTypeByLowerInfoWithThreadOwnData(info);
    }
  }
  return {};
}

/// 一个 thread 坐标对应一次 block buffer/vector 读取；长度为 1 的源维度固定读
/// 0。
struct ExtractThreadTileElement {
  OpBuilder &rewriter;
  Location loc;
  LowerInfo &info;
  Value tidx;
  ShapedType fullTy;
  Value fullVector;
  Value emit(ArrayRef<Value> tileIvs, Value currentTile) {
    auto fullIndices =
        buildMappedAccessIndices(rewriter, loc, info, tidx, tileIvs,
                                 static_cast<unsigned>(fullTy.getRank()));
    for (unsigned i = 0; i < fullIndices.size(); ++i) {
      if (fullTy.getDimSize(i) == 1)
        fullIndices[i] = createIndexConstant(rewriter, loc, 0);
    }
    SmallVector<int64_t, 2> staticFullPosition(fullIndices.size(),
                                               ShapedType::kDynamic);
    Value scalar;
    if (isa<MemRefType>(fullVector.getType())) {
      scalar =
          rewriter.create<affine::AffineLoadOp>(loc, fullVector, fullIndices);
    } else {
      scalar = rewriter.create<vector::ExtractOp>(
          loc, fullTy.getElementType(), fullVector, fullIndices,
          rewriter.getDenseI64ArrayAttr(staticFullPosition));
    }

    SmallVector<OpFoldResult, 2> tilePosition;
    tilePosition.reserve(tileIvs.size());
    for (Value iv : tileIvs) {
      tilePosition.push_back(iv);
    }
    return rewriter
        .create<vector::InsertOp>(loc, scalar, currentTile, tilePosition)
        .getResult();
  }
};

static FailureOr<Value> extractThreadTile(Value fullVector, VectorType tileTy,
                                          LowerInfo &info, OpBuilder &rewriter,
                                          Location loc, Operation *anchorOp) {
  auto fullTy = mlir::dyn_cast<ShapedType>(fullVector.getType());
  if (fullVector.getType() == tileTy)
    return fullVector;
  if (!fullTy || !isa<VectorType, MemRefType>(fullVector.getType()))
    return failure();
  if (fullTy.getElementType() != tileTy.getElementType() ||
      fullTy.getRank() != tileTy.getRank()) {
    return failure();
  }

  Value result =
      rewriter
          .create<arith::ConstantOp>(
              loc, tileTy, mlir::cast<TypedAttr>(rewriter.getZeroAttr(tileTy)))
          .getResult();

  auto tidx = findThreadIdxOp(anchorOp);

  ExtractThreadTileElement element{rewriter, loc,    info,
                                   tidx,     fullTy, fullVector};
  return VectorTileLoopNest(rewriter, loc, tileTy.getShape(),
                            {"thread_tile_extract", {}, {}})
      .emit(result, element);
}

/// 将 thread vector 的一个元素按 LowerInfo 写回完整 block vector。
struct InsertThreadTileElement {
  OpBuilder &rewriter;
  Location loc;
  LowerInfo &info;
  Value tidx;
  VectorType tileTy;
  VectorType fullTy;
  Value tileVector;
  Value emit(ArrayRef<Value> tileIvs, Value currentFull) {
    SmallVector<int64_t, 2> staticTilePosition(tileIvs.size(),
                                               ShapedType::kDynamic);
    auto scalar = rewriter.create<vector::ExtractOp>(
        loc, tileTy.getElementType(), tileVector, tileIvs,
        rewriter.getDenseI64ArrayAttr(staticTilePosition));

    auto fullIndices =
        buildMappedAccessIndices(rewriter, loc, info, tidx, tileIvs,
                                 static_cast<unsigned>(fullTy.getRank()));
    SmallVector<OpFoldResult, 2> fullPosition;
    fullPosition.reserve(fullIndices.size());
    for (Value index : fullIndices) {
      fullPosition.push_back(index);
    }
    return rewriter
        .create<vector::InsertOp>(loc, scalar.getResult(), currentFull,
                                  fullPosition)
        .getResult();
  }
};

static FailureOr<Value>
insertThreadTileIntoVector(Value tileVector, Value fullVector, LowerInfo &info,
                           ConversionPatternRewriter &rewriter, Location loc,
                           Operation *anchorOp) {
  auto tileTy = mlir::dyn_cast<VectorType>(tileVector.getType());
  auto fullTy = mlir::dyn_cast<VectorType>(fullVector.getType());
  if (!tileTy || !fullTy) {
    return failure();
  }
  if (tileTy == fullTy) {
    return tileVector;
  }
  if (fullTy.getElementType() != tileTy.getElementType() ||
      fullTy.getRank() != tileTy.getRank()) {
    return failure();
  }

  auto tidx = findThreadIdxOp(anchorOp);

  InsertThreadTileElement element{rewriter, loc,    info,      tidx,
                                  tileTy,   fullTy, tileVector};
  return VectorTileLoopNest(rewriter, loc, tileTy.getShape(),
                            {"thread_tile_insert", {}, {}})
      .emit(fullVector, element);
}

/// 仅广播源 shape 为 1 的维度，其他维度复用目标下标。
struct BroadcastTileElement {
  OpBuilder &rewriter;
  Location loc;
  VectorType sourceTy;
  Value sourceVector;
  Value emit(ArrayRef<Value> tileIvs, Value currentTile) {
    SmallVector<Value, 2> sourceIndices;
    sourceIndices.reserve(tileIvs.size());
    for (auto [idx, iv] : llvm::enumerate(tileIvs)) {
      sourceIndices.push_back(sourceTy.getDimSize(idx) == 1
                                  ? createIndexConstant(rewriter, loc, 0)
                                  : iv);
    }
    SmallVector<int64_t, 2> staticSourcePosition(sourceIndices.size(),
                                                 ShapedType::kDynamic);
    auto scalar = rewriter.create<vector::ExtractOp>(
        loc, sourceTy.getElementType(), sourceVector, sourceIndices,
        rewriter.getDenseI64ArrayAttr(staticSourcePosition));

    SmallVector<OpFoldResult, 2> tilePosition;
    tilePosition.reserve(tileIvs.size());
    for (Value iv : tileIvs) {
      tilePosition.push_back(iv);
    }
    return rewriter
        .create<vector::InsertOp>(loc, scalar.getResult(), currentTile,
                                  tilePosition)
        .getResult();
  }
};

static FailureOr<Value> broadcastLocalVectorToTile(Value sourceVector,
                                                   VectorType tileTy,
                                                   OpBuilder &rewriter,
                                                   Location loc) {
  auto sourceTy = mlir::dyn_cast<VectorType>(sourceVector.getType());
  if (!sourceTy || sourceTy == tileTy) {
    return sourceVector;
  }
  if (sourceTy.getElementType() != tileTy.getElementType() ||
      sourceTy.getRank() != tileTy.getRank()) {
    return failure();
  }

  bool needsBroadcast = false;
  for (int64_t dim = 0; dim < sourceTy.getRank(); ++dim) {
    int64_t sourceSize = sourceTy.getDimSize(dim);
    int64_t tileSize = tileTy.getDimSize(dim);
    if (sourceSize == tileSize) {
      continue;
    }
    if (sourceSize != 1) {
      return failure();
    }
    needsBroadcast = true;
  }
  if (!needsBroadcast) {
    return sourceVector;
  }

  Value result =
      rewriter
          .create<arith::ConstantOp>(
              loc, tileTy, mlir::cast<TypedAttr>(rewriter.getZeroAttr(tileTy)))
          .getResult();

  BroadcastTileElement element{rewriter, loc, sourceTy, sourceVector};
  return VectorTileLoopNest(rewriter, loc, tileTy.getShape())
      .emit(result, element);
}

static Operation *getDominanceRoot(Operation *op) {
  Operation *root = op;
  while (root->getParentOp()) {
    root = root->getParentOp();
  }
  return root;
}

/// 从替换表取值时检查定义支配使用点；不满足时交给调用方走其他物化路径。
static Value getVectorReplacement(Value value, Operation *user) {
  {
    auto it = s_buffer_replace.find(value);
    if (it != s_buffer_replace.end() &&
        mlir::isa<VectorType>(it->second.getType())) {
      if (user) {
        DominanceInfo dominance(getDominanceRoot(user));
        if (!dominance.properlyDominates(it->second, user)) {
          return {};
        }
      }
      return it->second;
    }
  }
  if (mlir::isa<VectorType>(value.getType())) {
    return value;
  }
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      castOp && castOp.getInputs().size() == 1 &&
      mlir::isa<VectorType>(castOp.getInputs()[0].getType())) {
    return castOp.getInputs()[0];
  }
  return {};
}

static Value getVectorValue(Value value) {
  if (mlir::isa<VectorType>(value.getType())) {
    return value;
  }
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      castOp && castOp.getInputs().size() == 1 &&
      mlir::isa<VectorType>(castOp.getInputs()[0].getType())) {
    return castOp.getInputs()[0];
  }
  return {};
}

static void eraseDeadUnrealizedConversionCasts(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> deadCasts;
    root->walk([&](UnrealizedConversionCastOp castOp) {
      if (llvm::all_of(castOp->getResults(),
                       [](Value result) { return result.use_empty(); })) {
        deadCasts.push_back(castOp.getOperation());
      }
    });
    for (Operation *op : deadCasts) {
      op->erase();
      changed = true;
    }
  }
}

static void eraseTriviallyDeadOps(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> deadOps;
    root->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op != root && !op->hasTrait<OpTrait::IsTerminator>() &&
          isOpTriviallyDead(op)) {
        deadOps.push_back(op);
      }
    });
    for (Operation *op : deadOps) {
      op->erase();
      changed = true;
    }
  }
}

static void foldThreadTileVectorLoads(Operation *root) {
  SmallVector<vector::LoadOp, 8> loadsToErase;
  root->walk([&](vector::LoadOp loadOp) {
    auto castOp = loadOp.getBase().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getInputs().size() != 1) {
      return;
    }
    Value vector = castOp.getInputs()[0];
    if (!mlir::isa<VectorType>(vector.getType()) ||
        vector.getType() != loadOp.getResult().getType()) {
      return;
    }
    loadOp.getResult().replaceAllUsesWith(vector);
    loadsToErase.push_back(loadOp);
  });

  for (auto loadOp : loadsToErase) {
    loadOp.erase();
  }
}

static Value findSingleExtractSource(affine::AffineForOp forOp,
                                     VectorType expectedSourceTy = {}) {
  Value source;
  bool mismatch = false;
  forOp.walk([&](vector::ExtractOp extractOp) {
    Value candidate = extractOp.getVector();
    if (expectedSourceTy && candidate.getType() != expectedSourceTy) {
      return;
    }
    if (!source) {
      source = candidate;
      return;
    }
    if (source != candidate) {
      mismatch = true;
    }
  });
  return mismatch ? Value{} : source;
}

static bool foldThreadTileExtractLoops(Operation *root) {
  SmallVector<affine::AffineForOp, 8> loops;
  root->walk<WalkOrder::PostOrder>([&](affine::AffineForOp forOp) {
    if (forOp->hasAttr("thread_tile_extract") && forOp.getNumResults() == 1) {
      loops.push_back(forOp);
    }
  });

  bool changed = false;
  IRRewriter rewriter(root->getContext());
  for (affine::AffineForOp forOp : loops) {
    auto resultTy = mlir::dyn_cast<VectorType>(forOp.getResult(0).getType());
    if (!resultTy) {
      continue;
    }
    Value source = findSingleExtractSource(forOp);
    if (!source) {
      continue;
    }
    bool isConvertedTile =
        llvm::any_of(s_buffer_replace,
                     [&](const auto &entry) { return entry.second == source; });
    if (!isConvertedTile)
      continue;
    DominanceInfo dominance(getDominanceRoot(forOp));
    if (!dominance.properlyDominates(source, forOp))
      continue;
    rewriter.setInsertionPoint(forOp);
    auto tile =
        broadcastLocalVectorToTile(source, resultTy, rewriter, forOp.getLoc());
    if (failed(tile))
      continue;
    rewriter.replaceOp(forOp, *tile);
    changed = true;
  }
  return changed;
}

static void promoteSingleIterationAffineFors(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<affine::AffineForOp, 16> loops;
    root->walk<WalkOrder::PostOrder>(
        [&](affine::AffineForOp forOp) { loops.push_back(forOp); });
    for (affine::AffineForOp forOp : loops) {
      if (succeeded(affine::promoteIfSingleIteration(forOp))) {
        changed = true;
      }
    }
  }
}

static Attribute convertScalarAttrForElementType(TypedAttr valueAttr,
                                                 Type elementType,
                                                 Builder &builder) {
  if (valueAttr.getType() == elementType) {
    return valueAttr;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    if (mlir::isa<FloatType>(elementType)) {
      return builder.getFloatAttr(elementType, floatAttr.getValueAsDouble());
    }
  }
  if (auto integerAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    if (mlir::isa<IntegerType, IndexType>(elementType)) {
      return builder.getIntegerAttr(elementType, integerAttr.getValue());
    }
  }
  return {};
}

static std::optional<LowerInfo> findThreadCarrierLowerInfo(Value value,
                                                           Operation *anchor) {
  if (s_info == nullptr) {
    return std::nullopt;
  }

  SmallVector<Value, 8> worklist{value};
  SmallVector<Value, 8> visited;

  auto enqueue = [&](Value candidate) {
    if (candidate && !llvm::is_contained(visited, candidate) &&
        !llvm::is_contained(worklist, candidate)) {
      worklist.push_back(candidate);
    }
  };

  while (!worklist.empty()) {
    Value candidate = worklist.pop_back_val();
    visited.push_back(candidate);

    if (auto *info = findLowerInfoForValue(candidate, anchor)) {
      return *info;
    }
    if (anchor) {
      if (auto *info = s_info->getNearestInferedInfo(candidate, anchor, true)) {
        return *info;
      }
      if (auto *info =
              s_info->getNearestInferedInfo(candidate, anchor, false)) {
        return *info;
      }
    }

    if (auto castOp = candidate.getDefiningOp<UnrealizedConversionCastOp>();
        castOp && castOp.getInputs().size() == 1) {
      enqueue(castOp.getInputs()[0]);
    }

    if (auto blockArg = mlir::dyn_cast<BlockArgument>(candidate);
        blockArg && blockArg.getArgNumber() != 0) {
      if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(
              blockArg.getOwner()->getParentOp())) {
        unsigned initIndex = blockArg.getArgNumber() - 1;
        if (initIndex < forOp.getInits().size()) {
          enqueue(forOp.getInits()[initIndex]);
        }
      }
    }

    if (auto result = mlir::dyn_cast<OpResult>(candidate)) {
      if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(result.getOwner());
          forOp && result.getResultNumber() < forOp.getNumRegionIterArgs()) {
        enqueue(forOp.getRegionIterArgs()[result.getResultNumber()]);
        enqueue(forOp.getInits()[result.getResultNumber()]);
      }
    }

    for (Operation *user : candidate.getUsers()) {
      for (Value result : user->getResults()) {
        enqueue(result);
      }
      if (auto copyOp = dyn_cast<frisk::CopyOp>(user)) {
        enqueue(copyOp.getSrcMemRef());
        enqueue(copyOp.getDstMemRef());
      }
    }
  }

  return std::nullopt;
}

static bool fitsWithinCarrier(VectorType tileTy, VectorType carrierTy) {
  if (tileTy.getRank() != carrierTy.getRank()) {
    return false;
  }
  for (int64_t i = 0; i < tileTy.getRank(); ++i) {
    int64_t tileDim = tileTy.getDimSize(i);
    int64_t carrierDim = carrierTy.getDimSize(i);
    if (tileDim <= 0) {
      return false;
    }
    if (carrierDim != ShapedType::kDynamic && tileDim > carrierDim) {
      return false;
    }
  }
  return true;
}

// Normalize after producers/copies have been lowered, before greedy folding
// destroys the explicit block-value -> thread-value correspondence.
/// 按 yield 的 thread 类型重建 affine.for 的 init/iter_arg/result/yield。
/// 处理内层循环后再处理外层循环，保留每个 carrier 原来的位置与初值。
static bool normalizeLoopCarriedBlockVectors(func::FuncOp kernel) {
  SmallVector<affine::AffineForOp, 8> loops;
  kernel.walk<WalkOrder::PreOrder>([&](affine::AffineForOp loop) {
    if (loop.getNumRegionIterArgs() && !loop->hasAttr("thread_tile_extract") &&
        !loop->hasAttr("thread_tile_insert"))
      loops.push_back(loop);
  });
  IRRewriter rewriter(kernel.getContext());
  bool changed = false;
  for (auto loop : llvm::reverse(loops)) {
    auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
    SmallVector<Value> tiles;
    SmallVector<std::optional<LowerInfo>, 2> layouts;
    bool needsRewrite = false;
    bool supported = true;
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
      Value yielded = yield.getOperand(i);
      Value tile = getVectorReplacement(yielded, yield);
      if (!tile)
        tile = yielded;
      auto oldTy = dyn_cast<VectorType>(arg.getType());
      auto tileTy = dyn_cast<VectorType>(tile.getType());
      if (tile.getType() != arg.getType() &&
          (!oldTy || !tileTy || !fitsWithinCarrier(tileTy, oldTy))) {
        supported = false;
        break;
      }
      tiles.push_back(tile);
      layouts.push_back(findThreadCarrierLowerInfo(arg, loop));
      needsRewrite |= tile.getType() != arg.getType();
    }
    if (!supported || !needsRewrite)
      continue;

    SmallVector<Value> inits;
    rewriter.setInsertionPoint(loop);
    for (auto [i, init] : llvm::enumerate(loop.getInits())) {
      Type tileTy = tiles[i].getType();
      Value converted = getVectorReplacement(init, loop);
      if (converted && converted.getType() == tileTy) {
        inits.push_back(converted);
      } else if (init.getType() == tileTy) {
        inits.push_back(init);
      } else if (auto constant = init.getDefiningOp<arith::ConstantOp>();
                 constant && isa<DenseElementsAttr>(constant.getValue()) &&
                 cast<DenseElementsAttr>(constant.getValue()).isSplat()) {
        auto splat = cast<DenseElementsAttr>(constant.getValue());
        auto attr = DenseElementsAttr::get(cast<VectorType>(tileTy),
                                           splat.getSplatValue<Attribute>());
        inits.push_back(
            rewriter.create<arith::ConstantOp>(loop.getLoc(), tileTy, attr));
      } else if (layouts[i]) {
        auto tile =
            extractThreadTile(init, cast<VectorType>(tileTy), *layouts[i],
                              rewriter, loop.getLoc(), loop);
        if (failed(tile)) {
          supported = false;
          break;
        }
        inits.push_back(*tile);
      } else {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;

    rewriter.setInsertionPoint(loop);
    auto newLoop = rewriter.create<affine::AffineForOp>(
        loop.getLoc(), loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
        loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
        loop.getStepAsInt(), inits);
    for (NamedAttribute attr : loop->getAttrs()) {
      StringRef name = attr.getName().getValue();
      if (name != "lowerBoundMap" && name != "upperBoundMap" &&
          name != "operandSegmentSizes" && name != "step")
        newLoop->setAttr(attr.getName(), attr.getValue());
    }
    newLoop->setAttr("thread_yield_normalized", rewriter.getBoolAttr(true));
    for (auto [i, oldArg] : llvm::enumerate(loop.getRegionIterArgs())) {
      s_buffer_replace[oldArg] = newLoop.getRegionIterArgs()[i];
      s_buffer_replace[loop.getResult(i)] = newLoop.getResult(i);
      if (layouts[i]) {
        registerConvertedValueLowerInfo(newLoop.getRegionIterArgs()[i],
                                        *layouts[i], newLoop);
        registerConvertedValueLowerInfo(newLoop.getResult(i), *layouts[i],
                                        newLoop);
      }
    }
    // Move the body instead of cloning it: producer SSA identities and their
    // entries in s_buffer_replace must survive the loop signature change.
    yield->setOperands(tiles);
    Block *body = newLoop.getBody();
    if (!body->empty())
      rewriter.eraseOp(body->getTerminator());
    rewriter.mergeBlocks(loop.getBody(), body, body->getArguments());
    rewriter.replaceOp(loop, newLoop.getResults());
    changed = true;
  }
  return changed;
}

static FailureOr<Value>
castFloatVectorElementType(Value vector, Type dstElementType,
                           ConversionPatternRewriter &rewriter, Location loc) {
  auto srcVecTy = mlir::dyn_cast<VectorType>(vector.getType());
  if (!srcVecTy) {
    return failure();
  }
  Type srcElementType = srcVecTy.getElementType();
  if (srcElementType == dstElementType) {
    return vector;
  }
  auto srcFloatTy = mlir::dyn_cast<FloatType>(srcElementType);
  auto dstFloatTy = mlir::dyn_cast<FloatType>(dstElementType);
  if (!srcFloatTy || !dstFloatTy) {
    return failure();
  }

  auto dstVecTy = VectorType::get(srcVecTy.getShape(), dstElementType);
  if (srcFloatTy.getWidth() < dstFloatTy.getWidth()) {
    return rewriter.create<arith::ExtFOp>(loc, dstVecTy, vector).getResult();
  }
  return rewriter.create<arith::TruncFOp>(loc, dstVecTy, vector).getResult();
}

// Shared 输入保留原策略：二元算子按 LowerInfo 逐元素取 tile；exp2 先生成
// copy_to_reg，待后续阶段展开成基址加连续偏移的 load。
enum class SharedOperandRead { MappedThreadTile, ContiguousCopyToReg };

/// 将逐元素算子的输入统一为结果 tile 类型。
/// 顺序不可随意调整：已有替换值 -> vector 广播/提取 -> 标量广播 -> shared
/// 读取 -> ZeroOp 常量 -> local vector.load。广播失败才尝试按布局提取。
static FailureOr<Value> materializeElementwiseOperand(
    Operation *op, Value original, Value adapted, LowerInfo &info,
    VectorType resultVecTy, SharedOperandRead sharedRead, Value zero,
    StringRef operationName, ConversionPatternRewriter &rewriter) {
  if (Value vector = getVectorReplacement(original, op))
    adapted = vector;
  else if (Value vector = getVectorReplacement(adapted, op))
    adapted = vector;

  // getVectorValue 同时处理直接 vector 和单输入 cast 后的 vector。
  if (Value vector = getVectorValue(adapted)) {
    if (vector.getType() == resultVecTy)
      return vector;
    auto broadcast =
        broadcastLocalVectorToTile(vector, resultVecTy, rewriter, op->getLoc());
    if (succeeded(broadcast))
      return *broadcast;
    auto tile = extractThreadTile(vector, resultVecTy, info, rewriter,
                                  op->getLoc(), op);
    if (succeeded(tile))
      return *tile;
    return rewriter.notifyMatchFailure(
        op, Twine(operationName) +
                " vector-backed operand type does not match result tile type");
  }
  if (isa<FloatType>(adapted.getType()))
    return rewriter
        .create<vector::BroadcastOp>(op->getLoc(), resultVecTy, adapted)
        .getResult();

  auto memTy = dyn_cast<MemRefType>(adapted.getType());
  if (!memTy)
    return rewriter.notifyMatchFailure(
        op, Twine(operationName) +
                " vector lowering expects vector, scalar, or memref operands");

  if (isSharedMemref(adapted)) {
    if (sharedRead == SharedOperandRead::MappedThreadTile)
      return extractThreadTile(adapted, resultVecTy, info, rewriter,
                               op->getLoc(), op);
    auto operands = buildLowerInfoMapOperands(
        rewriter, op->getLoc(), info, findThreadIdxOp(op), zero, zero, zero,
        zero, zero, zero, zero, zero);
    return rewriter
        .create<frisk::CopyToRegOp>(op->getLoc(), resultVecTy, adapted,
                                    operands, info.getAffineMap())
        .getResult();
  }
  if (auto zeroOp = original.getDefiningOp<frisk::ZeroOp>())
    return rewriter
        .create<arith::ConstantOp>(
            zeroOp->getLoc(), resultVecTy,
            cast<TypedAttr>(rewriter.getZeroAttr(resultVecTy)))
        .getResult();

  auto expectedTy = getConvertedBufferType(adapted, info);
  if (!expectedTy || expectedTy != resultVecTy)
    return rewriter.notifyMatchFailure(
        op, Twine(operationName) +
                " memref operand is not materialized as the result tile type");
  SmallVector<Value> indices(memTy.getRank(),
                             createIndexConstant(rewriter, op->getLoc(), 0));
  return rewriter
      .create<vector::LoadOp>(op->getLoc(), resultVecTy, adapted, indices)
      .getResult();
}

/// 匹配：Frisk Add/Sub/Mul/Div，结果必须有
/// LowerInfo；缺失的输入布局沿用结果布局。 改写：统一两侧输入为 thread
/// vector，再创建对应的浮点 arith 算子。
template <typename FromOpTy, typename ToOpTy>
class FriskBinaryOpConversion : public OpConversionPattern<FromOpTy> {
public:
  using Base = OpConversionPattern<FromOpTy>;
  using OpAdaptor = typename FromOpTy::Adaptor;
  using Base::Base;

  LogicalResult
  matchAndRewrite(FromOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get lowerInfo

    auto infoC = copyLowerInfoForValue(op.getResult(), op.getOperation());
    if (!infoC) {
      return rewriter.notifyMatchFailure(
          op, "LowerInfo not found for binary result");
    }
    auto infoA =
        copyLowerInfoForValue(op.getLhs(), op.getOperation()).value_or(*infoC);
    auto infoB =
        copyLowerInfoForValue(op.getRhs(), op.getOperation()).value_or(*infoC);

    // GEMM input LowerInfo describes one MMA's register budget. An elementwise
    // producer outside GEMM's loops must compute every M/N/K fragment instead.
    if (isSharedMemref(op.getResult())) {
      for (unsigned i = 0; i < 2; ++i) {
        infoC->thread_own_data_size[i] =
            infoC->ignoreDim == static_cast<int>(i) ? 1 :
            infoC->get_thread_widths()[i] * infoC->get_warp_repeat()[i] *
            infoC->warpInstUnroll[i] * infoC->get_block_repeat()[i];
      }
    }
    auto resultVecTy = getVecTypeByLowerInfoWithThreadOwnData(*infoC);
    auto zero = createIndexConstant(rewriter, op->getLoc(), 0);
    auto lhs = materializeElementwiseOperand(
        op, op.getLhs(), adaptor.getLhs(), infoA, resultVecTy,
        SharedOperandRead::MappedThreadTile, zero, "binary", rewriter);
    auto rhs = materializeElementwiseOperand(
        op, op.getRhs(), adaptor.getRhs(), infoB, resultVecTy,
        SharedOperandRead::MappedThreadTile, zero, "binary", rewriter);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }

    auto newOp = rewriter.create<ToOpTy>(op->getLoc(), *lhs, *rhs);
    registerConvertedValueLowerInfo(newOp.getResult(), *infoC,
                                    op.getOperation());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// 匹配：结果有 LowerInfo 的 Exp2。输入物化后生成逐元素 math.exp2。
/// shared 读取保留 copy_to_reg 策略，与二元算子的逐元素 mapped load 区分。
class FriskExp2OpConversion : public OpConversionPattern<frisk::Exp2Op> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(frisk::Exp2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto infoResult = copyLowerInfoForValue(op.getResult(), op.getOperation());
    if (!infoResult) {
      return rewriter.notifyMatchFailure(op,
                                         "LowerInfo not found for exp2 result");
    }
    auto infoOperand = copyLowerInfoForValue(op.getOperand(), op.getOperation())
                           .value_or(*infoResult);
    auto resultVecTy = getVecTypeByLowerInfoWithThreadOwnData(*infoResult);

    auto zero = createIndexConstant(rewriter, op->getLoc(), 0);
    auto operand = materializeElementwiseOperand(
        op, op.getOperand(), adaptor.getOperand(), infoOperand, resultVecTy,
        SharedOperandRead::ContiguousCopyToReg, zero, "exp2", rewriter);
    if (failed(operand)) {
      return failure();
    }

    auto newOp = rewriter.create<math::Exp2Op>(op->getLoc(), *operand);
    registerConvertedValueLowerInfo(newOp.getResult(), *infoResult,
                                    op.getOperation());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Reduce 的后备查找保留 nearest-inference 优先级，不能直接换成通用查询。
static std::optional<LowerInfo> findNearestReduceLowerInfo(Value value,
                                                           Operation *op) {
  if (auto *info = s_info->getLowerInfo(value, op)) {
    return *info;
  }
  if (auto *info = s_info->getNearestInferedInfo(value, op, true)) {
    return *info;
  }
  if (auto *info = s_info->getNearestInferedInfo(value, op, false)) {
    return *info;
  }
  for (auto user : value.getUsers()) {
    if (auto *info = s_info->getLowerInfo(value, user)) {
      return *info;
    }
  }
  for (auto &entry : *s_info) {
    if (entry.second.buffer == value)
      return entry.second;
  }
  return std::nullopt;
}

static FailureOr<LowerInfo> findReduceLowerInfo(Value buffer, Operation *op) {
  if (auto *info = findLowerInfoForValue(buffer, op)) {
    return *info;
  }

  if (auto info = findNearestReduceLowerInfo(buffer, op)) {
    return *info;
  }
  if (auto castOp = buffer.getDefiningOp<UnrealizedConversionCastOp>();
      castOp && castOp.getInputs().size() == 1) {
    Value source = castOp.getInputs()[0];
    if (auto info = findNearestReduceLowerInfo(source, op)) {
      return *info;
    }
    for (auto &entry : s_buffer_replace) {
      if (entry.second == source) {
        if (auto info = findNearestReduceLowerInfo(entry.first, op)) {
          return *info;
        }
      }
    }
  }
  for (auto &entry : s_buffer_replace) {
    if (entry.second == buffer) {
      if (auto info = findNearestReduceLowerInfo(entry.first, op)) {
        return *info;
      }
    }
  }
  return failure();
}

/// 读源只能复用已有 tile；写目标才允许分配，避免创建未初始化的读源。
static FailureOr<Value> getReduceLocalBuffer(LowerInfo &info,
                                             MemRefType originalTy,
                                             ArrayRef<int64_t> shape,
                                             bool forRead, Operation *op,
                                             OpBuilder &rewriter) {
  auto it = s_buffer_replace.find(info.buffer);
  if (it != s_buffer_replace.end()) {
    return it->second;
  }
  if (forRead) {
    return failure();
  }
  auto newTy = MemRefType::get(shape, originalTy.getElementType(), AffineMap{},
                               originalTy.getMemorySpace());
  auto newBuffer = rewriter.create<memref::AllocaOp>(op->getLoc(), newTy);
  s_buffer_replace[info.buffer] = newBuffer.getResult();
  return newBuffer.getResult();
}

static FailureOr<Value> createReduceIdentity(frisk::ReduceOp op, Type elemTy,
                                             OpBuilder &rewriter) {
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
}

static FailureOr<Value> combineReduceValues(frisk::ReduceOp op, Value lhs,
                                            Value rhs, OpBuilder &rewriter) {
  auto kind = op.getKind();
  if (kind == "add") {
    return rewriter.create<arith::AddFOp>(op->getLoc(), lhs, rhs).getResult();
  }
  if (kind == "mul") {
    return rewriter.create<arith::MulFOp>(op->getLoc(), lhs, rhs).getResult();
  }
  if (kind == "min") {
    return rewriter.create<arith::MinNumFOp>(op->getLoc(), lhs, rhs)
        .getResult();
  }
  if (kind == "max") {
    return rewriter.create<arith::MaxNumFOp>(op->getLoc(), lhs, rhs)
        .getResult();
  }
  return failure();
}

/// 匹配：同浮点元素类型、静态正归约长度，LowerInfo 只支持前两个维度。
/// 改写：线程内串行归约 -> warp 内 XOR shuffle -> 归约组 leader 写回。
/// 仅支持 add/mul/min/max；归约轴不能跨 warp，lane 数必须为 2 的幂。
class ReduceOpConversion : public OpConversionPattern<frisk::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(frisk::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tidx = findThreadIdxOp(op);

    // 获取src dst的LowerInfo
    auto srcInfoOr = findReduceLowerInfo(adaptor.getSrc(), op);
    auto dstInfoOr = findReduceLowerInfo(adaptor.getDst(), op);
    if (failed(srcInfoOr)) {
      srcInfoOr = findReduceLowerInfo(op.getSrc(), op);
    }
    if (failed(dstInfoOr)) {
      dstInfoOr = findReduceLowerInfo(op.getDst(), op);
    }
    if (failed(srcInfoOr) || failed(dstInfoOr)) {
      return op.emitOpError("LowerInfo not found for reduce operands");
    }
    auto srcInfo = *srcInfoOr;
    auto dstInfo = *dstInfoOr;
    srcInfo.buffer = adaptor.getSrc();
    dstInfo.buffer = adaptor.getDst();

    // 获取src 的 memrefType
    auto srcTy = mlir::cast<MemRefType>(adaptor.getSrc().getType());
    auto dstTy = mlir::cast<MemRefType>(adaptor.getDst().getType());
    auto elemTy = srcTy.getElementType();
    if (elemTy != dstTy.getElementType()) {
      return op.emitOpError("source and destination element types must match");
    }
    if (!mlir::isa<FloatType>(elemTy)) {
      return op.emitOpError(
          "thread-level reduce currently supports floating-point memrefs");
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
      return op.emitOpError(
          "thread-level reduce requires a positive static reduce extent");
    }

    // 单个线程持有的数据量

    std::array<int64_t, 2> dstThreadShape2D =
        dstInfo.get_thread_own_data_size();
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

    Value srcBuffer = adaptor.getSrc();
    Value dstBuffer = adaptor.getDst();
    Value srcVector = getVectorValue(adaptor.getSrc());
    VectorType srcVectorTy;
    if (srcVector) {
      srcVectorTy = mlir::dyn_cast<VectorType>(srcVector.getType());
      if (!srcVectorTy || srcVectorTy.getRank() != srcTy.getRank() ||
          srcVectorTy.getElementType() != elemTy) {
        return op.emitOpError(
            "vector-backed reduce source does not match source memref type");
      }
    }
    bool srcIsLocal = isLocalMemref(srcInfo.buffer);
    bool dstIsLocal = isLocalMemref(dstInfo.buffer);
    if (srcIsLocal) {
      auto replacement =
          getReduceLocalBuffer(srcInfo, srcTy, {}, true, op, rewriter);
      if (succeeded(replacement)) {
        srcBuffer = *replacement;
        srcTy = mlir::cast<MemRefType>(srcBuffer.getType());
        reduceExtent = srcTy.getDimSize(reduceDim);
      }
    }
    if (dstIsLocal) {
      auto replacement = getReduceLocalBuffer(dstInfo, dstTy, dstLoopShape,
                                              false, op, rewriter);
      if (failed(replacement)) {
        return failure();
      }
      dstBuffer = *replacement;
      dstTy = mlir::cast<MemRefType>(dstBuffer.getType());
    }

    // 1. 遍历目标 thread tile，每个输出元素使用独立的线程内累加器。
    std::vector<int> loopUbs;
    for (int64_t ub : dstLoopShape) {
      loopUbs.push_back(static_cast<int>(ub));
    }
    std::vector<Value> dstTileIvs;
    createNestedAffineFor(rewriter, op->getLoc(), loopUbs, dstTileIvs);

    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto accTy = MemRefType::get({1}, elemTy);
    auto acc = rewriter.create<memref::AllocaOp>(op->getLoc(), accTy);
    auto identity = createReduceIdentity(op, elemTy, rewriter);
    if (failed(identity)) {
      return op.emitOpError("unsupported reduce kind");
    }
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), *identity, acc,
                                           ValueRange{zeroIdx});

    auto dstIndices =
        dstIsLocal
            ? makeLocalIndices(dstTileIvs, dstTy.getRank())
            : buildMappedAccessIndices(rewriter, op->getLoc(), dstInfo, tidx,
                                       dstTileIvs, dstTy.getRank());

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
    auto redLoops = createNestedAffineFor(
        rewriter, op->getLoc(), {static_cast<int>(localReduceExtent)}, redIvs);

    SmallVector<Value, 2> srcTileIvs;
    if (srcTy.getRank() == dstTy.getRank()) {
      for (unsigned i = 0; i < srcTy.getRank(); ++i) {
        srcTileIvs.push_back(
            i == static_cast<unsigned>(reduceDim) ? redIvs[0] : dstTileIvs[i]);
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

    SmallVector<Value, 2> srcIndices;
    if (srcIsLocal && srcBuffer != op.getSrc()) {
      srcIndices.append(srcTileIvs.begin(), srcTileIvs.end());
    } else {
      srcIndices = buildMappedAccessIndices(rewriter, op->getLoc(), srcInfo,
                                            tidx, srcTileIvs, srcTy.getRank());
    }

    auto current = rewriter.create<affine::AffineLoadOp>(op->getLoc(), acc,
                                                         ValueRange{zeroIdx});
    Value srcValue;
    if (srcVector) {
      SmallVector<int64_t, 2> staticSrcPosition(srcTileIvs.size(),
                                                ShapedType::kDynamic);
      srcValue = rewriter
                     .create<vector::ExtractOp>(
                         op->getLoc(), elemTy, srcVector, srcTileIvs,
                         rewriter.getDenseI64ArrayAttr(staticSrcPosition))
                     .getResult();
    } else {
      srcValue =
          rewriter
              .create<affine::AffineLoadOp>(op->getLoc(), srcBuffer, srcIndices)
              .getResult();
    }
    auto next =
        combineReduceValues(op, current.getResult(), srcValue, rewriter);
    if (failed(next)) {
      return op.emitOpError("unsupported reduce kind");
    }
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), *next, acc,
                                           ValueRange{zeroIdx});

    rewriter.setInsertionPointAfter(redLoops.back());
    auto localResult = rewriter.create<affine::AffineLoadOp>(
        op->getLoc(), acc, ValueRange{zeroIdx});

    // 2. XOR shuffle 只合并同一 warp 内、归约轴方向上的 lane。
    auto warpLayout = srcInfo.get_warp_layout();
    auto warpLayoutOrder = srcInfo.base_layout.warp_layout_order;
    auto blockLayout = srcInfo.get_block_layout();
    if (blockLayout[reduceDim] != 1) {
      return op.emitOpError("shuffle reduce only supports reduce dimension "
                            "within a single warp layout");
    }
    int64_t reduceLaneExtent = warpLayout[reduceDim];
    if (reduceLaneExtent <= 0 || !llvm::isPowerOf2_64(reduceLaneExtent)) {
      return op.emitOpError("shuffle reduce requires power-of-two lane extent "
                            "along the reduce dimension");
    }
    int64_t reduceLaneStride = 1;
    if (warpLayoutOrder[0] == reduceDim) {
      reduceLaneStride = 1;
    } else if (warpLayoutOrder[1] == reduceDim) {
      reduceLaneStride = warpLayout[warpLayoutOrder[0]];
    } else {
      return op.emitOpError(
          "reduce dimension is not present in warp layout order");
    }

    Value reduced = localResult.getResult();
    for (int64_t laneOffset = 1; laneOffset < reduceLaneExtent;
         laneOffset <<= 1) {
      auto shuffled = rewriter.create<gpu::ShuffleOp>(
          op->getLoc(), reduced,
          static_cast<int32_t>(laneOffset * reduceLaneStride),
          static_cast<int32_t>(srcInfo.warp_threads), gpu::ShuffleMode::XOR);
      auto combined = combineReduceValues(
          op, reduced, shuffled.getShuffleResult(), rewriter);
      if (failed(combined)) {
        return op.emitOpError("unsupported reduce kind");
      }
      reduced = *combined;
    }

    // 3. 只有归约轴坐标为 0 的 lane 写回，避免同一输出位置被重复写入。
    Value laneId = modBy(rewriter, op->getLoc(), tidx, srcInfo.warp_threads);
    Value reduceLaneCoord =
        modBy(rewriter, op->getLoc(),
              floorDivBy(rewriter, op->getLoc(), laneId, reduceLaneStride),
              reduceLaneExtent);
    auto isReduceLeader = rewriter.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::eq, reduceLaneCoord, zeroIdx);
    auto ifOp = rewriter.create<scf::IfOp>(op->getLoc(), isReduceLeader, false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), reduced, dstBuffer,
                                           dstIndices);
    rewriter.setInsertionPointAfter(ifOp);
    Value oldSrc = op.getSrc();
    rewriter.eraseOp(op);
    if (auto castOp = oldSrc.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (oldSrc.use_empty()) {
        rewriter.eraseOp(castOp);
      }
    }
    return success();
  }
};

/// 将 frisk.block 的标量 region 搬入线程循环。
/// 各 buffer 分别使用自己的 LowerInfo：local 访问线程 tile，shared 访问 block
/// 坐标。 状态只在一次 Pattern 调用内有效；原有
/// blockRepeat/线程循环顺序保持不变。
class BlockLowering {
public:
  BlockLowering(frisk::BlockOp op, ConversionPatternRewriter &rewriter)
      : op(op), rewriter(rewriter) {}
  LogicalResult run() {
    collectLoadStoreOps();
    analyzeAccessedBuffers();
    materializeLocalBuffers();
    createThreadLoops();
    cloneBlockBody();
    replaceOldLocalBuffers();
    rewriter.eraseOp(op);
    return success();
  }

private:
  struct AccessedBufferInfo {
    Value buffer;
    std::optional<LowerInfo> lowerInfo;
  };
  frisk::BlockOp op;
  ConversionPatternRewriter &rewriter;
  gpu::ThreadIdOp tidx;
  std::vector<affine::AffineLoadOp> loadOps;
  std::vector<affine::AffineStoreOp> storeOps;
  std::array<int64_t, 2> threadLevelSize = {0, 0};
  std::array<int64_t, 2> blockRepeats = {0, 0};
  llvm::DenseMap<AllocBufferOp, std::array<int64_t, 2>> allocLocalsToReplace;
  std::vector<AccessedBufferInfo> bufferInfos;
  IRMapping mapper;
  IRMapping localMapper;
  std::vector<Value> threadOwnDataIvs;
  std::vector<Value> blockRepeatIvs;
  static std::array<int64_t, 2> get2DShape(Value buffer) {
    auto ty = cast<MemRefType>(buffer.getType());
    auto shape = ty.getShape();
    if (shape.size() == 1) {
      return {shape[0], 1};
    }
    return {shape[0], shape[1]};
  }

  void collectMaxThreadlevelSz(std::array<int64_t, 2> sz) {
    threadLevelSize[0] = std::max(threadLevelSize[0], sz[0]);
    threadLevelSize[1] = std::max(threadLevelSize[1], sz[1]);
  }

  static bool isThreadLocalTile(Value buffer) {
    if (auto alloc = buffer.getDefiningOp<AllocBufferOp>()) {
      return alloc.getMemorySpace() == int(friskMs::Local);
    }
    auto ty = cast<MemRefType>(buffer.getType());
    auto memorySpace = ty.getMemorySpaceAsInt();
    return memorySpace == int(friskMs::Local) || memorySpace == 5;
  }

  AccessedBufferInfo *findBufferInfo(Value buffer) {
    for (auto &info : bufferInfos) {
      if (info.buffer == buffer) {
        return &info;
      }
    }
    return nullptr;
  }

  void recordBufferInfo(Value buffer) {
    if (!isa<MemRefType>(buffer.getType()) ||
        findBufferInfo(buffer) != nullptr) {
      return;
    }

    AccessedBufferInfo recordedInfo{buffer, std::nullopt};
    if (auto *lowerInfo = s_info->getLowerInfo(buffer, op.getOperation())) {
      recordedInfo.lowerInfo = *lowerInfo;
      collectMaxThreadlevelSz(lowerInfo->get_thread_own_data_size());

    } else {
      // 前面的 block conversion 可能已经把原 frisk.alloc_buffer 替换为
      // memref.alloca 形式的 thread tile。这个新 value 已经是 lowered 后
      // 的形态，不会出现在 LowerInfoMap 中，因此直接用当前 memref shape
      // 作为循环 shape。
      collectMaxThreadlevelSz(get2DShape(buffer));
    }
    bufferInfos.push_back(recordedInfo);
  }

  AccessedBufferInfo *getBufferInfo(Value buffer) {
    if (auto mapped = mapper.lookupOrNull(buffer)) {
      for (auto &info : bufferInfos) {
        if (info.buffer == mapped) {
          return &info;
        }
      }
    }
    return findBufferInfo(buffer);
  }

  Value getLinearIvForDim(unsigned dim) {
    if (dim < threadOwnDataIvs.size()) {
      return threadOwnDataIvs[dim];
    }
    return createIndexConstant(rewriter, op->getLoc(), 0);
  }

  Value getRepeatIvForDim(unsigned dim) {
    if (dim < blockRepeatIvs.size()) {
      return blockRepeatIvs[dim];
    }
    return createIndexConstant(rewriter, op->getLoc(), 0);
  }

  SmallVector<Value, 2> buildAccessCoords(Value originalBuffer, unsigned rank) {
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
        coords.push_back(modBy(rewriter, op->getLoc(), linearIv, ownDataSize));
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
      Value br = modBy(rewriter, op->getLoc(), repeatIv, ownBlockRepeat);
      Value linearInTile = modBy(rewriter, op->getLoc(), linearIv, unrollWidth);

      brs.push_back(br);
      ius.push_back(
          modBy(rewriter, op->getLoc(),
                floorDivBy(rewriter, op->getLoc(), linearInTile, repeatWidth),
                instUnroll));
      Value withinInst =
          modBy(rewriter, op->getLoc(), linearInTile, repeatWidth);
      wrs.push_back(
          floorDivBy(rewriter, op->getLoc(), withinInst, threadWidth));
      regs.push_back(modBy(rewriter, op->getLoc(), withinInst, threadWidth));
    }

    auto mapOperands = buildLowerInfoMapOperands(
        rewriter, op->getLoc(), info, tidx, brs[0], brs[1], ius[0], ius[1],
        wrs[0], wrs[1], regs[0], regs[1]);
    return applyLowerInfoMap(rewriter, op->getLoc(), info, mapOperands, rank);
  }

  SmallVector<Value, 4> remapAffineOperands(ValueRange oldOperands,
                                            ArrayRef<Value> accessCoords) {
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(oldOperands.size());
    for (Value operand : oldOperands) {
      newOperands.push_back(remapValueForAccess(operand, accessCoords));
    }
    return newOperands;
  }

  Value remapValueForAccess(Value operand, ArrayRef<Value> accessCoords) {
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
  }

  void collectLoadStoreOps() {
    tidx = findThreadIdxOp(op);
    // 寻找内部所有 load store ops
    op->walk([&](Operation *childOp) {
      if (mlir::isa<affine::AffineLoadOp>(childOp)) {
        loadOps.push_back(mlir::cast<affine::AffineLoadOp>(childOp));
      }
      if (mlir::isa<affine::AffineStoreOp>(childOp)) {
        storeOps.push_back(mlir::cast<affine::AffineStoreOp>(childOp));
      }
    });
    assert(!storeOps.empty());

    // thread层面，每个buffer 当前tid持有的元素不一定一样。
  }

  void analyzeAccessedBuffers() {
    // 对每个affine.load 检查其memref，记录buffer Info。
    // 追踪来源 srcDefOp， 标记 allocLocalsToReplace[srcDefOp] = lowerInfo
    // 的thread总计算量
    for (auto loadOp : loadOps) {
      auto srcValue = loadOp.getMemref();
      recordBufferInfo(srcValue);
      auto srcDefOp = srcValue.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(srcValue);
      if (srcDefOp != nullptr && info != nullptr && info->lowerInfo &&
          isThreadLocalTile(srcValue) &&
          !allocLocalsToReplace.count(srcDefOp)) {
        allocLocalsToReplace[srcDefOp] =
            info->lowerInfo->get_thread_own_data_size();
      }
    }

    for (auto storeOp : storeOps) {
      auto dstVal = storeOp.getMemref();
      recordBufferInfo(dstVal);
      auto dstDefOp = dstVal.getDefiningOp<AllocBufferOp>();
      auto *info = findBufferInfo(dstVal);
      if (dstDefOp != nullptr && info != nullptr && info->lowerInfo &&
          isThreadLocalTile(dstVal) && !allocLocalsToReplace.count(dstDefOp)) {
        allocLocalsToReplace[dstDefOp] =
            info->lowerInfo->get_thread_own_data_size();
      }
    }

    // 统计该blockOp应该以多少 blockRepeat 为准。 每个子op的 blockRepeat可能不同
    for (auto e : bufferInfos) {
      if (e.lowerInfo) {
        auto br = e.lowerInfo->get_block_repeat();
        if (e.lowerInfo->ignoreDim >= 0 &&
            static_cast<unsigned>(e.lowerInfo->ignoreDim) < br.size()) {
          br[e.lowerInfo->ignoreDim] = 1;
        }
        blockRepeats[0] = std::max(blockRepeats[0], br[0]);
        blockRepeats[1] = std::max(blockRepeats[1], br[1]);
      }
    }
  }

  void materializeLocalBuffers() {
    assert(threadLevelSize[0] > 0 && threadLevelSize[1] > 0 &&
           "thread-level block size not inferred");
    assert(blockRepeats[0] > 0 && blockRepeats[1] > 0 && "br not inferred");
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
      } else {
        auto ty = MemRefType::get(sz, srcDefOp.getElementType(), AffineMap{},
                                  srcDefOp.getMemorySpace());
        auto newAlloc =
            rewriter.create<memref::AllocaOp>(srcDefOp->getLoc(), ty);
        replaceVal = newAlloc->getResult(0);
        s_buffer_replace[oldBuffer] = replaceVal;
      }
      mapper.map(oldBuffer, replaceVal);
      localMapper.map(oldBuffer, replaceVal);
    }
  }

  void createThreadLoops() {
    rewriter.setInsertionPoint(op);
    // frisk.blocOp 根据newbuffer的size，生成 nestedFor
    std::vector<int> thread_level_sz{threadLevelSize.begin(),
                                     threadLevelSize.end()};
    std::vector<int> blockop_br{blockRepeats.begin(), blockRepeats.end()};

    // 创建 thread_own_data_sz 循环
    createNestedAffineFor(rewriter, op->getLoc(), thread_level_sz,
                          threadOwnDataIvs);
    // block_repeat 循环
    createNestedAffineFor(rewriter, op->getLoc(), blockop_br, blockRepeatIvs);
    // 延用现有循环结构：iu/wr 由线性线程下标在 buildAccessCoords 中分解。
  }

  void cloneBlockBody() {
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
        auto accessCoords = buildAccessCoords(
            loadOp.getMemref(), loadOp.getAffineMap().getNumResults());
        auto mapOperands =
            remapAffineOperands(loadOp.getMapOperands(), accessCoords);
        auto newLoad = rewriter.create<affine::AffineLoadOp>(
            loadOp.getLoc(), memref, loadOp.getAffineMap(), mapOperands);
        mapper.map(loadOp.getResult(), newLoad.getResult());
        localMapper.map(loadOp.getResult(), newLoad.getResult());
        continue;
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(childOp)) {
        Value memref = mapper.lookupOrDefault(storeOp.getMemref());
        Value valueToStore = mapper.lookupOrDefault(storeOp.getValueToStore());
        auto accessCoords = buildAccessCoords(
            storeOp.getMemref(), storeOp.getAffineMap().getNumResults());
        auto mapOperands =
            remapAffineOperands(storeOp.getMapOperands(), accessCoords);
        rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), valueToStore,
                                               memref, storeOp.getAffineMap(),
                                               mapOperands);
        continue;
      }
      auto *cloned = rewriter.clone(childOp, mapper);
      // 后续普通 arith/math op 使用 mapper，后续 local store 使用 localMapper。
      // 因此不管当前 op 用哪套下标克隆，都把 result 同步给两套映射。
      for (auto [oldResult, newResult] :
           llvm::zip(childOp.getResults(), cloned->getResults())) {
        if (!mapper.lookupOrNull(oldResult)) {
          mapper.map(oldResult, newResult);
        }
        if (!localMapper.lookupOrNull(oldResult)) {
          localMapper.map(oldResult, newResult);
        }
      }
    }
  }

  void replaceOldLocalBuffers() {
    // blockOp body 外部（blockOp 之后）可能还有对旧 alloc 的 use（如 copy-out
    // 等） 用 replaceAllUsesExcept 只替换 blockOp 外部的 use
    for (auto &[srcDefOp, sz] : allocLocalsToReplace) {
      Value oldBuffer = srcDefOp->getResult(0);
      auto temp = mapper.lookupOrNull(oldBuffer);
      if (temp == nullptr) {
        continue;
      }
      auto newAlloc = temp.getDefiningOp<memref::AllocaOp>();
      // 只替换 blockOp 之外还残留的 use
      if (newAlloc != nullptr) {
        oldBuffer.replaceAllUsesExcept(newAlloc->getResult(0),
                                       SmallPtrSet<Operation *, 1>{op});
        rewriter.eraseOp(srcDefOp);
      }
    }
  }
};

/// 匹配有 affine.load/store 的 frisk.block；生成 local tile
/// 分配与标量线程循环。
class BlockOpConversion : public OpConversionPattern<frisk::BlockOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(frisk::BlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return BlockLowering(op, rewriter).run();
  }
};

static FailureOr<VectorType>
getThreadTileVectorType(LowerInfo &info, Type elementType, unsigned rank,
                        Operation *op, PatternRewriter &rewriter) {
  auto threadShape = info.get_thread_own_data_size();
  SmallVector<int64_t, 2> shape;
  shape.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    int64_t dim = i < threadShape.size() ? threadShape[i] : 1;
    if (info.ignoreDim >= 0 && static_cast<unsigned>(info.ignoreDim) == i) {
      dim = 1;
    }
    if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
      return rewriter.notifyMatchFailure(
          op, "layout convert expects positive static thread tile shape");
    }
    shape.push_back(dim);
  }
  return VectorType::get(shape, elementType);
}

/// 所有线程写入 scratch 并同步后，按目标布局读取一个元素。
struct LayoutReadBackElement {
  OpBuilder &rewriter;
  Location loc;
  LowerInfo &toInfo;
  Value tidx;
  unsigned rank;
  frisk::AllocBufferOp scratch;
  Value emit(ArrayRef<Value> toIvs, Value currentVector) {
    auto scratchReadIndices =
        buildMappedAccessIndices(rewriter, loc, toInfo, tidx, toIvs, rank);
    auto loaded = rewriter.create<affine::AffineLoadOp>(
        loc, scratch.getResult(), scratchReadIndices);
    SmallVector<OpFoldResult, 2> position;
    position.reserve(toIvs.size());
    for (Value iv : toIvs) {
      position.push_back(iv);
    }
    return rewriter
        .create<vector::InsertOp>(loc, loaded.getResult(), currentVector,
                                  position)
        .getResult();
  }
};

static FailureOr<std::vector<int>>
getLayoutConvertLoopBounds(VectorType vecTy, Operation *op,
                           PatternRewriter &rewriter) {
  std::vector<int> bounds;
  bounds.reserve(vecTy.getRank());
  for (int64_t dim : vecTy.getShape()) {
    if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
      return rewriter.notifyMatchFailure(
          op, "layout convert expects int-sized vector shape");
    }
    bounds.push_back(static_cast<int>(dim));
  }
  return bounds;
}

static FailureOr<Value>
materializeLayoutConvertThroughShared(frisk::ConvertLayoutOp op, Value source,
                                      LowerInfo fromInfo, LowerInfo toInfo,
                                      ConversionPatternRewriter &rewriter) {
  auto memTy = mlir::dyn_cast<MemRefType>(op.getMemref().getType());
  if (!memTy) {
    return rewriter.notifyMatchFailure(
        op, "layout convert source must be a memref");
  }
  if (!memTy.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, "layout convert expects static block tile shape");
  }
  unsigned rank = memTy.getRank();
  if (rank == 0 || rank > 2) {
    return rewriter.notifyMatchFailure(
        op, "layout convert currently supports rank-1/rank-2 tiles");
  }

  Type elementType = memTy.getElementType();
  auto fromVecTy = getThreadTileVectorType(fromInfo, elementType, rank,
                                           op.getOperation(), rewriter);
  auto toVecTy = getThreadTileVectorType(toInfo, elementType, rank,
                                         op.getOperation(), rewriter);
  if (failed(fromVecTy) || failed(toVecTy)) {
    return failure();
  }

  Location loc = op->getLoc();
  auto tidx = findThreadIdxOp(op);
  auto scratch = rewriter.create<frisk::AllocBufferOp>(
      loc,
      SmallVector<int64_t>(memTy.getShape().begin(), memTy.getShape().end()),
      elementType, 16, int(friskMs::Shared));

  Value sourceVector = getVectorReplacement(source, op.getOperation());
  if (!sourceVector) {
    sourceVector = getVectorValue(source);
  }
  if (!sourceVector) {
    sourceVector = getVectorReplacement(op.getMemref(), op.getOperation());
  }
  if (!sourceVector) {
    sourceVector = getVectorValue(op.getMemref());
  }

  if (sourceVector) {
    auto sourceVecTy = mlir::dyn_cast<VectorType>(sourceVector.getType());
    if (!sourceVecTy) {
      return failure();
    }
    if (sourceVecTy != *fromVecTy) {
      auto threadTile = extractThreadTile(sourceVector, *fromVecTy, fromInfo,
                                          rewriter, loc, op.getOperation());
      if (failed(threadTile)) {
        return rewriter.notifyMatchFailure(
            op, "layout convert source vector does not match old thread tile");
      }
      sourceVector = *threadTile;
    }
  }

  auto fromBounds = getLayoutConvertLoopBounds(*fromVecTy, op, rewriter);
  if (failed(fromBounds)) {
    return failure();
  }

  Value sourceMemref = source;
  auto sourceMemTy = mlir::dyn_cast<MemRefType>(sourceMemref.getType());
  if (!sourceVector && !sourceMemTy) {
    return rewriter.notifyMatchFailure(
        op, "layout convert source must lower from vector or memref");
  }

  std::vector<Value> fromIvs;
  auto writeLoops = createNestedAffineFor(rewriter, loc, *fromBounds, fromIvs);
  SmallVector<Value, 2> scratchWriteIndices =
      buildMappedAccessIndices(rewriter, loc, fromInfo, tidx, fromIvs, rank);
  Value scalar;
  if (sourceVector) {
    SmallVector<int64_t, 2> staticPosition(fromIvs.size(),
                                           ShapedType::kDynamic);
    scalar = rewriter
                 .create<vector::ExtractOp>(
                     loc, fromVecTy->getElementType(), sourceVector, fromIvs,
                     rewriter.getDenseI64ArrayAttr(staticPosition))
                 .getResult();
  } else {
    SmallVector<Value, 2> sourceIndices;
    if (isLocalMemref(sourceMemref) &&
        sourceMemTy.getShape() == fromVecTy->getShape()) {
      sourceIndices.append(fromIvs.begin(), fromIvs.end());
    } else {
      sourceIndices = scratchWriteIndices;
    }
    scalar =
        rewriter.create<affine::AffineLoadOp>(loc, sourceMemref, sourceIndices)
            .getResult();
  }
  rewriter.create<affine::AffineStoreOp>(loc, scalar, scratch.getResult(),
                                         scratchWriteIndices);
  if (!writeLoops.empty()) {
    rewriter.setInsertionPointAfter(writeLoops.front());
  }

  rewriter.create<frisk::SyncThreadsInBlockOp>(loc);

  // Keep the complete block tile addressable. Materializing only toVecTy here
  // captures the first MMA fragment and loses the consumer's M/K offsets.
  if (isSharedMemref(op->getResult(0)))
    return scratch.getResult();

  Value result = rewriter
                     .create<arith::ConstantOp>(
                         loc, *toVecTy,
                         mlir::cast<TypedAttr>(rewriter.getZeroAttr(*toVecTy)))
                     .getResult();

  auto toBounds = getLayoutConvertLoopBounds(*toVecTy, op, rewriter);
  if (failed(toBounds)) {
    return failure();
  }

  LayoutReadBackElement element{rewriter, loc, toInfo, tidx, rank, scratch};
  return VectorTileLoopNest(rewriter, loc, toVecTy->getShape())
      .emit(result, element);
}

/// 匹配：insertConvertLayoutOps 登记过新旧布局的 rank-1/rank-2 静态 tile。
/// 改写：旧布局写 shared scratch -> block 同步 -> 新布局读 thread vector。
class ConvertLayoutOpConversion
    : public OpConversionPattern<frisk::ConvertLayoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(frisk::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto it = convertLayoutInfo.find(op);
    if (it == convertLayoutInfo.end()) {
      return rewriter.notifyMatchFailure(op,
                                         "missing layout convert LowerInfo");
    }

    auto [fromInfo, toInfo] = it->getSecond();
    auto converted = materializeLayoutConvertThroughShared(
        op, adaptor.getMemref(), fromInfo, toInfo, rewriter);
    if (failed(converted)) {
      return failure();
    }

    registerConvertedValueLowerInfo(*converted, toInfo, op.getOperation());
    rewriter.replaceOp(op, *converted);
    if (auto castOp =
            adaptor.getMemref().getDefiningOp<UnrealizedConversionCastOp>()) {
      if (llvm::all_of(castOp->getResults(),
                       [](Value result) { return result.use_empty(); })) {
        rewriter.eraseOp(castOp);
      }
    }
    return success();
  }
};

struct GemmVectorFragmentElement {
  PatternRewriter &rewriter;
  Location loc;
  Value source;
  SmallVector<Value, 2> offsets;

  Value emit(ArrayRef<Value> ivs, Value result) {
    SmallVector<OpFoldResult, 2> sourcePosition, resultPosition;
    for (auto [i, iv] : llvm::enumerate(ivs)) {
      sourcePosition.push_back(addIndexValues(rewriter, loc, offsets[i], iv));
      resultPosition.push_back(iv);
    }
    Value scalar = rewriter.create<vector::ExtractOp>(loc, source, sourcePosition);
    return rewriter.create<vector::InsertOp>(loc, scalar, result, resultPosition);
  }
};

static FailureOr<Value> getCopyToRegVectorFragment(
    frisk::CopyToRegOp op, Value source, PatternRewriter &rewriter) {
  auto sourceTy = dyn_cast<VectorType>(source.getType());
  auto resultTy = dyn_cast<VectorType>(op.getResult().getType());
  if (!sourceTy || !resultTy)
    return failure();
  auto offsetAttr = op->getAttrOfType<AffineMapAttr>("gemm_thread_offset");
  if (!offsetAttr) {
    if (sourceTy == resultTy)
      return source;
    return failure();
  }
  if (sourceTy.getRank() != 2 || resultTy.getRank() != 2 ||
      sourceTy.getElementType() != resultTy.getElementType())
    return failure();

  SmallVector<Value, 2> offsets;
  for (AffineExpr expr : offsetAttr.getValue().getResults()) {
    auto map = AffineMap::get(7, 0, expr, rewriter.getContext());
    offsets.push_back(rewriter.create<affine::AffineApplyOp>(
        op.getLoc(), map, op.getMapOperands()));
  }
  Value init = rewriter.create<arith::ConstantOp>(
      op.getLoc(), resultTy, cast<TypedAttr>(rewriter.getZeroAttr(resultTy)));
  GemmVectorFragmentElement element{rewriter, op.getLoc(), source, offsets};
  return VectorTileLoopNest(rewriter, op.getLoc(), resultTy.getShape())
      .emit(init, element);
}

/// 早期清理：vector-backed GEMM 按当前循环的 thread offset 提取 MMA 片段。
/// 这里只折叠 cast，不展开真实 memref load，保留后续布局处理所需的信息。
class CopyToRegCastRewrite : public OpRewritePattern<frisk::CopyToRegOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(frisk::CopyToRegOp op,
                                PatternRewriter &rewriter) const override {
    Value srcMem = op.getSrcMemRef();
    auto resultVecTy = mlir::dyn_cast<VectorType>(op.getResult().getType());
    if (!resultVecTy) {
      return rewriter.notifyMatchFailure(op,
                                         "copy_to_reg result must be a vector");
    }

    auto castOp = srcMem.getDefiningOp<UnrealizedConversionCastOp>();
    if (castOp && castOp.getInputs().size() == 1) {
      Value srcVector = castOp.getInputs()[0];

      auto fragment = getCopyToRegVectorFragment(op, srcVector, rewriter);
      if (succeeded(fragment)) {
        rewriter.replaceOp(op, *fragment);
        if (srcMem.use_empty()) {
          rewriter.eraseOp(castOp);
        }
        return success();
      }
    }

    return failure();
  }
};

/// 对 offset map 给出的基址加上 vector 尾部维度下标，读取并转换一个浮点元素。
struct CopyToRegElement {
  PatternRewriter &rewriter;
  Location loc;
  frisk::CopyToRegOp op;
  MemRefType srcMemTy;
  VectorType resultVecTy;
  Value srcMem;
  Value emit(ArrayRef<Value> vectorIvs, Value currentVector) {
    SmallVector<Value> srcIndices;
    srcIndices.reserve(srcMemTy.getRank());
    unsigned loopStart = srcMemTy.getRank() - vectorIvs.size();
    for (unsigned i = 0; i < static_cast<unsigned>(srcMemTy.getRank()); ++i) {
      auto oneResultMap = AffineMap::get(
          op.getOffsetMap().getNumDims(), op.getOffsetMap().getNumSymbols(),
          op.getOffsetMap().getResult(i), rewriter.getContext());
      Value index = rewriter.create<affine::AffineApplyOp>(loc, oneResultMap,
                                                           op.getMapOperands());
      if (i >= loopStart) {
        index = addIndexValues(rewriter, loc, index, vectorIvs[i - loopStart]);
      }
      srcIndices.push_back(index);
    }

    Value scalar =
        rewriter.create<affine::AffineLoadOp>(loc, srcMem, srcIndices);
    if (scalar.getType() != resultVecTy.getElementType()) {
      auto srcFloatTy = mlir::dyn_cast<FloatType>(scalar.getType());
      auto dstFloatTy = mlir::dyn_cast<FloatType>(resultVecTy.getElementType());
      if (!srcFloatTy || !dstFloatTy) {
        return Value{};
      }
      if (srcFloatTy.getWidth() < dstFloatTy.getWidth()) {
        scalar =
            rewriter.create<arith::ExtFOp>(loc, dstFloatTy, scalar).getResult();
      } else if (srcFloatTy.getWidth() > dstFloatTy.getWidth()) {
        scalar = rewriter.create<arith::TruncFOp>(loc, dstFloatTy, scalar)
                     .getResult();
      }
    }

    SmallVector<OpFoldResult, 2> position;
    position.reserve(vectorIvs.size());
    for (Value iv : vectorIvs) {
      position.push_back(iv);
    }
    return rewriter
        .create<vector::InsertOp>(loc, scalar, currentVector, position)
        .getResult();
  }
};

/// 最终清理：先尝试复用 cast 后的同类型 vector，否则将 copy_to_reg 展开为
/// affine.load + 可选 extf/truncf + vector.insert，循环携带完整结果 vector。
class CopyToRegOpRewrite : public OpRewritePattern<frisk::CopyToRegOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(frisk::CopyToRegOp op,
                                PatternRewriter &rewriter) const override {
    Value srcMem = op.getSrcMemRef();
    auto resultVecTy = mlir::dyn_cast<VectorType>(op.getResult().getType());
    if (!resultVecTy) {
      return rewriter.notifyMatchFailure(op,
                                         "copy_to_reg result must be a vector");
    }

    auto castOp = srcMem.getDefiningOp<UnrealizedConversionCastOp>();
    if (castOp && castOp.getInputs().size() == 1) {
      Value srcVector = castOp.getInputs()[0];

      auto fragment = getCopyToRegVectorFragment(op, srcVector, rewriter);
      if (succeeded(fragment)) {
        rewriter.replaceOp(op, *fragment);
        if (srcMem.use_empty()) {
          rewriter.eraseOp(castOp);
        }
        return success();
      }
    }

    auto srcMemTy = mlir::dyn_cast<MemRefType>(srcMem.getType());
    if (!srcMemTy) {
      return rewriter.notifyMatchFailure(op,
                                         "copy_to_reg source must be a memref");
    }
    if (srcMemTy.getRank() != op.getOffsetMap().getNumResults()) {
      return rewriter.notifyMatchFailure(
          op, "copy_to_reg offset map result count must match source rank");
    }
    if (op.getOffsetMap().getNumInputs() != op.getMapOperands().size()) {
      return rewriter.notifyMatchFailure(
          op, "copy_to_reg offset map operand count mismatch");
    }
    if (resultVecTy.getRank() > srcMemTy.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "copy_to_reg vector rank is larger than source rank");
    }

    Location loc = op->getLoc();
    Value initVector =
        rewriter
            .create<arith::ConstantOp>(
                loc, resultVecTy,
                mlir::cast<TypedAttr>(rewriter.getZeroAttr(resultVecTy)))
            .getResult();

    CopyToRegElement element{rewriter, loc, op, srcMemTy, resultVecTy, srcMem};
    Value result = VectorTileLoopNest(rewriter, loc, resultVecTy.getShape())
                       .emit(initVector, element);
    if (!result) {
      return rewriter.notifyMatchFailure(
          op, "copy_to_reg element type conversion is unsupported");
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// 用于重写 frisk.copy 进行数据类型转换的情况。frisk.copy <3x3xf32> to <3x3xf16>
static FailureOr<Value> getCopyConvertLocalBuffer(LowerInfo *info,
                                                  Type elementType,
                                                  bool forRead, Operation *op,
                                                  OpBuilder &rewriter) {
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
}

static int64_t getEffectiveDimSize(LowerInfo *info, int dim, int64_t value) {
  return info->ignoreDim == dim ? int64_t{1} : value;
}

/// 一个寄存器 tile 的线程内基址：((br * unroll + iu) * repeat + wr) * width。
static Value buildCopyConvertTileBase(int dim, LowerInfo *srcInfo,
                                      ArrayRef<Value> outerIvs,
                                      OpBuilder &rewriter, Location loc) {
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
}

struct CopyOffset {
  SmallVector<Value> operands;
  AffineMap map;
};
/// local tile 使用不含 tid 的线程内 map；shared 使用包含 tid 的 block 访问
/// map。
static CopyOffset buildCopyConvertOffset(LowerInfo *info, bool useLocalTile,
                                         ArrayRef<Value> tileBaseIvs,
                                         Value tidx, Value zero,
                                         OpBuilder &rewriter, Location loc) {
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
    ius.push_back(modBy(
        rewriter, loc, floorDivBy(rewriter, loc, iv, repeatWidth), instUnroll));
    Value withinInst = modBy(rewriter, loc, iv, repeatWidth);
    wrs.push_back(floorDivBy(rewriter, loc, withinInst, threadWidth));
  }

  if (useLocalTile) {
    SmallVector<Value> wrXY{wrs[0], wrs[1]};
    SmallVector<Value> operands{brs[0],
                                brs[1],
                                ius[0],
                                ius[1],
                                flattenXY(rewriter, loc, wrXY,
                                          info->base_layout.warp_repeat_order,
                                          info->get_warp_repeat()),
                                zero};
    return {operands, buildThreadTileOffsetMap(rewriter, *info)};
  }

  auto operands =
      buildLowerInfoMapOperands(rewriter, loc, *info, tidx, brs[0], brs[1],
                                ius[0], ius[1], wrs[0], wrs[1], zero, zero);
  return {SmallVector<Value>{operands.begin(), operands.end()},
          info->getAffineMap()};
}

/// 匹配：源/目标 shape 相同、元素类型不同的 copy，线程宽度和 tile 数必须匹配。
/// 改写：按 br/iu/wr 遍历，读源寄存器片段 -> extf/truncf -> 写目标片段。
/// 这是浮点转换路径；不会自动扩展为整数转换或跨布局重排。
class CopyConvertOpRewrite : public OpConversionPattern<frisk::CopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 匹配模式检查
    auto srcTy = mlir::cast<MemRefType>(op.getSrcMemRef().getType());
    auto dstTy = mlir::cast<MemRefType>(op.getDstMemRef().getType());
    bool needConvert = false;
    if (srcTy.getShape() == dstTy.getShape() &&
        srcTy.getElementType() != dstTy.getElementType()) {
      needConvert = true;
    }
    if (!needConvert) {
      return failure();
    }

    auto tidx = findThreadIdxOp(op);
    auto srcInfo = s_info->getLowerInfo(op.getSrcMemRef(), op);
    auto dstInfo = s_info->getLowerInfo(op.getDstMemRef(), op);
    assert(srcInfo != nullptr && dstInfo != nullptr &&
           "copy-convert LowerInfo not found");
    auto [tw0, tw1] = srcInfo->get_thread_widths();
    auto [dstTw0, dstTw1] = dstInfo->get_thread_widths();
    assert(tw0 == dstTw0 && tw1 == dstTw1 &&
           "copy-convert expects source and destination thread tiles to match");

    bool srcIsLocal = isLocalMemref(srcInfo->buffer);
    bool dstIsLocal = isLocalMemref(dstInfo->buffer);
    auto srcBuffer = srcInfo->buffer;
    auto dstBuffer = dstInfo->buffer;
    if (srcIsLocal) {
      auto replacement = getCopyConvertLocalBuffer(
          srcInfo, srcTy.getElementType(), true, op, rewriter);
      if (failed(replacement)) {
        return op.emitOpError("local source has no thread-level replacement "
                              "for dtype conversion");
      }
      srcBuffer = *replacement;
    }
    if (dstIsLocal) {
      auto replacement = getCopyConvertLocalBuffer(
          dstInfo, dstTy.getElementType(), false, op, rewriter);
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

    auto srcTileCounts = srcInfo->get_block_repeat() * srcInfo->warpInstUnroll *
                         srcInfo->get_warp_repeat();
    auto dstTileCounts = dstInfo->get_block_repeat() * dstInfo->warpInstUnroll *
                         dstInfo->get_warp_repeat();
    for (int dim = 0; dim < 2; ++dim) {
      srcTileCounts[dim] =
          getEffectiveDimSize(srcInfo, dim, srcTileCounts[dim]);
      dstTileCounts[dim] =
          getEffectiveDimSize(dstInfo, dim, dstTileCounts[dim]);
      if (srcTileCounts[dim] != dstTileCounts[dim]) {
        return op.emitOpError(
            "copy-convert source/destination tile counts do not match");
      }
    }

    auto srcBr = srcInfo->get_block_repeat();
    auto srcWr = srcInfo->get_warp_repeat();
    std::vector<int> outerUbs = {
        int(getEffectiveDimSize(srcInfo, 0, srcBr[0])),
        int(getEffectiveDimSize(srcInfo, 1, srcBr[1])),
        int(getEffectiveDimSize(srcInfo, 0, srcInfo->warpInstUnroll[0])),
        int(getEffectiveDimSize(srcInfo, 1, srcInfo->warpInstUnroll[1])),
        int(getEffectiveDimSize(srcInfo, 0, srcWr[0])),
        int(getEffectiveDimSize(srcInfo, 1, srcWr[1]))};
    std::vector<const char *> outerLabels = {"br0",  "br1", "iwu0",
                                             "iwu1", "wr0", "wr1"};
    std::vector<Value> outerIvs;
    createNestedAffineFor(rewriter, loc, outerUbs, outerIvs, outerLabels);

    Value zero = createIndexConstant(rewriter, loc, 0);

    SmallVector<Value, 2> tileBaseIvs{
        buildCopyConvertTileBase(0, srcInfo, outerIvs, rewriter, loc),
        buildCopyConvertTileBase(1, srcInfo, outerIvs, rewriter, loc)};

    auto srcOffset = buildCopyConvertOffset(srcInfo, srcIsLocal, tileBaseIvs,
                                            tidx, zero, rewriter, loc);
    rewriter.create<frisk::CopyOp>(loc, srcBuffer, srcTile.getResult(),
                                   srcOffset.operands, srcOffset.map);

    std::vector<Value> regIvs;
    std::vector<const char *> regLabels = {"tw0", "tw1"};
    auto regLoops = createNestedAffineFor(
        rewriter, loc, std::vector<int>{int(tw0), int(tw1)}, regIvs, regLabels);

    auto srcValue = rewriter.create<affine::AffineLoadOp>(
        loc, srcTile.getResult(), ValueRange{regIvs[0], regIvs[1]});
    mlir::Value converted{};
    if (srcTy.getElementType().getIntOrFloatBitWidth() <
        dstTy.getElementType().getIntOrFloatBitWidth()) {
      converted = rewriter.create<arith::ExtFOp>(loc, dstTy.getElementType(),
                                                 srcValue.getResult());
    } else {
      converted = rewriter.create<arith::TruncFOp>(loc, dstTy.getElementType(),
                                                   srcValue.getResult());
    }
    rewriter.create<affine::AffineStoreOp>(loc, converted, dstTile.getResult(),
                                           ValueRange{regIvs[0], regIvs[1]});

    if (!regLoops.empty()) {
      rewriter.setInsertionPointAfter(regLoops.front());
    }
    auto dstOffset = buildCopyConvertOffset(dstInfo, dstIsLocal, tileBaseIvs,
                                            tidx, zero, rewriter, loc);
    rewriter.create<frisk::CopyOp>(loc, dstTile.getResult(), dstBuffer,
                                   dstOffset.operands, dstOffset.map);
    if (op.hasValueResult()) {
      rewriter.replaceOp(op, dstBuffer);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// 按既有优先级分派：vector -> global/shared -> 带 LowerInfo 的 view -> 普通
/// memref。 每次 match 创建独立对象，成员只保存本次改写的上下文，避免大型
/// lambda 隐式捕获。
class CopyLowering {
public:
  CopyLowering(frisk::CopyOp op, frisk::CopyOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter)
      : op(op), rewriter(rewriter), srcMem(adaptor.getSrc()),
        dstMem(adaptor.getDst()) {}
  LogicalResult run() {
    if (succeeded(lowerVectorCopy())) {
      return success();
    }

    srcMemType = mlir::cast<MemRefType>(srcMem.getType());
    dstMemType = mlir::cast<MemRefType>(dstMem.getType());
    if (srcMemType.getElementType() != dstMemType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "copy-to-base only handles same element type copies");
    }

    // 获取 srcmem dstmem 的 实际buffer
    srcInfo = resolveBuffer(srcMem);
    dstInfo = resolveBuffer(dstMem);

    if (succeeded(lowerGlobalSharedCopy())) {
      return success();
    }
    if (succeeded(lowerBufferViewCopyWithThreadMap())) {
      return success();
    }

    return lowerScalarCopy();
  }

private:
  struct BufferInfo {
    Value realBuffer;
    MemRefType realType;
    bool fromView = false;
    AffineMap viewMap;
    SmallVector<Value, 4> viewOperands;
  };
  struct CopyPlan {
    SmallVector<int64_t, 4> copyShape;
    bool hasMappedSide = false;
    bool isMapForSrc = true;
    AffineMap indexMap;
    SmallVector<Value, 4> indexMapOperands;
  };
  frisk::CopyOp op;
  ConversionPatternRewriter &rewriter;
  Value srcMem;
  Value dstMem;
  MemRefType srcMemType;
  MemRefType dstMemType;
  BufferInfo srcInfo;
  BufferInfo dstInfo;

  std::optional<LowerInfo> getCopyValueInfo(Value value) {
    return copyLowerInfoForValue(value, op.getOperation());
  }
  /// Shared/global destinations retain address semantics. Only local register
  /// destinations may be represented by an SSA replacement.
  LogicalResult lowerVectorCopy() {
    Value srcVector = getVectorReplacement(srcMem, op.getOperation());
    if (!srcVector) {
      srcVector = getVectorValue(srcMem);
    }
    Value dstVector = getVectorValue(dstMem);
    if (!srcVector) {
      return failure();
    }
    auto dstMemType = mlir::dyn_cast<MemRefType>(dstMem.getType());

    if (!dstVector && dstMemType) {
      auto casted = castFloatVectorElementType(
          srcVector, dstMemType.getElementType(), rewriter, op->getLoc());
      if (failed(casted)) {
        return failure();
      }

      if (isLocalMemref(dstMem)) {
        if (!mlir::isa<BlockArgument>(dstMem)) {
          s_buffer_replace[dstMem] = *casted;
          if (auto castOp = dstMem.getDefiningOp<UnrealizedConversionCastOp>();
              castOp && castOp.getInputs().size() == 1) {
            s_buffer_replace[castOp.getInputs()[0]] = *casted;
          }
        }
        if (op.hasValueResult()) {
          rewriter.replaceOp(op, dstMem);
        } else {
          rewriter.eraseOp(op);
        }
        eraseUnusedCast(srcMem);
        return success();
      }

      auto srcVecTy = mlir::cast<VectorType>((*casted).getType());
      auto loc = op->getLoc();
      std::vector<int> loopUpperBounds;
      loopUpperBounds.reserve(srcVecTy.getRank());
      for (int64_t dim : srcVecTy.getShape()) {
        if (dim < 0 || dim > std::numeric_limits<int>::max()) {
          return rewriter.notifyMatchFailure(
              op, "vector copy expects static int-sized vector shape");
        }
        loopUpperBounds.push_back(static_cast<int>(dim));
      }

      std::vector<Value> vectorIvs;
      auto loops =
          createNestedAffineFor(rewriter, loc, loopUpperBounds, vectorIvs);
      SmallVector<int64_t> staticPosition(vectorIvs.size(),
                                          ShapedType::kDynamic);
      auto scalar = rewriter.create<vector::ExtractOp>(
          loc, srcVecTy.getElementType(), *casted, vectorIvs,
          rewriter.getDenseI64ArrayAttr(staticPosition));

      SmallVector<Value, 4> tileIndices;
      auto copyInfo = getCopyValueInfo(srcMem);
      if (!copyInfo) {
        copyInfo = getCopyValueInfo(dstMem);
      }
      if (copyInfo && srcVecTy.getRank() <= 2) {
        auto threadOwnData = copyInfo->get_thread_own_data_size();
        bool matchesThreadTile = true;
        for (int64_t i = 0; i < srcVecTy.getRank(); ++i) {
          matchesThreadTile &= srcVecTy.getDimSize(i) == threadOwnData[i];
        }
        if (matchesThreadTile) {
          tileIndices = buildMappedAccessIndices(rewriter, loc, *copyInfo,
                                                 findThreadIdxOp(op), vectorIvs,
                                                 srcVecTy.getRank());
        }
      }
      if (tileIndices.empty()) {
        tileIndices.append(vectorIvs.begin(), vectorIvs.end());
      }

      unsigned dstRank = dstMemType.getRank();
      // Same-shape copies use the legacy () -> (2) sentinel, not an
      // address offset. The thread layout already supplies block coordinates.
      auto srcMemTy = dyn_cast<MemRefType>(srcMem.getType());
      bool wholeBufferCopy = srcMemTy &&
                             srcMemTy.getShape() == dstMemType.getShape();
      if ((!wholeBufferCopy &&
           (op.getOffsetMap().getNumResults() != dstRank ||
            op.getOffsetMap().getNumInputs() != op.getMapOperands().size())) ||
          tileIndices.size() > dstRank) {
        return rewriter.notifyMatchFailure(
            op, "vector-to-memref copy has incompatible offset map");
      }

      SmallVector<Value> dstIndices;
      dstIndices.reserve(dstRank);
      unsigned loopStart = dstRank - tileIndices.size();
      for (unsigned i = 0; i < dstRank; ++i) {
        Value index;
        if (wholeBufferCopy) {
          index = createIndexConstant(rewriter, loc, 0);
        } else {
          auto oneResultMap = AffineMap::get(
              op.getOffsetMap().getNumDims(), op.getOffsetMap().getNumSymbols(),
              op.getOffsetMap().getResult(i), rewriter.getContext());
          index = rewriter.create<affine::AffineApplyOp>(
              loc, oneResultMap, op.getMapOperands());
        }
        if (i >= loopStart) {
          index =
              addIndexValues(rewriter, loc, index, tileIndices[i - loopStart]);
        }
        dstIndices.push_back(index);
      }

      rewriter.create<affine::AffineStoreOp>(loc, scalar.getResult(), dstMem,
                                             dstIndices);
      if (!loops.empty()) {
        rewriter.setInsertionPointAfter(loops.front());
      }
      if (isSharedMemref(dstMem)) {
        rewriter.create<frisk::SyncThreadsInBlockOp>(loc);
      }
      if (op.hasValueResult()) {
        rewriter.replaceOp(op, dstMem);
      } else {
        rewriter.eraseOp(op);
      }
      eraseUnusedCast(srcMem);
      eraseUnusedAllocBuffer(srcMem);
      return success();
    }

    if (!dstVector) {
      return failure();
    }

    auto srcVecTy = mlir::cast<VectorType>(srcVector.getType());
    auto dstVecTy = mlir::cast<VectorType>(dstVector.getType());
    auto casted = castFloatVectorElementType(
        srcVector, dstVecTy.getElementType(), rewriter, op->getLoc());
    if (failed(casted)) {
      return failure();
    }
    srcVector = *casted;
    srcVecTy = mlir::cast<VectorType>(srcVector.getType());

    Value updatedDst = srcVector;
    if (srcVecTy != dstVecTy) {
      auto dstInfo = getCopyValueInfo(dstMem);
      if (!dstInfo) {
        return rewriter.notifyMatchFailure(
            op, "vector copy destination LowerInfo not found");
      }
      auto inserted =
          insertThreadTileIntoVector(srcVector, dstVector, *dstInfo, rewriter,
                                     op->getLoc(), op.getOperation());
      if (failed(inserted)) {
        return failure();
      }
      updatedDst = *inserted;
      s_buffer_replace[updatedDst] = srcVector;
    }

    if (!mlir::isa<BlockArgument>(dstMem)) {
      s_buffer_replace[dstMem] = updatedDst;
      if (auto castOp = dstMem.getDefiningOp<UnrealizedConversionCastOp>();
          castOp && castOp.getInputs().size() == 1) {
        s_buffer_replace[castOp.getInputs()[0]] = updatedDst;
      }
    }

    if (op.hasValueResult()) {
      if (auto dstInfo = getCopyValueInfo(dstMem)) {
        registerConvertedValueLowerInfo(updatedDst, *dstInfo,
                                        op.getOperation());
      }
      rewriter.replaceOp(op, updatedDst);
      eraseUnusedCast(srcMem);
      return success();
    }

    Operation *terminator = op->getBlock()->getTerminator();
    if (terminator && op->isBeforeInBlock(terminator)) {
      rewriter.modifyOpInPlace(terminator, [&]() {
        for (OpOperand &operand : terminator->getOpOperands()) {
          if (operand.get() == dstMem) {
            operand.set(updatedDst);
          }
        }
      });
    }

    rewriter.eraseOp(op);
    eraseUnusedCast(srcMem);
    return success();
  }

  /// view 路径：LowerInfo 计算 tile 坐标，view map 再加上底层 buffer 的基址。
  LogicalResult lowerBufferViewCopyWithThreadMap() {
    if (!s_info || (!srcInfo.fromView && !dstInfo.fromView) ||
        (srcInfo.fromView && dstInfo.fromView)) {
      return failure();
    }
    if (srcMemType.getShape() != dstMemType.getShape()) {
      return failure();
    }
    int64_t srcSpace = srcMemType.getMemorySpaceAsInt();
    int64_t dstSpace = dstMemType.getMemorySpaceAsInt();
    if ((srcSpace == int(friskMs::Global) &&
         dstSpace == int(friskMs::Shared)) ||
        (srcSpace == int(friskMs::Shared) &&
         dstSpace == int(friskMs::Global))) {
      return failure();
    }

    LowerInfo *copyInfo = s_info->getLowerInfo(srcMem, op.getOperation());
    if (copyInfo == nullptr) {
      copyInfo = s_info->getLowerInfo(dstMem, op.getOperation());
    }
    if (copyInfo == nullptr) {
      return failure();
    }

    const BufferInfo &viewSide = srcInfo.fromView ? srcInfo : dstInfo;
    const BufferInfo &directSide = srcInfo.fromView ? dstInfo : srcInfo;
    unsigned viewRank =
        srcInfo.fromView ? srcMemType.getRank() : dstMemType.getRank();
    if (directSide.realType.getRank() != static_cast<int64_t>(viewRank)) {
      return rewriter.notifyMatchFailure(
          op, "thread mapped direct side rank must match view rank");
    }
    if (viewSide.viewMap.getNumResults() != viewSide.realType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "buffer_view index map result count must match source rank");
    }
    if (viewSide.viewMap.getNumInputs() != viewSide.viewOperands.size()) {
      return rewriter.notifyMatchFailure(
          op, "buffer_view index map operands do not match map inputs");
    }
    if (viewRank > viewSide.realType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "thread mapped view rank is larger than real buffer rank");
    }

    auto tidx = findThreadIdxOp(op);
    auto loc = op->getLoc();
    std::vector<Value> iterVars;
    auto loops = copyInfo->getForLoops(rewriter, loc, iterVars);
    assert(iterVars.size() == 6 &&
           "thread mapped copy expects six LowerInfo loop iterators");

    SmallVector<Value, 7> mapOperands;
    mapOperands.push_back(tidx);
    mapOperands.append(iterVars.begin(), iterVars.end());
    auto tileIndices =
        applyLowerInfoMap(rewriter, loc, *copyInfo, mapOperands, viewRank);

    SmallVector<Value> srcIndices;
    SmallVector<Value> dstIndices;
    if (srcInfo.fromView) {
      srcIndices = buildViewIndices(srcInfo, tileIndices, loc);
      dstIndices = SmallVector<Value>(tileIndices.begin(), tileIndices.end());
    } else {
      srcIndices = SmallVector<Value>(tileIndices.begin(), tileIndices.end());
      dstIndices = buildViewIndices(dstInfo, tileIndices, loc);
    }

    auto value = rewriter.create<affine::AffineLoadOp>(loc, srcInfo.realBuffer,
                                                       srcIndices);
    AppendNameToLoc(value);
    auto storeOp = rewriter.create<affine::AffineStoreOp>(
        loc, value.getResult(), dstInfo.realBuffer, dstIndices);
    AppendNameToLoc(storeOp);
    if (!loops.empty()) {
      rewriter.setInsertionPointAfter(loops.front());
    }
    if (op.hasValueResult()) {
      rewriter.replaceOp(op, dstMem);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

  /// global/shared 路径：每线程连续搬运 ceil(N/thread_num)
  /// 个元素，尾部做边界检查。 shared -> global 在读前同步；global -> shared
  /// 在写后同步。
  LogicalResult lowerGlobalSharedCopy() {
    bool srcIsGlobal = isGlobalMemref(srcInfo.realBuffer);
    bool dstIsGlobal = isGlobalMemref(dstInfo.realBuffer);
    bool srcIsShared = isSharedMemref(srcInfo.realBuffer);
    bool dstIsShared = isSharedMemref(dstInfo.realBuffer);
    if (!((srcIsGlobal && dstIsShared) || (srcIsShared && dstIsGlobal))) {
      return failure();
    }

    auto copyPlan = computeCopyPlan();
    if (failed(copyPlan)) {
      return failure();
    }
    auto totalElements = productOfShape(copyPlan->copyShape);
    if (failed(totalElements)) {
      return failure();
    }
    auto threadNum = getThreadNum(op.getOperation(), rewriter);
    if (failed(threadNum)) {
      return failure();
    }
    int64_t chunkSize = (*totalElements + *threadNum - 1) / *threadNum;
    if (chunkSize > std::numeric_limits<int>::max()) {
      return rewriter.notifyMatchFailure(
          op, "copy-to-base expects int-sized per-thread copy chunk");
    }

    auto loc = op->getLoc();
    auto tileRank = copyPlan->copyShape.size();

    if (!copyPlan->hasMappedSide) {
      if (failed(validateDirectSide(srcInfo.realType, tileRank)) ||
          failed(validateDirectSide(dstInfo.realType, tileRank))) {
        return failure();
      }
    } else if (copyPlan->isMapForSrc) {
      if (failed(validateMappedSide(srcInfo.realType, *copyPlan)) ||
          failed(validateDirectSide(dstInfo.realType, tileRank))) {
        return failure();
      }
    } else {
      if (failed(validateDirectSide(srcInfo.realType, tileRank)) ||
          failed(validateMappedSide(dstInfo.realType, *copyPlan))) {
        return failure();
      }
    }

    if (srcIsShared) {
      rewriter.create<frisk::SyncThreadsInBlockOp>(loc);
    }

    if (*totalElements > 0) {
      auto tidx = findThreadIdxOp(op);
      std::vector<Value> copyIvs;
      auto loops = createNestedAffineFor(
          rewriter, loc, std::vector<int>{static_cast<int>(chunkSize)},
          copyIvs);

      auto d0 = rewriter.getAffineDimExpr(0);
      auto d1 = rewriter.getAffineDimExpr(1);
      auto linearMap =
          AffineMap::get(2, 0, d0 * chunkSize + d1, rewriter.getContext());
      Value linearIndex = rewriter.create<affine::AffineApplyOp>(
          loc, linearMap, ValueRange{tidx.getResult(), copyIvs[0]});

      auto inBoundsSet = IntegerSet::get(
          1, 0, SmallVector<AffineExpr>{-d0 + (*totalElements - 1)},
          SmallVector<bool>{false});
      auto ifOp = rewriter.create<affine::AffineIfOp>(
          loc, inBoundsSet, ValueRange{linearIndex}, false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());

      auto tileIndices = buildContiguousIndicesFromLinear(
          rewriter, loc, linearIndex, copyPlan->copyShape);

      FailureOr<SmallVector<Value>> srcIndices;
      FailureOr<SmallVector<Value>> dstIndices;
      if (!copyPlan->hasMappedSide) {
        srcIndices = buildDirectCopyIndices(srcInfo.realType.getRank(),
                                            tileIndices, "copy tile");
        dstIndices = buildDirectCopyIndices(dstInfo.realType.getRank(),
                                            tileIndices, "copy tile");
      } else if (copyPlan->isMapForSrc) {
        srcIndices = buildMappedCopyIndices(srcInfo.realType, *copyPlan,
                                            tileIndices, "copy tile");
        dstIndices = buildDirectCopyIndices(dstInfo.realType.getRank(),
                                            tileIndices, "copy tile");
      } else {
        srcIndices = buildDirectCopyIndices(srcInfo.realType.getRank(),
                                            tileIndices, "copy tile");
        dstIndices = buildMappedCopyIndices(dstInfo.realType, *copyPlan,
                                            tileIndices, "copy tile");
      }
      if (failed(srcIndices) || failed(dstIndices)) {
        return failure();
      }

      auto value = rewriter.create<affine::AffineLoadOp>(
          loc, srcInfo.realBuffer, *srcIndices);
      rewriter.create<affine::AffineStoreOp>(loc, value.getResult(),
                                             dstInfo.realBuffer, *dstIndices);
      rewriter.setInsertionPointAfter(loops.front());
    }

    if (dstIsShared) {
      rewriter.create<frisk::SyncThreadsInBlockOp>(loc);
    }
    if (op.hasValueResult()) {
      rewriter.replaceOp(op, dstMem);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

  /// 通用路径：以较小 buffer 为 copy tile，在较大 buffer 一侧应用 offset map。
  LogicalResult lowerScalarCopy() {
    auto copyPlan = computeCopyPlan();
    if (failed(copyPlan)) {
      return failure();
    }
    auto &copyShape = copyPlan->copyShape;
    bool hasMappedSide = copyPlan->hasMappedSide;
    bool isMapForSrc = copyPlan->isMapForSrc;

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

    FailureOr<SmallVector<Value>> srcIndices;
    FailureOr<SmallVector<Value>> dstIndices;
    if (!hasMappedSide) {
      srcIndices = buildDirectCopyIndices(srcInfo.realType.getRank(), copyIvs,
                                          "copy iteration");
      dstIndices = buildDirectCopyIndices(dstInfo.realType.getRank(), copyIvs,
                                          "copy iteration");
    } else if (isMapForSrc) {
      srcIndices = buildMappedCopyIndices(srcInfo.realType, *copyPlan, copyIvs,
                                          "copy iteration");
      dstIndices = buildDirectCopyIndices(dstInfo.realType.getRank(), copyIvs,
                                          "copy iteration");
    } else {
      srcIndices = buildDirectCopyIndices(srcInfo.realType.getRank(), copyIvs,
                                          "copy iteration");
      dstIndices = buildMappedCopyIndices(dstInfo.realType, *copyPlan, copyIvs,
                                          "copy iteration");
    }
    if (failed(srcIndices) || failed(dstIndices)) {
      return failure();
    }

    auto value = rewriter.create<affine::AffineLoadOp>(
        op->getLoc(), srcInfo.realBuffer, *srcIndices);
    rewriter.create<affine::AffineStoreOp>(op->getLoc(), value.getResult(),
                                           dstInfo.realBuffer, *dstIndices);

    if (!loops.empty()) {
      rewriter.setInsertionPointAfter(loops.front());
    }
    if (op.hasValueResult()) {
      rewriter.replaceOp(op, dstMem);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

  /// 仅解开一层 buffer_view；保留逻辑 shape 与物理 buffer 的区别。
  static BufferInfo resolveBuffer(Value buffer) {
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
  }

  /// 只计算访问计划，不生成 IR；同元素数但不同 shape 无法推断 map 作用侧。
  FailureOr<CopyPlan> computeCopyPlan() {
    CopyPlan plan;
    if (srcInfo.fromView || dstInfo.fromView) {
      if (srcInfo.fromView && dstInfo.fromView) {
        (void)rewriter.notifyMatchFailure(
            op,
            "copy-to-base does not support copy between two buffer_view ops");
        return failure();
      }
      if (srcMemType.getShape() != dstMemType.getShape()) {
        (void)rewriter.notifyMatchFailure(
            op, "buffer_view copy expects the other buffer to have the same "
                "shape");
        return failure();
      }
      plan.hasMappedSide = true;
      plan.isMapForSrc = srcInfo.fromView;
      plan.indexMap = srcInfo.fromView ? srcInfo.viewMap : dstInfo.viewMap;
      plan.indexMapOperands =
          srcInfo.fromView ? srcInfo.viewOperands : dstInfo.viewOperands;
      plan.copyShape.assign(srcMemType.getShape().begin(),
                            srcMemType.getShape().end());
      return plan;
    }
    if (srcMemType.getShape() == dstMemType.getShape()) {
      plan.copyShape.assign(srcMemType.getShape().begin(),
                            srcMemType.getShape().end());
      return plan;
    }

    auto shapeCompare =
        compareShapeSize(srcMemType.getShape(), dstMemType.getShape());
    if (failed(shapeCompare)) {
      return failure();
    }
    if (*shapeCompare == 0) {
      (void)rewriter.notifyMatchFailure(
          op, "copy-to-base cannot infer map side for different shapes with "
              "the same element count");
      return failure();
    }
    plan.hasMappedSide = true;
    plan.isMapForSrc = *shapeCompare > 0;
    plan.indexMap = op.getOffsetMap();
    plan.indexMapOperands.assign(op.getMapOperands().begin(),
                                 op.getMapOperands().end());
    auto copyShapeRef =
        plan.isMapForSrc ? dstMemType.getShape() : srcMemType.getShape();
    plan.copyShape.assign(copyShapeRef.begin(), copyShapeRef.end());
    return plan;
  }

  FailureOr<int64_t> productOfShape(ArrayRef<int64_t> shape) {
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
  }

  FailureOr<int> compareShapeSize(ArrayRef<int64_t> lhs,
                                  ArrayRef<int64_t> rhs) {
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
  }

  SmallVector<Value> buildViewIndices(const BufferInfo &viewInfo,
                                      ArrayRef<Value> tileIndices,
                                      Location loc) {
    unsigned rank = viewInfo.realType.getRank();
    SmallVector<Value> indices;
    indices.reserve(rank);
    unsigned loopStart = rank - tileIndices.size();
    for (unsigned i = 0; i < rank; ++i) {
      auto oneResultMap = AffineMap::get(
          viewInfo.viewMap.getNumDims(), viewInfo.viewMap.getNumSymbols(),
          viewInfo.viewMap.getResult(i), rewriter.getContext());
      Value index = rewriter.create<affine::AffineApplyOp>(
          loc, oneResultMap, viewInfo.viewOperands);
      if (i >= loopStart) {
        index =
            addIndexValues(rewriter, loc, index, tileIndices[i - loopStart]);
      }
      indices.push_back(index);
    }
    return indices;
  }

  LogicalResult validateDirectSide(MemRefType realType, size_t tileRank) {
    if (realType.getRank() != static_cast<int64_t>(tileRank)) {
      return rewriter.notifyMatchFailure(
          op, "direct copy side rank must match copy tile rank");
    }
    return success();
  }

  LogicalResult validateMappedSide(MemRefType realType, const CopyPlan &plan) {
    if (plan.indexMap.getNumResults() != realType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "index map result count must match mapped buffer rank");
    }
    if (plan.indexMap.getNumInputs() != plan.indexMapOperands.size()) {
      return rewriter.notifyMatchFailure(
          op, "index map operand count does not match map input count");
    }
    if (plan.copyShape.size() > static_cast<size_t>(realType.getRank())) {
      return rewriter.notifyMatchFailure(
          op, "copy tile rank is larger than mapped buffer rank");
    }
    return success();
  }

  FailureOr<SmallVector<Value>> buildDirectCopyIndices(unsigned rank,
                                                       ArrayRef<Value> indices,
                                                       StringRef indexKind) {
    if (rank != indices.size()) {
      (void)rewriter.notifyMatchFailure(
          op, Twine("direct copy side rank must match ") + indexKind + " rank");
      return failure();
    }
    return SmallVector<Value>{indices.begin(), indices.end()};
  }

  /// offset map 的结果是基址；tile 坐标右对齐到物理 buffer 的尾部维度。
  FailureOr<SmallVector<Value>>
  buildMappedCopyIndices(MemRefType realType, const CopyPlan &plan,
                         ArrayRef<Value> tileIndices, StringRef indexKind) {
    unsigned rank = realType.getRank();
    if (plan.indexMap.getNumResults() != rank) {
      (void)rewriter.notifyMatchFailure(
          op, "index map result count must match mapped buffer rank");
      return failure();
    }
    if (plan.indexMap.getNumInputs() != plan.indexMapOperands.size()) {
      (void)rewriter.notifyMatchFailure(
          op, "index map operand count does not match map input count");
      return failure();
    }
    if (tileIndices.size() > rank) {
      (void)rewriter.notifyMatchFailure(
          op, Twine(indexKind) + " rank is larger than mapped buffer rank");
      return failure();
    }

    SmallVector<Value> indices;
    indices.reserve(rank);
    unsigned loopStart = rank - tileIndices.size();
    for (unsigned i = 0; i < rank; ++i) {
      auto oneResultMap = AffineMap::get(
          plan.indexMap.getNumDims(), plan.indexMap.getNumSymbols(),
          plan.indexMap.getResult(i), rewriter.getContext());
      Value index = rewriter.create<affine::AffineApplyOp>(
          op->getLoc(), oneResultMap, plan.indexMapOperands);
      if (i >= loopStart) {
        index = addIndexValues(rewriter, op->getLoc(), index,
                               tileIndices[i - loopStart]);
      }
      indices.push_back(index);
    }
    return indices;
  }

  void eraseUnusedAllocBuffer(Value value) {
    if (auto alloc = value.getDefiningOp<frisk::AllocBufferOp>()) {
      if (value.use_empty()) {
        rewriter.eraseOp(alloc);
      }
    }
  }

  void eraseUnusedCast(Value value) {
    if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (value.use_empty()) {
        rewriter.eraseOp(castOp);
      }
    }
  }
};

/// 最终 copy 转换入口。各分支的匹配条件与实现见 CopyLowering。
class CopyOpRewrite : public OpConversionPattern<frisk::CopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return CopyLowering(op, adaptor, rewriter).run();
  }
};

/// 沿 cast 链检查后续用户是否仍要求地址语义，决定 SSA fill 是否可转成 vector
/// 常量。
static bool shouldKeepFillResultAsMemref(frisk::FillOp op) {
  if (!op.hasValueResult()) {
    return false;
  }

  SmallVector<Value, 4> worklist{op.getValueResult()};
  SmallVector<Value, 4> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (llvm::is_contained(visited, value)) {
      continue;
    }
    visited.push_back(value);

    for (Operation *user : value.getUsers()) {
      if (isa<frisk::GemmOp, frisk::CopyToRegOp, frisk::ConvertLayoutOp,
              affine::AffineLoadOp, affine::AffineStoreOp>(user)) {
        return true;
      }
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
        for (Value result : castOp->getResults()) {
          worklist.push_back(result);
        }
      }
    }
  }
  return false;
}

/// 三条路径：已有 vector 表示 -> splat 常量；可向量化的 SSA fill -> thread
/// 常量； 其余 -> memref 标量写循环。需要地址语义的用户必须继续使用 memref。
/// 对无结果的旧式 fill，同步更新替换表和当前 block 的 terminator。
class FillOpRewrite : public OpConversionPattern<frisk::FillOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(frisk::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 获取 value，构建for循环。将value 赋值给 memref的每个点
    auto memref = adaptor.getMemref();
    auto loc = op->getLoc();
    auto valueAttr = dyn_cast<TypedAttr>(op.getValueAttr());
    if (!valueAttr) {
      return rewriter.notifyMatchFailure(op, "fill value must be typed");
    }

    Value fillVector = getVectorReplacement(memref, op.getOperation());
    if (!fillVector) {
      fillVector = getVectorValue(memref);
    }
    if (fillVector &&
        !(op.hasValueResult() && shouldKeepFillResultAsMemref(op))) {
      auto vecTy = mlir::dyn_cast<VectorType>(fillVector.getType());
      if (!vecTy) {
        return failure();
      }
      Attribute elementAttr = convertScalarAttrForElementType(
          valueAttr, vecTy.getElementType(), rewriter);
      if (!elementAttr) {
        return rewriter.notifyMatchFailure(
            op, "fill value cannot be converted to vector element type");
      }
      auto denseAttr = DenseElementsAttr::get(vecTy, elementAttr);
      auto filled = rewriter.create<arith::ConstantOp>(
          loc, vecTy, mlir::cast<TypedAttr>(denseAttr));

      if (op.hasValueResult()) {
        if (auto *info = findLowerInfoForValue(memref, op.getOperation())) {
          registerConvertedValueLowerInfo(filled.getResult(), *info,
                                          op.getOperation());
        }
        rewriter.replaceOp(op, filled.getResult());
        if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp->use_empty()) {
            rewriter.eraseOp(castOp);
          }
        }
        return success();
      }

      if (!mlir::isa<BlockArgument>(memref)) {
        s_buffer_replace[memref] = filled.getResult();
        if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>();
            castOp && castOp.getInputs().size() == 1) {
          s_buffer_replace[castOp.getInputs()[0]] = filled.getResult();
        }
      }

      Operation *terminator = op->getBlock()->getTerminator();
      if (terminator && op->isBeforeInBlock(terminator)) {
        rewriter.modifyOpInPlace(terminator, [&]() {
          for (OpOperand &operand : terminator->getOpOperands()) {
            if (operand.get() == memref || operand.get() == fillVector) {
              operand.set(filled.getResult());
            }
          }
        });
      }

      rewriter.eraseOp(op);
      if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp->use_empty()) {
          rewriter.eraseOp(castOp);
        }
      }
      return success();
    }

    if (op.hasValueResult() && !shouldKeepFillResultAsMemref(op)) {
      LowerInfo *info =
          findLowerInfoForValue(op.getValueResult(), op.getOperation());
      if (!info) {
        info = findLowerInfoForValue(op.getMemref(), op.getOperation());
      }
      if (info) {
        auto vecTy = getVecTypeByLowerInfoWithThreadOwnData(*info);
        Attribute elementAttr = convertScalarAttrForElementType(
            valueAttr, vecTy.getElementType(), rewriter);
        if (!elementAttr) {
          return rewriter.notifyMatchFailure(
              op, "fill value cannot be converted to vector element type");
        }
        auto denseAttr = DenseElementsAttr::get(vecTy, elementAttr);
        auto filled = rewriter.create<arith::ConstantOp>(
            loc, vecTy, mlir::cast<TypedAttr>(denseAttr));
        registerConvertedValueLowerInfo(filled.getResult(), *info,
                                        op.getOperation());
        rewriter.replaceOp(op, filled.getResult());
        return success();
      }
    }

    auto memrefType = mlir::cast<MemRefType>(memref.getType());
    // Local memory spaces (including the legacy value 5) represent registers.
    bool isRegFill = isLocalMemref(memref);

    std::vector<int> loopUpperBounds;
    loopUpperBounds.reserve(memrefType.getRank());
    SmallVector<int64_t> fillShape(memrefType.getShape());
    if (s_info && isRegFill && memrefType.getRank() == 2) {
      LowerInfo *lowerInfo =
          s_info->getLowerInfo(op.getMemref(), op.getOperation());
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

    // Keep the memref value type in this conversion.  Replacing an alloc's
    // memref result with a vector here would invalidate all affine users;
    // vectorization must rewrite the complete use-def chain in a later pass.
    auto val = rewriter.create<arith::ConstantOp>(
        loc, memrefType.getElementType(), valueAttr);
    std::vector<Value> ivs;
    auto loops = createNestedAffineFor(rewriter, loc, loopUpperBounds, ivs);
    rewriter.create<affine::AffineStoreOp>(loc, val.getResult(), memref,
                                           ValueRange{ivs});
    if (!loops.empty()) {
      rewriter.setInsertionPointAfter(loops.front());
    }
    if (isRegFill && !loops.empty()) {
      rewriter.modifyOpInPlace(loops.front(), [&]() {
        loops.front()->setAttr(REG_FILL_SEMATIC, valueAttr);
      });
    }
    if (op.hasValueResult()) {
      rewriter.replaceOp(op, memref);
      return success();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// 最后一阶段的存储分配：Shared -> memref.alloc，Local -> memref.alloca。
/// 保留 shape、memory space 和 alignment；只面向这两种已支持的内存空间。
class AllocBufferOpConversion
    : public OpConversionPattern<frisk::AllocBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memtype = mlir::cast<MemRefType>(op.getResult().getType());
    if (memtype.getMemorySpaceAsInt() == (int)friskMs::Shared) {
      auto newOp = rewriter.create<memref::AllocOp>(op->getLoc(), memtype,
                                                    op.getAlignmentAttr());
      rewriter.replaceOp(op, newOp);
    } else if (memtype.getMemorySpaceAsInt() == (int)friskMs::Local) {
      auto newOp = rewriter.create<memref::AllocaOp>(op->getLoc(), memtype,
                                                     op.getAlignmentAttr());
      rewriter.replaceOp(op, newOp);
    }
    return llvm::success();
  }
};

static Operation *getMaterializationConsumer(OpBuilder &builder) {
  if (builder.getBlock() == nullptr) {
    return nullptr;
  }
  Block::iterator insertPt = builder.getInsertionPoint();
  if (insertPt == builder.getBlock()->end()) {
    return nullptr;
  }
  return &*insertPt;
}

static void setThreadTileAttrs(OpBuilder &builder,
                               UnrealizedConversionCastOp castOp,
                               LowerInfo *lowerInfo) {
  if (lowerInfo == nullptr) {
    return;
  }
  auto shape = lowerInfo->get_thread_own_data_size();
  SmallVector<int64_t, 2> threadTileShape(shape.begin(), shape.end());
  castOp->setAttr(
      "thread_tile_shape",
      DenseI64ArrayAttr::get(builder.getContext(), threadTileShape));
}

/// 临时桥接 block/thread 表示，同时记录布局；最终 full conversion 会禁止残留
/// cast。
template <bool ToThread>
static Value materializeThreadTileCast(OpBuilder &builder, Type resultType,
                                       ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return {};
  Operation *consumerOp = getMaterializationConsumer(builder);
  LowerInfo *lowerInfo =
      findLowerInfoForMaterialization(inputs[0], resultType, consumerOp);
  auto castOp =
      builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs);
  castOp->setAttr(ToThread ? "b->t" : "t->b", builder.getBoolAttr(true));
  setThreadTileAttrs(builder, castOp, lowerInfo);
  registerMaterializedLowerInfo(castOp, lowerInfo, consumerOp);
  return castOp.getResult(0);
}

static void configureThreadTileTypeConverter(TypeConverter &converter) {
  // tile 类型依赖每个使用点的 LowerInfo，不能通过无上下文的 Type -> Type
  // 映射推导。
  converter.addConversion([](Type type) { return type; });
  converter.addTargetMaterialization(materializeThreadTileCast<true>);
  converter.addSourceMaterialization(materializeThreadTileCast<false>);
}

static void addThreadLevelLegalDialects(ConversionTarget &target) {
  target.addLegalDialect<
      frisk::FriskDialect, arith::ArithDialect, affine::AffineDialect,
      math::MathDialect, func::FuncDialect, memref::MemRefDialect,
      scf::SCFDialect, gpu::GPUDialect, vector::VectorDialect>();
}

/// Pass 阶段顺序体现数据依赖：先建立 producer 的 thread 表示，再转换 consumer。
class ConvertFriskBaseToThreadLevelIR
    : public impl::ConvertFriskBaseToThreadLevelIRBase<
          ConvertFriskBaseToThreadLevelIR> {
public:
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    auto kernel = getOperation();
    if (!kernel->hasAttr("thread_num")) {
      return;
    }
    s_buffer_replace.clear();
    convertLayoutInfo.clear();
    if (s_hw == nullptr) {
      s_hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, context);
    }

    initializeLayouts(kernel, context);
    TypeConverter typeConverter;
    configureThreadTileTypeConverter(typeConverter);

    if (failed(lowerStructuredOps(kernel, typeConverter, context)) ||
        failed(lowerValueFills(kernel, typeConverter, context)) ||
        failed(lowerElementwiseOps(kernel, typeConverter, context)) ||
        failed(lowerReductions(kernel, typeConverter, context)) ||
        failed(lowerCopiesAndLayouts(kernel, typeConverter, context)) ||
        failed(lowerAllocations(kernel, context))) {
      signalPassFailure();
      return;
    }
    //  --------- finally : dce
    eraseTriviallyDeadOps(kernel);

    promoteSingleIterationAffineFors(kernel);

    llvm::outs() << "---- convert to thread level IR done!\n";
    llvm::outs().flush();
  }

private:
  void initializeLayouts(func::FuncOp kernel, MLIRContext *context) {
    // -------- step 1 ：进行 layoutInfer 得到block级别IR上，每个buffer的
    // 访问模式。
    s_info = LowerInfoAnalysis::run(kernel);
    llvm::outs() << "\n-------------- lowerinfo analyze done\n";
    llvm::outs().flush();
    llvm::outs() << "\n-------------- lowerinfo print done!\n";
    llvm::outs().flush();

    // 保留原实现从分析表首项取得函数属性的行为。DenseMap 的迭代顺序不稳定；
    // 这里的属性不作为本文件逐元素索引生成的依据，索引使用各自的 LowerInfo。
    auto warpLayout = s_info->begin()->getSecond().get_warp_layout();
    auto blockLayout = s_info->begin()->getSecond().get_block_layout();
    auto blockLayoutOrder =
        s_info->begin()->getSecond().get_block_layout_order();

    kernel->setAttr("warp_layout", DenseI64ArrayAttr::get(context, warpLayout));
    kernel->setAttr("block_layout",
                    DenseI64ArrayAttr::get(context, blockLayout));
    kernel->setAttr("block_layout_order",
                    DenseI64ArrayAttr::get(context, blockLayoutOrder));

    // 根据 layout推定结果，插入 convertLAyoutOp, 随即更新 s_info
    // (加入convertLayoutOp 得到的新ssa值的info 及后续users信息替换)
    insertConvertLayoutOps(*s_info);
    llvm::outs() << "\n-------------- after insertConvertLayoutOps\n"
                 << getOperation() << "\n";
    llvm::outs().flush();
  }

  /// 1. 展开 Block/GEMM/Mask；仅强制转换 shared 间同 shape、不同 dtype 的
  /// copy。
  LogicalResult lowerStructuredOps(func::FuncOp kernel,
                                   TypeConverter &typeConverter,
                                   MLIRContext *context) {
    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);

    target.addIllegalOp<BlockOp, frisk::GemmOp, frisk::MaskOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<frisk::CopyOp>([](frisk::CopyOp op) {
      auto src = mlir::dyn_cast<MemRefType>(op.getSrcMemRef().getType());
      auto dst = mlir::dyn_cast<MemRefType>(op.getDstMemRef().getType());
      if (!src || !dst) {
        return true;
      }
      return !(isSharedMemref(op.getSrcMemRef()) &&
               isSharedMemref(op.getDstMemRef()) &&
               src.getShape() == dst.getShape() &&
               src.getElementType() != dst.getElementType());
    });

    RewritePatternSet patterns(context);
    patterns.add<BlockOpConversion, CopyConvertOpRewrite, GemmOpConversion,
                 MaskOpConversion>(typeConverter, context);
    llvm::outs()
        << "---- after convert gemm/blockop/reduce/copy(convert datatype)\n";
    if (failed(applyPartialConversion(kernel, target, std::move(patterns))))
      return failure();
    eraseDeadUnrealizedConversionCasts(kernel);
    llvm::outs() << kernel << "\n";
    llvm::outs().flush();

    return success();
  }

  /// 2. 提前物化 SSA fill，使循环初值和后续逐元素算子可以使用 thread vector。
  LogicalResult lowerValueFills(func::FuncOp kernel,
                                TypeConverter &typeConverter,
                                MLIRContext *context) {
    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);
    target.addDynamicallyLegalOp<frisk::FillOp>([](frisk::FillOp op) {
      return !op.hasValueResult() || shouldKeepFillResultAsMemref(op);
    });
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet fillPatterns(context);
    fillPatterns.add<FillOpRewrite>(typeConverter, context);
    llvm::outs() << "---- after convert SSA value fills\n";
    if (failed(applyPartialConversion(kernel, target, std::move(fillPatterns))))
      return failure();
    eraseDeadUnrealizedConversionCasts(kernel);
    llvm::outs() << kernel << "\n";
    llvm::outs().flush();

    return success();
  }

  /// 3. GEMM/Mask/fill 结果就绪后，将逐元素算子转成 arith/math vector 运算。
  LogicalResult lowerElementwiseOps(func::FuncOp kernel,
                                    TypeConverter &typeConverter,
                                    MLIRContext *context) {
    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);
    target.addIllegalOp<frisk::AddOp, frisk::SubOp, frisk::DivOp, frisk::MulOp,
                        frisk::Exp2Op, frisk::MaskOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    patterns.add<FriskBinaryOpConversion<frisk::AddOp, arith::AddFOp>,
                 FriskBinaryOpConversion<frisk::SubOp, arith::SubFOp>,
                 FriskBinaryOpConversion<frisk::DivOp, arith::DivFOp>,
                 FriskBinaryOpConversion<frisk::MulOp, arith::MulFOp>,
                 FriskExp2OpConversion>(typeConverter, context);
    llvm::outs() << "---- after convert binary elementwise ops\n";
    if (failed(applyPartialConversion(kernel, target, std::move(patterns))))
      return failure();
    RewritePatternSet copyToRegCleanup(context);
    copyToRegCleanup.add<CopyToRegCastRewrite>(context);
    (void)applyPatternsGreedily(kernel, std::move(copyToRegCleanup));
    foldThreadTileVectorLoads(kernel);
    eraseDeadUnrealizedConversionCasts(kernel);
    llvm::outs() << kernel << "\n";
    llvm::outs().flush();

    return success();
  }

  /// 4. Reduce 消费上述 thread vector，执行线程内累加和 warp shuffle。
  LogicalResult lowerReductions(func::FuncOp kernel,
                                TypeConverter &typeConverter,
                                MLIRContext *context) {
    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);
    target.addIllegalOp<frisk::ReduceOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet reducePatterns(context);
    reducePatterns.add<ReduceOpConversion>(typeConverter, context);
    llvm::outs() << "---- after convert reduce ops\n";
    if (failed(
            applyPartialConversion(kernel, target, std::move(reducePatterns))))
      return failure();
    RewritePatternSet copyToRegCleanup(context);
    copyToRegCleanup.add<CopyToRegCastRewrite>(context);
    (void)applyPatternsGreedily(kernel, std::move(copyToRegCleanup));
    foldThreadTileVectorLoads(kernel);
    eraseDeadUnrealizedConversionCasts(kernel);
    llvm::outs() << kernel << "\n";
    llvm::outs().flush();

    return success();
  }

  /// 5. 展开剩余 copy/fill/relayout；必须先收缩循环携带值，再折叠提取循环。
  LogicalResult lowerCopiesAndLayouts(func::FuncOp kernel,
                                      TypeConverter &typeConverter,
                                      MLIRContext *context) {

    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);

    target.addIllegalOp<BlockOp, GemmOp, ReduceOp, ConvertLayoutOp, CopyOp,
                        FillOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    patterns.add<CopyOpRewrite, FillOpRewrite, ConvertLayoutOpConversion>(
        typeConverter, context);
    llvm::outs() << "---- after convert copy /fill/ convLayout \n";
    if (failed(applyPartialConversion(kernel, target, std::move(patterns))))
      return failure();
    normalizeLoopCarriedBlockVectors(kernel);
    foldThreadTileExtractLoops(kernel);
    eraseTriviallyDeadOps(kernel);
    RewritePatternSet copyToRegCleanup(context);
    copyToRegCleanup.add<CopyToRegOpRewrite>(context);
    (void)applyPatternsGreedily(kernel, std::move(copyToRegCleanup));
    foldThreadTileVectorLoads(kernel);
    eraseDeadUnrealizedConversionCasts(kernel);
    llvm::outs() << kernel << "\n";
    llvm::outs().flush();

    return success();
  }

  /// 6. 最后分配实际存储，并要求所有高层操作和临时 cast 都已消除。
  LogicalResult lowerAllocations(func::FuncOp kernel, MLIRContext *context) {
    ConversionTarget target(*context);
    addThreadLevelLegalDialects(target);
    target.addIllegalOp<KernelOp, ParallelOp, ForOp, BlockOp, GemmOp, ReduceOp,
                        ConvertLayoutOp, CopyOp, FillOp, AllocBufferOp,
                        UnrealizedConversionCastOp>();
    RewritePatternSet patterns(context);
    patterns.add<AllocBufferOpConversion>(context);
    if (failed(applyFullConversion(kernel, target, std::move(patterns)))) {
      return failure();
    }

    return success();
  }
};

} // end namespace

std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass() {
  return std::make_unique<ConvertFriskBaseToThreadLevelIR>();
}

} // namespace mlir::frisk
