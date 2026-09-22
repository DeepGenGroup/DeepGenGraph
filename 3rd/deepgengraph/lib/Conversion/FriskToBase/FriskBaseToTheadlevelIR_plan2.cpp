//===----------------------------------------------------------------------===//
// Frisk block-level -> thread-level tiling (plan2).
// Every pattern checks `tiled`, consumes explicit ToThreadTile bridges and
// produces block tiles through FromThreadTile. Loops keep their block-level
// signatures. Destination-style operations retain explicit writebacks at the
// original program point; slice-copy results alias the complete destination.
// Bridge elimination is a later stage.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
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

// Only layout analysis is shared. All block/thread value transitions live in IR.
static LowerInfoMap *s_info = nullptr;
static HWSpecification *s_hw = nullptr;
static DenseMap<frisk::ConvertLayoutOp, std::pair<LowerInfo, LowerInfo>>
    convertLayoutInfo;

static bool isTiled(Operation *op) { return op->hasAttr("tiled"); }
static void markTiled(Operation *op, OpBuilder &builder) {
  op->setAttr("tiled", builder.getBoolAttr(true));
}

static std::optional<LowerInfo> findLowerInfoForValue(Value value, Operation *consumer) {
  if (s_info) {
    if (auto *info = s_info->getLowerInfo(value, consumer))
      return *info;
    for (Operation *user : value.getUsers())
      if (auto *info = s_info->getLowerInfo(value, user))
        return *info;
    for (auto &entry : *s_info)
      if (entry.second.buffer == value)
        return entry.second;
  }
  // A preceding legacy vector update can change an operand before its pattern
  // runs. Recover its layout from the bridge itself, not an SSA replacement map.
  auto bridge = value.getDefiningOp<frisk::FromThreadTileOp>();
  if (!bridge)
    return std::nullopt;
  auto warpThreads = bridge->getAttrOfType<IntegerAttr>("warp_threads");
  auto ignored = bridge->getAttrOfType<IntegerAttr>("ignore_dim");
  if (!warpThreads || !ignored)
    return std::nullopt;
  LowerInfo info(warpThreads.getInt());
  info.buffer = value;
  info.ignoreDim = ignored.getInt();
  auto read = [&](StringRef name, coordXY_t &field) {
    auto attr = bridge->getAttrOfType<DenseI64ArrayAttr>(name);
    if (!attr || attr.size() != 2)
      return false;
    field = {attr[0], attr[1]};
    return true;
  };
  if (!read("thread_widths", info.base_layout.thread_creg) ||
      !read("thread_creg_order", info.base_layout.thread_creg_order) ||
      !read("warp_repeat", info.base_layout.warp_repeat) ||
      !read("warp_repeat_order", info.base_layout.warp_repeat_order) ||
      !read("warp_layout", info.base_layout.warp_layout) ||
      !read("warp_layout_order", info.base_layout.warp_layout_order) ||
      !read("warp_inst_unroll", info.warpInstUnroll) ||
      !read("block_repeat", info.block_repeat) ||
      !read("block_layout", info.block_layout) ||
      !read("block_layout_order", info.block_layout_order))
    return std::nullopt;
  info.thread_own_data_size = info.get_thread_widths() * info.get_warp_repeat() *
                              info.warpInstUnroll * info.get_block_repeat();
  return info;
}

// A/B layouts may only budget one MMA's registers. A bridge represents the
// complete tile, so include block_repeat and preserve singleton broadcast axes.
static FailureOr<VectorType> getFullThreadTileType(Value blockTile,
                                                   const LowerInfo &info) {
  auto type = dyn_cast<ShapedType>(blockTile.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 2)
    return failure();
  auto full = info.get_thread_widths() * info.get_warp_repeat() *
              info.warpInstUnroll * info.get_block_repeat();
  SmallVector<int64_t, 2> shape;
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    unsigned layoutDim = type.getRank() == 1 && info.ignoreDim == 0 ? 1 : dim;
    int64_t size = type.getDimSize(dim) == 1 ||
                          (type.getRank() == 2 && info.ignoreDim == dim)
                      ? 1 : full[layoutDim];
    if (size <= 0 || size > std::numeric_limits<int>::max())
      return failure();
    shape.push_back(size);
  }
  return VectorType::get(shape, type.getElementType());
}

// A copy slice can have different extents from the buffer whose layout was
// inferred. Preserve lane/warp ownership while recomputing the repeat counts.
static LogicalResult setCopyTileShape(LowerInfo &info, ArrayRef<int64_t> shape) {
  if (shape.empty() || shape.size() > 2)
    return failure();
  auto widths = info.get_block_widths();
  for (unsigned dim = 0; dim < shape.size(); ++dim) {
    unsigned axis = shape.size() == 1 && info.ignoreDim == 0 ? 1 : dim;
    if (shape[dim] == 1) {
      info.block_repeat[axis] = 1;
      info.ignoreDim = axis;
    } else {
      if (widths[axis] <= 0 || shape[dim] % widths[axis] != 0)
        return failure();
      info.block_repeat[axis] = shape[dim] / widths[axis];
      if (info.ignoreDim == static_cast<int>(axis))
        info.ignoreDim = -1;
    }
  }
  return success();
}

// Keep layout metadata on each bridge: equal vector shapes do not imply equal
// distributions across lanes. No Value -> Value replacement table is needed.
static void setTileLayout(Operation *bridge, const LowerInfo &info,
                          OpBuilder &builder) {
  LowerInfo layout = info;
  bridge->setAttr("tile_layout", AffineMapAttr::get(layout.getAffineMap()));
  bridge->setAttr("thread_widths", builder.getDenseI64ArrayAttr(info.get_thread_widths()));
  bridge->setAttr("warp_repeat", builder.getDenseI64ArrayAttr(info.get_warp_repeat()));
  bridge->setAttr("warp_inst_unroll", builder.getDenseI64ArrayAttr(info.warpInstUnroll));
  bridge->setAttr("thread_creg_order", builder.getDenseI64ArrayAttr(info.base_layout.thread_creg_order));
  bridge->setAttr("warp_repeat_order", builder.getDenseI64ArrayAttr(info.base_layout.warp_repeat_order));
  bridge->setAttr("warp_layout", builder.getDenseI64ArrayAttr(info.get_warp_layout()));
  bridge->setAttr("warp_layout_order", builder.getDenseI64ArrayAttr(info.base_layout.warp_layout_order));
  bridge->setAttr("block_repeat", builder.getDenseI64ArrayAttr(info.get_block_repeat()));
  bridge->setAttr("block_layout", builder.getDenseI64ArrayAttr(info.get_block_layout()));
  bridge->setAttr("block_layout_order", builder.getDenseI64ArrayAttr(info.get_block_layout_order()));
  bridge->setAttr("warp_threads", builder.getI64IntegerAttr(info.warp_threads));
  bridge->setAttr("ignore_dim", builder.getI64IntegerAttr(info.ignoreDim));
  markTiled(bridge, builder);
}

static Value toThreadTile(Value value, Type tileType, const LowerInfo &info,
                           OpBuilder &builder, Location loc) {
  auto bridge = builder.create<frisk::ToThreadTileOp>(loc, tileType, value);
  setTileLayout(bridge, info, builder);
  return bridge.getResult();
}

static Value fromThreadTile(Value value, Type blockType, const LowerInfo &info,
                             OpBuilder &builder, Location loc) {
  auto bridge = builder.create<frisk::FromThreadTileOp>(loc, blockType, value);
  setTileLayout(bridge, info, builder);
  return bridge.getResult();
}

static Value findThreadIdxOp(Operation *op, OpBuilder &builder) {
  // Materialize at the use site, so the value always dominates its consumer.
  return builder.create<gpu::ThreadIdOp>(op->getLoc(), gpu::Dimension::x);
}

static void insertConvertLayoutOps(LowerInfoMap &infoMap) {
  struct Conversion { Operation *user; Value input; LowerInfo from; LowerInfo to; };
  SmallVector<Conversion, 8> conversions;
  // Snapshot first: adding conversion layouts can rehash the analysis map.
  for (auto &entry : infoMap) {
    auto &info = entry.second;
    if (info.convertFrom && info.buffer && info.op && !isTiled(info.op))
      conversions.push_back({info.op, info.buffer, *info.convertFrom, info});
  }
  for (auto &conversion : conversions) {
    OpBuilder builder(conversion.user);
    auto op = builder.create<frisk::ConvertLayoutOp>(conversion.user->getLoc(),
        conversion.input.getType(), conversion.input, builder.getStringAttr("lowerinfo.convert"));
    infoMap.updateLowerInfoForLayoutConvertOp(op, conversion.to);
    convertLayoutInfo.insert({op, {conversion.from, conversion.to}});
    conversion.user->replaceUsesOfWith(conversion.input, op->getResult(0));
  }
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
    unsigned layoutDim = rank == 1 && info.ignoreDim == 0 ? 1 : i;
    if (rank == 2 && info.ignoreDim >= 0 &&
        static_cast<unsigned>(info.ignoreDim) == layoutDim) {
      indices.push_back(createIndexConstant(builder, loc, 0));
      continue;
    }
    auto oneResultMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                       map.getResult(layoutDim), builder.getContext());
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
    if (rank == 1 && info.ignoreDim == 0)
      iv = i == 1 ? tileIvs[0] : zero;
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


// Legacy destination-style operations write memory at the original program
// point. Local destinations are thread memrefs; shared/global destinations keep
// physical block addresses. The explicit To(From(...)) chain is foldable later.
static void writeBackBlockTile(Value block, Value destination, LowerInfo info,
                                OpBuilder &builder, Location loc) {
  auto bridge = block.getDefiningOp<frisk::FromThreadTileOp>();
  auto threadType = cast<ShapedType>(bridge.getThreadTile().getType());
  auto vectorType = VectorType::get(threadType.getShape(), threadType.getElementType());
  Value source = toThreadTile(block, vectorType, info, builder, loc);
  auto dstType = cast<MemRefType>(destination.getType());
  bool local = dstType.getMemorySpaceAsInt() == int(friskMs::Local) ||
               dstType.getMemorySpaceAsInt() == 5;
  Value target = destination;
  if (local)
    target = toThreadTile(destination,
        MemRefType::get(threadType.getShape(), threadType.getElementType()),
        info, builder, loc);
  Value tid;
  if (!local)
    tid = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);

  // Reduced/broadcast axes are replicated among lanes. Only the leader writes
  // each physical output element, avoiding overlapping shared/global stores.
  scf::IfOp leader;
  if (!local) {
    // A/B layouts may be replicated across warps that compute different C
    // fragments. Only the first instance of that layout writes shared memory.
    Value warp = floorDivBy(builder, loc, tid, info.warp_threads);
    Value warpCount = createIndexConstant(builder, loc, flat_size(info.get_block_layout()));
    Value isLeader = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                   warp, warpCount);
    SmallVector<unsigned, 2> replicatedAxes;
    if (info.ignoreDim >= 0)
      replicatedAxes.push_back(info.ignoreDim);
    for (unsigned dim = 0; dim < dstType.getRank(); ++dim) {
      unsigned axis = dstType.getRank() == 1 && info.ignoreDim == 0 ? 1 : dim;
      if (dstType.getDimSize(dim) == 1 && !llvm::is_contained(replicatedAxes, axis))
        replicatedAxes.push_back(axis);
    }
    for (unsigned dim : replicatedAxes) {
      auto order = info.base_layout.warp_layout_order;
      int64_t stride = order[0] == dim ? 1 : info.get_warp_layout()[order[0]];
      Value lane = modBy(builder, loc, tid, info.warp_threads);
      Value coordinate = modBy(builder, loc, floorDivBy(builder, loc, lane, stride),
                                info.get_warp_layout()[dim]);
      auto blockOrder = info.get_block_layout_order();
      int64_t warpStride = blockOrder[0] == dim ? 1 : info.get_block_layout()[blockOrder[0]];
      Value warpCoordinate = modBy(builder, loc, floorDivBy(builder, loc, warp, warpStride),
                                    info.get_block_layout()[dim]);
      coordinate = addIndexValues(builder, loc, coordinate, warpCoordinate);
      Value zero = createIndexConstant(builder, loc, 0);
      Value axisLeader = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                       coordinate, zero);
      isLeader = builder.create<arith::AndIOp>(loc, isLeader, axisLeader);
    }
    leader = builder.create<scf::IfOp>(loc, isLeader, false);
    builder.setInsertionPointToStart(&leader.getThenRegion().front());
  }
  std::vector<int> bounds(threadType.getShape().begin(), threadType.getShape().end());
  std::vector<Value> ivs;
  auto loops = createNestedAffineFor(builder, loc, bounds, ivs);
  markTiled(loops.front(), builder);
  SmallVector<OpFoldResult, 2> position(ivs.begin(), ivs.end());
  Value scalar = builder.create<vector::ExtractOp>(loc, source, position);
  SmallVector<Value, 2> indices(ivs.begin(), ivs.end());
  if (!local)
    indices = buildMappedAccessIndices(builder, loc, info, tid, ivs, dstType.getRank());
  for (unsigned dim = 0; dim < dstType.getRank(); ++dim)
    if (dstType.getDimSize(dim) == 1)
      indices[dim] = createIndexConstant(builder, loc, 0);
  auto store = builder.create<affine::AffineStoreOp>(loc, scalar, target, indices);
  markTiled(store, builder);
  builder.setInsertionPointAfter(loops.front());
  if (leader)
    builder.setInsertionPointAfter(leader);
  if (dstType.getMemorySpaceAsInt() == int(friskMs::Shared)) {
    auto barrier = builder.create<gpu::BarrierOp>(loc);
    markTiled(barrier, builder);
  }
}

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


static FailureOr<Value> materializeElementwiseOperand(
    Value original, Value adapted, const LowerInfo &info, VectorType resultType,
    ConversionPatternRewriter &rewriter, Location loc) {
  if (isa<FloatType>(original.getType())) {
    if (original.getType() != resultType.getElementType())
      return failure();
    return rewriter.create<vector::BroadcastOp>(loc, resultType, adapted).getResult();
  }
  auto type = getFullThreadTileType(original, info);
  if (failed(type))
    return failure();
  Value tile = toThreadTile(adapted, *type, info, rewriter, loc);
  auto casted = castFloatVectorElementType(tile, resultType.getElementType(),
                                          rewriter, loc);
  if (failed(casted))
    return failure();
  return broadcastLocalVectorToTile(*casted, resultType, rewriter, loc);
}

template <typename FromOp, typename ToOp>
class FriskBinaryOpTiling : public OpConversionPattern<FromOp> {
public:
  using OpConversionPattern<FromOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(FromOp op, typename FromOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto resultInfo = findLowerInfoForValue(op.getResult(), op);
    if (!resultInfo)
      return rewriter.notifyMatchFailure(op, "missing result layout");
    auto resultType = getFullThreadTileType(op.getResult(), *resultInfo);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op, "invalid result thread tile");
    auto lhsInfo = findLowerInfoForValue(op.getLhs(), op);
    auto rhsInfo = findLowerInfoForValue(op.getRhs(), op);
    auto lhs = materializeElementwiseOperand(op.getLhs(), adaptor.getLhs(),
        lhsInfo ? *lhsInfo : *resultInfo, *resultType, rewriter, op.getLoc());
    auto rhs = materializeElementwiseOperand(op.getRhs(), adaptor.getRhs(),
        rhsInfo ? *rhsInfo : *resultInfo, *resultType, rewriter, op.getLoc());
    if (failed(lhs) || failed(rhs))
      return rewriter.notifyMatchFailure(op, "incompatible operand thread tiles");
    auto result = rewriter.create<ToOp>(op.getLoc(), *lhs, *rhs);
    markTiled(result, rewriter);
    Value block = fromThreadTile(result.getResult(), op.getResult().getType(),
                                 *resultInfo, rewriter, op.getLoc());
    rewriter.replaceOp(op, block);
    return success();
  }
};

class Exp2OpTiling : public OpConversionPattern<frisk::Exp2Op> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::Exp2Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto info = findLowerInfoForValue(op.getResult(), op);
    if (!info)
      return rewriter.notifyMatchFailure(op, "missing exp2 layout");
    auto type = getFullThreadTileType(op.getResult(), *info);
    if (failed(type))
      return failure();
    auto inputInfo = findLowerInfoForValue(op.getOperand(), op);
    auto input = materializeElementwiseOperand(op.getOperand(), adaptor.getOperand(),
        inputInfo ? *inputInfo : *info, *type, rewriter, op.getLoc());
    if (failed(input))
      return failure();
    auto result = rewriter.create<math::Exp2Op>(op.getLoc(), *input);
    markTiled(result, rewriter);
    rewriter.replaceOp(op, fromThreadTile(result, op.getResult().getType(),
                                          *info, rewriter, op.getLoc()));
    return success();
  }
};

class ZeroOpTiling : public OpConversionPattern<frisk::ZeroOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::ZeroOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto info = findLowerInfoForValue(op.getResult(), op);
    if (!info)
      return rewriter.notifyMatchFailure(op, "missing zero layout");
    auto type = getFullThreadTileType(op.getResult(), *info);
    if (failed(type))
      return failure();
    auto zero = rewriter.create<arith::ConstantOp>(op.getLoc(), *type,
        cast<TypedAttr>(rewriter.getZeroAttr(*type)));
    markTiled(zero, rewriter);
    rewriter.replaceOp(op, fromThreadTile(zero, op.getResult().getType(),
                                          *info, rewriter, op.getLoc()));
    return success();
  }
};

class FillOpTiling : public OpConversionPattern<frisk::FillOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::FillOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto info = findLowerInfoForValue(op.getMemref(), op);
    if (!info && op.hasValueResult())
      info = findLowerInfoForValue(op.getValueResult(), op);
    if (!info)
      return rewriter.notifyMatchFailure(op, "missing fill layout");
    auto type = getFullThreadTileType(op.getMemref(), *info);
    auto attr = dyn_cast<TypedAttr>(op.getValueAttr());
    if (failed(type) || !attr)
      return failure();
    auto element = convertScalarAttrForElementType(attr, type->getElementType(), rewriter);
    if (!element)
      return failure();
    auto loc = op.getLoc();
    auto memType = MemRefType::get(type->getShape(), type->getElementType());
    Value tile = toThreadTile(adaptor.getMemref(), memType, *info, rewriter, loc);
    Value value = rewriter.create<arith::ConstantOp>(loc, type->getElementType(),
                                                     cast<TypedAttr>(element));
    std::vector<int> bounds(type->getShape().begin(), type->getShape().end());
    std::vector<Value> ivs;
    auto loops = createNestedAffineFor(rewriter, loc, bounds, ivs);
    auto store = rewriter.create<affine::AffineStoreOp>(loc, value, tile, ivs);
    markTiled(store, rewriter);
    rewriter.setInsertionPointAfter(loops.front());
    Value block = fromThreadTile(tile, op.getMemref().getType(), *info, rewriter, loc);
    writeBackBlockTile(block, adaptor.getMemref(), *info, rewriter, loc);
    if (op.hasValueResult())
      rewriter.replaceOp(op, block);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

class AllocBufferOpTiling : public OpConversionPattern<frisk::AllocBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::AllocBufferOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto type = cast<MemRefType>(op.getResult().getType());
    // Shared/global allocations remain physical block storage. Only register
    // allocations change representation, and their users still see block types.
    if (type.getMemorySpaceAsInt() == int(friskMs::Shared) ||
        type.getMemorySpaceAsInt() == int(friskMs::Global)) {
      auto alloc = rewriter.create<memref::AllocOp>(op.getLoc(), type, op.getAlignmentAttr());
      markTiled(alloc, rewriter);
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }
    auto info = findLowerInfoForValue(op.getResult(), op);
    if (!info)
      return rewriter.notifyMatchFailure(op, "missing local allocation layout");
    auto tileType = getFullThreadTileType(op.getResult(), *info);
    if (failed(tileType))
      return failure();
    auto alloc = rewriter.create<memref::AllocaOp>(op.getLoc(),
        MemRefType::get(tileType->getShape(), type.getElementType()), op.getAlignmentAttr());
    markTiled(alloc, rewriter);
    rewriter.replaceOp(op, fromThreadTile(alloc, type, *info, rewriter, op.getLoc()));
    return success();
  }
};

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
/// 对线程持有的每个元素执行 region，经 FromThreadTile 返回原 block 类型。
class MaskOpTiling : public OpConversionPattern<frisk::MaskOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(frisk::MaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto loc = op->getLoc();
    auto resultTy = mlir::dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "mask result must be a memref");
    }
    if (resultTy.getRank() > 2) {
      return rewriter.notifyMatchFailure(
          op, "mask vector lowering currently supports rank <= 2");
    }

    auto resultInfo = findLowerInfoForValue(op.getResult(), op);
    if (!resultInfo)
      return rewriter.notifyMatchFailure(op, "LowerInfo not found for mask");

    auto tileType = getFullThreadTileType(op.getResult(), *resultInfo);
    if (failed(tileType))
      return rewriter.notifyMatchFailure(op, "invalid mask thread tile");
    auto vecTy = *tileType;
    auto vectorShape = vecTy.getShape();
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

    auto tidx = findThreadIdxOp(op, rewriter);

    MaskTileElement element{rewriter, loc,     &*resultInfo, tidx,
                            resultTy, adaptor, body,       yieldOp};
    resultVector =
        VectorTileLoopNest(rewriter, loc, vectorShape, {{}, "mask_tw", {}})
            .emit(resultVector, element);
    rewriter.replaceOp(op, fromThreadTile(resultVector, op.getResult().getType(),
                                          *resultInfo, rewriter, loc));
    return success();
  }
};


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


struct ReduceTileElement {
  ConversionPatternRewriter &rewriter;
  frisk::ReduceOp op;
  Value source;
  VectorType sourceType;
  Value identity;
  int64_t laneExtent;
  int64_t laneStride;
  int64_t warpThreads;

  Value emit(ArrayRef<Value> outputIvs, Value output) {
    auto loc = op.getLoc();
    int64_t dim = op.getDim();
    auto loop = rewriter.create<affine::AffineForOp>(loc, 0,
        sourceType.getDimSize(dim), 1, ValueRange{identity});
    rewriter.setInsertionPointToStart(loop.getBody());
    SmallVector<OpFoldResult, 2> indices;
    unsigned out = 0;
    bool keepDim = outputIvs.size() == static_cast<size_t>(sourceType.getRank());
    for (int64_t i = 0; i < sourceType.getRank(); ++i) {
      if (i == dim) {
        indices.push_back(loop.getInductionVar());
        if (keepDim)
          ++out;
      } else {
        indices.push_back(outputIvs[out++]);
      }
    }
    Value value = rewriter.create<vector::ExtractOp>(loc, source, indices);
    Value combined = *combineReduceValues(op, loop.getRegionIterArgs()[0], value, rewriter);
    rewriter.create<affine::AffineYieldOp>(loc, combined);
    rewriter.setInsertionPointAfter(loop);
    Value reduced = loop.getResult(0);
    for (int64_t offset = 1; offset < laneExtent; offset <<= 1) {
      auto shuffle = rewriter.create<gpu::ShuffleOp>(loc, reduced,
          static_cast<int32_t>(offset * laneStride), static_cast<int32_t>(warpThreads),
          gpu::ShuffleMode::XOR);
      reduced = *combineReduceValues(op, reduced, shuffle.getShuffleResult(), rewriter);
    }
    SmallVector<OpFoldResult, 2> positions(outputIvs.begin(), outputIvs.end());
    return rewriter.create<vector::InsertOp>(loc, reduced, output, positions);
  }
};

class ReduceOpTiling : public OpConversionPattern<frisk::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto sourceInfo = findLowerInfoForValue(op.getSrc(), op);
    auto resultInfo = findLowerInfoForValue(op.getDst(), op);
    if (!sourceInfo || !resultInfo)
      return rewriter.notifyMatchFailure(op, "missing reduce layouts");
    auto sourceType = getFullThreadTileType(op.getSrc(), *sourceInfo);
    auto resultType = getFullThreadTileType(op.getDst(), *resultInfo);
    if (failed(sourceType) || failed(resultType) ||
        !isa<FloatType>(sourceType->getElementType()) ||
        sourceType->getElementType() != resultType->getElementType())
      return rewriter.notifyMatchFailure(op, "requires compatible floating-point tiles");
    int64_t dim = op.getDim();
    if (dim < 0 || dim >= sourceType->getRank() ||
        sourceInfo->get_block_layout()[dim] != 1)
      return rewriter.notifyMatchFailure(op, "reduction axis must lie within one warp");
    int64_t lanes = sourceInfo->get_warp_layout()[dim];
    auto order = sourceInfo->base_layout.warp_layout_order;
    if (lanes <= 0 || !llvm::isPowerOf2_64(lanes) ||
        (order[0] != dim && order[1] != dim))
      return rewriter.notifyMatchFailure(op, "invalid reduction lane layout");
    int64_t stride = order[0] == dim ? 1 : sourceInfo->get_warp_layout()[order[0]];
    // Retain the surviving source axes; a destination's inferred register
    // budget can still describe the unreduced matrix.
    SmallVector<int64_t, 2> shape;
    for (int64_t i = 0; i < sourceType->getRank(); ++i) {
      if (i != dim)
        shape.push_back(sourceType->getDimSize(i));
      else if (resultType->getRank() == sourceType->getRank())
        shape.push_back(1);
    }
    if (shape.empty() || shape.size() != static_cast<size_t>(resultType->getRank()))
      return failure();
    auto dstType = VectorType::get(shape, resultType->getElementType());
    auto identity = createReduceIdentity(op, sourceType->getElementType(), rewriter);
    if (failed(identity))
      return failure();
    auto loc = op.getLoc();
    Value source = toThreadTile(adaptor.getSrc(), *sourceType, *sourceInfo, rewriter, loc);
    Value initial = rewriter.create<arith::ConstantOp>(loc, dstType,
        cast<TypedAttr>(rewriter.getZeroAttr(dstType)));
    ReduceTileElement element{rewriter, op, source, *sourceType, *identity,
                               lanes, stride, sourceInfo->warp_threads};
    Value result = VectorTileLoopNest(rewriter, loc, shape).emit(initial, element);
    Value block = fromThreadTile(result, op.getDst().getType(), *resultInfo, rewriter, loc);
    writeBackBlockTile(block, adaptor.getDst(), *resultInfo, rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyOpTiling : public OpConversionPattern<frisk::CopyOp> {
public:
  explicit CopyOpTiling(MLIRContext *context) : OpConversionPattern(context) {
    // Updating a legacy vector destination can trigger legalization of the
    // next copy immediately. Each invocation erases one original copy and
    // creates none, so this recursion is bounded by the original IR.
    setHasBoundedRewriteRecursion();
  }
  LogicalResult matchAndRewrite(frisk::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto srcTy = dyn_cast<ShapedType>(op.getSrc().getType());
    auto dstTy = dyn_cast<ShapedType>(op.getDst().getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "copy expects static block tiles");
    Value source = adaptor.getSrc();
    Value destination = adaptor.getDst();
    Value layoutValue = op.getSrc();
    auto info = findLowerInfoForValue(layoutValue, op);
    if (!info) {
      layoutValue = op.getDst();
      info = findLowerInfoForValue(layoutValue, op);
    }
    if (!info)
      return rewriter.notifyMatchFailure(op, "missing copy layout");
    auto loc = op.getLoc();
    // The legacy () -> (rank) map denotes a whole-buffer copy. Other maps
    // describe a slice on the larger side, as in the old copy lowering.
    bool sameShape = srcTy.getShape() == dstTy.getShape();
    bool sourceSlice = !sameShape && srcTy.getNumElements() > dstTy.getNumElements();
    auto copyShape = sourceSlice ? dstTy.getShape() : srcTy.getShape();
    auto map = op.getOffsetMap();
    bool sentinel = map.getNumInputs() == 0 && map.getNumResults() == 1 &&
        isa<AffineConstantExpr>(map.getResult(0)) &&
        cast<AffineConstantExpr>(map.getResult(0)).getValue() == dstTy.getRank();
    bool zeroMap = llvm::all_of(map.getResults(), [](AffineExpr e) {
      auto c = dyn_cast<AffineConstantExpr>(e);
      return c && c.getValue() == 0;
    });
    if (!sameShape || (!sentinel && !zeroMap)) {
      auto mappedType = sourceSlice ? srcTy : dstTy;
      if (!isa<MemRefType>(mappedType) ||
          map.getNumResults() != mappedType.getRank() ||
          map.getNumInputs() != adaptor.getMapOperands().size())
        return rewriter.notifyMatchFailure(op, "incompatible copy offset map");
      auto view = rewriter.create<frisk::BufferViewOp>(loc,
          sourceSlice ? source : destination, adaptor.getMapOperands(), map, copyShape);
      markTiled(view, rewriter);
      if (sourceSlice)
        source = view;
      else
        destination = view;
      layoutValue = sourceSlice ? op.getDst() : op.getSrc();
    }
    if (!sameShape && failed(setCopyTileShape(*info, copyShape)))
      return rewriter.notifyMatchFailure(op, "copy slice must cover complete layout tiles");
    auto tileTy = getFullThreadTileType(layoutValue, *info);
    if (failed(tileTy))
      return failure();
    auto sourceTileTy = VectorType::get(tileTy->getShape(), srcTy.getElementType());
    Value tile = toThreadTile(source, sourceTileTy, *info, rewriter, loc);
    auto converted = castFloatVectorElementType(tile, dstTy.getElementType(), rewriter, loc);
    if (failed(converted))
      return rewriter.notifyMatchFailure(op, "unsupported copy element conversion");
    if (isa<VectorType>(dstTy)) {
      if (!sameShape)
        return rewriter.notifyMatchFailure(op, "vector destination requires a whole-tile copy");
      Value block = fromThreadTile(*converted, dstTy, *info, rewriter, loc);
      if (op.hasValueResult()) {
        rewriter.replaceOp(op, block);
      } else {
        // Legacy vector destinations denote an update to subsequent uses in
        // this iteration. Replace only dominated block-tile uses, including
        // yields, while retaining pre-copy reads and loop initial values.
        DominanceInfo dominance(op->getParentOfType<func::FuncOp>());
        rewriter.replaceUsesWithIf(op.getDst(), block, [&](OpOperand &use) {
          return use.getOwner() != op &&
                 dominance.properlyDominates(block.getDefiningOp(), use.getOwner());
        });
        rewriter.eraseOp(op);
      }
      return success();
    }
    auto dstTileTy = MemRefType::get(tileTy->getShape(), dstTy.getElementType());
    Value dstTile = toThreadTile(destination, dstTileTy, *info, rewriter, loc);
    SmallVector<Value> zeros(tileTy->getRank(), createIndexConstant(rewriter, loc, 0));
    auto store = rewriter.create<vector::StoreOp>(loc, *converted, dstTile, zeros);
    markTiled(store, rewriter);
    Value block = fromThreadTile(dstTile, destination.getType(), *info, rewriter, loc);
    writeBackBlockTile(block, destination, *info, rewriter, loc);
    if (op.hasValueResult()) {
      if (block.getType() != op.getValueResult().getType()) {
        // The FromThreadTile above represents only the copied slice. Its
        // writeback has updated the original destination in place, so the
        // copy result aliases that complete block-level memref. Do not tile
        // the enclosing buffer: it may be a rank-4 global tensor, and the
        // slice layout says nothing about elements outside this block's tile.
        block = adaptor.getDst();
      }
      rewriter.replaceOp(op, block);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// copy_to_reg already specifies one thread's contiguous slice. Preserve its
// address map in a view, then express the representation change explicitly.
class CopyToRegOpTiling : public OpConversionPattern<frisk::CopyToRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::CopyToRegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto resultType = cast<VectorType>(op.getResult().getType());
    if (op.getOffsetMap().getNumResults() != srcType.getRank() ||
        op.getOffsetMap().getNumInputs() != adaptor.getMapOperands().size())
      return rewriter.notifyMatchFailure(op, "incompatible copy_to_reg offset map");
    auto view = rewriter.create<frisk::BufferViewOp>(op.getLoc(), adaptor.getSrc(),
        adaptor.getMapOperands(), op.getOffsetMap(), resultType.getShape());
    markTiled(view, rewriter);
    auto tileType = VectorType::get(resultType.getShape(), srcType.getElementType());
    auto tile = rewriter.create<frisk::ToThreadTileOp>(op.getLoc(), tileType, view);
    markTiled(tile, rewriter);
    auto converted = castFloatVectorElementType(tile, resultType.getElementType(), rewriter, op.getLoc());
    if (failed(converted))
      return failure();
    auto block = rewriter.create<frisk::FromThreadTileOp>(op.getLoc(), resultType, *converted);
    markTiled(block, rewriter);
    rewriter.replaceOp(op, block.getResult());
    return success();
  }
};

struct LayoutReadBackElement {
  ConversionPatternRewriter &rewriter;
  Location loc;
  LowerInfo &info;
  Value tid;
  Value scratch;
  Value emit(ArrayRef<Value> ivs, Value output) {
    auto indices = buildMappedAccessIndices(rewriter, loc, info, tid, ivs, ivs.size());
    Value scalar = rewriter.create<affine::AffineLoadOp>(loc, scratch, indices);
    SmallVector<OpFoldResult, 2> position(ivs.begin(), ivs.end());
    return rewriter.create<vector::InsertOp>(loc, scalar, output, position);
  }
};

class ConvertLayoutOpTiling : public OpConversionPattern<frisk::ConvertLayoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto it = convertLayoutInfo.find(op);
    if (it == convertLayoutInfo.end() || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "missing source/destination layouts");
    auto [fromInfo, toInfo] = it->second;
    auto fromTy = getFullThreadTileType(op.getMemref(), fromInfo);
    auto toTy = getFullThreadTileType(op->getResult(0), toInfo);
    if (failed(fromTy) || failed(toTy))
      return failure();
    auto blockTy = cast<MemRefType>(op.getMemref().getType());
    auto loc = op.getLoc();
    Value source = toThreadTile(adaptor.getMemref(), *fromTy, fromInfo, rewriter, loc);
    auto scratchTy = MemRefType::get(blockTy.getShape(), blockTy.getElementType(),
        AffineMap{}, rewriter.getI64IntegerAttr(int(friskMs::Shared)));
    auto scratch = rewriter.create<memref::AllocOp>(loc, scratchTy);
    markTiled(scratch, rewriter);
    Value tid = findThreadIdxOp(op, rewriter);
    Value sourceBlock = fromThreadTile(source, blockTy, fromInfo, rewriter, loc);
    // Reuse the same leader selection and shared-memory barrier as other
    // writebacks; replicated source layouts must not race on scratch stores.
    writeBackBlockTile(sourceBlock, scratch, fromInfo, rewriter, loc);
    Value initial = rewriter.create<arith::ConstantOp>(loc, *toTy,
        cast<TypedAttr>(rewriter.getZeroAttr(*toTy)));
    LayoutReadBackElement element{rewriter, loc, toInfo, tid, scratch};
    Value result = VectorTileLoopNest(rewriter, loc, toTy->getShape()).emit(initial, element);
    // No thread may reuse scratch until all threads have completed their reads.
    auto readSync = rewriter.create<frisk::SyncThreadsInBlockOp>(loc);
    markTiled(readSync, rewriter);
    rewriter.replaceOp(op, fromThreadTile(result, op->getResult(0).getType(),
                                          toInfo, rewriter, loc));
    return success();
  }
};

/// Scalar block bodies use thread memrefs for memory accesses and block
/// coordinates for arithmetic on the original region IVs. IRMapping below is
/// local to cloning this region; it never records replacements across patterns.
class BlockOpTiling : public OpConversionPattern<frisk::BlockOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::BlockOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    struct Buffer {
      Value original;
      LowerInfo info;
      VectorType type;
      Value tile;
      bool written;
    };
    SmallVector<Buffer, 4> buffers;
    Block *body = op.getBody();
    auto addBuffer = [&](Value value, bool written) -> LogicalResult {
      for (auto &buffer : buffers) {
        if (buffer.original == value) {
          buffer.written |= written;
          return success();
        }
      }
      auto info = findLowerInfoForValue(value, op);
      if (!info)
        return failure();
      auto type = getFullThreadTileType(value, *info);
      if (failed(type))
        return failure();
      buffers.push_back({value, *info, *type, {}, written});
      return success();
    };
    for (Operation &child : body->without_terminator()) {
      if (child.getNumRegions())
        return rewriter.notifyMatchFailure(op, "nested block regions are unsupported");
      AffineMap map;
      ValueRange operands;
      if (auto load = dyn_cast<affine::AffineLoadOp>(child)) {
        if (failed(addBuffer(load.getMemref(), false)))
          return rewriter.notifyMatchFailure(op, "missing block input layout");
        map = load.getAffineMap();
        operands = load.getMapOperands();
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(child)) {
        if (failed(addBuffer(store.getMemref(), true)))
          return rewriter.notifyMatchFailure(op, "missing block output layout");
        map = store.getAffineMap();
        operands = store.getMapOperands();
      } else if (!isMemoryEffectFree(&child)) {
        return rewriter.notifyMatchFailure(op, "unsupported side effect in scalar block");
      }
      if (map) {
        if (!map.isProjectedPermutation(/*allowZeroInResults=*/true) ||
            llvm::any_of(operands, [&](Value v) {
              auto arg = dyn_cast<BlockArgument>(v);
              return !arg || arg.getOwner() != body;
            }))
          return rewriter.notifyMatchFailure(op, "block access requires projected region IVs");
        for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
          if (auto axis = dyn_cast<AffineDimExpr>(expr)) {
            if (cast<BlockArgument>(operands[axis.getPosition()]).getArgNumber() != dim)
              return rewriter.notifyMatchFailure(
                  op, "permuted distributed block axes require an explicit layout conversion");
          }
        }
      }
    }
    auto output = llvm::find_if(buffers, [](const Buffer &b) { return b.written; });
    if (output == buffers.end())
      return rewriter.notifyMatchFailure(op, "block has no output");
    auto shape = output->type.getShape();
    if (shape.size() != body->getNumArguments())
      return failure();
    for (auto &buffer : buffers) {
      if (buffer.type.getRank() != static_cast<int64_t>(shape.size()))
        return rewriter.notifyMatchFailure(op, "incompatible block tile ranks");
      for (unsigned i = 0; i < shape.size(); ++i)
        if (buffer.type.getDimSize(i) != shape[i] && buffer.type.getDimSize(i) != 1)
          return rewriter.notifyMatchFailure(op, "incompatible block tile extents");
    }
    auto loc = op.getLoc();
    for (auto &buffer : buffers) {
      auto type = MemRefType::get(buffer.type.getShape(), buffer.type.getElementType());
      Value adapted = rewriter.getRemappedValue(buffer.original);
      if (!adapted)
        return failure();
      buffer.tile = toThreadTile(adapted, type, buffer.info, rewriter, loc);
    }
    Value tid = findThreadIdxOp(op, rewriter);
    std::vector<int> bounds(shape.begin(), shape.end());
    std::vector<Value> ivs;
    auto loops = createNestedAffineFor(rewriter, loc, bounds, ivs);
    markTiled(loops.front(), rewriter);
    auto blockIndices = buildMappedAccessIndices(rewriter, loc, output->info, tid, ivs, ivs.size());
    IRMapping scalarMapping;
    for (auto [arg, index] : llvm::zip(body->getArguments(), blockIndices))
      scalarMapping.map(arg, index);
    auto getBuffer = [&](Value original) -> Buffer & {
      return *llvm::find_if(buffers, [&](const Buffer &b) { return b.original == original; });
    };
    auto access = [&](AffineMap map, ValueRange operands, Buffer &buffer) {
      SmallVector<Value, 2> mappedOperands, indices;
      for (Value value : operands)
        mappedOperands.push_back(ivs[cast<BlockArgument>(value).getArgNumber()]);
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        if (buffer.type.getDimSize(i) == 1) {
          indices.push_back(createIndexConstant(rewriter, loc, 0));
        } else {
          auto single = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
              map.getResult(i), rewriter.getContext());
          indices.push_back(rewriter.create<affine::AffineApplyOp>(loc, single, mappedOperands));
        }
      }
      return indices;
    };
    for (Operation &child : body->without_terminator()) {
      if (auto load = dyn_cast<affine::AffineLoadOp>(child)) {
        auto &buffer = getBuffer(load.getMemref());
        auto indices = access(load.getAffineMap(), load.getMapOperands(), buffer);
        Value value = rewriter.create<affine::AffineLoadOp>(loc, buffer.tile, indices);
        scalarMapping.map(load.getResult(), value);
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(child)) {
        auto &buffer = getBuffer(store.getMemref());
        auto indices = access(store.getAffineMap(), store.getMapOperands(), buffer);
        rewriter.create<affine::AffineStoreOp>(loc,
            scalarMapping.lookupOrDefault(store.getValueToStore()), buffer.tile, indices);
      } else {
        rewriter.clone(child, scalarMapping);
      }
    }
    rewriter.setInsertionPointAfter(loops.front());
    for (auto &buffer : buffers) {
      if (!buffer.written)
        continue;
      Value block = fromThreadTile(buffer.tile, buffer.original.getType(), buffer.info, rewriter, loc);
      writeBackBlockTile(block, rewriter.getRemappedValue(buffer.original), buffer.info, rewriter, loc);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// 从完整 thread tile 提取一个 MMA 片段。MN 为静态片段编号，K 为循环 IV；
/// vector.extract_strided_slice 不支持动态 offset，因此逐寄存器提取/组装。
static Value extractGemmTileFragment(OpBuilder &builder, Location loc,
                                     Value tile, VectorType fragmentType,
                                     int64_t mn, Value k, bool isA) {
  Value fragment = builder.create<arith::ConstantOp>(
      loc, fragmentType, cast<TypedAttr>(builder.getZeroAttr(fragmentType)));
  auto d0 = builder.getAffineDimExpr(0);
  for (int64_t row = 0; row < fragmentType.getDimSize(0); ++row) {
    for (int64_t col = 0; col < fragmentType.getDimSize(1); ++col) {
      int64_t staticIndex =
          isA ? mn * fragmentType.getDimSize(0) + row
              : mn * fragmentType.getDimSize(1) + col;
      Value dynamicIndex = createSingleDimAffineApply(
          builder, loc,
          isA ? d0 * fragmentType.getDimSize(1) + col
              : d0 * fragmentType.getDimSize(0) + row,
          k);
      SmallVector<int64_t, 2> position =
          isA ? SmallVector<int64_t, 2>{staticIndex, ShapedType::kDynamic}
              : SmallVector<int64_t, 2>{ShapedType::kDynamic, staticIndex};
      Value element = builder.create<vector::ExtractOp>(
          loc, fragmentType.getElementType(), tile, ValueRange{dynamicIndex},
          builder.getDenseI64ArrayAttr(position));
      fragment = builder.create<vector::InsertOp>(
          loc, element, fragment, ArrayRef<int64_t>{row, col});
    }
  }
  return fragment;
}

/// 只替换 block-tile SSA 值，thread-tile 的数据流完全由 IR 显式表达。
class GemmOpTiling : public OpConversionPattern<frisk::GemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(frisk::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr("tiled")) {
      return failure();
    }
    if (!s_info || !s_hw || s_hw->getKind() != HW_KIND_DCU) {
      return rewriter.notifyMatchFailure(op, "requires DCU layout analysis");
    }
    auto *infoA = s_info->getLowerInfo(op.getA(), op);
    auto *infoB = s_info->getLowerInfo(op.getB(), op);
    auto *infoC = s_info->getLowerInfo(op.getC(), op);
    if (!infoA || !infoB || !infoC || !infoA->mmaInst || !infoB->mmaInst ||
        !infoC->mmaInst) {
      return rewriter.notifyMatchFailure(op, "missing GEMM operand layouts");
    }
    auto instName = op->getAttrOfType<StringAttr>("inst_name");
    auto constraints = op->getAttrOfType<StringAttr>("inst_constraints");
    if (!instName || !constraints ||
        infoA->mmaInst->asm_str != infoB->mmaInst->asm_str ||
        infoA->mmaInst->asm_str != infoC->mmaInst->asm_str) {
      return rewriter.notifyMatchFailure(op, "inconsistent MMA instruction");
    }
    auto aType = dyn_cast<MemRefType>(op.getA().getType());
    auto bType = dyn_cast<MemRefType>(op.getB().getType());
    auto cType = dyn_cast<MemRefType>(op.getC().getType());
    if (!aType || !bType || !cType || aType.getRank() != 2 ||
        bType.getRank() != 2 || cType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "requires rank-two block tiles");
    }

    auto aCounts = infoA->get_block_repeat() * infoA->warpInstUnroll;
    auto bCounts = infoB->get_block_repeat() * infoB->warpInstUnroll;
    auto cCounts = infoC->get_block_repeat() * infoC->warpInstUnroll;
    auto aFragmentShape = infoA->get_thread_widths() * infoA->get_warp_repeat();
    auto bFragmentShape = infoB->get_thread_widths() * infoB->get_warp_repeat();
    auto cFragmentShape = infoC->get_thread_widths() * infoC->get_warp_repeat();
    for (unsigned dim = 0; dim < 2; ++dim) {
      if (aCounts[dim] <= 0 || bCounts[dim] <= 0 || cCounts[dim] <= 0 ||
          aFragmentShape[dim] <= 0 || bFragmentShape[dim] <= 0 ||
          cFragmentShape[dim] <= 0) {
        return rewriter.notifyMatchFailure(op, "invalid MMA tile dimensions");
      }
    }
    if (aCounts[1] != bCounts[0] || aCounts[0] != cCounts[0] ||
        bCounts[1] != cCounts[1] ||
        cCounts * cFragmentShape != infoC->get_thread_own_data_size()) {
      return rewriter.notifyMatchFailure(op, "incompatible MMA tile counts");
    }

    // A/B 的旧 thread_own_data_size 不含 block_repeat（复用寄存器的尺寸）。
    // ToThreadTile 在循环外产生完整 SSA tile，必须包含全部 MN/K 片段。
    auto threadTileA = VectorType::get(aCounts * aFragmentShape,
                                       aType.getElementType());
    auto threadTileB = VectorType::get(bCounts * bFragmentShape,
                                       bType.getElementType());
    auto threadTileC = VectorType::get(infoC->get_thread_own_data_size(),
                                       cType.getElementType());
    auto fragmentA = VectorType::get(aFragmentShape, aType.getElementType());
    auto fragmentB = VectorType::get(bFragmentShape, bType.getElementType());
    auto fragmentC = VectorType::get(cFragmentShape, cType.getElementType());
    auto loc = op.getLoc();
    Value ttileA = toThreadTile(adaptor.getA(), threadTileA, *infoA, rewriter, loc);
    Value ttileB = toThreadTile(adaptor.getB(), threadTileB, *infoB, rewriter, loc);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, threadTileC, cast<TypedAttr>(rewriter.getZeroAttr(threadTileC)));
    Value zeroFragment = rewriter.create<arith::ConstantOp>(
        loc, fragmentC, cast<TypedAttr>(rewriter.getZeroAttr(fragmentC)));

    // 静态展开 MN，K 循环通过 iter_args/yield 累加一个 MMA 片段。
    // GEMM 没有输入 C：每个片段都从零开始，不读取尚未定义的 op.getC()。
    for (int64_t m = 0; m < cCounts[0]; ++m) {
      for (int64_t n = 0; n < cCounts[1]; ++n) {
        auto kFor = rewriter.create<affine::AffineForOp>(
            loc, 0, aCounts[1], 1, ValueRange{zeroFragment});
        kFor->setAttr("iterLabel", rewriter.getStringAttr("k"));
        rewriter.setInsertionPointToStart(kFor.getBody());
        Value a = extractGemmTileFragment(rewriter, loc, ttileA, fragmentA,
                                          m, kFor.getInductionVar(), true);
        Value b = extractGemmTileFragment(rewriter, loc, ttileB, fragmentB,
                                          n, kFor.getInductionVar(), false);
        auto mma = rewriter.create<frisk::WarpMmaRROp>(
            loc, fragmentC, a, b, kFor.getRegionIterArgs()[0]);
        markTiled(mma, rewriter);
        mma->setAttr("inst_name", instName);
        mma->setAttr("inst_constraints", constraints);
        rewriter.create<affine::AffineYieldOp>(loc, mma.getResult());
        rewriter.setInsertionPointAfter(kFor);
        auto insert = rewriter.create<vector::InsertStridedSliceOp>(
            loc, kFor.getResult(0), result,
            ArrayRef<int64_t>{m * cFragmentShape[0], n * cFragmentShape[1]},
            ArrayRef<int64_t>{1, 1});
        insert->setAttr("frisk.mma_fragment", rewriter.getUnitAttr());
        result = insert.getResult();
      }
    }

    Value newBlockTile = fromThreadTile(result, op.getC().getType(), *infoC, rewriter, loc);
    rewriter.replaceOp(op, newBlockTile);
    return success();
  }
};


static bool isBlockTileOperation(Operation *op) {
  return isa<frisk::GemmOp, frisk::MaskOp, frisk::AddOp, frisk::SubOp,
      frisk::MulOp, frisk::DivOp, frisk::Exp2Op, frisk::CopyOp,
      frisk::CopyToRegOp, frisk::FillOp, frisk::ZeroOp, frisk::ReduceOp,
      frisk::BlockOp, frisk::ConvertLayoutOp, frisk::AllocBufferOp>(op);
}

// Fold ToThreadTile(FromThreadTile(threadTile)) without changing block-level
// users of the same FromThreadTile. Equal types alone do not imply equal lane
// ownership, so only cancel bridges describing the same layout.
static void foldThreadTilePairs(Operation *root) {
  static constexpr StringLiteral layoutAttrs[] = {
      "tile_layout", "thread_widths", "thread_creg_order", "warp_repeat",
      "warp_repeat_order", "warp_layout", "warp_layout_order",
      "warp_inst_unroll", "block_repeat", "block_layout", "block_layout_order",
      "warp_threads", "ignore_dim"};
  bool changed;
  do {
    changed = false;
    SmallVector<frisk::ToThreadTileOp> candidates;
    root->walk([&](frisk::ToThreadTileOp to) { candidates.push_back(to); });
    for (auto to : candidates) {
      auto from = to.getBlockTile().getDefiningOp<frisk::FromThreadTileOp>();
      if (!from)
        continue;
      if (llvm::any_of(layoutAttrs, [&](StringRef name) {
            return from->getAttr(name) != to->getAttr(name);
          }))
        continue;
      Value replacement = from.getThreadTile();
      if (replacement.getType() != to.getResult().getType()) {
        auto memrefType = dyn_cast<MemRefType>(replacement.getType());
        auto vectorType = dyn_cast<VectorType>(to.getResult().getType());
        if (!memrefType || !vectorType || vectorType.isScalable() ||
            memrefType.getShape() != vectorType.getShape() ||
            memrefType.getElementType() != vectorType.getElementType() ||
            !memrefType.isLastDimUnitStride())
          continue;
        // A thread memref is storage, not an SSA vector. Read it at the To
        // operation so writes between From and To remain visible.
        OpBuilder builder(to);
        SmallVector<Value> zeros;
        for (int64_t dim = 0; dim < memrefType.getRank(); ++dim)
          zeros.push_back(createIndexConstant(builder, to.getLoc(), 0));
        auto load = builder.create<vector::LoadOp>(
            to.getLoc(), vectorType, replacement, zeros);
        markTiled(load, builder);
        replacement = load.getResult();
      }
      to.getResult().replaceAllUsesWith(replacement);
      to.erase();
      if (from.getResult().use_empty())
        from.erase();
      changed = true;
    }
    // Folding can expose another pair, including across blocks whose textual
    // order differs from their dominance order.
  } while (changed);
}

class ConvertFriskBaseToThreadLevelIR
    : public impl::ConvertFriskBaseToThreadLevelIRBase<ConvertFriskBaseToThreadLevelIR> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<frisk::FriskDialect, arith::ArithDialect, affine::AffineDialect,
        vector::VectorDialect, math::MathDialect, memref::MemRefDialect,
        gpu::GPUDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto kernel = getOperation();
    if (!kernel->hasAttr("thread_num"))
      return;
    eraseTriviallyDeadOps(kernel);
    bool needsTiling = false;
    bool needsLayoutAnalysis = false;
    bool hasLayoutAnchor = false;
    kernel.walk([&](Operation *op) {
      hasLayoutAnchor |= isa<frisk::GemmOp>(op);
      if (!isBlockTileOperation(op) || isTiled(op))
        return;
      needsTiling = true;
      if (isa<frisk::CopyToRegOp>(op))
        return;
      if (auto alloc = dyn_cast<frisk::AllocBufferOp>(op)) {
        auto space = cast<MemRefType>(alloc.getResult().getType()).getMemorySpaceAsInt();
        if (space == int(friskMs::Shared) || space == int(friskMs::Global))
          return;
      }
      needsLayoutAnalysis = true;
    });
    // Running the pass again must not re-analyze or re-tile thread operations.
    if (!needsTiling) {
      foldThreadTilePairs(kernel);
      return;
    }
    auto *context = &getContext();
    s_hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, context);
    convertLayoutInfo.clear();
    // CopyToReg already supplies its complete per-thread slice. Do not invoke
    // the GEMM-anchored inference for operations that need no layout analysis.
    if (needsLayoutAnalysis && !hasLayoutAnchor) {
      kernel.emitError("thread tiling requires a GEMM anchor for layout inference");
      signalPassFailure();
      return;
    }
    LowerInfoMap emptyInfo;
    s_info = needsLayoutAnalysis ? LowerInfoAnalysis::run(kernel) : &emptyInfo;
    if (!s_info) {
      signalPassFailure();
      return;
    }
    if (s_info->begin() != s_info->end()) {
      const auto &info = s_info->begin()->second;
      kernel->setAttr("warp_layout", DenseI64ArrayAttr::get(context, info.get_warp_layout()));
      kernel->setAttr("block_layout", DenseI64ArrayAttr::get(context, info.get_block_layout()));
      kernel->setAttr("block_layout_order", DenseI64ArrayAttr::get(context, info.get_block_layout_order()));
    }
    insertConvertLayoutOps(*s_info);
    llvm::outs() << "----- after insertConvertLayoutOps:\n" << kernel << "\n"; llvm::outs().flush();

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isBlockTileOperation(op) || isTiled(op);
    });
    RewritePatternSet patterns(context);
    patterns.add<GemmOpTiling, MaskOpTiling, Exp2OpTiling, FillOpTiling,
        ZeroOpTiling, AllocBufferOpTiling, CopyOpTiling, CopyToRegOpTiling,
        ReduceOpTiling, BlockOpTiling, ConvertLayoutOpTiling>(context);
    patterns.add<FriskBinaryOpTiling<frisk::AddOp, arith::AddFOp>,
        FriskBinaryOpTiling<frisk::SubOp, arith::SubFOp>,
        FriskBinaryOpTiling<frisk::MulOp, arith::MulFOp>,
        FriskBinaryOpTiling<frisk::DivOp, arith::DivFOp>>(context);
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      convertLayoutInfo.clear();
      s_info = nullptr;
      signalPassFailure();
      return;
    }
    convertLayoutInfo.clear();
    s_info = nullptr;

    // 用 FromThreadTile 的 threadTile 替换配对的 ToThreadTile 结果；
    // FromThreadTile 仍有 block-tile 使用者时必须保留。
    foldThreadTilePairs(kernel);

    eraseTriviallyDeadOps(kernel);
    promoteSingleIterationAffineFors(kernel);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass() {
  return std::make_unique<ConvertFriskBaseToThreadLevelIR>();
}

} // namespace mlir::frisk
