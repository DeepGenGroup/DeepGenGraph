//===----------------------------------------------------------------------===//
// Frisk block-level -> thread-level tiling (plan2).
// Every pattern checks `tiled`, consumes explicit ToThreadTile bridges and
// produces block tiles through FromThreadTile. Loops keep their block-level
// signatures. Destination-style operations retain explicit writebacks at the
// original program point; slice-copy results alias the complete destination.
// Bridge elimination is a later stage.
//===----------------------------------------------------------------------===//

// 中文阅读导航：block tile -> thread tile -> fragment
// 配套文档：同目录 FriskBaseToTheadlevelIR_plan2_中文说明.md。
//
// 【概念】block tile 是整个线程块协作处理的逻辑矩阵；thread tile 是当前线程
// 实际持有的元素集合，在这里打包成小 vector，或放在线程私有 memref 中。
// thread tile 的相邻下标未必对应 block 矩阵中的相邻元素，必须结合布局解释。
// fragment 是 thread tile 中一次计算/搬运处理的寄存器小片段；MMA 的 fragment
// 只是单条 warp 指令在当前 lane 上的操作数，warp 内所有 lane 协作完成该指令。
// 普通二维计算布局中，逐维有：
//   fragmentShape = thread_widths * warp_repeat
//   fragmentCount = block_repeat * warpInstUnroll
//   threadTileShape = fragmentCount * fragmentShape
// 例如 thread tile 为 2x32、fragment 为 1x4，片段网格为 2x8，共 16 次迭代。
// fragment 内偏移是静态的；加上当前片段 origin 得到 thread tile 下标，
// 再结合 tid 和 LowerInfo 才能得到 block tile 的逻辑坐标。
//
// 【第一阶段】ConvertFriskBaseToThreadLevelIR：
//   LowerInfoAnalysis 推断每个 (Value, 使用它的 Operation) 的布局；
//   插入必要的 ConvertLayoutOp；各 pattern 将计算改写为线程级计算；
//   用 ToThreadTile(block) 读入，用 FromThreadTile(thread) 保留原 block 类型；
//   在线程 tile 内生成 fragment 循环和 load/compute/store region；
//   尽量折叠布局一致的 To(From(...))，暂时保留外层循环的 block 类型接口。
// 【第二阶段】FinalizeThreadTiling：
//   调整可转换的 affine 循环携带值，折叠桥接，展开剩余桥接为实际访存；
//   处理 buffer_view、cast 和私有暂存，再检查中间桥接是否全部消除。
//   此时仍保留 fragment 边界供后续融合/预取；lower-frisk-fragments 再内联
//   region、规划共享内存复用并按预算展开标记循环，它不在本文件中实现。
//
// 【LowerInfo 的三个用途】
//   1. 决定线程 tile 形状：thread_widths * warp_repeat * warpInstUnroll * block_repeat。
//   2. 决定元素归属：(threadIdx.x, 线程内下标) -> block 内逻辑坐标。
//   3. 判断生产者/消费者布局是否相容；不相容时通过共享内存交换数据。
// 以上乘法均为二维数组逐维相乘；单例轴、归约轴需要额外投影。
//
// 建议阅读顺序：两个 pass 的 runOnOperation -> getFullThreadTileType ->
// buildMappedAccessIndices -> FragmentTileLoopNest -> GemmOpTiling -> writeBackBlockTile ->
// ThreadTileAccess / ToThreadTileFinalization。桥接本身不等于内存写回，
// FromThreadTile 也不表示已经把所有线程的数据实际拼成一个大矩阵。
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "SharedBufferAliases.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

using friskMs = frisk::attr::MemorySpace;

// Only layout analysis is shared. All block/thread value transitions live in IR.
// 第一阶段使用的布局分析入口；key 是 (Value, Operation*)，不是单独一个 Value。
// 同一逻辑矩阵在不同消费者处可能需要不同分布，所以不能只按 SSA 值缓存布局。
// 第一阶段完成后清空 s_info；第二阶段依靠桥接属性恢复信息，不依赖旧分析对象。
static LowerInfoMap *s_info = nullptr;
static HWSpecification *s_hw = nullptr;
// 记录显式 ConvertLayoutOp 的源布局和目标布局，供该 op 的重写使用。
// 这里存的是布局描述，不是 block SSA 值到 thread SSA 值的替换表。
static DenseMap<frisk::ConvertLayoutOp, std::pair<LowerInfo, LowerInfo>>
    convertLayoutInfo;

// 判断是否已经处理过。当前实现只检查 tiled 属性是否存在，
// 即使属性值为 false 也会视为已处理；不要把它当作普通布尔条件读取。
static bool isTiled(Operation *op) { return op->hasAttr("tiled"); }
// 给新生成且不应再次切分的操作设置 tiled=true，使重复执行转换保持稳定。
static void markTiled(Operation *op, OpBuilder &builder) {
  op->setAttr("tiled", builder.getBoolAttr(true));
}

// 沿 buffer_view 的 source 链找到底层 buffer，用于识别真实内存空间。
// 这里只追踪来源，不计算偏移；实际坐标合成由 resolveViewAccess 完成。
// Stop at memref.transpose: its strided type is needed by copy vectorization
// and shared packing to avoid treating a transposed logical row as contiguous.
static Value getViewBuffer(Value value) {
  while (auto view = value.getDefiningOp<frisk::BufferViewOp>())
    value = view.getSource();
  return value;
}

// Which logical tile axis becomes the physical minor axis after folding all
// views/transposes? A stride-one axis alone is not enough for vector.load:
// it must become the final axis of the memref used by that operation.
static std::optional<unsigned> getPhysicalMinorCopyAxis(Value source) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || type.getRank() != 2)
    return std::nullopt;
  for (unsigned axis : {0u, 1u}) {
    Value buffer = source;
    unsigned mappedAxis = axis;
    while (true) {
      if (auto view = buffer.getDefiningOp<frisk::BufferViewOp>()) {
        mappedAxis += view.getSourceType().getRank() - view.getViewType().getRank();
        buffer = view.getSource();
      } else if (auto transpose = buffer.getDefiningOp<memref::TransposeOp>()) {
        mappedAxis = cast<AffineDimExpr>(
            transpose.getPermutation().getResult(mappedAxis)).getPosition();
        buffer = transpose.getIn();
      } else {
        break;
      }
    }
    auto rootType = cast<MemRefType>(buffer.getType());
    if (mappedAxis + 1 == unsigned(rootType.getRank()) &&
        rootType.isLastDimUnitStride())
      return axis;
  }
  return std::nullopt;
}

// 识别 global -> shared 搬运的特殊路径；忽略外层 view 后检查两端内存空间。
// 该路径按连续区间分工，不使用 MMA 的 LowerInfo 分布。
static bool isGlobalToSharedCopy(frisk::CopyOp op) {
  auto src = dyn_cast<MemRefType>(getViewBuffer(op.getSrc()).getType());
  auto dst = dyn_cast<MemRefType>(getViewBuffer(op.getDst()).getType());
  return src && dst && src.getMemorySpaceAsInt() == int(friskMs::Global) &&
         dst.getMemorySpaceAsInt() == int(friskMs::Shared);
}

// Plan a physical layout only for owned, non-escaping shared GEMM operands whose
// complete producer/consumer set is understood. This is independent of kernel
// names, attention shapes and surrounding graph operators. A logical memref
// remains row-major until pack-shared-memory rewrites ALL accesses together.
static void planSharedOperandPacking(func::FuncOp kernel, LowerInfoMap &layouts) {
  auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
  if (!threads || threads.getInt() <= 0)
    return;
  llvm::SmallPtrSet<Operation *, 16> visited;
  kernel.walk([&](frisk::AllocBufferOp alloc) {
    if (visited.contains(alloc))
      return;
    auto type = cast<MemRefType>(alloc.getResult().getType());
    if (type.getRank() != 2 || !type.hasStaticShape() ||
        !type.getLayout().isIdentity() ||
        type.getMemorySpaceAsInt() != int(friskMs::Shared))
      return;
    SharedBufferAliases aliases;
    if (!aliases.collect(alloc.getResult()))
      return;
    for (Operation *storage : aliases.allocations)
      visited.insert(storage);
    int64_t width = 0, pairStride = 0, axis = -1, instructionStride = 0;
    SmallVector<frisk::CopyOp> writers;
    for (OpOperand *use : aliases.accesses) {
      Operation *user = use->getOwner();
      Value buffer = use->get();
      if (auto gemm = dyn_cast<frisk::GemmOp>(user)) {
        auto *info = layouts.getLowerInfo(buffer, gemm);
        if (!info || !info->mmaInst || gemm.getA() == gemm.getB())
          return;
        // Infer the contiguous register axis from the operand layout. Packing
        // along columns interleaves A's row packets, avoiding the large row
        // stride between lanes. The same mechanism applies to ordinary GEMM
        // and fused producers; it does not identify P or attention by name.
        int64_t candidateAxis;
        if (gemm.getB() == buffer && info->get_thread_widths()[1] == 1)
          candidateAxis = 0;
        else if (gemm.getA() == buffer && info->get_thread_widths()[0] == 1)
          candidateAxis = 1;
        else
          return;
        int64_t candidate = info->get_thread_widths()[candidateAxis];
        if (candidate <= 0)
          return;
        // Pair the same lane's fragments from two K instructions, rather
        // than the fragments owned by adjacent lanes. The latter would make
        // an eight-element packet contain another lane's operands.
        int64_t stride = info->get_warpInst_widths()[candidateAxis];
        int64_t candidateStride =
            candidateAxis == 0 &&
            candidate * 2 * type.getElementTypeBitWidth() == 128 &&
            stride >= candidate && stride % candidate == 0 &&
            type.getDimSize(0) % (2 * stride) == 0 &&
            info->get_block_layout()[0] == 1 ? stride : 0;
        if (width && (width != candidate || pairStride != candidateStride || axis != candidateAxis))
          return;
        width = candidate;
        axis = candidateAxis;
        pairStride = candidateStride;
        if (instructionStride && instructionStride != stride)
          return;
        instructionStride = stride;
      } else if (auto copy = dyn_cast<frisk::CopyOp>(user)) {
        // Offset/sliced and escaping copy aliases conservatively keep their
        // original layout. Buffer-view origins on the source are supported.
        auto map = copy.getOffsetMap();
        auto sourceType = dyn_cast<MemRefType>(copy.getSrc().getType());
        if (use->getOperandNumber() != 1 || !sourceType ||
            sourceType.getShape() != type.getShape() ||
            map.getNumInputs() != 0 || map.getNumResults() != 1 ||
            !isa<AffineConstantExpr>(map.getResult(0)) ||
            cast<AffineConstantExpr>(map.getResult(0)).getValue() != type.getRank())
          return;
        writers.push_back(copy);
      } else {
        return;
      }
    }
    if (writers.empty() || width < 2 || !llvm::isPowerOf2_64(width) ||
        width * type.getElementTypeBitWidth() > 128 ||
        type.getDimSize(axis) % width != 0 ||
        (axis == 0 && (type.getNumElements() % threads.getInt() != 0 ||
         (type.getNumElements() / threads.getInt()) % width != 0)))
      return;
    // Both row-major and column-major GLOBAL inputs can feed row-axis packing.
    // The copy chooses ownership using physical strides; GEMM LowerInfo above
    // still determines the shared packet axis/width and all consumer addresses.
    if (axis == 0 && llvm::any_of(writers, [](frisk::CopyOp copy) {
          return !isGlobalToSharedCopy(copy) ||
              !getPhysicalMinorCopyAxis(copy.getSrc());
        }))
      return;
    auto set = [&](StringRef name, int64_t value) {
      for (Operation *storage : aliases.allocations)
        storage->setAttr(name, IntegerAttr::get(
            IntegerType::get(kernel.getContext(), 64), value));
    };
    set("frisk.shared_pack_axis", axis);
    set("frisk.shared_pack", width);
    // A computed producer can own strided columns even though the consumer
    // reads contiguous register fragments. Put the producer's lane residue
    // outside the row dimension and pair the consumer's successive K packets.
    // Both sides then have contiguous subsets, without changing lane ownership.
    if (axis == 1 && instructionStride > 0 &&
        type.getDimSize(1) % (2 * instructionStride) == 0) {
      int64_t storeStride = 0;
      bool compatible = llvm::all_of(writers, [&](frisk::CopyOp copy) {
        auto *producer = copy.getSrc().getDefiningOp();
        auto *info = layouts.getLowerInfo(copy.getSrc(), copy);
        if (!producer || !producer->hasTrait<frisk::ComputedTile>() || !info ||
            info->get_thread_widths()[1] != 1 ||
            info->get_block_layout()[1] != 1 ||
            info->base_layout.warp_layout_order[0] != 0)
          return false;
        int64_t stride = info->get_warp_layout()[1];
        if (stride < 2 || instructionStride % stride ||
            !llvm::isPowerOf2_64(stride) || (storeStride && storeStride != stride))
          return false;
        storeStride = stride;
        return true;
      });
      if (compatible) {
        set("frisk.shared_store_stride", storeStride);
        set("frisk.shared_store_pair", instructionStride);
      }
    }
    if (pairStride && (type.getNumElements() / threads.getInt()) % (2 * width) == 0)
      set("frisk.shared_pair_stride", pairStride);
  });
}

// 把 affine map 应用到 operands，并合成已有 affine.apply，返回单个索引值。
// 用于将切片偏移与局部坐标拼成底层 buffer 的访问坐标。
static Value composeAccessIndex(OpBuilder &b, Location loc, AffineMap map,
                                ValueRange operands) {
  return affine::makeComposedAffineApply(
      b, loc, map, llvm::to_vector_of<OpFoldResult>(operands));
}

// 上一个函数的单表达式版本：将 expr 包装成无 symbol 的单结果 map。
static Value composeAccessIndex(OpBuilder &b, Location loc, AffineExpr expr,
                                ValueRange operands) {
  return composeAccessIndex(
      b, loc, AffineMap::get(operands.size(), 0, expr, b.getContext()), operands);
}

// A buffer_view is an index mapping, not a physical memref descriptor.
// Its map gives the source origin; local coordinates occupy the trailing
// source axes. Compose views and transpose permutations out to the input.
// 将 view 内 indices 原地改写为真实 buffer 的 indices，同时更新 buffer。
// 每层 view 的 index_map 给出切片起点；局部坐标叠加到源的末尾若干轴。
// 例如二维 tile 位于四维张量 [batch, head, row0, col0]，局部 [i,j]
// 变为 [batch, head, row0+i, col0+j]；嵌套 view 从内向外依次组合。
// 保留原 buffer 类型，因此真实行跨度不会被误写成切片的宽度。
static void resolveViewAccess(OpBuilder &b, Location loc, Value &buffer,
                              SmallVectorImpl<Value> &indices) {
  while (true) {
    if (auto transpose = buffer.getDefiningOp<memref::TransposeOp>()) {
      SmallVector<Value> mapped(indices.size());
      for (auto [dim, expr] : llvm::enumerate(transpose.getPermutation().getResults()))
        mapped[cast<AffineDimExpr>(expr).getPosition()] = indices[dim];
      indices.assign(mapped.begin(), mapped.end());
      buffer = transpose.getIn();
      continue;
    }
    auto view = buffer.getDefiningOp<frisk::BufferViewOp>();
    if (!view)
      break;
    auto map = view.getIndexMap();
    unsigned leading = map.getNumResults() - indices.size();
    SmallVector<Value> mapped;
    for (unsigned dim = 0; dim < map.getNumResults(); ++dim) {
      Value origin = composeAccessIndex(
          b, loc, map.getSubMap({dim}), view.getIndices());
      if (dim >= leading)
        origin = composeAccessIndex(
            b, loc, b.getAffineDimExpr(0) + b.getAffineDimExpr(1),
            ValueRange{origin, indices[dim - leading]});
      mapped.push_back(origin);
    }
    indices.assign(mapped.begin(), mapped.end());
    buffer = view.getSource();
  }
}

// 查询 value 在 consumer 处的布局，按以下顺序寻找：
//   精确 (value, consumer) -> value 的其他用户 -> 表中同 buffer 的条目。
// 后两项是兼容改写过程中旧分析键失配的回退，不代表求出了新的最优布局。
// 若分析表不可用或未命中，尝试从定义 value 的 FromThreadTile 属性恢复。
// 返回 LowerInfo 副本，便于调用方调整切片/广播轴而不修改共享分析结果。
// 桥接恢复只重建地址映射所需字段，不恢复 mmaInst、convertFrom 等分析关系。
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
// 输入 block tile 及其布局，返回当前线程持有“完整 tile”的 vector 类型。
// 对普通二维轴：T[d] = thread_widths[d] * warp_repeat[d] *
//                         warpInstUnroll[d] * block_repeat[d]。
// warp_layout、block_layout 不直接乘入 T，因为它们分配的是其他线程/warp 的工作。
// 不能直接使用旧 thread_own_data_size：GEMM A/B 的该字段可能只预算一次 MMA。
// 原 shape 为 1 的轴或二维 ignoreDim 轴压到 1；一维且 ignoreDim=0 时，
// 唯一的物理轴对应布局的第 1 轴。只接受静态 rank 1/2 和正的有效大小。
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
// 将已知 buffer 的布局调整到实际 copy 切片的大小，只更新副本 info。
// 保持 lane/warp 分工不变，用 shape[d] / get_block_widths()[axis] 重算 block_repeat。
// 切片必须整除一个布局覆盖单元；单例轴设置 ignoreDim，变成非单例时取消该轴忽略。
// 因此这个函数不支持任意尾块补零或带掩码的不整除切片。
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
// 把 LowerInfo 的地址映射及各层布局序列化到桥接 op 上。
// tile_layout 保存最终七输入、二输出 affine map；其余属性保留分解参数，
// 用于从线程内 packed 下标构造 map 操作数、比较布局、或恢复 LowerInfo。
// 相同 vector shape 不意味着 lane 所有权相同，不能只记录形状。
// getAffineMap 会更新 LowerInfo 内部辅助字段，所以这里对副本调用。
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

// 创建 block -> thread 的显式表示边界，并附上完整布局属性。
// tileType 决定返回 SSA vector 还是可写的线程 memref；此时尚未展开实际 load。
// 后续可能与 FromThreadTile 抵消，也可能在 finalization 生成取数/布局交换。
static Value toThreadTile(Value value, Type tileType, const LowerInfo &info,
                           OpBuilder &builder, Location loc) {
  auto bridge = builder.create<frisk::ToThreadTileOp>(loc, tileType, value);
  setTileLayout(bridge, info, builder);
  return bridge.getResult();
}

// 将线程数据包装为原 blockType，供尚未改写的用户和循环接口继续使用。
// 这是逻辑表示桥接，不会自动分配完整 block buffer，也不会自动写回目的地址。
static Value fromThreadTile(Value value, Type blockType, const LowerInfo &info,
                             OpBuilder &builder, Location loc) {
  auto bridge = builder.create<frisk::FromThreadTileOp>(loc, blockType, value);
  setTileLayout(bridge, info, builder);
  return bridge.getResult();
}

// 在当前 builder 插入点生成 gpu.thread_id x；尽管名字叫 find，实际会创建 op。
// 在使用位置生成保证支配关系。这里的 tid 是整个 block 内的线程编号，
// LowerInfo 的 map 还会把它分解为 laneId 和 warpId。
static Value findThreadIdxOp(Operation *op, OpBuilder &builder) {
  // Materialize at the use site, so the value always dominates its consumer.
  return builder.create<gpu::ThreadIdOp>(op->getLoc(), gpu::Dimension::x);
}

// A whole-tile store into independent shared storage can use the producer's
// register distribution. The consumer redistributes by loading that storage;
// converting the registers first would introduce a second LDS round trip.
// Keep slices and potentially aliasing storage on the existing path. A copy
// result aliases the physical destination and does not constrain store layout.
static bool canStoreProducerLayout(frisk::CopyOp copy, Value input) {
  SharedBufferAliases aliases;
  if (input != copy.getSrc() || !aliases.collect(copy.getDst()))
    return false;
  auto src = dyn_cast<MemRefType>(input.getType());
  auto dst = dyn_cast<MemRefType>(copy.getDst().getType());
  if (!src || !dst || !src.hasStaticShape() ||
      src.getShape() != dst.getShape() ||
      dst.getMemorySpaceAsInt() != int(friskMs::Shared))
    return false;
  // Use the IR's value/storage contract, not a list of producer op names.
  // In particular, memory-effect-free views are still aliases, not values.
  Operation *producer = input.getDefiningOp();
  if (!producer || !producer->hasTrait<frisk::ComputedTile>())
    return false;
  auto map = copy.getOffsetMap();
  if (map.getNumInputs() != 0)
    return false;
  bool whole = map.getNumResults() == 1 &&
      map.getResult(0) == getAffineConstantExpr(dst.getRank(), copy.getContext());
  bool zero = map.getNumResults() == unsigned(dst.getRank()) &&
      llvm::all_of(map.getResults(), [](AffineExpr e) {
        auto c = dyn_cast<AffineConstantExpr>(e);
        return c && c.getValue() == 0;
      });
  return whole || zero;
}

// Owned shared storage has an address layout, not a register distribution.
// Copy results retain that identity; each reader can load its requested lanes
// directly. Do not classify computed memref-typed values as physical storage.
static bool isSharedCopyStorage(Value value) {
  SharedBufferAliases aliases;
  if (!aliases.collect(value))
    return false;
  // Other destination-style operations can still forward their temporary
  // registers through a ConvertLayoutOp (e.g. singleton-axis reductions).
  // Leave those candidates intact until their own lowering decides whether
  // the allocation must actually be materialized.
  return llvm::any_of(aliases.accesses, [&](OpOperand *use) {
    return isa<frisk::CopyOp>(use->getOwner()) && use->getOperandNumber() == 1;
  });
}

static void insertConvertLayoutOps(LowerInfoMap &infoMap) {
  struct Conversion { Operation *user; Value input; LowerInfo from; LowerInfo to; };
  SmallVector<Conversion, 8> conversions;
  // Snapshot first: adding conversion layouts can rehash the analysis map.
  for (auto &entry : infoMap) {
    auto &info = entry.second;
    if (auto copy = dyn_cast_or_null<frisk::CopyOp>(info.op);
        copy && isGlobalToSharedCopy(copy))
      continue;
    if (info.convertFrom && info.buffer && info.op && !isTiled(info.op))
      conversions.push_back({info.op, info.buffer, *info.convertFrom, info});
  }
  for (auto &conversion : conversions) {
    if (isSharedCopyStorage(conversion.input)) {
      // The previous user's access distribution does not constrain this
      // memory read. Keep the consumer layout without staging through scratch.
      infoMap.getLowerInfo(conversion.input, conversion.user)->convertFrom = nullptr;
      continue;
    }
    if (auto copy = dyn_cast<frisk::CopyOp>(conversion.user);
        copy && canStoreProducerLayout(copy, conversion.input)) {
      // Change only this copy's source layout. Destination readers (including
      // MMA) retain their own layouts and the store's existing block barrier.
      auto *info = infoMap.getLowerInfo(conversion.input, conversion.user);
      *info = conversion.from;
      info->buffer = conversion.input;
      info->op = conversion.user;
      info->pos = LowerInfo::BufPos::In;
      info->convertFrom = nullptr;
      continue;
    }
    OpBuilder builder(conversion.user);
    auto op = builder.create<frisk::ConvertLayoutOp>(conversion.user->getLoc(),
        conversion.input.getType(), conversion.input, builder.getStringAttr("lowerinfo.convert"));
    infoMap.updateLowerInfoForLayoutConvertOp(op, conversion.to);
    convertLayoutInfo.insert({op, {conversion.from, conversion.to}});
    conversion.user->replaceUsesOfWith(conversion.input, op->getResult(0));
  }
}

// 生成 index 类型常量，供循环下标、取模、向量位置和地址计算共用。
static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

// 将只依赖一个 index 值的表达式构造成 affine.apply，返回计算后的 index。
static Value createSingleDimAffineApply(OpBuilder &builder, Location loc,
                                        AffineExpr expr, Value operand) {
  auto map = AffineMap::get(1, 0, expr, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, operand);
}

// 计算 operand % divisor，用来拆出某层局部坐标；除数为 1 时直接返回 0。
static Value modBy(OpBuilder &builder, Location loc, Value operand,
                   int64_t divisor) {
  assert(divisor > 0 && "affine modulo divisor must be positive");
  if (divisor == 1) {
    return createIndexConstant(builder, loc, 0);
  }
  auto d0 = builder.getAffineDimExpr(0);
  return createSingleDimAffineApply(builder, loc, d0 % divisor, operand);
}

// 计算 operand floordiv divisor，用来去掉更内层坐标；除数为 1 时返回原值。
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

// 生成 lhs + rhs 的 affine 索引表达式，例如 mask 的起点加 block 内坐标。
static Value addIndexValues(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, d0 + d1, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, ValueRange{lhs, rhs});
}

// 按 order 将二维坐标展平：flat = xy[order[0]] +
// xy[order[1]] * layout[order[0]]，其中 order[0] 是变化最快的轴。
// 例如 shape=[2,3]、xy=[1,2]：order=[0,1] 得 1+2*2；
// order=[1,0] 得 2+1*3。不能把 order 固定理解为 C 数组行优先。
// 只有一个元素时直接生成 0；这与 LowerInfo 中 UnflattenIndexToXY 互为对应。
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
// 把分层坐标整理成 LowerInfo::getAffineMap 约定的七个实参。
// br/iu 保留每轴独立参数；wr/reg 分别按各自 order 展平成一个参数。
// tid 已隐含 lane 与 warp 坐标，所以不再额外传入 laneId、warpId。
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

// 应用 LowerInfo 的二维地址 map，并投影成实际 rank 个索引。
// 二维 ignoreDim 轴返回 0；归约掉第 0 轴的一维结果使用 map 的第 1 个结果。
// 返回的是 block tile 内逻辑坐标；如果底层还有 buffer_view，需要再加 view 偏移。
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
// 核心逆打包函数：输入当前线程的 tileIvs，输出它们在 block tile 中的坐标。
// 记 W=thread_widths[d]、R=warp_repeat[d]、U=warpInstUnroll[d]：
//   br = iv / (U*R*W)，iu = (iv / (R*W)) % U，
//   wr = (iv % (R*W)) / W，reg = iv % W。
// 线程内部排列为 [br][iu][wr][reg]，reg 最内层；这里不把 tid 加到 iv 上，
// 而是把 tid 与上述分层坐标一起交给 LowerInfo map。
// 在 LowerInfo.h 中，每轴 block 坐标是：
//   br*blockWidth + warpCoord*warpInstWidth*U + iu*warpInstWidth
//   + wr*warpWidth + laneCoord*W + reg。
// laneCoord 来自 tid%warp_threads；warpCoord 来自 tid/warp_threads，
// 各自再按 warp_layout_order、block_layout_order 拆成二维。
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


// 控制 fragment 网格循环的属性：marker 标记已处理，label 用于阅读 IR，
// unrollFull 请求后续 lower-frisk-fragments 在预算允许时完全展开循环。
struct FragmentLoopOptions {
  StringRef marker;
  StringRef label;
  bool unrollFull = false;
};

// A fragment region preserves the scheduling boundary even when its target
// implementation needs several scalar loads or register insertions.
// 创建一个无条件执行的单 block region，将 body 生成的操作包在同一调度边界内。
// body 捕获外部 SSA 值；有结果时用 fragment_yield 导出，无结果时仅保留副作用。
// 这里构造的是 IR 容器，并没有执行计算，也不保证 region 对应单条机器指令。
template <typename Callback>
static Value buildFragment(OpBuilder &b, Location loc, Type resultType,
                           StringRef kind, Callback body) {
  SmallVector<Type> types;
  if (resultType)
    types.push_back(resultType);
  auto fragment = b.create<frisk::FragmentOp>(loc, types, b.getStringAttr(kind));
  fragment.getBody().push_back(new Block);
  b.setInsertionPointToStart(&fragment.getBody().front());
  Value value = body();
  b.create<frisk::FragmentYieldOp>(loc, value ? ValueRange{value} : ValueRange{});
  b.setInsertionPointAfter(fragment);
  return resultType ? fragment.getResult(0) : Value{};
}

static Value transferFragment(OpBuilder &b, Location loc, Value tile,
                              VectorType fragmentType, ArrayRef<Value> origins,
                              Value fragment = {});

// A fragment is one instruction's per-thread register tile. Singleton and
// reduced axes are projected before constructing the fragment iteration space.
// 从完整 thread tile 的形状求单片段形状 F=W*R，不包含 U、BR 两层重复。
// 单例轴固定为 1；rank-1 且忽略第 0 轴时，使用布局的第 1 轴参数。
// 要求每轴完整长度能被片段宽度整除，后面的循环因此无需处理残缺片段。
static SmallVector<int64_t> getFragmentShape(ArrayRef<int64_t> shape,
                                             const LowerInfo &info) {
  auto widths = info.get_thread_widths() * info.get_warp_repeat();
  SmallVector<int64_t> fragment;
  for (auto [dim, size] : llvm::enumerate(shape)) {
    unsigned axis = shape.size() == 1 && info.ignoreDim == 0 ? 1 : dim;
    int64_t width = size == 1 ? 1 : widths[axis];
    assert(width > 0 && size % width == 0 && "incomplete register fragment");
    fragment.push_back(width);
  }
  return fragment;
}

// Flatten only the fragment grid (br * instUnroll + iu), never the register
// shape. The body receives a fragment origin; registers inside it are static.
// One SSA carrier may be a vector, a memref, or a scalar reduction accumulator.
// 两层遍历：外层生成 IR 中的 affine.for 遍历片段，内层在 C++ 中枚举寄存器。
// shape 是完整线程形状 T，fragment 是单片段形状 F；网格大小 G[d]=T[d]/F[d]。
// 注意网格的行优先展平只决定遍历次序，元素归属仍由 LowerInfo 的布局决定。
class FragmentTileLoopNest {
public:
  // 未显式传入 fragmentShape 时，把整个线程 tile 当作一个 fragment。
  FragmentTileLoopNest(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
                    FragmentLoopOptions options = {},
                    ArrayRef<int64_t> fragmentShape = {})
      : builder(builder), loc(loc), shape(shape), options(options),
        fragment(fragmentShape.empty() ? shape : fragmentShape) {}

  template <typename Callback>
  // body 接收片段起点和当前累积值，返回更新后的值；空 initial 表示仅有副作用。
  // initial 非空时通过 iter_args/yield 串起各次更新，避免生成脱离 SSA 的可变值。
  Value emitFragments(Value initial, Callback body) {
    int64_t count = 1;
    for (auto [size, width] : llvm::zip(shape, fragment)) {
      assert(width > 0 && size % width == 0);
      count *= size / width;
    }
    SmallVector<Value> init;
    if (initial)
      init.push_back(initial);
    auto loop = builder.create<affine::AffineForOp>(loc, 0, count, 1, init);
    loop->setAttr("frisk.fragment_loop", builder.getUnitAttr());
    loop->setAttr("frisk.fragment_shape", builder.getDenseI64ArrayAttr(fragment));
    loop->setAttr("frisk.tile_shape", builder.getDenseI64ArrayAttr(shape));
    if (!options.marker.empty())
      loop->setAttr(options.marker, builder.getBoolAttr(true));
    if (options.unrollFull)
      loop->setAttr("frisk.loopUnrollFull", builder.getBoolAttr(true));
    if (!options.label.empty())
      loop->setAttr("iterLabel", builder.getStringAttr(options.label));
    builder.setInsertionPointToStart(loop.getBody());
    // 将一维片段编号拆成网格坐标，再乘 F 得到线程 vector 内的 origin。
    // 例如 T=2x32、F=1x4：origin=[iv/8, (iv%8)*4]。
    SmallVector<Value> origins(shape.size());
    int64_t stride = 1;
    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      int64_t extent = shape[dim] / fragment[dim];
      origins[dim] = extent == 1 ? createIndexConstant(builder, loc, 0)
          : composeAccessIndex(builder, loc,
              (builder.getAffineDimExpr(0).floorDiv(stride) % extent) * fragment[dim],
              loop.getInductionVar());
      stride *= extent;
    }
    Value result = body(origins, initial ? loop.getRegionIterArgs()[0] : Value{});
    if (initial) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<affine::AffineYieldOp>(loc, result);
    }
    builder.setInsertionPointAfter(loop);
    return initial ? loop.getResult(0) : Value{};
  }

  template <typename BodyEmitter>
  // 把逐元素 emitter 适配成 fragment：indices=origin+静态寄存器偏移。
  // vector carrier：body 返回标量，先组装片段，再插回完整线程 vector；
  // 标量/memref carrier：依次传递 body 返回值（例如归约累加器）；
  // 无 carrier：body 只生成 store 等副作用，fragment 不返回值。
  Value emit(Value initial, BodyEmitter &body, StringRef kind = "compute") {
    return emitFragments(initial, [&](ArrayRef<Value> origins, Value current) {
      auto vectorType = initial ? dyn_cast<VectorType>(initial.getType()) : VectorType{};
      Type resultType = vectorType ? Type(VectorType::get(fragment, vectorType.getElementType()))
                                   : initial ? initial.getType() : Type{};
      Value result = buildFragment(builder, loc, resultType, kind, [&]() -> Value {
        Value packed = vectorType ? builder.create<arith::ConstantOp>(
            loc, resultType, builder.getZeroAttr(resultType)).getResult() : current;
        int64_t registers = ShapedType::getNumElements(fragment);
        for (int64_t reg = 0; reg < registers; ++reg) {
          SmallVector<Value> indices(shape.size());
          SmallVector<OpFoldResult> local(shape.size());
          int64_t remaining = reg;
          for (int dim = shape.size() - 1; dim >= 0; --dim) {
            int64_t offset = remaining % fragment[dim];
            remaining /= fragment[dim];
            local[dim] = builder.getIndexAttr(offset);
            indices[dim] = offset == 0 ? origins[dim]
                : composeAccessIndex(builder, loc, builder.getAffineDimExpr(0) + offset,
                                     origins[dim]);
          }
          Value element = body.emit(indices, vectorType ? Value{} : packed);
          packed = vectorType ? builder.create<vector::InsertOp>(loc, element, packed, local)
                              : element;
        }
        return packed;
      });
      return vectorType ? transferFragment(builder, loc, current,
          cast<VectorType>(resultType), origins, result) : result;
    });
  }

private:
  OpBuilder &builder;
  Location loc;
  SmallVector<int64_t> shape;
  FragmentLoopOptions options;
  SmallVector<int64_t> fragment;
};

// Dynamic vector slice offsets are not supported by extract_strided_slice.
// Gather/scatter the statically sized register fragment using scalar positions.
// fragment 为空时，从 tile 的 origins 起点抽取片段；非空时将片段插回 tile。
// 局部寄存器坐标静态展开，完整 tile 坐标允许动态；这里只搬运当前线程的 SSA
// vector 元素，不进行跨线程通信，也不直接产生 shared/global 内存访问。
static Value transferFragment(OpBuilder &b, Location loc, Value tile,
                              VectorType fragmentType, ArrayRef<Value> origins,
                              Value fragment) {
  bool insert = bool(fragment);
  if (!insert)
    fragment = b.create<arith::ConstantOp>(loc, fragmentType,
                                         b.getZeroAttr(fragmentType));
  for (int64_t reg = 0; reg < fragmentType.getNumElements(); ++reg) {
    SmallVector<OpFoldResult> local(fragmentType.getRank()), position(local.size());
    int64_t remaining = reg;
    for (int dim = fragmentType.getRank() - 1; dim >= 0; --dim) {
      int64_t offset = remaining % fragmentType.getDimSize(dim);
      remaining /= fragmentType.getDimSize(dim);
      local[dim] = b.getIndexAttr(offset);
      position[dim] = offset == 0 ? origins[dim]
          : composeAccessIndex(b, loc, b.getAffineDimExpr(0) + offset, origins[dim]);
    }
    if (insert) {
      Value scalar = b.create<vector::ExtractOp>(loc, fragment, local);
      tile = b.create<vector::InsertOp>(loc, scalar, tile, position);
    } else {
      Value scalar = b.create<vector::ExtractOp>(loc, tile, position);
      fragment = b.create<vector::InsertOp>(loc, scalar, fragment, local);
    }
  }
  return insert ? tile : fragment;
}

template <typename Callback>
// 通用逐片段计算：零初始化完整结果 -> compute(origin, fragmentType)
// -> 片段结果插回完整 vector。算术 op 在 region 内，插回 tile 在 region 外，
// 便于后续融合 pass 分析各迭代写入哪些位置。
static Value mapFragments(OpBuilder &b, Location loc, VectorType type,
                          ArrayRef<int64_t> fragment, Callback compute) {
  auto fragmentType = VectorType::get(fragment, type.getElementType());
  Value initial = b.create<arith::ConstantOp>(loc, type, b.getZeroAttr(type));
  return FragmentTileLoopNest(b, loc, type.getShape(), {"tiled", "fragment"}, fragment)
      .emitFragments(initial, [&](ArrayRef<Value> origins, Value tile) {
        Value value = buildFragment(b, loc, fragmentType, "compute", [&]() {
          Value result = compute(origins, fragmentType);
          markTiled(result.getDefiningOp(), b);
          return result;
        });
        return transferFragment(b, loc, tile, fragmentType, origins, value);
      });
}

// Legacy destination-style operations write memory at the original program
// point. Local destinations are thread memrefs; shared/global destinations keep
// physical block addresses. The explicit To(From(...)) chain is foldable later.
// 把 FromThreadTile 包装的计算结果实际写入 destination，保留原 op 的副作用。
// 输入 block 必须直接由 FromThreadTile 定义；先 To(block) 获得可读取的 vector。
// local/空间 5：目标转为线程私有 memref，直接以线程局部 iv 写入。
// shared/global：目标仍是物理 block 存储，以 LowerInfo 映射后的坐标写入。
// 若归约轴、广播轴或跨 warp 复用导致多个线程拥有同一逻辑元素，仅 leader 写。
// shared 写完后在条件分支外发出 barrier，保证所有线程都参加同步。
// 调用点位置很重要：不能把写回随意移到消费者处，否则会改变可见的内存顺序。
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
  auto emitStore = [&](ArrayRef<Value> ivs, Value) -> Value {
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
  return {};
  };
  struct StoreBody {
    decltype(emitStore) &fn;
    Value emit(ArrayRef<Value> ivs, Value current) { return fn(ivs, current); }
  } body{emitStore};
  FragmentTileLoopNest(builder, loc, threadType.getShape(), {"tiled", "store"},
                     getFragmentShape(threadType.getShape(), info)).emit({}, body, "store");
  if (leader)
    builder.setInsertionPointAfter(leader);
  if (dstType.getMemorySpaceAsInt() == int(friskMs::Shared)) {
    auto barrier = builder.create<gpu::BarrierOp>(loc);
    markTiled(barrier, builder);
  }
}

// 局部广播的单元素 emitter：源的单例轴固定取 0，其他轴使用当前 tile 下标。
// 返回一个标量，由 FragmentTileLoopNest::emit 组装片段；不执行跨 lane 通信。
struct BroadcastTileElement {
  OpBuilder &rewriter;
  Location loc;
  VectorType sourceTy;
  Value sourceVector;
  // 为目标 tile 的一个位置取源标量；currentTile 在 vector 模式下不使用。
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

    return scalar.getResult();
  }
};

// 将线程内 sourceVector 广播到 tileTy；要求元素类型、rank 相同，
// 每个不匹配的源维度必须为 1。相同类型直接复用，否则用逐元素循环构造结果。
// 正确的跨线程元素归属应由调用前的 ToThreadTile 布局保证。
static FailureOr<Value> broadcastLocalVectorToTile(Value sourceVector,
                                                   VectorType tileTy,
                                                   OpBuilder &rewriter,
                                                   Location loc,
                                                   const LowerInfo &info) {
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
  return FragmentTileLoopNest(rewriter, loc, tileTy.getShape(), {},
                            getFragmentShape(tileTy.getShape(), info))
      .emit(result, element);
}


// 后序扫描并反复删除 MLIR 判定为 trivially dead 的操作，直到不再变化。
// 避免把未使用但仍有写内存副作用的操作当作普通死 SSA 计算删除。
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


// 反复展开只有一次迭代的 affine.for，让循环体直接进入父 block。
// 这是单次迭代简化，与 frisk.loopUnrollFull 请求的多次循环完全展开不同。
static void promoteSingleIterationAffineFors(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<affine::AffineForOp, 16> loops;
    root->walk<WalkOrder::PostOrder>(
        [&](affine::AffineForOp forOp) { loops.push_back(forOp); });
    for (affine::AffineForOp forOp : loops) {
      if (!forOp->hasAttr("frisk.fragment_loop") &&
          succeeded(affine::promoteIfSingleIteration(forOp))) {
        changed = true;
      }
    }
  }
}

// 把 fill 使用的标量常量属性转换为目标元素类型。
// 支持浮点属性到浮点、整数属性到整数/index；不支持时返回空属性。
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


// 保持 vector 形状，只转换浮点元素位宽；同类型直接返回。
// 位宽增大生成 arith.extf，其他不同浮点类型走 arith.truncf；非浮点转换失败。
// 它不是通用数值转换器，后面的 finalizeNumericCast 有独立且更细的类型检查。
static FailureOr<Value>
castFloatVectorElementType(Value vector, Type dstElementType,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LowerInfo *info = nullptr) {
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
  SmallVector<int64_t> fragment = info ? getFragmentShape(srcVecTy.getShape(), *info)
                                      : SmallVector<int64_t>(srcVecTy.getShape());
  return mapFragments(rewriter, loc, dstVecTy, fragment,
      [&](ArrayRef<Value> origins, VectorType dst) -> Value {
        auto src = VectorType::get(fragment, srcElementType);
        Value value = transferFragment(rewriter, loc, vector, src, origins);
        if (srcFloatTy.getWidth() < dstFloatTy.getWidth())
          return rewriter.create<arith::ExtFOp>(loc, dst, value);
        return rewriter.create<arith::TruncFOp>(loc, dst, value);
      });
}


// 为逐元素计算准备完整线程操作数：标量直接广播，矩阵按结果布局取数，
// 再转换元素类型并广播单例轴。相同 vector 形状不代表相同的 block 元素归属。
static FailureOr<Value> materializeElementwiseOperand(
    Value original, Value adapted, const LowerInfo &resultInfo, VectorType resultType,
    ConversionPatternRewriter &rewriter, Location loc) {
  if (isa<FloatType>(original.getType())) {
    if (original.getType() != resultType.getElementType())
      return failure();
    return rewriter.create<vector::BroadcastOp>(loc, resultType, adapted).getResult();
  }
  // Each operand must supply the logical coordinates owned by the result.
  // Equal packed vector shapes do not guarantee equal row ownership: e.g.
  // block_repeat=[2,1] and warp_inst_unroll=[2,1] can both yield two rows
  // per thread while exchanging rows between warps. Project the result layout
  // onto singleton operand axes before broadcasting locally. The bridge keeps
  // the producer's layout separate and materializes a conversion when needed.
  LowerInfo info = resultInfo;
  info.buffer = original;
  auto shape = dyn_cast<ShapedType>(original.getType());
  if (!shape || !shape.hasStaticShape())
    return failure();
  if (shape.getRank() == 2) {
    info.ignoreDim = -1;
    for (unsigned dim = 0; dim < 2; ++dim) {
      if (shape.getDimSize(dim) == 1) {
        info.ignoreDim = dim;
        info.thread_own_data_size[dim] = 1;
      }
    }
  }
  auto type = getFullThreadTileType(original, info);
  if (failed(type))
    return failure();
  Value tile = toThreadTile(adapted, *type, info, rewriter, loc);
  auto casted = castFloatVectorElementType(tile, resultType.getElementType(),
                                          rewriter, loc, &resultInfo);
  if (failed(casted))
    return failure();
  return broadcastLocalVectorToTile(*casted, resultType, rewriter, loc, resultInfo);
}

// add/sub/mul/div 共用的重写模板：查询结果 LowerInfo -> 推导线程形状 ->
// 物化两端线程操作数 -> 逐 fragment 创建 arith 向量计算 ->
// 将片段插回完整线程 vector -> FromThreadTile 恢复原结果类型。
// matchAndRewrite 成功后替换原 op；缺布局或不兼容的广播/类型则匹配失败。
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
    auto lhs = materializeElementwiseOperand(op.getLhs(), adaptor.getLhs(),
        *resultInfo, *resultType, rewriter, op.getLoc());
    auto rhs = materializeElementwiseOperand(op.getRhs(), adaptor.getRhs(),
        *resultInfo, *resultType, rewriter, op.getLoc());
    if (failed(lhs) || failed(rhs))
      return rewriter.notifyMatchFailure(op, "incompatible operand thread tiles");
    auto fragment = getFragmentShape(resultType->getShape(), *resultInfo);
    Value result = mapFragments(rewriter, op.getLoc(), *resultType, fragment,
        [&](ArrayRef<Value> origins, VectorType type) -> Value {
          Value a = transferFragment(rewriter, op.getLoc(), *lhs, type, origins);
          Value b = transferFragment(rewriter, op.getLoc(), *rhs, type, origins);
          auto newop = rewriter.create<ToOp>(op.getLoc(), a, b);
          newop->setAttr("origin",rewriter.getStringAttr(op->getName().getStringRef()));
          return newop;
        });
    Value block = fromThreadTile(result, op.getResult().getType(),
                                 *resultInfo, rewriter, op.getLoc());
    rewriter.replaceOp(op, block);
    return success();
  }
};

// 将 block 级 exp2 改为 math.exp2 的线程 vector 计算，再桥接回 block 类型。
// 优先使用输入自己的 LowerInfo 取数；没有输入信息时回退到结果布局。
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
    Value result = mapFragments(rewriter, op.getLoc(), *type,
        getFragmentShape(type->getShape(), *info),
        [&](ArrayRef<Value> origins, VectorType fragmentType) -> Value {
          Value fragment = transferFragment(rewriter, op.getLoc(), *input, fragmentType, origins);
          return rewriter.create<math::Exp2Op>(op.getLoc(), fragment);
        });
    rewriter.replaceOp(op, fromThreadTile(result, op.getResult().getType(),
                                          *info, rewriter, op.getLoc()));
    return success();
  }
};

// Numeric conversion supports scalars and vectors. Equal-width floating point
// formats are not bitcasts; reject formats requiring a separate conversion.
static FailureOr<Value> finalizeNumericCast(OpBuilder &b, Location loc,
                                           Value value, Type resultType) {
  if (value.getType() == resultType)
    return value;
  Type src = getElementTypeOrSelf(value.getType());
  Type dst = getElementTypeOrSelf(resultType);
  if (auto sf = dyn_cast<FloatType>(src)) {
    if (auto df = dyn_cast<FloatType>(dst)) {
      if (sf.getWidth() < df.getWidth())
        return b.create<arith::ExtFOp>(loc, resultType, value).getResult();
      if (sf.getWidth() > df.getWidth())
        return b.create<arith::TruncFOp>(loc, resultType, value).getResult();
    }
    if (auto di = dyn_cast<IntegerType>(dst); di && di.isSignless())
      return b.create<arith::FPToSIOp>(loc, resultType, value).getResult();
  }
  if (auto si = dyn_cast<IntegerType>(src); si && si.isSignless()) {
    if (isa<FloatType>(dst))
      return b.create<arith::SIToFPOp>(loc, resultType, value).getResult();
    if (auto di = dyn_cast<IntegerType>(dst); di && di.isSignless()) {
      if (si.getWidth() < di.getWidth())
        return b.create<arith::ExtSIOp>(loc, resultType, value).getResult();
      return b.create<arith::TruncIOp>(loc, resultType, value).getResult();
    }
  }
  if ((isa<IndexType>(src) && isa<IntegerType>(dst)) ||
      (isa<IntegerType>(src) && isa<IndexType>(dst)))
    return b.create<arith::IndexCastOp>(loc, resultType, value).getResult();
  return failure();
}

// Casts share the same fragment boundary as arithmetic, including integer
// conversions. Scalar casts have no fragment iteration space.
class CastOpTiling : public OpConversionPattern<frisk::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto result = dyn_cast<ShapedType>(op.getResult().getType());
    if (!result) {
      auto value = finalizeNumericCast(rewriter, op.getLoc(), adaptor.getOperand(),
                                       op.getResult().getType());
      if (failed(value))
        return failure();
      rewriter.replaceOp(op, *value);
      return success();
    }
    auto source = dyn_cast<ShapedType>(op.getOperand().getType());
    if (!source || !source.hasStaticShape() || !result.hasStaticShape() ||
        (isa<VectorType>(source) && cast<VectorType>(source).isScalable()) ||
        (isa<VectorType>(result) && cast<VectorType>(result).isScalable()) ||
        source.getShape() != result.getShape())
      return rewriter.notifyMatchFailure(op, "cast requires matching static shapes");
    auto info = findLowerInfoForValue(op.getResult(), op);
    if (!info)
      info = findLowerInfoForValue(op.getOperand(), op);
    if (!info && (isa<MemRefType>(source) || isa<MemRefType>(result)))
      return rewriter.notifyMatchFailure(op, "missing cast layout");
    auto type = info ? getFullThreadTileType(op.getResult(), *info)
                     : FailureOr<VectorType>(cast<VectorType>(result));
    if (failed(type))
      return failure();
    auto srcType = VectorType::get(type->getShape(), source.getElementType());
    Value input = info ? toThreadTile(adaptor.getOperand(), srcType, *info, rewriter, op.getLoc())
                       : adaptor.getOperand();
    SmallVector<int64_t> shape = info ? getFragmentShape(type->getShape(), *info)
                                      : SmallVector<int64_t>(type->getShape());
    bool supported = true;
    Value value = mapFragments(rewriter, op.getLoc(), *type, shape,
        [&](ArrayRef<Value> origins, VectorType dst) -> Value {
          Value fragment = transferFragment(rewriter, op.getLoc(), input,
              VectorType::get(shape, source.getElementType()), origins);
          auto converted = finalizeNumericCast(rewriter, op.getLoc(), fragment, dst);
          if (failed(converted)) {
            supported = false;
            // ConversionPatternRewriter rolls back this speculative IR.
            return rewriter.create<arith::ConstantOp>(op.getLoc(), dst, rewriter.getZeroAttr(dst));
          }
          return *converted;
        });
    if (!supported)
      return rewriter.notifyMatchFailure(op, "unsupported fragment numeric conversion");
    if (info)
      value = fromThreadTile(value, result, *info, rewriter, op.getLoc());
    rewriter.replaceOp(op, value);
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
    Value zero = mapFragments(rewriter, op.getLoc(), *type,
        getFragmentShape(type->getShape(), *info),
        [&](ArrayRef<Value>, VectorType fragmentType) -> Value {
          return rewriter.create<arith::ConstantOp>(op.getLoc(), fragmentType,
                                                  rewriter.getZeroAttr(fragmentType));
        });
    rewriter.replaceOp(op, fromThreadTile(zero, op.getResult().getType(),
                                          *info, rewriter, op.getLoc()));
    return success();
  }
};

// fill 有写目标内存的语义：先生成线程 memref，循环写入目标常量，
// 再用 writeBackBlockTile 保留物理写回；有结果时返回 block 表示，否则删除原 op。
// 与 ZeroOpTiling 不同，不能只用零/常量 vector 替换而丢失写内存副作用。
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
    struct FillElement {
      OpBuilder &builder;
      Location loc;
      Value value, tile;
      Value emit(ArrayRef<Value> ivs, Value) {
        auto store = builder.create<affine::AffineStoreOp>(loc, value, tile, ivs);
        markTiled(store, builder);
        return {};
      }
    } elementBody{rewriter, loc, value, tile};
    FragmentTileLoopNest(rewriter, loc, type->getShape(), {"tiled", "fill"},
                      getFragmentShape(type->getShape(), *info)).emit({}, elementBody, "store");
    Value block = fromThreadTile(tile, op.getMemref().getType(), *info, rewriter, loc);
    writeBackBlockTile(block, adaptor.getMemref(), *info, rewriter, loc);
    if (op.hasValueResult())
      rewriter.replaceOp(op, block);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

// shared/global 分配保留原来的 block 大小和内存空间。
// 其他分配根据 LowerInfo 缩小为线程 tile 大小的 memref.alloca，
// 再用 FromThreadTile 保持用户看到的 block 类型。
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
      if (auto pack = op->getAttr("frisk.shared_pack"))
        alloc->setAttr("frisk.shared_pack", pack);
      if (auto axis = op->getAttr("frisk.shared_pack_axis"))
        alloc->setAttr("frisk.shared_pack_axis", axis);
      if (auto stride = op->getAttr("frisk.shared_pair_stride"))
        alloc->setAttr("frisk.shared_pair_stride", stride);
      for (StringRef name : {"frisk.shared_store_stride", "frisk.shared_store_pair"})
        if (auto attr = op->getAttr(name)) alloc->setAttr(name, attr);
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

// mask 的逐元素 emitter：线程下标先映射为 block 坐标，再叠加 starts。
// 用这些逻辑坐标替换原 region 参数，克隆标量表达式，将 mask_yield 的值
// 插入当前线程 vector。否则使用局部 iv 会让不同线程误算同一组 mask 坐标。
struct MaskTileElement {
  ConversionPatternRewriter &rewriter;
  Location loc;
  LowerInfo *resultInfo;
  Value tidx;
  MemRefType resultTy;
  frisk::MaskOp::Adaptor adaptor;
  Block *body;
  frisk::MaskYieldOp yieldOp;
  // 计算 mask 中一个线程拥有的元素；IRMapping 仅用于本次 region 克隆。
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
    return scalar;
  }
};

/// 匹配：rank <= 2、带 LowerInfo 的 memref mask，region yield 一个标量。
/// 对线程持有的每个元素执行 region，经 FromThreadTile 返回原 block 类型。
// 校验 mask 的静态结果形状、LowerInfo、region 参数数和单标量 yield，
// 创建线程 vector，并通过 MaskTileElement 遍历填充，最终桥接回原结果类型。
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
        FragmentTileLoopNest(rewriter, loc, vectorShape, {"tiled", "mask_fragment"},
                           getFragmentShape(vectorShape, *resultInfo))
            .emit(resultVector, element);
    rewriter.replaceOp(op, fromThreadTile(resultVector, op.getResult().getType(),
                                          *resultInfo, rewriter, loc));
    return success();
  }
};


// 为浮点归约生成单位元：add=0、mul=1、min=+∞、max=-∞。
// 未知 kind 返回 failure，调用方据此拒绝该归约。
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

// 按 reduce.kind 合并两个浮点值，生成 addf/mulf/minnumf/maxnumf。
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


// 对一个输出位置先在线程内部归约，再在同一 warp 的相关 lanes 间归约。
// laneExtent 是归约轴上的 lane 数；laneStride 由该轴在 warp_layout_order
// 中的位置决定，因此 XOR shuffle 的偏移不是一律 1、2、4。
struct ReduceTileElement {
  ConversionPatternRewriter &rewriter;
  frisk::ReduceOp op;
  Value source;
  VectorType sourceType;
  Value identity;
  int64_t laneExtent;
  int64_t laneStride;
  int64_t warpThreads;
  int64_t fragmentWidth;

  // 沿源线程 vector 的归约轴循环累加，再按 offset*laneStride 做 XOR shuffle，
  // 最后返回归约标量，由外层 emitter 组装输出片段。支持保留或去掉归约轴。
  Value emit(ArrayRef<Value> outputIvs, Value output) {
    auto loc = op.getLoc();
    int64_t dim = op.getDim();
    struct ReductionElement {
      ReduceTileElement &self;
      ArrayRef<Value> outputIvs;
      Value emit(ArrayRef<Value> reductionIvs, Value accumulator) {
        SmallVector<OpFoldResult, 2> indices;
        unsigned out = 0;
        bool keepDim = outputIvs.size() == static_cast<size_t>(self.sourceType.getRank());
        for (int64_t i = 0; i < self.sourceType.getRank(); ++i) {
          if (i == self.op.getDim()) {
            indices.push_back(reductionIvs[0]);
            if (keepDim)
              ++out;
          } else {
            indices.push_back(outputIvs[out++]);
          }
        }
        Value value = self.rewriter.create<vector::ExtractOp>(self.op.getLoc(), self.source, indices);
        return *combineReduceValues(self.op, accumulator, value, self.rewriter);
      }
    } element{*this, outputIvs};
    // Keep the original left-to-right FP association across all fragments.
    Value reduced = FragmentTileLoopNest(rewriter, loc, {sourceType.getDimSize(dim)},
        {"tiled", "reduce_axis_fragment"}, {fragmentWidth}).emit(identity, element);
    for (int64_t offset = 1; offset < laneExtent; offset <<= 1) {
      auto shuffle = rewriter.create<gpu::ShuffleOp>(loc, reduced,
          static_cast<int32_t>(offset * laneStride), static_cast<int32_t>(warpThreads),
          gpu::ShuffleMode::XOR);
      reduced = *combineReduceValues(op, reduced, shuffle.getShuffleResult(), rewriter);
    }
    return reduced;
  }
};

// Compose the same tid/thread-index decomposition as buildMappedAccessIndices.
// Project singleton axes before comparing: their layout metadata may differ
// even though every lane holds exactly the same surviving coordinates.
static AffineMap effectiveThreadTileMap(LowerInfo info, ShapedType block,
                                        VectorType tile, MLIRContext *ctx) {
  auto zero = getAffineConstantExpr(0, ctx);
  SmallVector<AffineExpr> br, iu, wr, reg;
  for (unsigned axis = 0; axis < 2; ++axis) {
    int dim = tile.getRank() == 1 ? (axis == (info.ignoreDim == 0 ? 1 : 0) ? 0 : -1)
                                  : int(axis);
    AffineExpr iv = dim < 0 || tile.getDimSize(dim) == 1
        ? zero : getAffineDimExpr(dim + 1, ctx);
    int64_t width = info.get_thread_widths()[axis];
    int64_t repeat = info.get_warp_repeat()[axis];
    int64_t unroll = info.warpInstUnroll[axis];
    br.push_back(iv.floorDiv(width * repeat * unroll));
    iu.push_back(iv.floorDiv(width * repeat) % unroll);
    wr.push_back((iv % (width * repeat)).floorDiv(width));
    reg.push_back(iv % width);
  }
  auto flatten = [](ArrayRef<AffineExpr> xy, coordXY_t order, coordXY_t shape) {
    return xy[order[0]] + xy[order[1]] * shape[order[0]];
  };
  SmallVector<AffineExpr> operands{
      getAffineDimExpr(0, ctx), br[0], br[1], iu[0], iu[1],
      flatten(wr, info.base_layout.warp_repeat_order, info.get_warp_repeat()),
      flatten(reg, info.base_layout.thread_creg_order, info.get_thread_widths())};
  auto map = info.getAffineMap().replaceDimsAndSymbols(
      operands, {}, tile.getRank() + 1, 0);
  SmallVector<AffineExpr> results;
  for (int dim = 0; dim < tile.getRank(); ++dim) {
    int axis = tile.getRank() == 1 && info.ignoreDim == 0 ? 1 : dim;
    results.push_back(block.getDimSize(dim) == 1 ||
                              (tile.getRank() == 2 && info.ignoreDim == axis)
                          ? zero : map.getResult(axis));
  }
  return simplifyAffineMap(AffineMap::get(tile.getRank() + 1, 0, results, ctx));
}

// A hoisted reduction temporary may serve independent producer/reader pairs
// in several blocks. Forward only locally proven pairs, never across a branch
// or loop boundary. Equal vector shapes alone do not prove lane ownership.
static bool forwardReductionTemporary(frisk::ReduceOp op, Value result,
                                      LowerInfo resultInfo,
                                      ConversionPatternRewriter &rewriter) {
  Value dst = op.getDst();
  Operation *allocation = dst.getDefiningOp();
  if (!isa_and_nonnull<frisk::AllocBufferOp, memref::AllocOp>(allocation) ||
      cast<MemRefType>(dst.getType()).getMemorySpaceAsInt() != int(friskMs::Shared))
    return false;
  struct LocalPair {
    frisk::ReduceOp writer;
    SmallVector<OpOperand *> readers;
  };
  DenseMap<Block *, LocalPair> pairs;
  for (OpOperand &use : dst.getUses()) {
    auto &pair = pairs[use.getOwner()->getBlock()];
    if (auto writer = dyn_cast<frisk::ReduceOp>(use.getOwner())) {
      if (use.getOperandNumber() != 1 || pair.writer) return false;
      pair.writer = writer;
    } else if (auto convert = dyn_cast<frisk::ConvertLayoutOp>(use.getOwner())) {
      // A layout conversion can also feed a destination-style writer. Such
      // a use is not a read of this reduction and cannot become an SSA value.
      if (!llvm::all_of(convert->getUsers(), [](Operation *reader) {
            return reader->hasTrait<frisk::ComputedTile>() ||
                   isa<frisk::CopyToRegOp>(reader);
          }))
        return false;
      pair.readers.push_back(&use);
    } else if (use.getOwner()->hasTrait<frisk::ComputedTile>())
      pair.readers.push_back(&use);
    else
      return false;
  }
  for (auto &[block, pair] : pairs) {
    if (!pair.writer)
      return false;
    for (OpOperand *use : pair.readers) {
      Operation *reader = use->getOwner();
      if (!pair.writer->isBeforeInBlock(reader))
        return false;
      // A nested region between this pair might write the same allocation.
      // Known readers can occur between one another; nothing else may touch it.
      for (Operation *next = pair.writer->getNextNode(); next != reader;
           next = next->getNextNode()) {
        bool touches = false;
        next->walk([&](Operation *nested) {
          for (OpOperand &operand : nested->getOpOperands())
            if (operand.get() == dst && !llvm::is_contained(pair.readers, &operand))
              touches = true;
        });
        if (touches) return false;
      }
    }
  }
  auto &readers = pairs[op->getBlock()].readers;
  if (readers.empty()) return false;
  SmallVector<LowerInfo, 2> targets;
  auto resultType = cast<VectorType>(result.getType());
  auto blockType = cast<ShapedType>(dst.getType());
  for (OpOperand *use : readers) {
    std::optional<LowerInfo> target;
    if (auto consumer = dyn_cast<frisk::ConvertLayoutOp>(use->getOwner())) {
      auto found = convertLayoutInfo.find(consumer);
      if (found != convertLayoutInfo.end()) target = found->second.second;
    } else {
      target = findLowerInfoForValue(dst, use->getOwner());
    }
    if (!target) return false;
    auto type = getFullThreadTileType(dst, *target);
    if (failed(type) || *type != resultType ||
        effectiveThreadTileMap(resultInfo, blockType, resultType, op.getContext()) !=
        effectiveThreadTileMap(*target, blockType, resultType, op.getContext()))
      return false;
    targets.push_back(*target);
  }
  // Validate every reader before changing any of them. They all observe the
  // same local reduction, so the physical store and its barrier are redundant.
  for (auto [use, target] : llvm::zip(readers, targets)) {
    Operation *reader = use->getOwner();
    Value forwarded = fromThreadTile(result, dst.getType(), target,
                                      rewriter, op.getLoc());
    if (auto consumer = dyn_cast<frisk::ConvertLayoutOp>(reader)) {
      convertLayoutInfo.erase(consumer);
      rewriter.replaceOp(consumer, forwarded);
    } else {
      rewriter.modifyOpInPlace(reader, [&] { use->set(forwarded); });
    }
  }
  return true;
}

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
                               lanes, stride, sourceInfo->warp_threads,
                               getFragmentShape(sourceType->getShape(), *sourceInfo)[dim]};
    Value result = FragmentTileLoopNest(rewriter, loc, shape, {"tiled", "reduce_fragment"},
        getFragmentShape(shape, *resultInfo)).emit(initial, element);
    if (forwardReductionTemporary(op, result, *resultInfo, rewriter)) {
      rewriter.eraseOp(op);
      return success();
    }
    Value block = fromThreadTile(result, op.getDst().getType(), *resultInfo, rewriter, loc);
    writeBackBlockTile(block, adaptor.getDst(), *resultInfo, rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }
};

// Partition a logical row-major tile into consecutive per-thread intervals.
// Vector stores never cross a row, even when an interval spans several rows.
// A short final interval is guarded, while every thread reaches the barrier.
// global -> shared 的专用搬运：源的物理连续轴决定搬运顺序，
// GEMM LowerInfo 决定后续 shared packing，二者通过逻辑坐标衔接。
// 每线程负责 ceil(总元素数/thread_num) 个元素；向量宽度取该数量与末轴长度的 gcd，
// 保证一次 vector.store 不跨行。连续源显式使用至多 128-bit 的读取包；
// 非连续源保留标量读取，随后组装向量，可做浮点位宽转换。
// 尾部无任务线程跳过访存，但所有线程都在分支外执行 block 同步。
// 源/目标 view 的偏移会合成到真实 buffer；带结果的 copy 返回传入的完整目标。
static LogicalResult lowerGlobalToSharedCopy(
    frisk::CopyOp op, Value source, Value destination, Value result,
    ArrayRef<int64_t> shape, ConversionPatternRewriter &rewriter) {
  auto kernel = op->getParentOfType<func::FuncOp>();
  auto threadsAttr = kernel->getAttrOfType<IntegerAttr>("thread_num");
  if (!threadsAttr || threadsAttr.getInt() <= 0 || shape.empty() ||
      llvm::any_of(shape, [](int64_t size) { return size <= 0; }))
    return rewriter.notifyMatchFailure(op, "copy requires positive shape and thread_num");
  auto srcType = cast<MemRefType>(source.getType());
  auto dstType = cast<MemRefType>(destination.getType());
  auto physicalDst = cast<MemRefType>(getViewBuffer(destination).getType());
  if (!physicalDst.isLastDimUnitStride())
    return rewriter.notifyMatchFailure(op, "vector copy requires unit destination row stride");
  auto loc = op.getLoc();
  bool columnMajorSource = getPhysicalMinorCopyAxis(source) == 0;
  int64_t total = ShapedType::getNumElements(shape);
  int64_t perThread = llvm::divideCeil(total, threadsAttr.getInt());
  SharedBufferAliases aliases;
  bool owned = aliases.collect(getViewBuffer(destination));
  auto packing = owned ? dyn_cast_or_null<IntegerAttr>(aliases.commonAttribute("frisk.shared_pack"))
                       : IntegerAttr{};
  auto packAxis = owned ? dyn_cast_or_null<IntegerAttr>(aliases.commonAttribute("frisk.shared_pack_axis"))
                        : IntegerAttr{};
  bool columnPacking = packing && packAxis && packAxis.getInt() == 1;
  int64_t pack = packing && !columnPacking ? packing.getInt() : 1;
  // For a transposed input, transfer ownership follows the physical source
  // order [0,1]. Shared packing remains a separate consumer-driven mapping.
  if (columnMajorSource)
    pack = 1;
  if (pack < 2 || !llvm::isPowerOf2_64(pack) || shape.size() != 2 ||
      pack > 128 / dstType.getElementTypeBitWidth() || shape[0] % pack ||
      total % threadsAttr.getInt() || perThread % pack)
    pack = 1;
  int64_t width = std::gcd(perThread, shape.back() * pack);
  int64_t minorLanes = 1, minorRepeats = 1;
  if (columnMajorSource) {
    int64_t limit = std::max<int64_t>(1, 128 / srcType.getElementTypeBitWidth());
    width = 1;
    while (width * 2 <= limit && shape[0] % (width * 2) == 0)
      width *= 2;
    // A 128-byte contiguous lane group balances GLOBAL coalescing with the
    // consumer's packed LDS stores. Letting all lanes advance along D can
    // concentrate writes in the same banks. In LowerInfo terms this adjusts
    // the transfer's lane shape as well as its fastest-axis-first order.
    int64_t laneLimit = 1024 / (width * srcType.getElementTypeBitWidth());
    while (minorLanes * 2 <= laneLimit &&
           (shape[0] / width) % (minorLanes * 2) == 0 &&
           threadsAttr.getInt() % (minorLanes * 2) == 0)
      minorLanes *= 2;
    minorRepeats = shape[0] / (width * minorLanes);
    perThread = minorRepeats *
        llvm::divideCeil(shape[1], threadsAttr.getInt() / minorLanes) * width;
  }
  auto vectorType = VectorType::get({width}, srcType.getElementType());
  Value tid = findThreadIdxOp(op, rewriter);
  auto chunk = rewriter.create<affine::AffineForOp>(loc, 0, perThread / width, 1);
  markTiled(chunk, rewriter);
  chunk->setAttr("frisk.fragment_loop", rewriter.getUnitAttr());
  chunk->setAttr("frisk.fragment_shape", rewriter.getDenseI64ArrayAttr({width}));
  chunk->setAttr("iterLabel", rewriter.getStringAttr("copy_transfer_fragment"));
  if (columnMajorSource)
    chunk->setAttr("frisk.loopUnrollFull", rewriter.getBoolAttr(true));
  rewriter.setInsertionPointToStart(chunk.getBody());
  Value start = composeAccessIndex(
      rewriter, loc,
      columnMajorSource
          ? (rewriter.getAffineDimExpr(1).floorDiv(minorRepeats) *
                 (threadsAttr.getInt() / minorLanes) +
             rewriter.getAffineDimExpr(0).floorDiv(minorLanes)) * shape[0] +
                ((rewriter.getAffineDimExpr(1) % minorRepeats) * minorLanes +
                 rewriter.getAffineDimExpr(0) % minorLanes) * width
          : rewriter.getAffineDimExpr(0) * perThread +
                rewriter.getAffineDimExpr(1) * width,
      ValueRange{tid, chunk.getInductionVar()});
  scf::IfOp active;
  if (total % perThread != 0 || total / perThread < threadsAttr.getInt()) {
    Value limit = createIndexConstant(rewriter, loc, total);
    Value inBounds = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, start, limit);
    active = rewriter.create<scf::IfOp>(loc, inBounds, false);
    rewriter.setInsertionPointToStart(&active.getThenRegion().front());
  }
  auto coordinates = [&](Value flat, int64_t reg = 0) {
    if (columnMajorSource) {
      // Same fastest-axis-first convention as LowerInfo's *_order fields.
      auto xy = UnflattenIndexToXY(rewriter.getAffineDimExpr(0),
                                  coordXY_t{0, 1}, coordXY_t{shape[0], shape[1]});
      return SmallVector<Value>{composeAccessIndex(rewriter, loc, xy[0], flat),
                                composeAccessIndex(rewriter, loc, xy[1], flat)};
    }
    if (pack > 1) {
      auto d = rewriter.getAffineDimExpr(0);
      // Each thread transfers a consecutive interval in the basic row-pack
      // layout. Optional K pairing later permutes its four-element chunks;
      // keep this producer ownership to preserve 128-bit GLOBAL loads.
      // Retain logical
      // coordinates here, so even an unexecuted packing pass is semantically
      // valid. The later pass folds the inverse permutation and widens stores.
      // width divides N*pack and is a multiple of pack; start is a multiple
      // of width. The packet cannot cross a packed row. Keep its shared base
      // explicit, avoiding hard-to-simplify floorDiv(start + reg) expressions.
      return SmallVector<Value>{
          composeAccessIndex(rewriter, loc, d.floorDiv(shape[1] * pack) * pack + reg % pack, flat),
          composeAccessIndex(rewriter, loc, d.floorDiv(pack) % shape[1] + reg / pack, flat)};
    }
    SmallVector<Value> indices(shape.size());
    int64_t stride = 1;
    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      indices[dim] = composeAccessIndex(
          rewriter, loc,
          rewriter.getAffineDimExpr(0).floorDiv(stride) % shape[dim], flat);
      stride *= shape[dim];
    }
    return indices;
  };
  Value packet = buildFragment(rewriter, loc, vectorType, "load", [&]() {
    if (columnMajorSource) {
      auto at = coordinates(start);
      Value buffer = source;
      resolveViewAccess(rewriter, loc, buffer, at);
      return Value(rewriter.create<vector::LoadOp>(loc, vectorType, buffer, at));
    }
    Value packet = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
    if (cast<MemRefType>(getViewBuffer(source).getType()).isLastDimUnitStride()) {
      // A transfer contains pack rows by width/pack columns (one row for
      // ordinary copies). Bound GLOBAL loads independently of the thread's
      // transfer size and of LDS packing. In particular, scalar loads assembled
      // into a large vector can become an oversized, element-aligned LLVM load
      // that the backend scalarizes again. Explicit native-size packets avoid
      // relying on SLP and do not assume stronger alignment for external buffers
      // or offset views. Only exact elements of this row are accessed.
      int64_t columns = width / pack;
      int64_t limit = std::max<int64_t>(1, 128 / srcType.getElementTypeBitWidth());
      for (int64_t row = 0; row < pack; ++row) {
        for (int64_t col = 0; col < columns;) {
          int64_t count = 1;
          while (count * 2 <= std::min(limit, columns - col))
            count *= 2;
          auto at = coordinates(start, row);
          at.back() = composeAccessIndex(rewriter, loc,
              rewriter.getAffineDimExpr(0) + col, at.back());
          Value buffer = source;
          resolveViewAccess(rewriter, loc, buffer, at);
          auto loadType = VectorType::get({count}, srcType.getElementType());
          Value values = rewriter.create<vector::LoadOp>(loc, loadType, buffer, at);
          for (int64_t i = 0; i < count; ++i) {
            Value value = rewriter.create<vector::ExtractOp>(loc, values,
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr(i)});
            packet = rewriter.create<vector::InsertOp>(loc, value, packet,
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr((col + i) * pack + row)});
          }
          col += count;
        }
      }
      return packet;
    }
  // Static register positions within one transfer fragment; only the packet
  // grid is represented by an affine loop, just like computation fragments.
  for (int64_t reg = 0; reg < width; ++reg) {
    Value flat = composeAccessIndex(
        rewriter, loc, rewriter.getAffineDimExpr(0) + reg, start);
    auto indices = pack > 1 ? coordinates(start, reg) : coordinates(flat);
    Value buffer = source;
    resolveViewAccess(rewriter, loc, buffer, indices);
    Value scalar = rewriter.create<affine::AffineLoadOp>(loc, buffer, indices);
    packet = rewriter.create<vector::InsertOp>(
        loc, scalar, packet, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(reg)});
  }
    return packet;
  });
  auto converted = castFloatVectorElementType(
      packet, dstType.getElementType(), rewriter, loc);
  if (failed(converted))
    return rewriter.notifyMatchFailure(op, "unsupported global copy element conversion");
  auto indices = coordinates(start);
  Value buffer = destination;
  resolveViewAccess(rewriter, loc, buffer, indices);
  buildFragment(rewriter, loc, Type{}, "store", [&]() -> Value {
    if (pack == 1 && !columnPacking && !columnMajorSource) {
      auto store = rewriter.create<vector::StoreOp>(loc, *converted, buffer, indices);
      markTiled(store, rewriter);
    } else {
      for (int64_t reg = 0; reg < width; ++reg) {
        // Column packing keeps the original row-major producer intervals.
        // Emit scalar logical stores so the physical pass can remap them all,
        // then coalesce exactly the packets that remain contiguous.
        auto at = pack > 1 ? coordinates(start, reg) : coordinates(composeAccessIndex(
            rewriter, loc, rewriter.getAffineDimExpr(0) + reg, start));
        Value target = destination;
        resolveViewAccess(rewriter, loc, target, at);
        Value scalar = rewriter.create<vector::ExtractOp>(loc, *converted,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(reg)});
        auto store = rewriter.create<memref::StoreOp>(loc, scalar, target, at);
        markTiled(store, rewriter);
      }
    }
    return {};
  });
  rewriter.setInsertionPointAfter(chunk);
  auto sync = rewriter.create<frisk::SyncThreadsInBlockOp>(loc);
  markTiled(sync, rewriter);
  if (op.hasValueResult())
    rewriter.replaceOp(op, result);
  else
    rewriter.eraseOp(op);
  return success();
}

// 处理整块/切片 copy，并保留写目标及返回目标的语义。
//   1. 解析 offset_map；旧式 ()->(rank) 是整块复制哨兵，不是实际偏移。
//   2. 必要时对较大一端建立 buffer_view，并调整切片的 block_repeat。
//   3. global->shared 走连续搬运；其他路径按 LowerInfo 取得源线程 tile。
//   4. vector 目标更新 SSA；memref 目标写线程暂存，再执行物理写回。
//   5. 若切片类型不同于 copy 结果类型，返回完整原目标，不能只返回切片。
// 旧式无结果 vector copy 只替换被新定义支配的后续使用，以保留 copy 前的读取。
class CopyOpTiling : public OpConversionPattern<frisk::CopyOp> {
public:
  // 注册为允许有界递归的 pattern：一次改写删除一个旧 copy 且不创建新 copy，
  // 即使替换后续 vector 使用立即触发另一个 copy 重写，也会有限终止。
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
    bool naiveCopy = isGlobalToSharedCopy(op);
    std::optional<LowerInfo> info;
    if (!naiveCopy) {
      info = findLowerInfoForValue(layoutValue, op);
      if (!info) {
        layoutValue = op.getDst();
        info = findLowerInfoForValue(layoutValue, op);
      }
      if (!info)
        return rewriter.notifyMatchFailure(op, "missing copy layout");
    }
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
    if (naiveCopy)
      return lowerGlobalToSharedCopy(op, source, destination, adaptor.getDst(),
                                     copyShape, rewriter);
    if (!sameShape && failed(setCopyTileShape(*info, copyShape)))
      return rewriter.notifyMatchFailure(op, "copy slice must cover complete layout tiles");
    auto tileTy = getFullThreadTileType(layoutValue, *info);
    if (failed(tileTy))
      return failure();
    auto sourceTileTy = VectorType::get(tileTy->getShape(), srcTy.getElementType());
    Value tile = toThreadTile(source, sourceTileTy, *info, rewriter, loc);
    auto converted = castFloatVectorElementType(tile, dstTy.getElementType(), rewriter, loc, &*info);
    if (failed(converted))
      return rewriter.notifyMatchFailure(op, "unsupported copy element conversion");
    if (srcTy.getElementType() == dstTy.getElementType()) {
      auto resultType = VectorType::get(tileTy->getShape(), dstTy.getElementType());
      converted = mapFragments(rewriter, loc, resultType,
          getFragmentShape(tileTy->getShape(), *info),
          [&](ArrayRef<Value> origins, VectorType fragmentType) {
            return transferFragment(rewriter, loc, tile, fragmentType, origins);
          });
    }
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
    Value block = fromThreadTile(*converted, destination.getType(), *info, rewriter, loc);
    writeBackBlockTile(block, destination, *info, rewriter, loc);
    if (op.hasValueResult()) {
      auto destinationSpace = cast<MemRefType>(dstTy).getMemorySpaceAsInt();
      if (destinationSpace == int(friskMs::Shared) ||
          destinationSpace == int(friskMs::Global) ||
          block.getType() != op.getValueResult().getType()) {
        // The FromThreadTile above represents only the copied slice. Its
        // writeback has updated the original destination in place, so the
        // copy result aliases that complete block-level memref. Do not tile
        // the enclosing buffer: it may be a rank-4 global tensor, and the
        // slice layout says nothing about elements outside this block's tile.
        // Whole physical copies also return the destination, not the source
        // register snapshot: readers use their own layouts and observe later
        // writes through either alias, without an extra layout scratch buffer.
        block = adaptor.getDst();
      }
      rewriter.replaceOp(op, block);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// 识别 copy_to_reg 的整块形式：源和结果 shape 相同，offset_map 为 ()->(rank)。
// 这种形式的 vector 仍表示逻辑 block tile，不能误认为已经是线程寄存器片段。
static bool isWholeTileCopyToReg(frisk::CopyToRegOp op) {
  auto src = cast<MemRefType>(op.getSrc().getType());
  auto dst = cast<VectorType>(op.getResult().getType());
  auto map = op.getOffsetMap();
  return src.getShape() == dst.getShape() && map.getNumInputs() == 0 &&
         map.getNumResults() == 1 &&
         map.getResult(0) == getAffineConstantExpr(src.getRank(), op.getContext());
}

// A rank sentinel denotes a whole block tile (used by pipeline prologue and
// epilogue). Other maps specify one thread's contiguous slice.
// copy_to_reg 分两种语义：整块形式使用 LowerInfo 缩成线程 tile，再桥接回
// 原 block vector；显式偏移形式已经描述一个线程的连续切片，直接建 view。
// 后者不附 tile_layout，后续要求切片与线程形状一致，按局部索引读取即可。
class CopyToRegOpTiling : public OpConversionPattern<frisk::CopyToRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::CopyToRegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isTiled(op))
      return failure();
    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto resultType = cast<VectorType>(op.getResult().getType());
    if (isWholeTileCopyToReg(op)) {
      auto info = findLowerInfoForValue(op.getSrc(), op);
      if (!info)
        info = findLowerInfoForValue(op.getResult(), op);
      if (!info)
        return rewriter.notifyMatchFailure(op, "missing whole-tile copy_to_reg layout");
      auto tileType = getFullThreadTileType(op.getSrc(), *info);
      if (failed(tileType))
        return failure();
      Value tile = toThreadTile(adaptor.getSrc(), *tileType, *info,
                                rewriter, op.getLoc());
      auto converted = castFloatVectorElementType(
          tile, resultType.getElementType(), rewriter, op.getLoc(), &*info);
      if (failed(converted))
        return failure();
      rewriter.replaceOp(op, fromThreadTile(*converted, resultType, *info,
                                            rewriter, op.getLoc()));
      return success();
    }
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

// 布局转换的回读 emitter：以目标 LowerInfo 把线程下标映射到共享 scratch，
// 读回标量并组装新的线程 vector；生产者在 scratch 中使用的是源布局地址。
struct LayoutReadBackElement {
  ConversionPatternRewriter &rewriter;
  Location loc;
  LowerInfo &info;
  Value tid;
  Value scratch;
  // 从共享中转矩阵读取目标线程应持有的一个元素，返回更新后的 vector。
  Value emit(ArrayRef<Value> ivs, Value output) {
    auto indices = buildMappedAccessIndices(rewriter, loc, info, tid, ivs, ivs.size());
    Value scalar = rewriter.create<affine::AffineLoadOp>(loc, scratch, indices);
    return scalar;
  }
};

// 落实显式布局转换：按 fromInfo 取源寄存器 -> 按源坐标写 shared scratch ->
// 同步 -> 按 toInfo 坐标读成新寄存器 vector -> 再同步 -> FromThreadTile。
// 第一次同步由 writeBackBlockTile 生成，第二次保护 scratch 不被过早复用。
// 转换改变的是线程/lane 对元素的持有关系，矩阵的逻辑值和 block 类型保持不变。
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
    Value result = FragmentTileLoopNest(rewriter, loc, toTy->getShape(), {"tiled", "layout_fragment"},
        getFragmentShape(toTy->getShape(), toInfo)).emit(initial, element, "load");
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
// 把 frisk.block 的标量 region 分配给线程执行。
// 先收集读写 buffer 的 LowerInfo，选择一个输出线程形状作为局部循环范围。
// 特别区分两套坐标：region 参数参与标量计算时用 block 坐标；
// 线程私有 memref 的 load/store 使用局部 iv（单例轴为 0）。
// 当前仅支持平直 region 和受限的投影访问；嵌套 region、未知副作用、
// 需要跨分布轴置换的访问会失败。所有写过的 buffer 最后显式写回。
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
    auto emitBlock = [&](ArrayRef<Value> ivs, Value) -> Value {
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
    return {};
    };
    struct BlockElement {
      decltype(emitBlock) &fn;
      Value emit(ArrayRef<Value> ivs, Value current) { return fn(ivs, current); }
    } blockElement{emitBlock};
    FragmentTileLoopNest(rewriter, loc, shape, {"tiled", "block_fragment"},
                      getFragmentShape(shape, output->info)).emit({}, blockElement, "store");
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

/// 只替换 block-tile SSA 值，thread-tile 的数据流完全由 IR 显式表达。
// 将一个 block GEMM 改写为每线程寄存器片段参与的 WarpMmaRROp。
// 要求 DCU、二维 memref A/B/C、三者 LowerInfo 绑定相同 MMA 指令。
// 每轴片段大小 = thread_widths * warp_repeat；
// 每轴片段数量 = block_repeat * warpInstUnroll；两者相乘才是完整线程 tile。
// A 是 [M,K]，B 是 [K,N]，C 是 [M,N]，所以需检查三者片段数量相容。
// M/N 片段网格展平成一个 IR 循环，内层 K 循环逐片段累加；C 片段从零开始，
// 结束后插入完整 C 线程 vector，再用 FromThreadTile 替换原 GEMM 结果。
// op.getC() 在这里是输出 SSA 值，不是 GEMM 的输入累加器。
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

    // 把“片段数量”和“单片段寄存器形状”分开计算。
    // 例如某轴 W=2、R=2、U=1、BR=3，则片段大小=4、片段数=3、完整长度=12。
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
    // Bridge 类型记录完整 MN/K 布局；Finalize 只在片段使用处物化实际读。
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
    Value initial = rewriter.create<arith::ConstantOp>(
        loc, threadTileC, rewriter.getZeroAttr(threadTileC));
    Value zeroFragment = rewriter.create<arith::ConstantOp>(
        loc, fragmentC, rewriter.getZeroAttr(fragmentC));
    // 外层枚举 C 的 (m,n) 片段；cOrigin 是线程 vector 坐标，除以片段宽度
    // 才得到片段编号 m/n。MN 与 K 循环保留到后续融合、调度完成之后。
    Value result = FragmentTileLoopNest(rewriter, loc, threadTileC.getShape(),
        {"tiled", "gemm_mn"}, cFragmentShape).emitFragments(initial,
        [&](ArrayRef<Value> cOrigin, Value output) -> Value {
          Value m = floorDivBy(rewriter, loc, cOrigin[0], cFragmentShape[0]);
          Value n = floorDivBy(rewriter, loc, cOrigin[1], cFragmentShape[1]);
          // 每个 C[m,n] 的累加器独立从零开始，K 循环执行
          // acc = MMA(A[m,k], B[k,n], acc)，iter_args 保存上一轮 fragment。
          auto kFor = rewriter.create<affine::AffineForOp>(
              loc, 0, aCounts[1], 1, ValueRange{zeroFragment});
          kFor->setAttr("iterLabel", rewriter.getStringAttr("gemm_k"));
          kFor->setAttr("frisk.fragment_loop", rewriter.getUnitAttr());
          kFor->setAttr("frisk.loopUnrollFull", rewriter.getBoolAttr(true));
          rewriter.setInsertionPointToStart(kFor.getBody());
          Value k = kFor.getInductionVar();
          auto operand = [&](Value block, VectorType tileType, VectorType fragType,
                             LowerInfo &info, Value row, Value col) {
            return buildFragment(rewriter, loc, fragType, "load", [&]() {
              // 在使用位置描述完整 A/B 布局，但只抽取当前 (m,k)/(k,n) 片段。
              // fragment_source 让 Finalize 把这些 extract 原地改为实际 load，
              // 从而无需先加载包含所有 K 片段的完整线程 vector。
              Value tile = toThreadTile(block, tileType, info, rewriter, loc);
              tile.getDefiningOp()->setAttr("frisk.fragment_source", rewriter.getUnitAttr());
              SmallVector<Value> origin{
                  composeAccessIndex(rewriter, loc, rewriter.getAffineDimExpr(0) * fragType.getDimSize(0), row),
                  composeAccessIndex(rewriter, loc, rewriter.getAffineDimExpr(0) * fragType.getDimSize(1), col)};
              return transferFragment(rewriter, loc, tile, fragType, origin);
            });
          };
          Value a = operand(adaptor.getA(), threadTileA, fragmentA, *infoA, m, k);
          Value b = operand(adaptor.getB(), threadTileB, fragmentB, *infoB, k, n);
          Value mma = buildFragment(rewriter, loc, fragmentC, "compute", [&]() -> Value {
            auto op = rewriter.create<frisk::WarpMmaRROp>(
                loc, fragmentC, a, b, kFor.getRegionIterArgs()[0]);
            markTiled(op, rewriter);
            op->setAttr("inst_name", instName);
            op->setAttr("inst_constraints", constraints);
            return op;
          });
          rewriter.create<affine::AffineYieldOp>(loc, mma);
          rewriter.setInsertionPointAfter(kFor);
          // K 累加结束后把 C fragment 插回当前完整线程结果；外层循环携带它。
          return transferFragment(rewriter, loc, output, fragmentC, cOrigin, kFor.getResult(0));
        });

    Value newBlockTile = fromThreadTile(result, op.getC().getType(), *infoC, rewriter, loc);
    rewriter.replaceOp(op, newBlockTile);
    return success();
  }
};


// 列举本阶段需要处理的 block 级 op，供预扫描和 ConversionTarget 共用。
// 普通控制流和已生成的线程计算不属于这份待转换列表。
static bool isBlockTileOperation(Operation *op) {
  return isa<frisk::GemmOp, frisk::MaskOp, frisk::AddOp, frisk::SubOp,
      frisk::MulOp, frisk::DivOp, frisk::Exp2Op, frisk::CastOp, frisk::CopyOp,
      frisk::CopyToRegOp, frisk::FillOp, frisk::ZeroOp, frisk::ReduceOp,
      frisk::BlockOp, frisk::ConvertLayoutOp, frisk::AllocBufferOp>(op);
}

// Fold ToThreadTile(FromThreadTile(threadTile)) without changing block-level
// users of the same FromThreadTile. Equal types alone do not imply equal lane
// ownership, so only cancel bridges describing the same layout.
// 消除布局等价的 ToThreadTile(FromThreadTile(x))，反复处理直到稳定。
// 逐项比较布局属性，不能只比较 shape；同类型时直接复用 x。
// 若 x 是线程 memref 而 To 要 vector，必须在 To 原位置加载，
// 以看见 From 和 To 之间的写入。From 还有其他 block 用户时不能删除。
// 这里只消除成对桥接，不展开独立 To，也不强行消除所有 From。
static void foldThreadTilePairs(Operation *root) {
  static constexpr StringLiteral layoutAttrs[] = {
      "tile_layout", "thread_widths", "thread_creg_order", "warp_repeat",
      "warp_repeat_order", "warp_layout", "warp_layout_order",
      "warp_inst_unroll", "block_layout", "block_layout_order",
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
        if (to->hasAttr("frisk.fragment_source"))
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

#define GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

// 第一阶段 pass：由 block 计算建立线程计算和显式表示边界。
// runOnOperation 的步骤：筛选 kernel -> 判断布局需求 -> LowerInfo 推断 ->
// 插入布局转换 -> partial conversion -> 折叠桥接/清理。
// 共享内存复用推迟到 fragment 调度完成后。
// 只有带 thread_num 的函数会处理；需要推断时必须存在 GEMM 布局锚点。
// 该阶段不会统一改写循环签名，也不保证桥接已经全部消失。
class ConvertFriskBaseToThreadLevelIR
    : public impl::ConvertFriskBaseToThreadLevelIRBase<ConvertFriskBaseToThreadLevelIR> {
public:
  // 声明本 pass 会创建的 dialect，让 pass 管理器提前加载相应 IR 定义。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<frisk::FriskDialect, arith::ArithDialect, affine::AffineDialect,
        vector::VectorDialect, math::MathDialect, memref::MemRefDialect,
        gpu::GPUDialect, scf::SCFDialect>();
  }

  // 对当前 func.func 执行本阶段转换；具体处理顺序见下方分支和类说明。
  void runOnOperation() override {
    auto kernel = getOperation();
    if (!kernel->hasAttr("thread_num"))
      return;
    eraseTriviallyDeadOps(kernel);
    // 预扫描区分“有操作要改写”和“需要 GEMM 推导布局”。
    // 显式线程 copy_to_reg、global->shared 搬运、shared/global 分配可不依赖布局分析。
    bool needsTiling = false;
    bool needsLayoutAnalysis = false;
    bool hasLayoutAnchor = false;
    kernel.walk([&](Operation *op) {
      hasLayoutAnchor |= isa<frisk::GemmOp>(op);
      if (!isBlockTileOperation(op) || isTiled(op))
        return;
      needsTiling = true;
      if (auto cast = dyn_cast<frisk::CastOp>(op);
          cast && !isa<MemRefType>(cast.getOperand().getType()) &&
          !isa<MemRefType>(cast.getResult().getType()))
        return;
      if (auto copy = dyn_cast<frisk::CopyToRegOp>(op);
          copy && !isWholeTileCopyToReg(copy))
        return;
      if (auto copy = dyn_cast<frisk::CopyOp>(op);
          copy && isGlobalToSharedCopy(copy))
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
      // Shared storage planning is deferred until after fragment scheduling.
      return;
    }
    auto *context = &getContext();
    s_hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, context);
    convertLayoutInfo.clear();
    // Explicit CopyToReg slices need no GEMM-anchored layout analysis.
    if (needsLayoutAnalysis && !hasLayoutAnchor) {
      kernel.emitError("thread tiling requires a GEMM anchor for layout inference");
      signalPassFailure();
      return;
    }
    // 分析从 MMA 硬件描述确定 GEMM 布局，再向前/向后传播给其他操作，
    // 最后解决同一值不同使用点的布局冲突，必要时设置 convertFrom。
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
    if (packSharedOperands)
      planSharedOperandPacking(kernel, *s_info);
    llvm::outs() << "----- after insertConvertLayoutOps:\n" << kernel << "\n"; llvm::outs().flush();

    // 未 tiled 的目标 Frisk op 必须被转换；其他 op 动态合法。
    // 因此 pattern 返回 failure 不等于默默保留原计算：若没有其他合法改写，
    // applyPartialConversion 会失败并使 pass 报错。
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isBlockTileOperation(op) || isTiled(op);
    });
    RewritePatternSet patterns(context);
    patterns.add<GemmOpTiling, MaskOpTiling, Exp2OpTiling, CastOpTiling, FillOpTiling,
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
    // Shared storage planning is deferred until after fragment scheduling.
  }
};

} // namespace

// 第一阶段的工厂入口，对应命令行 --convert-friskbase-to-thread。
std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass() {
  return std::make_unique<ConvertFriskBaseToThreadLevelIR>();
}

namespace{

// Finalization uses the layout recorded on the bridge, never the expired
// LowerInfo analysis. Packed coordinates are [br, iu, wr, reg] per axis.
// 第二阶段的轻量布局解码器，完全从桥接属性获得地址计算信息。
// read 校验并读取 map/形状/分层参数；indices 把线程局部 iv 映射到 block 地址。
// 它复现 buildMappedAccessIndices 的分解公式，但不要求旧 LowerInfoMap 仍有效。
struct ThreadTileAccess {
  AffineMap map;
  coordXY_t widths, repeats, unroll, regOrder, repeatOrder;
  int64_t ignoreDim = -1;

  // 校验静态 rank 1/2、相同 rank 和元素类型；有 tile_layout 时还校验七输入
  // 二输出 map、ignore_dim 和正的分层尺寸/合法 order。
  // 没有布局属性只接受 block 与 thread 同 shape，表示已经是线程连续切片。
  LogicalResult read(Operation *op, ShapedType block, ShapedType thread) {
    if ((isa<VectorType>(block) && cast<VectorType>(block).isScalable()) ||
        (isa<VectorType>(thread) && cast<VectorType>(thread).isScalable()))
      return failure();
    if (!block.hasStaticShape() || !thread.hasStaticShape() ||
        block.getRank() != thread.getRank() || block.getRank() < 1 ||
        block.getRank() > 2 ||
        block.getElementType() != thread.getElementType())
      return failure();
    auto attr = op->getAttrOfType<AffineMapAttr>("tile_layout");
    if (!attr)
      return success(block.getShape() == thread.getShape());
    map = attr.getValue();
    auto ignored = op->getAttrOfType<IntegerAttr>("ignore_dim");
    if (!ignored || ignored.getInt() < -1 || ignored.getInt() > 1 ||
        map.getNumDims() != 7 || map.getNumSymbols() != 0 ||
        map.getNumResults() != 2)
      return failure();
    ignoreDim = ignored.getInt();
    auto pair = [&](StringRef name, coordXY_t &out, bool order = false) {
      auto a = op->getAttrOfType<DenseI64ArrayAttr>(name);
      if (!a || a.size() != 2)
        return false;
      out = {a[0], a[1]};
      return order ? ((a[0] == 0 && a[1] == 1) ||
                      (a[0] == 1 && a[1] == 0))
                   : a[0] > 0 && a[1] > 0;
    };
    return success(pair("thread_widths", widths) &&
                   pair("warp_repeat", repeats) &&
                   pair("warp_inst_unroll", unroll) &&
                   pair("thread_creg_order", regOrder, true) &&
                   pair("warp_repeat_order", repeatOrder, true));
  }

  // 由 packed iv 拆出 br/iu/wr/reg，按属性中的 order 展平，再调用保存的 map。
  // 无 map 则直接返回局部 iv；单例轴或二维 ignore_dim 轴固定为 0。
  // 此处仍返回 tile 内坐标，buffer_view 的物理偏移由另一个 pattern 合成。
  SmallVector<Value> indices(OpBuilder &b, Location loc, Value tid,
                             ArrayRef<Value> ivs, ShapedType block) const {
    if (!map)
      return SmallVector<Value>(ivs);
    Value zero = createIndexConstant(b, loc, 0);
    SmallVector<Value> br, iu, wr, reg;
    for (unsigned axis = 0; axis < 2; ++axis) {
      Value iv = axis < ivs.size() ? ivs[axis] : zero;
      if (ivs.size() == 1 && ignoreDim == 0)
        iv = axis == 1 ? ivs[0] : zero;
      int64_t repeatWidth = widths[axis] * repeats[axis];
      br.push_back(floorDivBy(b, loc, iv, repeatWidth * unroll[axis]));
      iu.push_back(modBy(b, loc, floorDivBy(b, loc, iv, repeatWidth),
                         unroll[axis]));
      wr.push_back(modBy(b, loc, floorDivBy(b, loc, iv, widths[axis]),
                         repeats[axis]));
      reg.push_back(modBy(b, loc, iv, widths[axis]));
    }
    SmallVector<Value> operands{
        tid, br[0], br[1], iu[0], iu[1],
        flattenXY(b, loc, wr, repeatOrder, repeats),
        flattenXY(b, loc, reg, regOrder, widths)};
    SmallVector<Value> result;
    for (unsigned dim = 0; dim < ivs.size(); ++dim) {
      unsigned axis = ivs.size() == 1 && ignoreDim == 0 ? 1 : dim;
      if (block.getDimSize(dim) == 1 ||
          (ivs.size() == 2 && ignoreDim == axis))
        result.push_back(zero);
      else
        result.push_back(b.create<affine::AffineApplyOp>(
            loc, map.getSubMap({axis}), operands));
    }
    return result;
  }
};

// Fold a view into each scalar memory access, preserving the original buffer's
// physical strides. In particular, a K slice keeps the full K row stride.
// Leave unsupported/escaping uses visible for the finalization diagnostic.
// 把 buffer_view 折叠到每个标量 memref/affine load/store 的访问地址中。
// 一次重写一个用户，贪心驱动重复执行；无用户时删除 view。
// 保留底层 buffer 的真实 stride；不支持或逃逸的用户留给最终诊断报告。
class BufferViewFinalization : public OpRewritePattern<frisk::BufferViewOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::BufferViewOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getView().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    for (Operation *user : op.getView().getUsers()) {
      Value buffer = op.getView(), stored;
      SmallVector<Value> indices;
      AffineMap map;
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        if (load.getMemref() != buffer)
          continue;
        indices.assign(load.getIndices().begin(), load.getIndices().end());
      } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemref() != buffer)
          continue;
        stored = store.getValueToStore();
        indices.assign(store.getIndices().begin(), store.getIndices().end());
      } else if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
        if (load.getMemRef() != buffer)
          continue;
        map = load.getAffineMap();
        indices.assign(load.getMapOperands().begin(), load.getMapOperands().end());
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
        if (store.getMemRef() != buffer)
          continue;
        stored = store.getValueToStore();
        map = store.getAffineMap();
        indices.assign(store.getMapOperands().begin(), store.getMapOperands().end());
      } else {
        continue;
      }
      rewriter.setInsertionPoint(user);
      if (map) {
        SmallVector<Value> coordinates;
        for (unsigned dim = 0; dim < map.getNumResults(); ++dim)
          coordinates.push_back(composeAccessIndex(
              rewriter, user->getLoc(), map.getSubMap({dim}), indices));
        indices = std::move(coordinates);
      }
      resolveViewAccess(rewriter, user->getLoc(), buffer, indices);
      if (stored) {
        if (map)
          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              user, stored, buffer, indices);
        else
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              user, stored, buffer, indices);
      } else {
        if (map)
          rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(user, buffer, indices);
        else
          rewriter.replaceOpWithNewOp<memref::LoadOp>(user, buffer, indices);
      }
      return success();
    }
    return failure();
  }
};

// 消除 frisk.cast。普通标量/vector 直接数值转换；memref 表示的 block 值
// 先从生产者 From 或消费者 To 获取线程布局，再构造 To -> 数值转换 -> From。
// 这样后续可折叠桥接，让只需寄存器转换的操作保留在线程内部。
class CastFinalization : public OpRewritePattern<frisk::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::CastOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    Type resultType = op.getResult().getType();
    if (!isa<MemRefType>(input.getType()) && !isa<MemRefType>(resultType)) {
      auto converted = finalizeNumericCast(rewriter, op.getLoc(), input, resultType);
      if (failed(converted))
        return rewriter.notifyMatchFailure(op, "unsupported numeric conversion");
      rewriter.replaceOp(op, *converted);
      return success();
    }
    // Cast the owned registers, then keep the block representation for users.
    // The following bridge fold removes it without allocating a block buffer.
    Operation *layout = nullptr;
    VectorType threadType;
    if (auto from = input.getDefiningOp<frisk::FromThreadTileOp>()) {
      layout = from;
      auto type = cast<ShapedType>(from.getThreadTile().getType());
      threadType = VectorType::get(type.getShape(), type.getElementType());
    } else {
      for (auto user : op.getResult().getUsers()) {
        if (auto to = dyn_cast<frisk::ToThreadTileOp>(user)) {
          layout = to;
          auto type = cast<ShapedType>(to.getResult().getType());
          threadType = VectorType::get(type.getShape(), getElementTypeOrSelf(input.getType()));
          break;
        }
      }
    }
    if (!layout || !isa<MemRefType>(resultType))
      return rewriter.notifyMatchFailure(op, "memref cast needs a thread layout");
    auto to = rewriter.create<frisk::ToThreadTileOp>(op.getLoc(), threadType, input);
    to->setAttrs(layout->getAttrs());
    auto converted = finalizeNumericCast(rewriter, op.getLoc(), to,
        VectorType::get(threadType.getShape(), getElementTypeOrSelf(resultType)));
    if (failed(converted)) {
      rewriter.eraseOp(to);
      return rewriter.notifyMatchFailure(op, "unsupported numeric conversion");
    }
    auto from = rewriter.create<frisk::FromThreadTileOp>(op.getLoc(), resultType, *converted);
    from->setAttrs(layout->getAttrs());
    rewriter.replaceOp(op, from.getResult());
    return success();
  }
};

// Move the explicit representation boundary across affine loop carriers. This
// keeps only owned registers live across iterations and also exposes layout
// changes at loop exits as To(From(...)), where they can be handled explicitly.
// 在满足条件时，把 affine.for 的 block vector 携带值缩为 thread vector。
// 要求 yield 来自 From、两侧都是不同类型的 vector，且 iter_arg/循环结果
// 仅被 To 使用。不是对任意循环、memref 携带值或 scf.for 的通用类型转换。
// 循环前对 init 加 To；新循环携带线程值；循环体入口暂加 From 兼容旧用户；
// yield 改交线程值；循环出口再 From 恢复外部接口，之后折叠冗余桥接。
// 这解释了第一阶段为何能先保留 block 签名，第二阶段再收缩活跃寄存器集合。
class ThreadTileLoopFinalization : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<affine::AffineYieldOp>(op.getBody()->getTerminator());
    SmallVector<frisk::FromThreadTileOp> carriers(op.getNumResults());
    bool changed = false;
    auto onlyToUsers = [](Value value) {
      return llvm::all_of(value.getUsers(), [](Operation *user) {
        return isa<frisk::ToThreadTileOp>(user);
      });
    };
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto from = yield.getOperand(i).getDefiningOp<frisk::FromThreadTileOp>();
      if (from && isa<VectorType>(from.getResult().getType()) &&
          isa<VectorType>(from.getThreadTile().getType()) &&
          from.getResult().getType() != from.getThreadTile().getType() &&
          onlyToUsers(op.getRegionIterArgs()[i]) && onlyToUsers(op.getResult(i))) {
        carriers[i] = from;
        changed = true;
      }
    }
    if (!changed)
      return failure();
    auto loc = op.getLoc();
    SmallVector<Value> initial(op.getInits());
    for (unsigned i = 0; i < carriers.size(); ++i) {
      if (!carriers[i])
        continue;
      auto to = rewriter.create<frisk::ToThreadTileOp>(
          loc, carriers[i].getThreadTile().getType(), initial[i]);
      to->setAttrs(carriers[i]->getAttrs());
      initial[i] = to;
    }
    auto loop = rewriter.create<affine::AffineForOp>(
        loc, op.getLowerBoundOperands(), op.getLowerBoundMap(),
        op.getUpperBoundOperands(), op.getUpperBoundMap(), op.getStepAsInt(), initial);
    for (auto attr : op->getDiscardableAttrs())
      loop->setAttr(attr.getName(), attr.getValue());
    rewriter.setInsertionPointToStart(loop.getBody());
    SmallVector<Value> arguments{loop.getInductionVar()};
    for (unsigned i = 0; i < carriers.size(); ++i) {
      Value arg = loop.getRegionIterArgs()[i];
      if (carriers[i]) {
        auto from = rewriter.create<frisk::FromThreadTileOp>(
            loc, op.getRegionIterArgs()[i].getType(), arg);
        from->setAttrs(carriers[i]->getAttrs());
        arg = from;
      }
      arguments.push_back(arg);
    }
    rewriter.modifyOpInPlace(yield, [&] {
      for (unsigned i = 0; i < carriers.size(); ++i)
        if (carriers[i])
          yield.setOperand(i, carriers[i].getThreadTile());
    });
    // The builder only inserts an implicit terminator for loops without inits.
    if (!loop.getBody()->empty() &&
        loop.getBody()->back().hasTrait<OpTrait::IsTerminator>())
      rewriter.eraseOp(&loop.getBody()->back());
    rewriter.mergeBlocks(op.getBody(), loop.getBody(), arguments);
    rewriter.setInsertionPointAfter(loop);
    SmallVector<Value> results(loop.getResults());
    for (unsigned i = 0; i < carriers.size(); ++i) {
      if (!carriers[i])
        continue;
      auto from = rewriter.create<frisk::FromThreadTileOp>(
          loc, op.getResult(i).getType(), results[i]);
      from->setAttrs(carriers[i]->getAttrs());
      results[i] = from;
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Copy destinations already have an explicit block writeback. A dominating
// whole-tile store initializes their scratch storage, so do not read the
// (possibly still uninitialized) block allocation before that store.
// 判断 To 产生的线程 memref 是否先被一个全 tile、零起点 vector.store 覆盖，
// 且该 store 支配所有其他使用。若是，可只分配 scratch 而不读取旧 block 内容。
// 这是避免无意义/未初始化读取的保守判断，不证明任意标量循环都能完全覆盖。
static bool isFullyOverwritten(frisk::ToThreadTileOp op, MemRefType type) {
  for (Operation *user : op.getResult().getUsers()) {
    auto store = dyn_cast<vector::StoreOp>(user);
    if (!store || store.getBase() != op.getResult() ||
        store.getVectorType().getShape() != type.getShape() ||
        !llvm::all_of(store.getIndices(), [](Value index) {
          auto c = index.getDefiningOp<arith::ConstantIndexOp>();
          return c && c.value() == 0;
        }))
      continue;
    DominanceInfo dominance(op->getParentOfType<func::FuncOp>());
    if (llvm::all_of(op.getResult().getUsers(), [&](Operation *other) {
          return other == user || dominance.properlyDominates(user, other);
        }))
      return true;
  }
  return false;
}

// 物化 To 时的逐元素读取器：先取得 block 坐标，再从 memref load 或
// block vector extract，最后插入线程 vector 的局部 iv 位置。
struct ReadThreadTileElement {
  OpBuilder &builder;
  Location loc;
  Value source, tid;
  ShapedType blockType;
  const ThreadTileAccess &access;
  // 读取当前线程负责的一个逻辑元素，并返回更新后的线程 vector。
  Value emit(ArrayRef<Value> ivs, Value tile) {
    auto indices = access.indices(builder, loc, tid, ivs, blockType);
    Value scalar;
    if (isa<MemRefType>(blockType))
      scalar = builder.create<memref::LoadOp>(loc, source, indices);
    else
      scalar = builder.create<vector::ExtractOp>(
          loc, source, SmallVector<OpFoldResult>(indices.begin(), indices.end()));
    return scalar;
  }
};

// 将逻辑 ToThreadTile 桥接落地：同布局桥接抵消；不同布局经 shared 交换；
// 普通来源按映射读入线程 tile；GEMM fragment 来源只在 extract 使用点加载。
class ToThreadTileFinalization : public OpRewritePattern<frisk::ToThreadTileOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::ToThreadTileOp op,
                                PatternRewriter &rewriter) const override {
    auto block = cast<ShapedType>(op.getBlockTile().getType());
    auto thread = cast<ShapedType>(op.getResult().getType());
    ThreadTileAccess access;
    if (failed(access.read(op, block, thread)))
      return rewriter.notifyMatchFailure(op, "invalid or missing tile layout");
    auto memref = dyn_cast<MemRefType>(thread);
    if (memref && (!memref.getLayout().isIdentity() ||
                   memref.getMemorySpaceAsInt() != 0))
      return rewriter.notifyMatchFailure(op, "expected private thread scratch");
    auto loc = op.getLoc();
    Value source = op.getBlockTile();
    // fragment_source 的所有用户必须是同一 block 中的标量 extract。
    // 先完整验证用户，再逐个把线程位置映射成物理地址并替换为 memref.load；
    // 动态的 m/n/k 起点仍留在 fragment region 内，供后续预取调度使用。
    auto materializeFragment = [&](Value buffer, ShapedType physical,
                                   const ThreadTileAccess &mapping) -> LogicalResult {
      SmallVector<vector::ExtractOp> reads;
      for (Operation *user : op.getResult().getUsers()) {
        auto read = dyn_cast<vector::ExtractOp>(user);
        if (!read || read->getBlock() != op->getBlock() || isa<VectorType>(read.getType()))
          return failure();
        reads.push_back(read);
      }
      Value tid = mapping.map ? findThreadIdxOp(op, rewriter) : Value{};
      for (auto read : reads) {
        rewriter.setInsertionPoint(read);
        SmallVector<Value> positions;
        for (OpFoldResult index : read.getMixedPosition())
          positions.push_back(isa<Value>(index) ? cast<Value>(index)
              : createIndexConstant(rewriter, loc, cast<IntegerAttr>(cast<Attribute>(index)).getInt()));
        auto indices = mapping.indices(rewriter, loc, tid, positions, physical);
        rewriter.replaceOpWithNewOp<memref::LoadOp>(read, buffer, indices);
      }
      rewriter.eraseOp(op);
      return success();
    };
    bool exchanged = false;
    if (auto from = source.getDefiningOp<frisk::FromThreadTileOp>()) {
      // This also handles the temporary pair emitted by writeBackBlockTile.
      auto fromAttrs = NamedAttrList(from->getAttrs());
      auto toAttrs = NamedAttrList(op->getAttrs());
      fromAttrs.erase("frisk.fragment_source");
      toAttrs.erase("frisk.fragment_source");
      // block_repeat budgets the full tile; it is absent from the per-element
      // address mapping. Actual thread shapes are checked below. In particular,
      // broadcasting a loop-carried rowsum changes this budget, not its values.
      fromAttrs.erase("block_repeat");
      toAttrs.erase("block_repeat");
      if (fromAttrs == toAttrs) {
        Value replacement = from.getThreadTile();
        if (replacement.getType() == thread) {
          rewriter.replaceOp(op, replacement);
          return success();
        }
        auto storageType = dyn_cast<MemRefType>(replacement.getType());
        auto vectorType = dyn_cast<VectorType>(thread);
        if (storageType && vectorType &&
            storageType.getShape() == vectorType.getShape() &&
            storageType.getElementType() == vectorType.getElementType() &&
            storageType.isLastDimUnitStride()) {
          if (op->hasAttr("frisk.fragment_source"))
            return materializeFragment(replacement, storageType, ThreadTileAccess{});
          SmallVector<Value> zeros(thread.getRank(), createIndexConstant(rewriter, loc, 0));
          rewriter.replaceOpWithNewOp<vector::LoadOp>(op, vectorType, replacement, zeros);
          return success();
        }
        return rewriter.notifyMatchFailure(op, "incompatible thread storage");
      }
      auto info = findLowerInfoForValue(source, op);
      if (!info || !isa<VectorType>(thread))
        return rewriter.notifyMatchFailure(op, "unsupported thread layout transition");
      auto scratchType = MemRefType::get(block.getShape(), block.getElementType(),
          AffineMap{}, rewriter.getI64IntegerAttr(int(friskMs::Shared)));
      source = rewriter.create<memref::AllocOp>(loc, scratchType);
      writeBackBlockTile(op.getBlockTile(), source, *info, rewriter, loc);
      block = scratchType;
      exchanged = true;
    }
    if (op->hasAttr("frisk.fragment_source") && isa<MemRefType>(block) && !exchanged)
      return materializeFragment(source, block, access);
    Value storage;
    if (memref) {
      storage = rewriter.create<memref::AllocaOp>(loc, memref);
      if (isFullyOverwritten(op, memref)) {
        rewriter.replaceOp(op, storage);
        return success();
      }
    }
    auto vectorType = VectorType::get(thread.getShape(), thread.getElementType());
    Value initial = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));
    Value tid = access.map ? findThreadIdxOp(op, rewriter) : Value{};
    ReadThreadTileElement element{rewriter, loc, source, tid, block, access};
    auto unroll = op->getAttrOfType<BoolAttr>("frisk.loopUnrollFull");
    SmallVector<int64_t> fragment(thread.getShape());
    if (access.map) {
      for (unsigned dim = 0; dim < fragment.size(); ++dim) {
        unsigned axis = fragment.size() == 1 && access.ignoreDim == 0 ? 1 : dim;
        fragment[dim] = thread.getDimSize(dim) == 1 ? 1
            : access.widths[axis] * access.repeats[axis];
      }
    }
    FragmentTileLoopNest loops(rewriter, loc, thread.getShape(),
                            {"tiled", "load_fragment", unroll && unroll.getValue()},
                            fragment);
    Value tile = loops.emit(initial, element, "load");
    if (exchanged)
      rewriter.create<gpu::BarrierOp>(loc);
    if (storage) {
      SmallVector<Value> zeros(thread.getRank(), createIndexConstant(rewriter, loc, 0));
      rewriter.create<vector::StoreOp>(loc, tile, storage, zeros);
      tile = storage;
    }
    rewriter.replaceOp(op, tile);
    return success();
  }
};

// copy_to_reg already has a thread-shaped result and needs no redistribution.
// 仅消除没有 tile_layout 且输入输出类型相同的 From，例如显式切片 copy_to_reg。
// 它不负责把分散在多个线程的值聚合为真实 block 矩阵。
class FromThreadVectorFinalization : public OpRewritePattern<frisk::FromThreadTileOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::FromThreadTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("tile_layout") ||
        op.getResult().getType() != op.getThreadTile().getType())
      return failure();
    rewriter.replaceOp(op, op.getThreadTile());
    return success();
  }
};

// Thread scratch used only to store one vector and read it back is an SSA
// value in disguise. Reject aliases/escaping uses and additional writes, so
// forwarding never changes observable memory or crosses a possible clobber.
// 把只用于“一次 vector.store + 若干同位置 vector.load”的私有 alloca
// 转发为原 SSA vector，要求 store 支配所有 load、类型/索引匹配、没有其他用户。
// 拒绝别名、逃逸和额外写入，避免把可能变化的存储错误地当成不变 SSA 值。
class ForwardThreadScratch : public OpRewritePattern<memref::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType().getMemorySpaceAsInt() != 0)
      return failure();
    vector::StoreOp store;
    SmallVector<vector::LoadOp> loads;
    for (Operation *user : op.getResult().getUsers()) {
      if (auto write = dyn_cast<vector::StoreOp>(user)) {
        if (store || write.getBase() != op.getResult())
          return failure();
        store = write;
      } else if (auto read = dyn_cast<vector::LoadOp>(user)) {
        if (read.getBase() != op.getResult())
          return failure();
        loads.push_back(read);
      } else {
        return failure();
      }
    }
    if (!store || loads.empty())
      return failure();
    DominanceInfo dominance(op->getParentOfType<func::FuncOp>());
    for (auto load : loads) {
      if (load.getType() != store.getValueToStore().getType() ||
          !llvm::equal(load.getIndices(), store.getIndices()) ||
          !dominance.properlyDominates(store.getOperation(), load.getOperation()))
        return failure();
    }
    for (auto load : loads)
      rewriter.replaceOp(load, store.getValueToStore());
    rewriter.eraseOp(store);
    rewriter.eraseOp(op);
    return success();
  }
};

#undef  GEN_PASS_DEF_CONVERTFRISKBASETOTHREADLEVELIR
#define GEN_PASS_DEF_FINALIZETHREADTILING
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"


// 第二阶段 pass，消除线程 tiling 的中间表示。
// 顺序有语义意义：先 cast/循环携带值重写和桥接抵消，再实际展开访存，
// 随后消除纯暂存、检查是否仍有 To/From/view/cast。
// fragment region/循环在此保留；共享内存复用和标记循环展开由后续 pass 完成。
// 最终检查失败表示存在尚不支持的布局或使用方式，不会假装已经完成 lowering。
// WarpMmaRROp 等并不由这里全部降成机器指令；它仍需后续专门的转换 pass。
class FinalizeThreadTiling : public impl::FinalizeThreadTilingBase< FinalizeThreadTiling>{
public:
  // 声明本 pass 会创建的 dialect，让 pass 管理器提前加载相应 IR 定义。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<frisk::FriskDialect, arith::ArithDialect, affine::AffineDialect,
        vector::VectorDialect, math::MathDialect, memref::MemRefDialect,
        gpu::GPUDialect, scf::SCFDialect>();
  }
  // 对当前 func.func 执行本阶段转换；具体处理顺序见下方分支和类说明。
  void runOnOperation() override {
    auto kernel = getOperation();
    if (!kernel->hasAttr("thread_num")){
      return;
    }
    // Keep cast/bridge cancellation ahead of materialization, otherwise a
    // register-only conversion would accidentally become a shared-memory copy.
    // 先把类型转换和循环边界移动到线程值上，暴露更多可抵消的 To(From(...))。
    // 若提前物化 load/scratch，原本纯寄存器计算可能变成多余的共享内存搬运。
    RewritePatternSet casts(&getContext());
    casts.add<CastFinalization, ThreadTileLoopFinalization>(&getContext());
    if (failed(applyPatternsGreedily(kernel, std::move(casts)))) {
      signalPassFailure();
      return;
    }
    foldThreadTilePairs(kernel);
    RewritePatternSet patterns(&getContext());
    patterns.add<BufferViewFinalization, ToThreadTileFinalization,
                 FromThreadVectorFinalization>(&getContext());
    if (failed(applyPatternsGreedily(kernel, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet forwarding(&getContext());
    forwarding.add<ForwardThreadScratch>(&getContext());
    if (failed(applyPatternsGreedily(kernel, std::move(forwarding)))) {
      signalPassFailure();
      return;
    }
    eraseTriviallyDeadOps(kernel);
    // 完成中间桥接消除的强制检查：仍有任何桥接/view/cast 就报告具体 op 并失败。
    WalkResult result = kernel.walk([&](Operation *op) {
      if (isa<frisk::ToThreadTileOp, frisk::FromThreadTileOp,
              frisk::BufferViewOp, frisk::CastOp>(op)) {
        op->emitError("could not finalize thread tile; unsupported layout or use");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    // Shared storage planning is deferred until after fragment scheduling.

    // Fragment regions and loops are deliberately preserved here. The late
    // lower-frisk-fragments pass runs only after fusion/software pipelining.

  }
};

}


// 第二阶段的工厂入口，对应命令行 --finalize-thread-tiling。
std::unique_ptr<mlir::Pass> createFinalizeThreadTilingPass(){
  return std::make_unique<FinalizeThreadTiling>();
}

} // namespace mlir::frisk
