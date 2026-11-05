// ===========================================================================
// PackSharedMemory（pack-shared-memory）：packSharedMemory 是 LowerFriskFragments（优化汇总.py 第 15 项）的第一步。
//
// 作用：
//   把带 frisk.shared_pack 标注的二维 LDS alloc 改成一维物理布局，所有访问按同一个置换重写，
//   DMA 附上反向映射 frisk.source_map。相邻访问合并成更宽的向量读写。K 的 swizzle（第 6 项打的属性）
//   也在这里落实。外部 ABI 和全局张量布局不变。
//
// 输入属性（由 ConvertFriskBaseToFragmentIR 写在 alloc 上）：
//   frisk.shared_pack         一个包的元素数（如 16-bit 元素取 8 表示 128-bit 包）。
//   frisk.shared_pack_axis    沿第 0 轴还是第 1 轴打包。
//   frisk.shared_pair_stride  成对打包的跨度。
//   frisk.shared_store_stride / frisk.shared_store_pair  写入侧的跨度和配对。
//   frisk.shared_packet_rows  行组内交错完整的列包，0 表示旧行为。
//   frisk.shared_swizzle_src / frisk.shared_swizzle_dst  字节地址 XOR：dst 位 ^= src 位。
// 实际映射公式见 ShmDataIODesc（lib/Analysis/ShmDataIODesc.cpp）。
//
// 要点：一个连通的别名分量必须整体一起改。一旦不满足条件就不改；但如果要求了 swizzle 却没有改成，
// packSharedMemory 末尾会直接报错，因为 swizzle 是用户明确请求的。
// ===========================================================================
// Physical LDS packing is separate from lane ownership / logical tile layout.
// A complete owned allocation is permuted together with all its accesses;
// no global tensor layout or kernel ABI is changed.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "SharedBufferAliases.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_PACKSHAREDMEMORY
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

// Compose index expressions without mutating IR. Combining the operands in a
// single map lets canonicalization prove equality even through affine.apply.
static AffineMap compose(AffineMap map, SmallVectorImpl<Value> &operands) {
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  return simplifyAffineMap(map);
}

// 证明 next - first 恒等于 distance。
static bool consecutive(Value first, Value next, int64_t distance) {
  auto *ctx = first.getContext();
  SmallVector<Value> operands{next, first};
  auto map = compose(AffineMap::get(2, 0,
      getAffineDimExpr(0, ctx) - getAffineDimExpr(1, ctx), ctx), operands);
  auto constant = dyn_cast<AffineConstantExpr>(map.getResult(0));
  return constant && constant.getValue() == distance;
}

// 一次需要重写的访问。
//   op / buffer  访问 op 和它用的 memref 别名。
//   indices      逻辑坐标的操作数。
//   map          affine 访问的 map，memref 访问时为空。
//   stored       store 的值，load 时为空。
struct Access {
  Operation *op;
  Value buffer;
  SmallVector<Value> indices;
  AffineMap map;
  Value stored;
};

// 收集别名分量的全部访问。遇到任何未支持的 op（调用、subview、已有 vector 访问等）返回 false，整体放弃。
// 矩阵读有额外要求：包必须正好 128-bit，读的元素在物理上要连续。
static bool collect(const SharedBufferAliases &aliases, SmallVectorImpl<Access> &accesses,
                    SmallVectorImpl<memref::DeallocOp> &deallocs) {
  for (OpOperand *use : aliases.accesses) {
    Operation *user = use->getOwner();
    Value buffer = use->get();
    if (auto load = dyn_cast<memref::LoadOp>(user))
      accesses.push_back({user, buffer, llvm::to_vector(load.getIndices()), {}, {}});
    else if (auto store = dyn_cast<memref::StoreOp>(user))
      accesses.push_back({user, buffer, llvm::to_vector(store.getIndices()), {}, store.getValue()});
    else if (auto load = dyn_cast<affine::AffineLoadOp>(user))
      accesses.push_back({user, buffer, llvm::to_vector(load.getMapOperands()), load.getAffineMap(), {}});
    else if (auto store = dyn_cast<affine::AffineStoreOp>(user))
      accesses.push_back({user, buffer, llvm::to_vector(store.getMapOperands()), store.getAffineMap(), store.getValue()});
    else if (auto read = dyn_cast<ReadMatrixFromShmOp>(user)) {
      // Matrix reads use one contiguous 16-byte input packet per lane. Only
      // move its base when the packing contract preserves the whole packet.
      auto axis = aliases.commonAttribute("frisk.shared_pack_axis");
      auto width = dyn_cast_or_null<IntegerAttr>(aliases.commonAttribute("frisk.shared_pack"));
      auto axisInt = dyn_cast_or_null<IntegerAttr>(axis);
      auto type = cast<MemRefType>(buffer.getType());
      if (read.isBlockForm() || !axisInt || axisInt.getInt() != 1 || !width ||
          width.getInt() * type.getElementTypeBitWidth() != 128 ||
          (read.getM() != 32 || (read.getN() != 16 && read.getN() != 8)) ||
          read.getTranspose() || read.getAlt() != 0 || read.getOffset() != 0)
        return false;
      for (Operation *storage : aliases.allocations)
        if (storage->hasAttr("frisk.shared_pair_stride") ||
            storage->hasAttr("frisk.shared_store_stride")) return false;
      accesses.push_back({user, buffer, llvm::to_vector(read.getIndices()), {}, {}});
    }
    else if (auto free = dyn_cast<memref::DeallocOp>(user))
      deallocs.push_back(free);
    else if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(user)) {
      if (use->getOperandNumber() != 1 ||
          cast<MemRefType>(buffer.getType()).getNumElements() > (1 << 20) ||
          cast<MemRefType>(copy.getSrc().getType()).getShape() !=
              cast<MemRefType>(buffer.getType()).getShape() ||
          copy->hasAttr("frisk.source_map"))
        return false;
    }
    else
      // Includes escaping descriptors, calls, atomics, subviews and
      // preexisting vector accesses whose footprint has not been proven.
      return false;
  }
  return !accesses.empty();
}

// Merge only exact adjacent addresses with no intervening memory operation,
// control flow, barrier, or non-speculatable computation. Reads move to the
// first read; writes to the last write. This never reads padding or a tail.
// 把相邻地址的标量 load/store 合并成 vector，宽度取不超过 128-bit 的 2 的幂。
//   group    当前连续访问。
//   first    group 的第一个下标。
//   writing  group 是读还是写（不混合）。
//   flush    把 group 发射成向量访问。
// 读合并到第一条的位置，写合并到最后一条的位置。
static void coalesce(Block *block, Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  int64_t limit = 128 / type.getElementTypeBitWidth();
  SmallVector<Operation *> group;
  Value first;
  bool writing = false;
  auto flush = [&]() {
    unsigned offset = 0;
    while (offset < group.size()) {
      unsigned width = 1;
      while (2 * width <= std::min<uint64_t>(limit, group.size() - offset))
        width *= 2;
      if (width < 2) { ++offset; continue; }
      Operation *begin = group[offset], *end = group[offset + width - 1];
      IRRewriter b(begin->getContext());
      b.setInsertionPoint(writing ? end : begin);
      auto loc = begin->getLoc();
      auto vectorType = VectorType::get({int64_t(width)}, type.getElementType());
      Value index = writing ? cast<memref::StoreOp>(begin).getIndices()[0]
                            : cast<memref::LoadOp>(begin).getIndices()[0];
      if (writing) {
        Value packet = b.create<arith::ConstantOp>(loc, vectorType, b.getZeroAttr(vectorType));
        for (unsigned i = 0; i < width; ++i)
          packet = b.create<vector::InsertOp>(loc,
              cast<memref::StoreOp>(group[offset + i]).getValue(), packet,
              ArrayRef<OpFoldResult>{b.getIndexAttr(i)});
        b.create<vector::StoreOp>(loc, packet, buffer, ValueRange{index});
        for (unsigned i = 0; i < width; ++i)
          b.eraseOp(group[offset + i]);
      } else {
        Value packet = b.create<vector::LoadOp>(loc, vectorType, buffer, ValueRange{index});
        SmallVector<Value> elements;
        for (unsigned i = 0; i < width; ++i)
          elements.push_back(b.create<vector::ExtractOp>(loc, packet,
              ArrayRef<OpFoldResult>{b.getIndexAttr(i)}));
        for (unsigned i = 0; i < width; ++i)
          b.replaceOp(group[offset + i], elements[i]);
      }
      offset += width;
    }
    group.clear();
  };
  // Collect first: rewriting a group can erase nodes preceding this iterator.
  SmallVector<Operation *> operations;
  for (Operation &op : *block)
    operations.push_back(&op);
  for (Operation *op : operations) {
    Value index;
    bool write = false;
    if (auto load = dyn_cast<memref::LoadOp>(op); load && load.getMemref() == buffer)
      index = load.getIndices()[0];
    else if (auto store = dyn_cast<memref::StoreOp>(op); store && store.getMemref() == buffer) {
      index = store.getIndices()[0];
      write = true;
    }
    if (index) {
      if (!group.empty() && (writing != write ||
          !consecutive(first, index, group.size())))
        flush();
      if (group.empty()) { first = index; writing = write; }
      group.push_back(op);
    } else if (op->getNumRegions() || !isMemoryEffectFree(op) || !isSpeculatable(op)) {
      flush();
    }
  }
  flush();
}

// 处理一块 alloc 及其别名分量。全部检查都在改写前做完。
static void packAllocation(memref::AllocOp alloc,
                           llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (visited.contains(alloc))
    return;
  SharedBufferAliases aliases;
  if (!aliases.collect(alloc.getResult()))
    return;
  for (Operation *storage : aliases.allocations)
    visited.insert(storage);
  // Never permute one branch independently. Even a missing annotation on an
  // alternative invalidates the entire component, including direct accesses.
  // 分量里的所有 alloc 必须属性完全一致。例如 select 的两个分支不能只改一支。
  for (Operation *storage : aliases.allocations) {
    if (!isa<memref::AllocOp>(storage))
      return;
    for (StringRef name : {"frisk.shared_pack", "frisk.shared_pack_axis",
                           "frisk.shared_pair_stride", "frisk.shared_store_stride",
                           "frisk.shared_store_pair", "frisk.shared_packet_rows",
                           "frisk.shared_swizzle_src", "frisk.shared_swizzle_dst"})
      if (storage->getAttr(name) != alloc->getAttr(name))
        return;
  }
  auto attr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_pack");
  auto type = alloc.getType();
  if (!attr || type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != 3 ||
      !type.getElementType().isIntOrFloat())
    return;
  auto axisAttr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_pack_axis");
  int64_t axis = axisAttr ? axisAttr.getInt() : 0;
  if (axis != 0 && axis != 1)
    return;
  int64_t pack = attr.getInt();
  if (pack < 2 || !llvm::isPowerOf2_64(pack) ||
      pack * type.getElementTypeBitWidth() > 128 ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      type.getDimSize(axis) % pack)
    return;
  auto pairAttr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_pair_stride");
  int64_t pairStride = pairAttr ? pairAttr.getInt() : 0;
  auto storeAttr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_store_stride");
  auto storePairAttr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_store_pair");
  int64_t storeStride = storeAttr ? storeAttr.getInt() : 0;
  int64_t storePair = storePairAttr ? storePairAttr.getInt() : 0;
  auto packetRowsAttr = alloc->getAttrOfType<IntegerAttr>("frisk.shared_packet_rows");
  int64_t packetRows = packetRowsAttr ? packetRowsAttr.getInt() : 0;
  if ((storeAttr || storePairAttr) &&
      (axis != 1 || !storeAttr || !storePairAttr || storeStride < 2 ||
       !llvm::isPowerOf2_64(storeStride) || storePair < storeStride ||
       storePair > type.getDimSize(1) / 2 || storePair % storeStride ||
       type.getDimSize(1) % (2 * storePair) || pairAttr))
    return;
  if (pairAttr && (axis != 0 || pairStride < pack || pairStride > type.getDimSize(0) / 2 || pairStride % pack ||
      type.getDimSize(0) % (2 * pairStride) ||
      2 * pack * type.getElementTypeBitWidth() > 128))
    return;
  // io：物理布局描述，提供“逻辑 -> 物理”和“物理 -> 逻辑”的映射。
  auto io = ShmDataIODesc::getPacked(
      {type.getDimSize(0), type.getDimSize(1)},
      type.getElementTypeBitWidth() / 8, axis, pack, pairStride,
      storeStride, storePair, packetRows);
  if (type.getElementTypeBitWidth() % 8 || !io)
    return;
  auto swizzleSrc = alloc->getAttrOfType<IntegerAttr>("frisk.shared_swizzle_src");
  auto swizzleDst = alloc->getAttrOfType<IntegerAttr>("frisk.shared_swizzle_dst");
  if (swizzleSrc || swizzleDst)
    if (!swizzleSrc || !swizzleDst ||
        !io->setXorSwizzle(swizzleSrc.getInt(), swizzleDst.getInt())) return;
  SmallVector<Access> accesses;
  SmallVector<memref::DeallocOp> deallocs;
  if (!collect(aliases, accesses, deallocs))
    return;

  // 矩阵读只传一个基址，隐式读连续的 pack 个元素，所以必须证明这些元素在物理上真的连续。
  for (auto &access : accesses)
    if (isa<ReadMatrixFromShmOp>(access.op)) {
      // Moving just the lane pointer is correct only if every element read
      // implicitly by the instruction stays consecutive in physical storage.
      auto packetIO = *io;
      packetIO.swizzleSourceBit = packetIO.swizzleTargetBit = -1;
      auto logical = AffineMap::getMultiDimIdentityMap(2, alloc.getContext());
      auto last = AffineMap::get(2, 0,
          {logical.getResult(0), logical.getResult(1) + pack - 1}, alloc.getContext());
      auto delta = AffineMap::get(2, 0,
          packetIO.mapLogicalToPhysical(last).getResult(0) -
          packetIO.mapLogicalToPhysical(logical).getResult(0), alloc.getContext());
      SmallVector<Value> operands(access.indices);
      auto proven = compose(delta, operands).getResult(0);
      auto constant = dyn_cast<AffineConstantExpr>(proven);
      if (!constant || constant.getValue() != pack - 1) return;
      if (swizzleSrc) {
        // The XOR preserves aligned 16-byte packets, including their implicit
        // matrix-reader elements. Prove alignment before composing the XOR.
        auto residue = AffineMap::get(2, 0,
            packetIO.mapLogicalToPhysical(logical).getResult(0) % pack);
        operands.assign(access.indices.begin(), access.indices.end());
        auto aligned = dyn_cast<AffineConstantExpr>(compose(residue, operands).getResult(0));
        if (!aligned || aligned.getValue()) return;
      }
    }

  // Recheck the actual source type after view finalization, before mutation.
  if (swizzleSrc)
    for (OpOperand *use : aliases.accesses)
      if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(use->getOwner()))
        if (selectAsyncDMATransferBytes(copy, 16,
                io->getPhysicalToLogicalMap(alloc.getContext())) != 16) return;

  // 开始改写：physical = 一维 memref，buffers[旧别名] = 新的 memref。
  IRRewriter b(alloc.getContext());
  auto physical = MemRefType::get({type.getNumElements()}, type.getElementType(),
                                 AffineMap{}, type.getMemorySpace());
  DenseMap<Value, Value> buffers;
  for (Value alias : aliases.values)
    if (!SharedBufferAliases::isAllocation(alias)) {
      // Includes selector results and loop iter_args/results. Their incoming
      // descriptors are rewritten together below, preserving control flow.
      alias.setType(physical);
      buffers[alias] = alias;
    }
  for (Operation *allocation : aliases.allocations) {
    auto old = cast<memref::AllocOp>(allocation);
    b.setInsertionPoint(old);
    auto storage = b.create<memref::AllocOp>(old.getLoc(), physical, old.getAlignmentAttr());
    storage->setAttr("frisk.packed_shape", b.getDenseI64ArrayAttr(type.getShape()));
    storage->setAttr("frisk.packed_axis", b.getI64IntegerAttr(axis));
    storage->setAttr("frisk.packed_width", b.getI64IntegerAttr(pack));
    if (packetRows)
      storage->setAttr("frisk.packed_packet_rows", b.getI64IntegerAttr(packetRows));
    if (pairStride)
      storage->setAttr("frisk.packed_pair_stride", b.getI64IntegerAttr(pairStride));
    if (storeStride) {
      storage->setAttr("frisk.packed_store_stride", b.getI64IntegerAttr(storeStride));
      storage->setAttr("frisk.packed_store_pair", b.getI64IntegerAttr(storePair));
    }
    if (auto tiled = old->getAttr("tiled")) storage->setAttr("tiled", tiled);
    buffers[old.getResult()] = storage;
    old.getResult().replaceAllUsesWith(storage);
  }
  // The same descriptor maps consumers to physical LDS and inverts that
  // placement for GM producers. Preserve source strides in the copy's memref.
  // inverse：物理 LDS 元素 -> 逻辑坐标。DMA lowering 用 frisk.source_map 计算每个 lane 的 GM 地址。
  auto inverse = io->getPhysicalToLogicalMap(b.getContext());
  for (OpOperand *use : aliases.accesses)
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(use->getOwner()))
      copy->setAttr("frisk.source_map", AffineMapAttr::get(inverse));
  // blocks：需要做访问合并的 (基本块, buffer)。下面的循环重写每个访问的下标。
  llvm::SetVector<std::pair<Block *, Value>> blocks;
  for (auto &access : accesses) {
    b.setInsertionPoint(access.op);
    auto logical = compose(access.map ? access.map :
        b.getMultiDimIdentityMap(2), access.indices);
    auto map = io->mapLogicalToPhysical(logical);
    Value index = b.create<affine::AffineApplyOp>(access.op->getLoc(), map, access.indices);
    Value storage = buffers.lookup(access.buffer);
    blocks.insert({access.op->getBlock(), storage});
    if (auto read = dyn_cast<ReadMatrixFromShmOp>(access.op)) {
      b.replaceOpWithNewOp<ReadMatrixFromShmOp>(read, read.getResult().getType(),
          storage, ValueRange{index}, read.getM(), read.getN(),
          read.getElementBitWidth(), read.getTranspose(), read.getAlt(), read.getOffset());
    } else if (access.stored)
      b.replaceOpWithNewOp<memref::StoreOp>(access.op, access.stored, storage, ValueRange{index});
    else
      b.replaceOpWithNewOp<memref::LoadOp>(access.op, storage, ValueRange{index});
  }
  for (Operation *allocation : aliases.allocations)
    b.eraseOp(allocation);
  for (auto [block, storage] : blocks) coalesce(block, storage);
}

class PackSharedMemory : public impl::PackSharedMemoryBase<PackSharedMemory> {
  void runOnOperation() override {
    if (failed(packSharedMemory(getOperation()))) signalPassFailure();
  }
};
} // namespace

LogicalResult packSharedMemory(func::FuncOp kernel) {
  SmallVector<memref::AllocOp> allocations;
  kernel.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
  llvm::SmallPtrSet<Operation *, 16> visited;
  for (auto alloc : allocations) packAllocation(alloc, visited);
  bool unhandled = false;
  kernel.walk([&](memref::AllocOp alloc) {
    if (alloc->hasAttr("frisk.shared_swizzle_src") || alloc->hasAttr("frisk.shared_swizzle_dst")) {
      alloc.emitError("cannot safely apply requested shared swizzle to every access / alias while retaining 128-bit DMA");
      unhandled = true;
    }
  });
  return failure(unhandled);
}

std::unique_ptr<Pass> createPackSharedMemoryPass() {
  return std::make_unique<PackSharedMemory>();
}
} // namespace mlir::frisk
