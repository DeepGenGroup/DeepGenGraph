// Physical LDS packing is separate from lane ownership / logical tile layout.
// A complete owned allocation is permuted together with all its accesses;
// no global tensor layout or kernel ABI is changed.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
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

static bool consecutive(Value first, Value next, int64_t distance) {
  auto *ctx = first.getContext();
  SmallVector<Value> operands{next, first};
  auto map = compose(AffineMap::get(2, 0,
      getAffineDimExpr(0, ctx) - getAffineDimExpr(1, ctx), ctx), operands);
  auto constant = dyn_cast<AffineConstantExpr>(map.getResult(0));
  return constant && constant.getValue() == distance;
}

struct Access {
  Operation *op;
  Value buffer;
  SmallVector<Value> indices;
  AffineMap map;
  Value stored;
};

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
    else if (auto free = dyn_cast<memref::DeallocOp>(user))
      deallocs.push_back(free);
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
  for (Operation *storage : aliases.allocations) {
    if (!isa<memref::AllocOp>(storage))
      return;
    for (StringRef name : {"frisk.shared_pack", "frisk.shared_pack_axis",
                           "frisk.shared_pair_stride", "frisk.shared_store_stride",
                           "frisk.shared_store_pair"})
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
  SmallVector<Access> accesses;
  SmallVector<memref::DeallocOp> deallocs;
  if (!collect(aliases, accesses, deallocs))
    return;

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
  auto row = b.getAffineDimExpr(0), col = b.getAffineDimExpr(1);
  auto layout = AffineMap::get(2, 0,
      (row.floorDiv(pack) * type.getDimSize(1) + col) * pack + row % pack);
  if (axis == 1)
    // A bijection for any positive M and N divisible by pack: transpose the
    // grid of row packets, preserving each lane's contiguous register packet.
    // No padding, bank-count assumption, or logical ownership change.
    layout = AffineMap::get(2, 0,
        (col.floorDiv(pack) * type.getDimSize(0) + row) * pack + col % pack);
  if (storeStride) {
    // A bijection: column = laneResidue + storeStride * registerIndex.
    // Transpose the lane-residue and row axes, then interleave two instruction
    // groups within registerIndex. No padding or cross-thread shuffle is used.
    auto reg = col.floorDiv(storeStride);
    int64_t group = storePair / storeStride;
    auto paired = reg.floorDiv(2 * group) * (2 * group) +
                  (reg % group) * 2 + (reg.floorDiv(group) % 2);
    layout = AffineMap::get(2, 0,
        (col % storeStride * type.getDimSize(0) + row) *
            (type.getDimSize(1) / storeStride) + paired);
  }
  llvm::SetVector<std::pair<Block *, Value>> blocks;
  for (auto &access : accesses) {
    b.setInsertionPoint(access.op);
    AffineMap map;
    if (pairStride) {
      // Simplify the fragment number before introducing the outer pairing
      // permutation. Otherwise (base*4 + register) floordiv 32 obscures that
      // all four registers have the same fragment number, defeating exact
      // adjacent-address proofs in coalesce().
      auto logical = compose(access.map ? access.map :
          b.getMultiDimIdentityMap(2), access.indices);
      auto simplify = [&](AffineExpr expr) {
        return simplifyAffineExpr(expr, logical.getNumDims(), logical.getNumSymbols());
      };
      auto fragment = simplify(logical.getResult(0).floorDiv(pack));
      auto element = simplify(logical.getResult(0) % pack);
      int64_t groups = pairStride / pack;
      auto group = fragment.floorDiv(2 * groups) * groups + fragment % groups;
      auto column = logical.getResult(1);
      if (2 * pack * type.getElementTypeBitWidth() == 128 &&
          type.getDimSize(1) % 8 == 0) {
        // Permute whole 16-byte packets within each eight-column group.
        // Pairing K alone puts the copy lanes' 64-bit stores at the same bank
        // offset. Fold the producer column group into the packet's low bits.
        // Keep columns 0:3 and 4:7 separate for the paired LDS read phases;
        // the row-group high bit is equal for each pair of read lane groups.
        // This is a bijection, requires no padding, and leaves both halves of
        // a K packet adjacent. Express it in affine arithmetic so adjacency
        // remains provable after fragment unrolling.
        auto columnGroup = column.floorDiv(8);
        column = columnGroup * 8 +
            ((column.floorDiv(4) + columnGroup.floorDiv(4) + group.floorDiv(2)) % 2) * 4 +
            (column + columnGroup) % 4;
      }
      auto physicalIndex = (group * type.getDimSize(1) + column) * (2 * pack) +
                           (fragment.floorDiv(groups) % 2) * pack + element;
      map = simplifyAffineMap(AffineMap::get(logical.getNumDims(), logical.getNumSymbols(), physicalIndex));
    } else {
      map = compose(access.map ? layout.compose(access.map) : layout, access.indices);
    }
    Value index = b.create<affine::AffineApplyOp>(access.op->getLoc(), map, access.indices);
    Value storage = buffers.lookup(access.buffer);
    blocks.insert({access.op->getBlock(), storage});
    if (access.stored)
      b.replaceOpWithNewOp<memref::StoreOp>(access.op, access.stored, storage, ValueRange{index});
    else
      b.replaceOpWithNewOp<memref::LoadOp>(access.op, storage, ValueRange{index});
  }
  for (Operation *allocation : aliases.allocations)
    b.eraseOp(allocation);
  for (auto [block, storage] : blocks) coalesce(block, storage);
}

class PackSharedMemory : public impl::PackSharedMemoryBase<PackSharedMemory> {
  void runOnOperation() override { packSharedMemory(getOperation()); }
};
} // namespace

void packSharedMemory(func::FuncOp kernel) {
  SmallVector<memref::AllocOp> allocations;
  kernel.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
  llvm::SmallPtrSet<Operation *, 16> visited;
  for (auto alloc : allocations) packAllocation(alloc, visited);
}

std::unique_ptr<Pass> createPackSharedMemoryPass() {
  return std::make_unique<PackSharedMemory>();
}
} // namespace mlir::frisk
