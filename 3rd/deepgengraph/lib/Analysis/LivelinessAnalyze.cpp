#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace mlir::frisk {
namespace {

constexpr StringLiteral poolAttr = "frisk.shm_pool";
constexpr uint64_t maxBytes = std::numeric_limits<int64_t>::max();

bool isShared(MemRefType type) {
  auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
  return space && space.getInt() == 3;
}

bool isBarrier(Operation *op) {
  return isa<gpu::BarrierOp, SyncThreadsInBlockOp>(op);
}

// These regions finish synchronously. async.execute and parallel regions must
// not be treated as sequential operations when computing storage lifetimes.
Operation *getTopLevelUser(Operation *op, Block *entry) {
  while (op->getBlock() != entry) {
    op = op->getParentOp();
    if (!op || !isa<scf::ForOp, scf::IfOp, affine::AffineForOp,
                    affine::AffineIfOp>(op))
      return nullptr;
  }
  return op;
}

bool isSynchronousAccess(Operation *op) {
  // MemoryEffectOpInterface alone is insufficient: async copies also report
  // reads/writes, but their last operand use is not their completion point.
  return isa<memref::LoadOp, memref::StoreOp, memref::CopyOp,
             memref::DimOp, memref::AssumeAlignmentOp,
             affine::AffineLoadOp, affine::AffineStoreOp,
             affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
             vector::LoadOp, vector::StoreOp, vector::TransferReadOp,
             vector::TransferWriteOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp>(op);
}

void analyzeUses(ShmLiveInterval &info, Block *entry,
                 const DenseMap<Operation *, unsigned> &positions) {
  SmallVector<Value> worklist{info.allocation.getResult()};
  DenseSet<Value> visited;
  bool hasUse = false;
  info.firstUse = std::numeric_limits<unsigned>::max();
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    info.aliases.push_back(value);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      Operation *top = getTopLevelUser(user, entry);
      if (!top) {
        info.skipReason = "use inside unsupported/async control flow";
        return;
      }
      auto recordUse = [&] {
        unsigned position = positions.lookup(top);
        info.firstUse = std::min(info.firstUse, position);
        info.lastUse = std::max(info.lastUse, position);
        hasUse = true;
      };
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == value && user->getNumRegions() == 0 &&
            isMemoryEffectFree(user)) {
          for (Value result : user->getResults())
            if (isa<BaseMemRefType>(result.getType()))
              worklist.push_back(result);
          continue; // Constructing a descriptor does not touch storage.
        }
      }
      if (isa<memref::CastOp, arith::SelectOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // Loop inits also reach the result along the zero-trip path.
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        for (auto init : llvm::enumerate(loop.getInitArgs()))
          if (init.value() == value) {
            worklist.push_back(loop.getRegionIterArgs()[init.index()]);
            worklist.push_back(loop.getResult(init.index()));
          }
        recordUse();
        continue;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(user)) {
        for (auto init : llvm::enumerate(loop.getInits()))
          if (init.value() == value) {
            worklist.push_back(loop.getRegionIterArgs()[init.index()]);
            worklist.push_back(loop.getResult(init.index()));
          }
        recordUse();
        continue;
      }
      if (isa<scf::YieldOp, affine::AffineYieldOp>(user)) {
        Operation *parent = user->getParentOp();
        unsigned index = use.getOperandNumber();
        if (isa<scf::IfOp, affine::AffineIfOp, scf::ForOp,
                affine::AffineForOp>(parent)) {
          worklist.push_back(parent->getResult(index));
          if (auto loop = dyn_cast<scf::ForOp>(parent))
            worklist.push_back(loop.getRegionIterArgs()[index]);
          if (auto loop = dyn_cast<affine::AffineForOp>(parent))
            worklist.push_back(loop.getRegionIterArgs()[index]);
          recordUse(); // Covers loop backedges and branch-selected aliases.
          continue;
        }
      }
      if (!isSynchronousAccess(user)) {
        info.skipReason = "unsupported or escaping use: " +
                          user->getName().getStringRef().str();
        return;
      }
      if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(user))
        info.alignment = std::max<uint64_t>(info.alignment, assume.getAlignment());
      recordUse();
    }
  }
  if (!hasUse) {
    info.skipReason = "no storage uses";
    return;
  }
  info.reusable = true;
}

bool overlaps(uint64_t a, uint64_t sizeA, uint64_t b, uint64_t sizeB) {
  return a < b + sizeB && b < a + sizeA;
}

} // namespace

bool ShmLivenessResult::interferes(unsigned lhs, unsigned rhs) const {
  const auto &a = buffers[lhs];
  const auto &b = buffers[rhs];
  return lhs == rhs || !a.reusable || !b.reusable ||
         (a.firstUse <= b.lastUse && b.firstUse <= a.lastUse);
}

ShmLivenessResult analyzeShmLiveness(func::FuncOp kernel) {
  ShmLivenessResult result;
  if (kernel.isExternal())
    return result;
  Block *entry = &kernel.getBody().front();
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *entry) {
    positions[&op] = result.programOrder.size();
    result.programOrder.push_back(&op);
  }
  kernel.walk([&](memref::AllocOp alloc) {
    auto type = alloc.getType();
    if (!isShared(type))
      return;
    result.buffers.emplace_back();
    auto &info = result.buffers.back();
    info.allocation = alloc;
    auto skip = [&](StringRef reason) { info.skipReason = reason.str(); };
    if (!llvm::hasSingleElement(kernel.getBody()))
      return skip("multi-block function");
    if (alloc->hasAttr(poolAttr))
      return skip("existing shared-memory pool");
    if (!getTopLevelUser(alloc, entry))
      return skip("allocation inside unsupported control flow");
    if (!type.hasStaticShape() || !type.getLayout().isIdentity())
      return skip("dynamic shape or non-identity layout");
    Type element = type.getElementType();
    if (!isa<IntegerType, FloatType>(element))
      return skip("unsupported element storage size");
    unsigned bits = element.getIntOrFloatBitWidth();
    if (bits < 8 || bits % 8 || !llvm::isPowerOf2_64(bits / 8))
      return skip("unsupported element storage size");
    uint64_t size = bits / 8;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || uint64_t(dim) > maxBytes / size)
        return skip("empty or overflowing allocation");
      size *= dim;
    }
    info.sizeBytes = size;
    info.alignment = std::max<uint64_t>(bits / 8, alloc.getAlignment().value_or(1));
    analyzeUses(info, entry, positions);
  });
  return result;
}

ShmReusePlan planShmReuse(const ShmLivenessResult &liveness) {
  ShmReusePlan plan;
  SmallVector<unsigned> order;
  for (auto indexed : llvm::enumerate(liveness.buffers)) {
    const auto &info = indexed.value();
    if (!info.reusable)
      continue;
    if (info.sizeBytes > maxBytes - plan.originalBytes)
      return {};
    plan.originalBytes += info.sizeBytes;
    plan.alignment = std::max(plan.alignment, info.alignment);
    order.push_back(indexed.index());
  }
  llvm::stable_sort(order, [&](unsigned a, unsigned b) {
    return liveness.buffers[a].sizeBytes > liveness.buffers[b].sizeBytes;
  });
  for (unsigned index : order) {
    const auto &info = liveness.buffers[index];
    uint64_t offset = 0;
    // Move past conflicts until the lowest aligned free byte range is found.
    bool conflict;
    do {
      conflict = false;
      for (const auto &placed : plan.placements) {
        const auto &other = liveness.buffers[placed.bufferIndex];
        if (!liveness.interferes(index, placed.bufferIndex) ||
            !overlaps(offset, info.sizeBytes, placed.offsetBytes, other.sizeBytes))
          continue;
        uint64_t end = placed.offsetBytes + other.sizeBytes;
        if (info.alignment > maxBytes - end)
          return {};
        offset = llvm::alignTo(end, info.alignment);
        if (info.sizeBytes > maxBytes - offset)
          return {};
        conflict = true;
        break;
      }
    } while (conflict);
    plan.placements.push_back({index, offset});
    plan.pooledBytes = std::max(plan.pooledBytes, offset + info.sizeBytes);
  }
  return plan;
}

void dumpShmLiveness(const ShmLivenessResult &liveness, llvm::raw_ostream &os) {
  for (auto indexed : llvm::enumerate(liveness.buffers)) {
    const auto &info = indexed.value();
    os << "shm[" << indexed.index() << "] " << info.sizeBytes
       << " bytes, align " << info.alignment;
    if (info.reusable)
      os << ", live [" << info.firstUse << ", " << info.lastUse << "]";
    else
      os << ", skipped: " << info.skipReason;
    os << '\n';
  }
}

bool reuseSharedMemory(func::FuncOp kernel, ShmReuseStats *stats) {
  if (stats)
    *stats = {};
  if (!kernel->hasAttr("thread_num"))
    return false;
  auto liveness = analyzeShmLiveness(kernel);
  auto plan = planShmReuse(liveness);
  if (stats) {
    stats->originalBytes = plan.originalBytes;
    stats->pooledBytes = plan.originalBytes;
  }
  if (plan.placements.size() < 2 || plan.pooledBytes >= plan.originalBytes)
    return false;

  // Operand liveness does not guarantee every warp has finished reading.
  SmallVector<std::pair<unsigned, unsigned>> handoffs;
  for (auto a : plan.placements) {
    const auto &earlier = liveness.buffers[a.bufferIndex];
    for (auto b : plan.placements) {
      const auto &later = liveness.buffers[b.bufferIndex];
      if (earlier.lastUse >= later.firstUse ||
          !overlaps(a.offsetBytes, earlier.sizeBytes, b.offsetBytes, later.sizeBytes))
        continue;
      bool synchronized = false;
      for (unsigned i = earlier.lastUse + 1; i < later.firstUse; ++i)
        synchronized |= isBarrier(liveness.programOrder[i]);
      if (!synchronized)
        handoffs.emplace_back(later.firstUse, earlier.lastUse);
    }
  }
  // One barrier can satisfy several handoffs. Process the earliest deadline
  // first, and reuse a previously selected barrier whenever it covers the gap.
  llvm::sort(handoffs);
  SmallVector<unsigned> barrierPositions;
  for (auto handoff : handoffs)
    if (barrierPositions.empty() || barrierPositions.back() <= handoff.second)
      barrierPositions.push_back(handoff.first);
  OpBuilder builder(kernel.getContext());
  for (unsigned position : barrierPositions) {
    Operation *before = liveness.programOrder[position];
    builder.setInsertionPoint(before);
    builder.create<gpu::BarrierOp>(before->getLoc());
  }
  builder.setInsertionPointToStart(&kernel.getBody().front());
  auto poolType = MemRefType::get({int64_t(plan.pooledBytes)}, builder.getI8Type(),
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(3));
  auto pool = builder.create<memref::AllocOp>(
      kernel.getLoc(), poolType, builder.getI64IntegerAttr(plan.alignment));
  pool->setAttr(poolAttr, builder.getUnitAttr());
  pool->setAttr("tiled", builder.getBoolAttr(true));
  for (auto placement : plan.placements) {
    auto alloc = liveness.buffers[placement.bufferIndex].allocation;
    builder.setInsertionPoint(alloc);
    Value offset = builder.create<arith::ConstantIndexOp>(
        alloc.getLoc(), placement.offsetBytes);
    auto view = builder.create<memref::ViewOp>(alloc.getLoc(), alloc.getType(),
                                              pool, offset, ValueRange{});
    alloc.getResult().replaceAllUsesWith(view.getResult());
    alloc.erase();
  }
  if (stats) {
    stats->pooledBytes = plan.pooledBytes;
    stats->reusedAllocations = plan.placements.size();
    stats->insertedBarriers = barrierPositions.size();
  }
  return true;
}

} // namespace mlir::frisk
