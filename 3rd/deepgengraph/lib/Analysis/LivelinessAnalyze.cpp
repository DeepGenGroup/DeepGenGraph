#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <optional>
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
        info.accesses.push_back(user);
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

Block *getLoopBody(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return loop.getBody();
  if (auto loop = dyn_cast<affine::AffineForOp>(op))
    return loop.getBody();
  return nullptr;
}

bool isCollectiveLoop(Operation *op, Block *entry);

// Only scalar kernel arguments, block-wide GPU queries and pure arithmetic
// derived from them are uniform. Loads, thread IDs and loop-carried values are
// deliberately not inferred uniform, even if a loop already contains a barrier.
bool isBlockUniform(Value value, Block *entry) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == entry)
      return isa<IndexType, IntegerType, FloatType>(value.getType());
    auto *parent = arg.getOwner()->getParentOp();
    return getLoopBody(parent) == arg.getOwner() && arg.getArgNumber() == 0 &&
           isCollectiveLoop(parent, entry);
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::ConstantLike>() ||
      isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op))
    return true;
  if (op->getNumRegions() || !isMemoryEffectFree(op) ||
      !(isa<affine::AffineApplyOp>(op) ||
        op->getName().getDialectNamespace() == "arith"))
    return false;
  return llvm::all_of(op->getOperands(),
                      [&](Value operand) { return isBlockUniform(operand, entry); });
}

bool isCollectiveLoop(Operation *op, Block *entry) {
  if (!getLoopBody(op))
    return false;
  if (op->getBlock() != entry &&
      !isCollectiveLoop(op->getParentOp(), entry))
    return false; // Never enter a conditional or unsupported region.
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return isBlockUniform(loop.getLowerBound(), entry) &&
           isBlockUniform(loop.getUpperBound(), entry) &&
           isBlockUniform(loop.getStep(), entry);
  auto loop = cast<affine::AffineForOp>(op);
  return llvm::all_of(loop.getLowerBoundOperands(),
                      [&](Value v) { return isBlockUniform(v, entry); }) &&
         llvm::all_of(loop.getUpperBoundOperands(),
                      [&](Value v) { return isBlockUniform(v, entry); });
}

// Evaluate the common affine thread-to-storage mapping without changing IR.
// Unknown indices conservatively disable the full-overwrite proof.
std::optional<int64_t> evaluateIndex(Value value, int64_t thread) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.getSExtValue();
  if (auto id = value.getDefiningOp<gpu::ThreadIdOp>()) {
    if (id.getDimension() == gpu::Dimension::x)
      return thread;
    return std::nullopt;
  }
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> operands;
    for (Value operand : apply.getMapOperands()) {
      auto index = evaluateIndex(operand, thread);
      if (!index)
        return std::nullopt;
      operands.push_back(IntegerAttr::get(IndexType::get(value.getContext()), *index));
    }
    SmallVector<Attribute> results;
    if (succeeded(apply.getAffineMap().constantFold(operands, results)))
      return cast<IntegerAttr>(results.front()).getInt();
  }
  return std::nullopt;
}

// An allocation outside the loop is not necessarily iteration-local: partial
// stores, a read before the write, or a rotated pipeline buffer can carry data
// across the backedge. Require the first storage access to overwrite the whole
// allocation on every iteration. For tiled vector stores, check the actual
// address mapping across all threads, rather than trusting the 'tiled' marker.
bool fullyOverwrites(Operation *op, const ShmLiveInterval &info,
                     func::FuncOp kernel) {
  auto allocation = info.allocation;
  Value buffer = allocation.getResult();
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return copy.getTarget() == buffer && copy.getSource() != buffer;
  ValueRange indices;
  int64_t width = 1;
  if (auto store = dyn_cast<vector::StoreOp>(op)) {
    auto vectorType = store.getVectorType();
    if (store.getBase() != buffer || vectorType.getRank() != 1 ||
        vectorType.isScalable())
      return false;
    indices = store.getIndices();
    width = vectorType.getDimSize(0);
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    if (store.getMemref() != buffer)
      return false;
    indices = store.getIndices();
  } else {
    return false;
  }
  auto type = allocation.getType();
  int64_t elements = type.getNumElements();
  auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
  // Bound compile-time work; larger or unknown footprints stay conservative.
  if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 ||
      elements > (1 << 20) || type.getRank() == 0)
    return false;
  llvm::SmallBitVector written(elements);
  for (int64_t thread = 0; thread < threads.getInt(); ++thread) {
    int64_t offset = 0;
    for (auto [dim, index] : llvm::enumerate(indices)) {
      auto coordinate = evaluateIndex(index, thread);
      if (!coordinate || *coordinate < 0 || *coordinate >= type.getDimSize(dim))
        return false;
      if (dim + 1 == indices.size() && width > type.getDimSize(dim) - *coordinate)
        return false;
      offset = offset * type.getDimSize(dim) + *coordinate;
    }
    written.set(offset, offset + width);
  }
  return written.all();
}

void findIterationLocalLoops(ShmLiveInterval &info, func::FuncOp kernel) {
  if (!info.reusable || info.aliases.size() != 1)
    return; // Alias/iter_arg cases retain the existing whole-loop lifetime.
  Block *entry = &kernel.getBody().front();
  for (Operation *loop = info.accesses.front()->getParentOp();
       loop && loop != kernel.getOperation(); loop = loop->getParentOp()) {
    Block *body = getLoopBody(loop);
    if (!body || !isCollectiveLoop(loop, entry))
      continue;
    Operation *first = nullptr;
    bool allInside = llvm::all_of(info.accesses, [&](Operation *access) {
      if (!loop->isProperAncestor(access))
        return false;
      auto *top = getTopLevelUser(access, body);
      if (!top)
        return false;
      if (!first || top->isBeforeInBlock(first))
        first = top;
      return true;
    });
    if (allInside &&
        (loop->isProperAncestor(info.allocation) ||
         fullyOverwrites(first, info, kernel)))
      info.iterationLocalLoops.push_back(loop);
  }
}

struct Separation {
  Block *block;
  SmallVector<Operation *> order;
  unsigned earlierFirst, earlierLast, laterFirst, laterLast;
};

// Descend only when BOTH buffers are confined to one collective loop and their
// contents do not survive its backedge. Other regions remain atomic; in
// particular, barriers will never be inserted inside a divergent conditional.
std::optional<Separation> findSeparation(const ShmLiveInterval &a,
                                       const ShmLiveInterval &b, Block *block) {
  Separation result;
  result.block = block;
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *block) {
    positions[&op] = result.order.size();
    result.order.push_back(&op);
  }
  auto range = [&](const ShmLiveInterval &info) {
    unsigned first = std::numeric_limits<unsigned>::max(), last = 0;
    for (Operation *access : info.accesses) {
      auto *top = getTopLevelUser(access, block);
      assert(top && "access must be inside the common scope");
      unsigned pos = positions.lookup(top);
      first = std::min(first, pos);
      last = std::max(last, pos);
    }
    return std::make_pair(first, last);
  };
  auto [af, al] = range(a);
  auto [bf, bl] = range(b);
  if (bl < af) {
    std::swap(af, bf);
    std::swap(al, bl);
  }
  if (al < bf) {
    result.earlierFirst = af;
    result.earlierLast = al;
    result.laterFirst = bf;
    result.laterLast = bl;
    return result;
  }
  if (af == al && af == bf && bf == bl) {
    Operation *loop = result.order[af];
    if (llvm::is_contained(a.iterationLocalLoops, loop) &&
        llvm::is_contained(b.iterationLocalLoops, loop))
      return findSeparation(a, b, getLoopBody(loop));
  }
  return std::nullopt;
}

} // namespace

bool ShmLivenessResult::interferes(unsigned lhs, unsigned rhs) const {
  const auto &a = buffers[lhs];
  const auto &b = buffers[rhs];
  return lhs == rhs || !a.reusable || !b.reusable ||
         !findSeparation(a, b, programOrder.front()->getBlock());
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
    findIterationLocalLoops(info, kernel);
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
  bool hasBridges = false;
  kernel.walk([&](Operation *op) {
    hasBridges |= isa<ToThreadTileOp, FromThreadTileOp>(op);
  });
  if (hasBridges)
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
  struct ScopeHandoffs {
    SmallVector<Operation *> order;
    SmallVector<std::pair<unsigned, unsigned>> gaps;
  };
  SmallVector<ScopeHandoffs> scopes;
  DenseMap<Block *, unsigned> scopeIndices;
  for (auto [i, a] : llvm::enumerate(plan.placements)) {
    const auto &lhs = liveness.buffers[a.bufferIndex];
    for (auto b : ArrayRef(plan.placements).drop_front(i + 1)) {
      const auto &rhs = liveness.buffers[b.bufferIndex];
      if (!overlaps(a.offsetBytes, lhs.sizeBytes, b.offsetBytes, rhs.sizeBytes))
        continue;
      auto separation = findSeparation(lhs, rhs, &kernel.getBody().front());
      assert(separation && "overlapping placements must have disjoint lifetimes");
      auto [it, inserted] = scopeIndices.try_emplace(separation->block, scopes.size());
      if (inserted)
        scopes.push_back({separation->order, {}});
      auto &scope = scopes[it->second];
      auto hasBarrier = [&](unsigned begin, unsigned end) {
        return llvm::any_of(ArrayRef(scope.order).slice(begin, end - begin),
                            isBarrier);
      };
      if (!hasBarrier(separation->earlierLast + 1, separation->laterFirst))
        scope.gaps.emplace_back(separation->laterFirst, separation->earlierLast);
      // The last user of this iteration must finish before the first writer of
      // the next one. A prefix barrier also covers the wrapped handoff; otherwise
      // require a barrier at the loop tail. Even a zero-trip path stays safe.
      if (separation->block != &kernel.getBody().front() &&
          !hasBarrier(0, separation->earlierFirst) &&
          !hasBarrier(separation->laterLast + 1, scope.order.size()))
        scope.gaps.emplace_back(scope.order.size() - 1, separation->laterLast);
    }
  }
  // One barrier can satisfy several handoffs. Process the earliest deadline
  // first, and reuse a previously selected barrier whenever it covers the gap.
  SmallVector<Operation *> barrierPositions;
  for (auto &scope : scopes) {
    llvm::sort(scope.gaps);
    std::optional<unsigned> previous;
    for (auto [deadline, lastUse] : scope.gaps)
      if (!previous || *previous <= lastUse) {
        barrierPositions.push_back(scope.order[deadline]);
        previous = deadline;
      }
  }
  OpBuilder builder(kernel.getContext());
  for (Operation *before : barrierPositions) {
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
