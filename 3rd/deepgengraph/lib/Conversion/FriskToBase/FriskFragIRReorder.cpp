//===----------------------------------------------------------------------===//
// Fuse affine fragment loops without losing SSA vector/memref carriers.
// Scheduling A(all), B(all) as (A(i), B(i)) reverses B(i) / A(j), j > i.
// Check precisely those dependences; never infer independence from a marker.
//===----------------------------------------------------------------------===//
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <functional>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>
#include <set>

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_FRISKFRAGIRREORDER
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
using Loop = affine::AffineForOp;
using Coordinate = SmallVector<int64_t>;
using Footprint = std::set<Coordinate>;

// Evaluation is a bounded proof, not a heuristic. Unknown/dynamic coordinates
// cause rejection. Small static fragment loops are the intended workload.
static constexpr int64_t maxIterations = 256;
static std::optional<int64_t> evaluate(Value value, Loop loop, int64_t iv,
                                      unsigned depth = 0) {
  if (depth > 64)
    return std::nullopt;
  if (value == loop.getInductionVar())
    return iv;
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.getSExtValue();
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> inputs;
    for (Value operand : apply.getMapOperands()) {
      auto result = evaluate(operand, loop, iv, depth + 1);
      if (!result)
        return std::nullopt;
      inputs.push_back(IntegerAttr::get(IndexType::get(value.getContext()), *result));
    }
    SmallVector<Attribute> results;
    if (succeeded(apply.getAffineMap().constantFold(inputs, results)))
      return cast<IntegerAttr>(results.front()).getInt();
  }
  return std::nullopt;
}

static std::optional<Coordinate> position(ArrayRef<OpFoldResult> indices,
                                           Loop loop, int64_t iv) {
  Coordinate result;
  for (OpFoldResult index : indices) {
    if (auto attr = dyn_cast<Attribute>(index))
      result.push_back(cast<IntegerAttr>(attr).getInt());
    else if (auto number = evaluate(cast<Value>(index), loop, iv))
      result.push_back(*number);
    else
      return std::nullopt;
  }
  return result;
}

static SmallVector<int64_t> iterations(Loop loop) {
  SmallVector<int64_t> values;
  if (!loop.hasConstantBounds())
    return values;
  for (int64_t i = loop.getConstantLowerBound(); i < loop.getConstantUpperBound();) {
    if (values.size() == maxIterations)
      return {};
    values.push_back(i);
    if (__builtin_add_overflow(i, loop.getStepAsInt(), &i))
      return {};
  }
  return values;
}

// A memref yield transports a descriptor, not a snapshot of the contents.
// Only invariant descriptors are resolved; swapped/selected aliases stay unknown.
static Value invariantMemref(Value value) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    Loop loop;
    unsigned number;
    if (auto result = dyn_cast<OpResult>(value)) {
      loop = dyn_cast<Loop>(result.getOwner());
      number = result.getResultNumber();
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      loop = dyn_cast_or_null<Loop>(arg.getOwner()->getParentOp());
      if (!loop || arg.getArgNumber() == 0)
        return value;
      number = arg.getArgNumber() - 1;
    } else {
      return value;
    }
    if (!loop || !isa<MemRefType>(value.getType()))
      return value;
    Value yielded = loop.getBody()->getTerminator()->getOperand(number);
    if (yielded != loop.getRegionIterArgs()[number] &&
        yielded != loop.getInits()[number])
      return value;
    value = loop.getInits()[number];
  }
  return value;
}

static bool belongsToIteration(Operation *op, Loop loop) {
  while (op->getParentOp() != loop) {
    op = op->getParentOp();
    if (!op || !isa<frisk::FragmentOp>(op))
      return false;
  }
  return true;
}

// Prove an insert chain writes distinct scalar positions on each iteration.
// Restrict carrier uses to this chain and own-position scalar extracts; opaque
// vector arithmetic/reductions on the whole carrier are not pointwise updates.
static bool vectorWrites(Loop loop, unsigned number,
                         SmallVectorImpl<Footprint> &writes) {
  Value arg = loop.getRegionIterArgs()[number];
  Value value = loop.getBody()->getTerminator()->getOperand(number);
  SmallVector<vector::InsertOp> inserts;
  llvm::SmallPtrSet<Operation *, 16> chain;
  SmallVector<Value> versions{value};
  while (value != arg) {
    auto insert = value.getDefiningOp<vector::InsertOp>();
    if (!insert || insert->getBlock() != loop.getBody() ||
        isa<VectorType>(insert.getValueToStore().getType()))
      return false;
    inserts.push_back(insert);
    chain.insert(insert);
    value = insert.getDest();
    versions.push_back(value);
  }
  auto ivs = iterations(loop);
  if (ivs.empty() || ivs.size() * inserts.size() > 65536)
    return false;
  Footprint all;
  for (int64_t iv : ivs) {
    Footprint current;
    for (auto insert : inserts) {
      auto index = position(insert.getMixedPosition(), loop, iv);
      if (!index || !current.insert(*index).second || !all.insert(*index).second)
        return false;
    }
    writes.push_back(std::move(current));
  }
  for (Value version : versions) {
    for (OpOperand &use : version.getUses()) {
      if (use.getOwner() == loop.getBody()->getTerminator() ||
          (chain.contains(use.getOwner()) && use.getOperandNumber() == 1))
        continue;
      auto extract = dyn_cast<vector::ExtractOp>(use.getOwner());
      if (!extract || !belongsToIteration(extract, loop) ||
          isa<VectorType>(extract.getType()) || use.getOperandNumber() != 0)
        return false;
      for (auto [n, iv] : llvm::enumerate(ivs)) {
        auto index = position(extract.getMixedPosition(), loop, iv);
        if (!index || !writes[n].count(*index))
          return false;
      }
    }
  }
  return true;
}

static bool checkVectorUse(Value result, Loop a, Loop b, unsigned number) {
  SmallVector<Footprint> writes;
  if (!vectorWrites(a, number, writes))
    return false;
  auto ivs = iterations(b);
  if (ivs.size() != writes.size())
    return false;
  for (OpOperand &use : result.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == b) {
      // Bounds cannot depend on a vector. Replace this init by A's init only
      // when B overwrites exactly the same positions and reads its own slice.
      unsigned init = use.getOperandNumber() - b.getNumControlOperands();
      SmallVector<Footprint> consumerWrites;
      if (init >= b.getNumIterOperands() || !vectorWrites(b, init, consumerWrites) ||
          writes != consumerWrites)
        return false;
    } else if (b->isAncestor(owner)) {
      auto extract = dyn_cast<vector::ExtractOp>(owner);
      if (!extract || !belongsToIteration(owner, b) ||
          use.getOperandNumber() != 0 || isa<VectorType>(extract.getType()))
        return false;
      for (auto [n, iv] : llvm::enumerate(ivs)) {
        auto index = position(extract.getMixedPosition(), b, iv);
        if (!index || !writes[n].count(*index))
          return false;
      }
    }
  }
  return true;
}

struct Access {
  Value buffer;
  bool write;
  SmallVector<OpFoldResult> indices;
  AffineMap map;
};
static bool collectAccesses(Loop loop, SmallVectorImpl<Access> &accesses) {
  auto result = loop.walk([&](Operation *op) {
    Value buffer;
    bool write = false;
    ValueRange indices;
    AffineMap map;
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      buffer = load.getMemref(); indices = load.getMapOperands(); map = load.getAffineMap();
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      buffer = store.getMemref(); indices = store.getMapOperands(); map = store.getAffineMap(); write = true;
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      buffer = load.getMemref(); indices = load.getIndices();
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      buffer = store.getMemref(); indices = store.getIndices(); write = true;
    } else if (isa<Loop, affine::AffineYieldOp, frisk::FragmentOp, frisk::FragmentYieldOp>(op)) {
      return WalkResult::advance();
    } else {
      // Includes barriers, calls, allocation/free, atomics and unknown effects.
      return isMemoryEffectFree(op) ? WalkResult::advance() : WalkResult::interrupt();
    }
    accesses.push_back({invariantMemref(buffer), write,
                       SmallVector<OpFoldResult>(indices.begin(), indices.end()), map});
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

static std::optional<Coordinate> address(const Access &access, Loop loop, int64_t iv) {
  auto operands = position(access.indices, loop, iv);
  if (!operands || !access.map)
    return operands;
  SmallVector<Attribute> attrs, results;
  for (int64_t value : *operands)
    attrs.push_back(IntegerAttr::get(IndexType::get(loop.getContext()), value));
  if (failed(access.map.constantFold(attrs, results)))
    return std::nullopt;
  Coordinate coord;
  for (Attribute attr : results)
    coord.push_back(cast<IntegerAttr>(attr).getInt());
  return coord;
}

static bool memorySafe(Loop a, Loop b, AliasAnalysis &aliases) {
  SmallVector<Access> first, second;
  if (!collectAccesses(a, first) || !collectAccesses(b, second))
    return false;
  auto ivs = iterations(a);
  // Limit quadratic proof cost; exceeding the budget leaves the loops intact.
  if (first.size() * second.size() * ivs.size() * ivs.size() > 1048576)
    return false;
  for (const Access &x : first) {
    for (const Access &y : second) {
      if ((!x.write && !y.write) || aliases.alias(x.buffer, y.buffer).isNo())
        continue;
      if (x.buffer != y.buffer || ivs.empty() ||
          !cast<MemRefType>(x.buffer.getType()).getLayout().isIdentity())
        return false;
      for (unsigned i = 0; i < ivs.size(); ++i) {
        auto earlier = address(y, b, ivs[i]);
        if (!earlier)
          return false;
        for (unsigned j = i + 1; j < ivs.size(); ++j) {
          auto later = address(x, a, ivs[j]);
          if (!later || *earlier == *later)
            return false;
        }
      }
    }
  }
  return true;
}

static bool tryFuse(Loop a, Loop b, AliasAnalysis &aliases) {
  if (a->hasAttr("frisk.pipelined") || b->hasAttr("frisk.pipelined"))
    return false;
  if (a.getLowerBoundMap() != b.getLowerBoundMap() ||
      a.getUpperBoundMap() != b.getUpperBoundMap() || a.getStep() != b.getStep() ||
      !llvm::equal(a.getLowerBoundOperands(), b.getLowerBoundOperands()) ||
      !llvm::equal(a.getUpperBoundOperands(), b.getUpperBoundOperands()))
    return false;
  // Crossing pure, speculatable setup is safe only if it does not consume A.
  for (Operation *op = a->getNextNode(); op != b; op = op->getNextNode()) {
    if (!isMemoryEffectFree(op) || !isSpeculatable(op) || op->getNumRegions())
      return false;
    for (Value operand : op->getOperands())
      if (operand.getDefiningOp() == a)
        return false;
  }
  for (auto [n, result] : llvm::enumerate(a.getResults())) {
    bool consumed = llvm::any_of(result.getUsers(), [&](Operation *op) {
      return op == b || b->isAncestor(op);
    });
    if (!consumed)
      continue;
    if (isa<MemRefType>(result.getType())) {
      if (invariantMemref(result) == result)
        return false;
    } else if (!isa<VectorType>(result.getType()) || !checkVectorUse(result, a, b, n)) {
      return false;
    }
  }
  if (!memorySafe(a, b, aliases))
    return false;

  OpBuilder builder(b);
  SmallVector<Value> inits(a.getInits());
  for (Value init : b.getInits()) {
    if (auto result = dyn_cast<OpResult>(init); result && result.getOwner() == a)
      init = a.getInits()[result.getResultNumber()];
    inits.push_back(init);
  }
  auto fused = builder.create<Loop>(a.getLoc(), a.getLowerBoundOperands(),
      a.getLowerBoundMap(), a.getUpperBoundOperands(), a.getUpperBoundMap(),
      a.getStepAsInt(), inits);
  // Keep common scheduling metadata only; a mismatching fragment shape must
  // not become a false promise for a later scheduler.
  for (NamedAttribute attr : a->getAttrs())
    if (b->getAttr(attr.getName()) == attr.getValue() &&
        attr.getName() != "operandSegmentSizes")
      fused->setAttr(attr.getName(), attr.getValue());
  fused->setAttr("frisk.fused", builder.getUnitAttr());
  builder.setInsertionPointToStart(fused.getBody());
  IRMapping mapping;
  mapping.map(a.getInductionVar(), fused.getInductionVar());
  mapping.map(b.getInductionVar(), fused.getInductionVar());
  unsigned firstCount = a.getNumResults();
  for (auto [old, replacement] : llvm::zip(a.getRegionIterArgs(),
                                         fused.getRegionIterArgs().take_front(firstCount)))
    mapping.map(old, replacement);
  for (auto [old, replacement] : llvm::zip(b.getRegionIterArgs(),
                                         fused.getRegionIterArgs().drop_front(firstCount)))
    mapping.map(old, replacement);
  for (Operation &op : a.getBody()->without_terminator())
    builder.clone(op, mapping);
  SmallVector<Value> yields;
  for (auto [result, value] : llvm::zip(a.getResults(), a.getBody()->getTerminator()->getOperands())) {
    Value yielded = mapping.lookupOrDefault(value);
    yields.push_back(yielded);
    mapping.map(result, yielded);
  }
  // A vector used as B's initial value was complete before fusion. Overlay
  // A's current slice on B's carried value before B reads it, retaining all
  // previously computed B slices. Merely substituting A's init is incorrect.
  for (auto [n, init] : llvm::enumerate(b.getInits())) {
    auto result = dyn_cast<OpResult>(init);
    if (!result || result.getOwner() != a || !isa<VectorType>(init.getType()))
      continue;
    Value value = a.getBody()->getTerminator()->getOperand(result.getResultNumber());
    SmallVector<vector::InsertOp> inserts;
    while (auto insert = value.getDefiningOp<vector::InsertOp>()) {
      inserts.push_back(insert);
      value = insert.getDest();
    }
    Value updated = fused.getRegionIterArgs()[firstCount + n];
    for (auto insert : llvm::reverse(inserts)) {
      SmallVector<OpFoldResult> indices;
      for (OpFoldResult index : insert.getMixedPosition())
        indices.push_back(isa<Value>(index) ? OpFoldResult(mapping.lookupOrDefault(cast<Value>(index))) : index);
      updated = builder.create<vector::InsertOp>(insert.getLoc(),
          mapping.lookupOrDefault(insert.getValueToStore()), updated, indices);
    }
    mapping.map(b.getRegionIterArgs()[n], updated);
  }
  for (Operation &op : b.getBody()->without_terminator())
    builder.clone(op, mapping);
  for (Value value : b.getBody()->getTerminator()->getOperands())
    yields.push_back(mapping.lookupOrDefault(value));
  if (!inits.empty())
    builder.create<affine::AffineYieldOp>(b.getLoc(), yields);
  a.replaceAllUsesWith(fused.getResults().take_front(firstCount));
  b.replaceAllUsesWith(fused.getResults().drop_front(firstCount));
  b.erase();
  a.erase();
  return true;
}

// A one-iteration lookahead schedule for read-only fragment loops:
//   prologue: load(0)
//   steady:   next = load(i+1); current = compute(i, carriedLoad)
//   epilogue: compute(last, carriedLoad)
// No speculative last load, no reordered accumulator updates, no movement
// across a barrier or a write (including writes in another fragment region).
static bool pipelineFragments(Loop loop) {
  if (!loop->hasAttr("frisk.fragment_loop") || loop->hasAttr("frisk.pipelined"))
    return false;
  auto ivs = iterations(loop);
  if (ivs.size() < 2)
    return false;
  bool readsMemory = false, hasCompute = false;
  SmallVector<frisk::FragmentOp> loads;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto fragment = dyn_cast<frisk::FragmentOp>(op);
    if (!fragment)
      continue;
    hasCompute |= fragment.getKind() == "compute";
    if ((fragment.getKind() == "load" || fragment.getKind() == "gather") &&
        fragment.getNumResults())
      loads.push_back(fragment);
  }
  if (loads.empty() || !hasCompute)
    return false;
  // Validate the entire iteration, not just the proposed load groups. An
  // unknown effect, collective, nested loop or write is a scheduling boundary.
  auto safe = loop.walk([&](Operation *op) {
    if (op == loop || isa<frisk::FragmentOp, frisk::FragmentYieldOp,
                          affine::AffineYieldOp>(op))
      return WalkResult::advance();
    if (isa<affine::AffineLoadOp, memref::LoadOp, vector::LoadOp>(op))
      return WalkResult::advance();
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return WalkResult::interrupt();
    // Register MMA stays at the original iteration's compute point.
    if (isa<gpu::ShuffleOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (safe.wasInterrupted())
    return false;

  llvm::SmallPtrSet<Operation *, 16> groups, slice;
  for (auto load : loads) {
    groups.insert(load);
    auto valid = load.walk([&](Operation *op) {
      if (op == load || isa<frisk::FragmentYieldOp>(op))
        return WalkResult::advance();
      if (isa<affine::AffineLoadOp, memref::LoadOp, vector::LoadOp>(op)) {
        readsMemory = true;
        return WalkResult::advance();
      }
      // No collective or non-speculatable computation moves to the prologue.
      return !op->getNumRegions() && isMemoryEffectFree(op) && isSpeculatable(op)
          ? WalkResult::advance() : WalkResult::interrupt();
    });
    if (valid.wasInterrupted())
      return false;
  }
  if (!readsMemory)
    return false;
  std::function<bool(Value)> addDependency = [&](Value value) {
    if (value == loop.getInductionVar())
      return true;
    if (auto arg = dyn_cast<BlockArgument>(value))
      return arg.getOwner() != loop.getBody();
    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != loop.getBody())
      return true;
    if (slice.contains(def))
      return true;
    if (groups.contains(def)) {
      slice.insert(def);
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(def->getRegions(), captures);
      return llvm::all_of(captures, addDependency);
    }
    if (def->getNumRegions() || !isMemoryEffectFree(def) || !isSpeculatable(def))
      return false;
    slice.insert(def);
    return llvm::all_of(def->getOperands(), addDependency);
  };
  for (auto load : loads)
    for (Value value : load.getResults())
      if (!addDependency(value))
        return false;

  OpBuilder b(loop);
  Location loc = loop.getLoc();
  auto cloneLoads = [&](Value iv, StringRef stage) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), iv);
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (!slice.contains(&op))
        continue;
      Operation *copy = b.clone(op, mapping);
      if (groups.contains(&op))
        copy->setAttr("frisk.pipeline_stage", b.getStringAttr(stage));
    }
    SmallVector<Value> values;
    for (auto load : loads)
      for (Value value : load.getResults())
        values.push_back(mapping.lookup(value));
    return values;
  };
  auto cloneCompute = [&](Value iv, ValueRange carriers, ValueRange prefetched) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), iv);
    mapping.map(loop.getRegionIterArgs(), carriers);
    unsigned position = 0;
    for (auto load : loads)
      for (Value value : load.getResults())
        mapping.map(value, prefetched[position++]);
    for (Operation &op : loop.getBody()->without_terminator())
      if (!groups.contains(&op))
        b.clone(op, mapping);
    SmallVector<Value> values;
    for (Value value : loop.getBody()->getTerminator()->getOperands())
      values.push_back(mapping.lookupOrDefault(value));
    return values;
  };

  Value first = b.create<arith::ConstantIndexOp>(loc, ivs.front());
  SmallVector<Value> prefetched = cloneLoads(first, "prologue");
  SmallVector<Value> inits(loop.getInits());
  inits.append(prefetched);
  auto steady = b.create<Loop>(loc, ivs.front(), ivs.back(), loop.getStepAsInt(), inits);
  // Bounds/step belong to the newly constructed loop, not the old upper bound.
  for (NamedAttribute attr : loop->getAttrs())
    if (attr.getName().strref().starts_with("frisk.") || attr.getName() == "iterLabel")
      steady->setAttr(attr.getName(), attr.getValue());
  steady->setAttr("frisk.pipelined", b.getUnitAttr());
  b.setInsertionPointToStart(steady.getBody());
  Value next = b.create<affine::AffineApplyOp>(loc,
      AffineMap::get(1, 0, b.getAffineDimExpr(0) + loop.getStepAsInt()), steady.getInductionVar());
  SmallVector<Value> nextLoads = cloneLoads(next, "prefetch");
  unsigned carriers = loop.getNumResults();
  auto values = cloneCompute(steady.getInductionVar(),
      steady.getRegionIterArgs().take_front(carriers),
      steady.getRegionIterArgs().drop_front(carriers));
  values.append(nextLoads);
  b.create<affine::AffineYieldOp>(loc, values);
  b.setInsertionPointAfter(steady);
  Value last = b.create<arith::ConstantIndexOp>(loc, ivs.back());
  auto results = cloneCompute(last, steady.getResults().take_front(carriers),
                              steady.getResults().drop_front(carriers));
  loop.replaceAllUsesWith(results);
  loop.erase();
  return true;
}

#undef GEN_PASS_DEF_FRISKFRAGIRREORDER
#define GEN_PASS_DEF_LOWERFRISKFRAGMENTS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

class LowerFriskFragments : public impl::LowerFriskFragmentsBase<LowerFriskFragments> {
  void runOnOperation() override {
    packSharedMemory(getOperation());
    SmallVector<frisk::FragmentOp> fragments;
    getOperation().walk<WalkOrder::PostOrder>([&](frisk::FragmentOp op) {
      fragments.push_back(op);
    });
    for (auto fragment : fragments) {
      auto &body = fragment.getBody().front();
      auto yield = cast<frisk::FragmentYieldOp>(body.getTerminator());
      fragment.replaceAllUsesWith(yield.getOperands());
      while (&body.front() != yield.getOperation())
        body.front().moveBefore(fragment);
      fragment.erase();
    }
    // The prefetch schedule is now fixed, and all physical reads are visible.
    // Allocate shared storage only against these final operation lifetimes.
    reuseSharedMemory(getOperation());
    SmallVector<Loop> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) {
      auto full = loop->getAttrOfType<BoolAttr>("frisk.loopUnrollFull");
      if (full && full.getValue())
        loops.push_back(loop);
    });
    uint64_t remaining = 65536;
    for (auto loop : loops) {
      auto count = affine::getConstantTripCount(loop);
      if (!count || *count == 0 || *count > 64)
        continue;
      uint64_t size = 0;
      loop.getBody()->walk([&](Operation *) { ++size; });
      uint64_t cost = size * (*count - 1);
      if (cost > remaining)
        continue;
      if (failed(affine::loopUnrollFull(loop))) {
        loop.emitError("failed late fragment unrolling");
        signalPassFailure();
        return;
      }
      remaining -= cost;
    }
  }
};

class FriskFragIRReorder : public impl::FriskFragIRReorderBase<FriskFragIRReorder> {
  void runOnOperation() override {
    // Restart after each mutation: nested loop handles and alias caches may
    // otherwise refer to erased operations. Each successful step removes a loop.
    bool changed;
    do {
      changed = false;
      AliasAnalysis aliases(getOperation());
      getOperation().walk<WalkOrder::PreOrder>([&](Loop a) {
        for (Operation *next = a->getNextNode(); next; next = next->getNextNode()) {
          if (auto b = dyn_cast<Loop>(next)) {
            changed = tryFuse(a, b, aliases);
            break;
          }
          if (!isMemoryEffectFree(next) || !isSpeculatable(next) || next->getNumRegions())
            break;
        }
        return changed ? WalkResult::interrupt() : WalkResult::advance();
      });
    } while (changed);
    SmallVector<Loop> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) { loops.push_back(loop); });
    for (auto loop : loops)
      pipelineFragments(loop);
  }
};
} // namespace
std::unique_ptr<Pass> createFriskFragIRReorderPass() {
  return std::make_unique<FriskFragIRReorder>();
}
std::unique_ptr<Pass> createLowerFriskFragmentsPass() {
  return std::make_unique<LowerFriskFragments>();
}
} // namespace mlir::frisk
