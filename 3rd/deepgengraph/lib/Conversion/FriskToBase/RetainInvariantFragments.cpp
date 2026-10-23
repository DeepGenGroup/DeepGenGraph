// Retain immutable, thread-distributed operand values across a collective loop.
// Run after thread tiling / loop specialization, before shared packing/reuse.
// This deliberately matches dataflow and effects, never argument roles or names.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace mlir::frisk {
#define GEN_PASS_DECL_RETAININVARIANTFRAGMENTS
#define GEN_PASS_DEF_RETAININVARIANTFRAGMENTS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
namespace {
using Loop = affine::AffineForOp;
struct Range { int64_t lo, hi; };
using Bounds = std::optional<Range>;

// Small, conservative index-range proof. Refuse arithmetic outside signed
// 64-bit range, unknown values, and division whose sign is not established.
static Bounds combine(Bounds a, Bounds b, char kind) {
  if (!a || !b) return {};
  __int128 lo, hi;
  if (kind == '+') { lo = (__int128)a->lo + b->lo; hi = (__int128)a->hi + b->hi; }
  else if (kind == '-') { lo = (__int128)a->lo - b->hi; hi = (__int128)a->hi - b->lo; }
  else if (kind == '*') {
    __int128 v[] = {(__int128)a->lo*b->lo, (__int128)a->lo*b->hi,
                    (__int128)a->hi*b->lo, (__int128)a->hi*b->hi};
    lo = *std::min_element(v, v+4); hi = *std::max_element(v, v+4);
  } else {
    if (a->lo < 0 || b->lo <= 0) return {};
    if (kind == '/') { lo = a->lo / b->hi; hi = a->hi / b->lo; }
    else { lo = 0; hi = std::min(a->hi, b->hi - 1); }
  }
  if (lo < std::numeric_limits<int64_t>::min() ||
      hi > std::numeric_limits<int64_t>::max()) return {};
  return Range{int64_t(lo), int64_t(hi)};
}
static Bounds range(Value v, unsigned depth = 0);
static Bounds expression(AffineExpr e, AffineMap map, ValueRange values,
                         unsigned depth) {
  if (depth > 64) return {};
  if (auto c = dyn_cast<AffineConstantExpr>(e)) return Range{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e)) return range(values[d.getPosition()], depth+1);
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return range(values[map.getNumDims()+s.getPosition()], depth+1);
  auto b = dyn_cast<AffineBinaryOpExpr>(e);
  if (!b) return {};
  char kind;
  switch (e.getKind()) {
  case AffineExprKind::Add: kind = '+'; break;
  case AffineExprKind::Mul: kind = '*'; break;
  case AffineExprKind::FloorDiv: kind = '/'; break;
  case AffineExprKind::Mod: kind = '%'; break;
  default: return {};
  }
  return combine(expression(b.getLHS(), map, values, depth+1),
                 expression(b.getRHS(), map, values, depth+1), kind);
}
static Bounds range(Value v, unsigned depth) {
  if (depth > 64 || !v.getType().isIndex()) return {};
  APInt constant;
  if (matchPattern(v, m_ConstantInt(&constant))) {
    if (constant.getBitWidth() > 64) return {};
    return Range{constant.getSExtValue(), constant.getSExtValue()};
  }
  Operation *op = v.getDefiningOp();
  if (!op) return {};
  if (isa<gpu::BlockIdOp>(op)) {
    auto n = op->getAttrOfType<IntegerAttr>("range");
    if (n && n.getInt() > 0) return Range{0, n.getInt()-1};
  }
  if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
    return expression(apply.getAffineMap().getResult(0), apply.getAffineMap(),
                      apply.getMapOperands(), depth+1);
  char kind;
  if (isa<arith::AddIOp>(op)) kind = '+';
  else if (isa<arith::SubIOp>(op)) kind = '-';
  else if (isa<arith::MulIOp>(op)) kind = '*';
  else if (isa<arith::DivUIOp>(op)) kind = '/';
  else if (isa<arith::RemUIOp>(op)) kind = '%';
  else return {};
  return combine(range(op->getOperand(0), depth+1),
                 range(op->getOperand(1), depth+1), kind);
}
static bool nonempty(Loop loop) {
  int64_t lower = std::numeric_limits<int64_t>::min();
  int64_t upper = std::numeric_limits<int64_t>::max();
  for (AffineExpr e : loop.getLowerBoundMap().getResults()) {
    auto r = expression(e, loop.getLowerBoundMap(), loop.getLowerBoundOperands(), 0);
    if (!r) return false;
    lower = std::max(lower, r->hi);
  }
  for (AffineExpr e : loop.getUpperBoundMap().getResults()) {
    auto r = expression(e, loop.getUpperBoundMap(), loop.getUpperBoundOperands(), 0);
    if (!r) return false;
    upper = std::min(upper, r->lo);
  }
  return upper > lower;
}

static Operation *topLevel(Operation *op, Loop loop) {
  while (op && op->getBlock() != loop.getBody()) op = op->getParentOp();
  return op;
}
static bool inside(Value value, Operation *root) {
  Operation *owner = value.getDefiningOp();
  if (auto arg = dyn_cast<BlockArgument>(value)) owner = arg.getOwner()->getParentOp();
  return owner && (owner == root || root->isAncestor(owner));
}
static SmallVector<Value> captures(Operation *root) {
  SmallVector<Value> result;
  root->walk([&](Operation *op) {
    for (Value v : op->getOperands())
      if (!inside(v, root) && !llvm::is_contained(result, v)) result.push_back(v);
  });
  return result;
}
static bool space(Value v, unsigned n) {
  auto type = dyn_cast<MemRefType>(v.getType());
  return type && type.getMemorySpaceAsInt() == n;
}

// No unknown effects or potentially aliasing global writes may change the
// producer between iterations (or before its original first execution).
static bool immutableSource(Loop loop, Value source, AliasAnalysis &aa) {
  bool safe = true;
  loop.walk([&](Operation *op) {
    if (!safe || op == loop.getOperation()) return;
    if (isa<gpu::BarrierOp, WaitBufferReadyOp>(op)) return;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) {
      if (!space(copy.getDst(), 3)) safe = false;
      return;
    }
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return;
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) { safe = false; return; }
    SmallVector<MemoryEffects::EffectInstance> list;
    effects.getEffects(list);
    for (auto effect : list) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
      Value target = effect.getValue();
      if (!target || (!space(target, 3) && !aa.alias(source, target).isNo()))
        safe = false;
    }
  });
  return safe;
}

struct Candidate {
  Loop loop;
  AsyncCopyGmToShmOp copy;
  memref::AllocOp storage;
  WaitBufferReadyOp wait;
  llvm::SmallPtrSet<Operation *, 32> movable;
  llvm::SmallPtrSet<Operation *, 32> rejected;
  llvm::SmallPtrSet<Operation *, 32> selected;
  llvm::SmallPtrSet<Operation *, 32> data;

  // Region internals may use their own IVs/carriers, but may only read this
  // dedicated allocation. Never promote MMA, shuffle, conditional control flow
  // or another memory source as an incidental part of an operand's slice.
  bool allowed(Operation *root) {
    bool safe = true;
    root->walk([&](Operation *op) {
      if (!safe) return;
      if (op == storage || op == copy || op == wait) return;
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        safe = load.getMemref() == storage.getResult(); return;
      }
      if (auto forOp = dyn_cast<Loop>(op)) {
        safe = forOp.hasConstantBounds() &&
               forOp.getConstantUpperBound() >= forOp.getConstantLowerBound();
        return;
      }
      if (isa<FragmentOp, FragmentYieldOp, affine::AffineYieldOp>(op)) return;
      StringRef dialect = op->getName().getDialectNamespace();
      safe = !op->getNumRegions() && isMemoryEffectFree(op) &&
          (dialect == "arith" || dialect == "vector" ||
           isa<affine::AffineApplyOp, gpu::ThreadIdOp, gpu::BlockIdOp,
               memref::SubViewOp, memref::CastOp, memref::TransposeOp>(op));
    });
    return safe;
  }
  bool canMove(Operation *op) {
    if (movable.contains(op)) return true;
    if (rejected.contains(op) || op->getBlock() != loop.getBody()) return false;
    rejected.insert(op); // Bound recursion, including unexpected SSA cycles.
    if (!allowed(op)) return false;
    for (Value value : captures(op)) {
      if (!inside(value, loop)) continue;
      auto *def = value.getDefiningOp();
      if (!def || def->getBlock() != loop.getBody() || !canMove(def)) return false;
    }
    rejected.erase(op);
    movable.insert(op);
    return true;
  }
  void select(Operation *op) {
    if (!selected.insert(op).second) return;
    for (Value v : captures(op))
      if (auto *def = v.getDefiningOp(); def && def->getBlock() == loop.getBody())
        select(def);
  }
  std::optional<unsigned> analyze() {
    storage = copy.getDst().getDefiningOp<memref::AllocOp>();
    if (!storage || storage->getBlock() != loop.getBody() ||
        !storage.getType().hasStaticShape() || !space(copy.getSrc(), 1) ||
        !space(copy.getDst(), 3)) return {};
    SmallVector<Operation *> readers;
    for (Operation *user : storage->getUsers()) {
      if (user == copy) continue;
      if (auto ready = dyn_cast<WaitBufferReadyOp>(user)) {
        if (wait || ready->getBlock() != loop.getBody()) return {};
        wait = ready; continue;
      }
      auto load = dyn_cast<memref::LoadOp>(user);
      if (!load || !loop->isAncestor(user)) return {}; // No escaping alias.
      readers.push_back(topLevel(user, loop));
    }
    if (!wait || readers.empty() || !copy->isBeforeInBlock(wait)) return {};
    for (auto *reader : readers)
      if (!wait->isBeforeInBlock(reader) || !canMove(reader)) return {};
    if (!canMove(copy) || !canMove(wait)) return {};
    select(copy); select(wait);
    // Grow through invariant register computations, stopping at mixed/variant
    // consumers. The final cut, not every intermediate, determines the budget.
    for (Operation &op : *loop.getBody()) {
      bool depends = llvm::is_contained(readers, &op);
      for (Value v : captures(&op))
        if (data.contains(v.getDefiningOp())) depends = true;
      if (depends && canMove(&op)) { data.insert(&op); select(&op); }
    }
    uint64_t bits = 0;
    for (Operation *op : data) {
      for (Value value : op->getResults()) {
        bool live = llvm::any_of(value.getUsers(), [&](Operation *user) {
          return !selected.contains(topLevel(user, loop));
        });
        if (!live) continue;
        Type type = value.getType();
        if (auto vector = dyn_cast<VectorType>(type)) {
          if (vector.isScalable()) return {};
          bits += vector.getNumElements() * vector.getElementTypeBitWidth();
        } else if (type.isIntOrFloat()) bits += type.getIntOrFloatBitWidth();
        else return {};
      }
    }
    if (!bits || bits > unsigned(-1)) return {};
    return (bits + 31) / 32;
  }
  void apply() {
    // Moving, rather than cloning, preserves FP precision and eliminates the
    // repeated DMA/read/compute chain. Subsequent packing and liveness analysis
    // see all reads in their final scope and protect the storage handoff.
    for (Operation &op : llvm::make_early_inc_range(*loop.getBody()))
      if (selected.contains(&op)) op.moveBefore(loop);
  }
};

class RetainInvariantFragments
    : public impl::RetainInvariantFragmentsBase<RetainInvariantFragments> {
  void runOnOperation() override {
    if (!maxWords || getOperation().isExternal() ||
        !getOperation()->hasAttr("thread_num")) return;
    // Only entry-level collective loops in v1: nested retention scopes would
    // require an overlapping-live-range budget, rather than independent sums.
    SmallVector<Loop> loops;
    for (Operation &op : getOperation().front())
      if (auto loop = dyn_cast<Loop>(op)) loops.push_back(loop);
    unsigned count = 0;
    for (Loop loop : loops) {
      if (loop->hasAttr("frisk.retained_words") || !nonempty(loop) ||
          !isInBlockUniformControlFlow(loop.getBody()->getTerminator())) continue;
      if (loop.hasConstantBounds() &&
          (__int128)loop.getConstantUpperBound() - loop.getConstantLowerBound() <= loop.getStepAsInt()) continue;
      unsigned used = 0;
      SmallVector<AsyncCopyGmToShmOp> copies;
      for (Operation &op : *loop.getBody())
        if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) copies.push_back(copy);
      for (auto copy : copies) {
        if (used == maxWords) break;
        AliasAnalysis aa(getOperation());
        if (!immutableSource(loop, copy.getSrc(), aa)) continue;
        Candidate candidate{loop, copy};
        auto words = candidate.analyze();
        if (!words || *words > maxWords - used) continue;
        candidate.apply(); used += *words; ++count;
      }
      if (used) loop->setAttr("frisk.retained_words", IntegerAttr::get(
          IntegerType::get(&getContext(), 32), used));
    }
    llvm::errs() << "RetainInvariantFragments: " << count
                 << " operand tiles retained; budget " << maxWords << " words/thread\n";
  }
};
} // namespace
std::unique_ptr<Pass> createRetainInvariantFragmentsPass() {
  return std::make_unique<RetainInvariantFragments>();
}
} // namespace mlir::frisk
