// Pair and prefetch physical LDS operands after thread unrolling/address proofs.
// No LLVM pointer reconstruction or exported-IR scheduling is needed.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "ThreadSchedulePlan.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <limits>

namespace mlir::frisk {
namespace {
using namespace thread_schedule;

bool candidate(Operation *op) {
  auto load = dyn_cast<LDSLoadOp>(op);
  if (!load || load.getAlignment() < 8 || load.getResult().use_empty()) return false;
  auto type = dyn_cast<VectorType>(load.getResult().getType());
  // The current target models paired 64-bit lane reads. Their element type
  // and MMA operand role are independent of the pairing instruction.
  if (!type || type.isScalable() ||
      type.getNumElements() * type.getElementTypeBitWidth() != 64) return false;
  return llvm::all_of(load.getResult().getUses(), [&](OpOperand &use) {
    return mma(use.getOwner()) && use.getOperandNumber() < 2 &&
           use.getOwner()->getBlock() == op->getBlock();
  });
}
bool protectedLoad(Operation *op) {
  return protectedLDSPair(op);
}
bool reduction(Operation *op) {
  return isa<arith::AddFOp, arith::MaxNumFOp, arith::MinNumFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}
bool sameAddress(LDSLoadOp a, LDSLoadOp b) {
  return a.getSource() == b.getSource() && llvm::equal(a.getIndices(), b.getIndices());
}

// CSE ordinary reads only inside a straight-line, write-free epoch. Recognized
// register MMACs are pure even when thread legalization uses side-effecting asm.
// Any unknown call, memory operation, synchronization or region ends the epoch.
void deduplicateLoads(func::FuncOp f) {
  f.walk([&](Block *block) {
    SmallVector<LDSLoadOp> available;
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (auto load = dyn_cast<LDSLoadOp>(&op)) {
        if (protectedLoad(load)) { available.clear(); continue; }
        auto previous = llvm::find_if(available, [&](LDSLoadOp old) {
          return sameAddress(old, load) && old.getByteOffset() == load.getByteOffset() &&
                 old.getResult().getType() == load.getResult().getType() &&
                 old->getAttrs() == load->getAttrs();
        });
        if (previous != available.end()) {
          load.getResult().replaceAllUsesWith(previous->getResult());
          load.erase();
        } else available.push_back(load);
      } else if (!pure(&op) && !mma(&op)) available.clear();
    }
  });
}

bool hoist(Ops &order, ArrayRef<Operation *> roots, Operation *before) {
  DenseMap<Operation *, unsigned> pos;
  for (auto [n, op] : llvm::enumerate(order)) pos[op] = n;
  if (!pos.contains(before)) return false;
  unsigned at = pos[before];
  llvm::SmallPtrSet<Operation *, 32> moving, visiting;
  std::function<bool(Operation *, bool)> collect = [&](Operation *op, bool root) {
    if (!pos.contains(op) || pos[op] < at || moving.contains(op)) return true;
    if (op == before) return root;
    if ((!root && !address(op)) || !visiting.insert(op).second) return false;
    for (Value value : op->getOperands())
      if (auto *def = value.getDefiningOp()) if (!collect(def, false)) return false;
    visiting.erase(op); moving.insert(op); return true;
  };
  for (Operation *op : roots) if (!collect(op, true)) return false;
  Ops next;
  for (Operation *op : order) {
    if (op == before) for (Operation *node : order) if (moving.contains(node)) next.push_back(node);
    if (!moving.contains(op)) next.push_back(op);
  }
  order = std::move(next);
  return true;
}

struct Stats { unsigned regions = 0, pairs = 0, moved = 0, anchors = 0, unpaired = 0; uint64_t nextPair = 0; };
void schedule(Ops region, unsigned distance, StringRef marker, Stats &stats) {
  if (region.empty()) return;
  Ops original = region, mmas;
  SmallVector<LDSLoadOp> loads;
  DenseMap<Operation *, unsigned> pos, first;
  for (auto [n, op] : llvm::enumerate(region)) {
    pos[op] = n;
    if (candidate(op)) loads.push_back(cast<LDSLoadOp>(op));
    if (mma(op)) mmas.push_back(op);
  }
  if (loads.size() < 2 || mmas.size() < 2) { stats.unpaired += loads.size(); return; }
  for (auto load : loads) {
    first[load] = mmas.size();
    for (auto [n, op] : llvm::enumerate(mmas))
      if (op->getOperand(0) == load.getResult() || op->getOperand(1) == load.getResult()) { first[load] = n; break; }
  }
  struct Pair { Operation *a, *b; unsigned first; };
  SmallVector<Pair> pairs;
  llvm::SmallPtrSet<Operation *, 32> paired;
  for (auto a : loads) {
    if (paired.contains(a) || first[a] == mmas.size()) continue;
    Operation *best = nullptr;
    unsigned bestClass = 3, bestUse = std::numeric_limits<unsigned>::max();
    for (auto b : loads) {
      if (a == b || paired.contains(b) || pos[b] < pos[a] ||
          first[b] == mmas.size() || !sameAddress(a, b)) continue;
      int64_t oa = a.getByteOffset(), ob = b.getByteOffset();
      uint64_t delta = oa < ob ? uint64_t(ob) - uint64_t(oa) : uint64_t(oa) - uint64_t(ob);
      unsigned kind = delta && delta % 8 == 0 && delta <= 255 * 8 ? 0 :
                      delta && delta % 512 == 0 && delta <= 255 * 512 ? 1 : 2;
      if (kind == 2 || kind > bestClass || (kind == bestClass && first[b] >= bestUse)) continue;
      Ops trial = region;
      auto it = llvm::find(trial, a.getOperation());
      if (std::next(it) == trial.end() || !hoist(trial, {b}, *std::next(it))) continue;
      best = b; bestClass = kind; bestUse = first[b];
    }
    if (best) {
      paired.insert(a); paired.insert(best);
      pairs.push_back({a, best, std::min(first[a], first[best])});
    }
  }
  for (auto p : pairs) {
    auto it = llvm::find(region, p.a);
    if (std::next(it) == region.end() || !hoist(region, {p.b}, *std::next(it))) return;
  }
  llvm::SmallPtrSet<Operation *, 32> anchors;
  unsigned moved = 0;
  for (auto p : pairs) {
    unsigned start = p.first > distance ? p.first - distance : 0;
    for (unsigned n = start; n < p.first; ++n) {
      auto target = llvm::find(region, mmas[n]);
      if (llvm::find(region, p.a) < target || llvm::find(region, p.b) < target) continue;
      if (hoist(region, {p.a, p.b}, mmas[n])) { anchors.insert(mmas[n]); ++moved; break; }
    }
  }
  if (anchors.empty()) return;
  pos.clear();
  for (auto [n, op] : llvm::enumerate(region)) pos[op] = n;
  for (auto p : pairs) for (Operation *anchor : anchors)
    if (pos[p.a] < pos[anchor] && pos[anchor] < pos[p.b]) return;
  LocalSchedulePlan plan{original, region};
  if (!plan.validate()) return;
  Operation *end = original.back()->getNextNode();
  plan.apply();
  OpBuilder b(end);
  for (auto p : pairs) {
    auto tag = b.getI64IntegerAttr(stats.nextPair++);
    for (Operation *op : {p.a, p.b}) op->setAttr(LDSPairMarker, tag);
  }
  // Iterate in original MMA order for deterministic guard insertion.
  for (Operation *op : mmas) if (anchors.contains(op)) {
    guard(op, marker);
    guard(op->getNextNode(), marker);
  }
  ++stats.regions; stats.pairs += pairs.size(); stats.moved += moved;
  stats.anchors += anchors.size();
  stats.unpaired += loads.size() - 2 * pairs.size();
}

class ThreadPrefetchPairedLDSPass
    : public PassWrapper<ThreadPrefetchPairedLDSPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadPrefetchPairedLDSPass)
  ThreadPrefetchPairedLDSPass() = default;
  ThreadPrefetchPairedLDSPass(const ThreadPrefetchPairedLDSPass &other)
      : PassWrapper<ThreadPrefetchPairedLDSPass, OperationPass<ModuleOp>>(other) {}
  explicit ThreadPrefetchPairedLDSPass(const PairedLDSPrefetchOptions &o) { distance = o.distance; window = o.window; }
  Pass::Option<unsigned> distance{*this, "distance", llvm::cl::init(0)};
  Pass::Option<std::string> window{*this, "window", llvm::cl::init("register"),
      llvm::cl::desc("address: address/views only; register: ordinary register ALU; reduction: reduction windows")};
  StringRef getArgument() const final { return "frisk-thread-prefetch-paired-lds"; }
  StringRef getDescription() const final { return "Pair and prefetch LDS operands across proven ThreadIR register windows"; }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<FriskDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    auto module = this->getOperation();
    if (distance > 8) { module.emitError("paired LDS distance must be 0..8"); return this->signalPassFailure(); }
    if (window != "address" && window != "register" && window != "reduction") {
      module.emitError("window must be address, register, or reduction");
      return this->signalPassFailure();
    }
    if (!distance) return;
    if (failed(checkGuard(module))) return this->signalPassFailure();
    std::string marker = "frisk.lds_prefetch." + window.getValue();
    bool conflict = false;
    module.walk([&](func::FuncOp f) {
      if (auto old = f->getAttrOfType<StringAttr>(marker))
        if (old.getValue() != std::to_string(distance)) {
          f.emitError("conflicting paired LDS distance"); conflict = true;
        }
    });
    if (conflict) return this->signalPassFailure();
    module.walk([&](func::FuncOp f) {
      if (f.isExternal() || protectedFunction(f)) return;
      foldVectorViews(f);
      deduplicateLoads(f);
      SmallVector<Ops, 4> regions;
      unsigned reductionRegions = 0;
      f.walk([&](Block *block) {
        Ops region;
        bool hasReduction = false;
        auto flush = [&]() {
          if (hasReduction) ++reductionRegions;
          if (!region.empty() && (window != "reduction" || hasReduction)) regions.push_back(std::move(region));
          region.clear(); hasReduction = false;
        };
        for (Operation &op : *block) {
          bool local = reduction(&op);
          bool computation = window == "register" ? pure(&op) :
              window == "reduction" && (local || isa<arith::SelectOp>(&op));
          if (address(&op) || wrapper(&op) || mma(&op) || computation ||
              (candidate(&op) && !protectedLoad(&op))) {
            region.push_back(&op); hasReduction |= local;
          } else flush();
        }
        flush();
      });
      Stats stats;
      f.walk([&](Operation *op) {
        if (auto pair = op->getAttrOfType<IntegerAttr>(LDSPairMarker))
          if (pair.getInt() >= 0) stats.nextPair = std::max(stats.nextPair, uint64_t(pair.getInt()) + 1);
      });
      for (Ops &region : regions) schedule(std::move(region), distance, marker, stats);
      if (stats.regions) markFunction(f, marker, std::to_string(distance));
      llvm::errs() << "Thread LDS prefetch: window=" << window << ", "
                   << "distance=" << distance << ", regions=" << stats.regions
                   << ", pairs=" << stats.pairs << ", moved=" << stats.moved
                   << ", anchors=" << stats.anchors << ", unpaired=" << stats.unpaired;
      if (window == "reduction") llvm::errs() << ", reduction-regions=" << reductionRegions;
      llvm::errs() << ", function=" << f.getName() << '\n';
    });
  }
};
} // namespace
std::unique_ptr<Pass> createThreadPrefetchPairedLDSPass(const PairedLDSPrefetchOptions &options) {
  return std::make_unique<ThreadPrefetchPairedLDSPass>(options);
}
} // namespace mlir::frisk
