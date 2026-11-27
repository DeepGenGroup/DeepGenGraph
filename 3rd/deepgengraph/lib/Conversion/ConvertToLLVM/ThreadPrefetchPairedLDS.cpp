// Pair and prefetch physical LDS operands after thread unrolling/address proofs.
// No LLVM pointer reconstruction or exported-IR scheduling is needed.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/FragmentLDSPrefetch.h"
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

// FragmentIR chooses the issue window before physical addresses are known.
// A read2 may close two outstanding reads across independent MMAs, but never
// across the first use of the earlier read. This is physical pair formation,
// not another distance-based prefetch schedule. Restricting closure to a single
// MMA gap splits swizzled read pairs at each logical read-ahead anchor.
void pairFragmentReads(Ops region, Stats &stats) {
  SmallVector<LDSLoadOp> loads;
  for (Operation *op : region)
    if (candidate(op) && !protectedLoad(op)) loads.push_back(cast<LDSLoadOp>(op));
  llvm::SmallPtrSet<Operation *, 16> paired;
  for (auto a : loads) {
    if (paired.contains(a)) continue;
    LDSLoadOp best;
    unsigned bestClass = 2;
    for (auto b : loads) {
      if (a == b || paired.contains(b) || !a->isBeforeInBlock(b) || !sameAddress(a, b)) continue;
      if (llvm::any_of(a.getResult().getUsers(), [&](Operation *use) {
            return use->isBeforeInBlock(b);
          })) continue;
      int64_t oa = a.getByteOffset(), ob = b.getByteOffset();
      uint64_t delta = oa < ob ? uint64_t(ob) - uint64_t(oa) : uint64_t(oa) - uint64_t(ob);
      unsigned kind = delta && delta % 8 == 0 && delta <= 255 * 8 ? 0 :
                      delta && delta % 512 == 0 && delta <= 255 * 512 ? 1 : 2;
      if (kind >= bestClass) continue;
      Ops trial = region;
      auto it = llvm::find(trial, a.getOperation());
      if (std::next(it) == trial.end() || !hoist(trial, {b}, *std::next(it))) continue;
      best = b; bestClass = kind;
    }
    if (!best) continue;
    Ops trial = region;
    auto it = llvm::find(trial, a.getOperation());
    if (!hoist(trial, {best}, *std::next(it))) continue;
    LocalSchedulePlan plan{region, trial};
    if (!plan.validate()) continue;
    plan.apply(); region = std::move(trial);
    auto tag = IntegerAttr::get(IntegerType::get(a.getContext(), 64), stats.nextPair++);
    for (Operation *op : {a.getOperation(), best.getOperation()}) {
      paired.insert(op); op->setAttr(LDSPairMarker, tag);
    }
    ++stats.pairs;
  }
  stats.unpaired += loads.size() - paired.size();
}

void finalizeFragmentSchedule(func::FuncOp f, Stats &stats) {
  SmallVector<Operation *> anchors;
  llvm::SmallPtrSet<Operation *, 32> issuedBefore;
  bool planned = f->hasAttr(FragmentLDSSchedule);
  f.walk([&](Block *block) {
    Ops region;
    bool pending = false;
    auto flush = [&]() { pairFragmentReads(std::move(region), stats); region.clear(); };
    for (Operation &op : *block) {
      if (candidate(&op)) pending = true;
      else if (mma(&op)) {
        if (pending && op.hasAttr(FragmentLDSAnchor)) issuedBefore.insert(&op);
        pending = false;
      } else if (!address(&op) && !wrapper(&op) && !pure(&op)) pending = false;
      if (address(&op) || wrapper(&op) || pure(&op) ||
          (planned && mma(&op)) ||
          (candidate(&op) && !protectedLoad(&op)))
        region.push_back(&op);
      else flush(); // DMA, waits, writes, regions and existing guards bound pairs.
    }
    flush();
  });
  // Pair closure can empty a logical issue slot. Do not fence an MMA with no
  // outstanding read in that slot: redundant guards inhibit backend scheduling
  // and can separate a newly closed pair from other useful register work.
  f.walk([&](Block *block) {
    SmallVector<LDSLoadOp> pending;
    bool opaque = false;
    for (Operation &op : *block) {
      if (candidate(&op)) pending.push_back(cast<LDSLoadOp>(&op));
      else if (mma(&op)) {
        if (op.hasAttr(FragmentLDSAnchor)) {
          // Only remove a boundary whose physical issue slot was emptied by
          // pair closure. Uncanonicalized or unsupported reads must retain
          // their original protection even though candidate() cannot see them.
          if (issuedBefore.contains(&op) && pending.empty() && !opaque)
            op.removeAttr(FragmentLDSAnchor);
          else anchors.push_back(&op);
        }
        pending.clear();
        opaque = false;
      } else if (!address(&op) && !wrapper(&op) && !pure(&op)) {
        pending.clear();
        opaque = true;
      }
    }
  });
  for (Operation *op : anchors) {
    if (isa<LLVM::InlineAsmOp>(op)) {
      // Preserve the ordering contract when register MMAs become pure during
      // LLVM export. Otherwise O3 can move the MMA out of its guard bracket.
      op->setAttr("frisk.register_mma", UnitAttr::get(op->getContext()));
      op->setAttr("frisk.scheduled_mma", UnitAttr::get(op->getContext()));
    }
    guardMemoryAndMMA(op, "frisk.lds_prefetch.fragment");
    guardMemoryAndMMA(op->getNextNode(), "frisk.lds_prefetch.fragment");
    op->removeAttr(FragmentLDSAnchor);
    ++stats.anchors;
  }
}

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
    guardMemoryAndMMA(op, marker);
    guardMemoryAndMMA(op->getNextNode(), marker);
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
      llvm::cl::desc("address/register/reduction: legacy prefetch; fragment: pair and protect the FragmentIR schedule")};
  StringRef getArgument() const final { return "frisk-thread-prefetch-paired-lds"; }
  StringRef getDescription() const final { return "Pair and prefetch LDS operands across proven ThreadIR register windows"; }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<FriskDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    auto module = this->getOperation();
    if (distance > 8) { module.emitError("paired LDS distance must be 0..8"); return this->signalPassFailure(); }
    if (window != "address" && window != "register" && window != "reduction" && window != "fragment") {
      module.emitError("window must be address, register, reduction, or fragment");
      return this->signalPassFailure();
    }
    if (!distance && window != "fragment") return;
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
      if (window == "fragment") {
        Stats stats;
        f.walk([&](Operation *op) {
          if (auto pair = op->getAttrOfType<IntegerAttr>(LDSPairMarker))
            if (pair.getInt() >= 0) stats.nextPair = std::max(stats.nextPair, uint64_t(pair.getInt()) + 1);
        });
        finalizeFragmentSchedule(f, stats);
        if (stats.pairs || stats.anchors) markFunction(f, marker, std::to_string(distance));
        llvm::errs() << "Thread LDS finalize: fragment-plan, pairs=" << stats.pairs
                     << ", anchors=" << stats.anchors << ", unpaired=" << stats.unpaired
                     << ", function=" << f.getName() << '\n';
        return;
      }
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
