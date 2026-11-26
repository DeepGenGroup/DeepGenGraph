// Schedule LDS -> register reads while MMA operands still have fragment types.
// Physical read2 selection and compiler guards are materialized in ThreadIR.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Conversion/FragmentLDSPrefetch.h"
#include "deepgengraph/Conversion/VectorCanonicalization.h"
#include "deepgengraph/Analysis/ScheduleAttrs.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace mlir::frisk {
namespace {
bool authored(Operation *op) {
  for (; op; op = op->getParentOp())
    if (hasPipelineScheduleAttr(op) || op->hasAttr("cpsat.preserve_schedule"))
      return true;
  return false;
}

bool sharedRead(Operation *op) {
  auto load = dyn_cast<LoadFragOp>(op);
  return load && cast<MemRefType>(load.getSource().getType()).getMemorySpaceAsInt() == 3;
}

bool candidate(LoadFragOp load) {
  return sharedRead(load) && !load.getResult().use_empty() &&
      llvm::all_of(load.getResult().getUses(), [&](OpOperand &use) {
        return isa<GemmFragOp>(use.getOwner()) && use.getOperandNumber() < 2 &&
               use.getOwner()->getBlock() == load->getBlock();
      });
}

bool localReduction(Operation *op) {
  if (auto reduce = dyn_cast<ReduceFragOp>(op))
    return reduce.getLaneExtent() == 1 && isMemoryEffectFree(op);
  return isa<arith::AddFOp, arith::MaxNumFOp, arith::MinNumFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

// Hoist only checked address dependencies, never an FP producer, another load
// or a collective. A failed trial leaves both IR and its schedule untouched.
bool moveRead(LoadFragOp load, Operation *target) {
  llvm::SmallPtrSet<Operation *, 16> slice;
  SmallVector<Operation *> order;
  std::function<bool(Operation *)> collect = [&](Operation *op) {
    if (op->getBlock() != target->getBlock() || op->isBeforeInBlock(target)) return true;
    if (op == target) return false;
    if (!slice.insert(op).second) return true;
    if (op != load && (!isOrdinaryRegisterComputation(op) ||
        !llvm::all_of(op->getResultTypes(), [](Type t) {
          return t.isIndex() || isa<IntegerType, BaseMemRefType>(t);
        }))) return false;
    for (Value v : op->getOperands())
      if (auto *def = v.getDefiningOp())
        if (!collect(def)) return false;
    order.push_back(op);
    return true;
  };
  if (!collect(load)) return false;
  for (Operation *op : order) op->moveBefore(target);
  return true;
}

struct Stats { unsigned unrolled = 0, moved = 0, anchors = 0; };

// Follow register dataflow, not textual stage numbers. An accumulator carried
// from a previous MMA epoch has already paid the startup cost. Packing and
// inserting fragments can hide that dependence at this level.
bool hasMMAProducer(Value value, Block *block,
                    llvm::function_ref<bool(Operation *)> accept) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 32> seen;
  while (!worklist.empty() && seen.size() < 1024) {
    Operation *op = worklist.pop_back_val().getDefiningOp();
    if (!op || op->getBlock() != block || !seen.insert(op).second) continue;
    if (isa<GemmFragOp>(op)) {
      if (accept(op)) return true;
      continue;
    }
    if (!isOrdinaryRegisterComputation(op)) continue;
    llvm::append_range(worklist, op->getOperands());
  }
  return false;
}

void schedule(ArrayRef<Operation *> region, unsigned distance, Stats &stats) {
  SmallVector<Operation *> mmas;
  SmallVector<LoadFragOp> loads;
  for (Operation *op : region) {
    if (isa<GemmFragOp>(op)) mmas.push_back(op);
    if (auto load = dyn_cast<LoadFragOp>(op); load && candidate(load)) loads.push_back(load);
  }
  if (mmas.empty()) return;
  bool startup = !hasMMAProducer(mmas.front()->getOperand(2),
      mmas.front()->getBlock(), [](Operation *) { return true; });
  for (auto load : loads) {
    auto first = llvm::find_if(mmas, [&](Operation *mma) {
      return mma->getOperand(0) == load.getResult() || mma->getOperand(1) == load.getResult();
    });
    unsigned consumer = first - mmas.begin();
    if (first == mmas.end() || consumer == 0) continue;
    unsigned start = consumer > distance ? consumer - distance : 0;
    // Bound only a fresh accumulation's startup. Applying this at every wait
    // drains the steady-state read window and repeats a one-pair lgkmcnt(0)
    // startup for every tile. Carried accumulators retain the full distance.
    if (startup && consumer > 1) start = std::max(1u, start);
    for (unsigned i = start; i < consumer; ++i) {
      Operation *anchor = mmas[i];
      bool moved = !load->isBeforeInBlock(anchor);
      if (moved && !moveRead(load, anchor)) continue;
      // Also protect overlap already exposed by the loop lookahead pass.
      if (!anchor->hasAttr(FragmentLDSAnchor)) {
        anchor->setAttr(FragmentLDSAnchor, UnitAttr::get(anchor->getContext()));
        ++stats.anchors;
      }
      stats.moved += moved;
      break;
    }
  }
}

class PrefetchFragmentLDSPass
    : public PassWrapper<PrefetchFragmentLDSPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchFragmentLDSPass)
  PrefetchFragmentLDSPass() = default;
  PrefetchFragmentLDSPass(const PrefetchFragmentLDSPass &other) : PassWrapper(other) {}
  PrefetchFragmentLDSPass(unsigned d, unsigned r) { distance = d; reductionDistance = r; }
  Option<unsigned> distance{*this, "distance", llvm::cl::init(0),
      llvm::cl::desc("Maximum independent MMAs ahead of the first consumer (0..8)")};
  Option<unsigned> reductionDistance{*this, "reduction-distance", llvm::cl::init(0),
      llvm::cl::desc("Read-ahead across lane-local reductions (0..8)")};
  StringRef getArgument() const final { return "frisk-prefetch-fragment-lds"; }
  StringRef getDescription() const final { return "Overlap fragment LDS reads with independent register MMAs"; }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<FriskDialect, affine::AffineDialect, arith::ArithDialect>();
  }
  void runOnOperation() final {
    auto f = getOperation();
    if (distance > 8 || reductionDistance > 8) {
      f.emitError("fragment LDS prefetch distances must be 0..8");
      return signalPassFailure();
    }
    if (f.isExternal() || (!distance && !reductionDistance) || authored(f)) return;
    std::string config = std::to_string(distance) + ":" + std::to_string(reductionDistance);
    if (auto old = f->getAttrOfType<StringAttr>(FragmentLDSSchedule)) {
      if (old.getValue() != config) {
        f.emitError("conflicting fragment LDS prefetch distances"); signalPassFailure();
      }
      return;
    }
    Stats stats;
    // Expose only small MMA fragment loops, never the enclosing tile/DMA loop.
    // This is the unrolling otherwise performed after fragment lowering.
    SmallVector<affine::AffineForOp> loops;
    f.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) {
      if (loop->hasAttr("frisk.fragment_loop") && !authored(loop) &&
          loop.hasConstantBounds() &&
          loop.walk([](GemmFragOp) { return WalkResult::interrupt(); }).wasInterrupted())
        loops.push_back(loop);
    });
    uint64_t budget = 16384;
    for (auto loop : loops) {
      int64_t span;
      if (__builtin_sub_overflow(loop.getConstantUpperBound(), loop.getConstantLowerBound(), &span) || span <= 0)
        continue;
      uint64_t trips = (uint64_t(span) - 1) / loop.getStepAsInt() + 1;
      uint64_t size = 0, cost;
      loop.walk([&](Operation *) { ++size; });
      if (trips > 256 || __builtin_mul_overflow(size, trips - 1, &cost) || cost > budget) continue;
      if (succeeded(affine::loopUnrollFull(loop))) { budget -= cost; ++stats.unrolled; }
    }
    // Folding static fragment coordinates exposes real MMA uses and CSE
    // shares repeated reads before increasing their register live ranges.
    OpPassManager cleanup(func::FuncOp::getOperationName());
    cleanup.addPass(createVectorSafeCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, f))) return signalPassFailure();

    auto runWindow = [&](unsigned d, bool reductions) {
      if (!d) return;
      f.walk([&](Block *block) {
        SmallVector<Operation *> region;
        bool hasReduction = false;
        auto flush = [&](Operation *boundary = nullptr) {
          unsigned effective = d;
          // The lane-local part of a fragment reduction is exposed only later
          // in ThreadIR. Reserve its shorter read-ahead window now; a second
          // scheduling sweep cannot undo reads already hoisted by distance.
          if (auto reduce = dyn_cast_or_null<ReduceFragOp>(boundary);
              reduce && reductionDistance && hasMMAProducer(
                  reduce->getOperand(0), block, [&](Operation *mma) {
                    return llvm::is_contained(region, mma);
                  }))
            effective = std::min(effective, unsigned(reductionDistance));
          if (!reductions || hasReduction) schedule(region, effective, stats);
          region.clear(); hasReduction = false;
        };
        for (Operation &op : *block) {
          bool local = localReduction(&op);
          bool allowed = !authored(&op) &&
              (sharedRead(&op) || isa<GemmFragOp>(&op) ||
               (local ? reductions : isOrdinaryRegisterComputation(&op)));
          if (allowed) { region.push_back(&op); hasReduction |= local; }
          else flush(&op);
        }
        flush();
      });
    };
    runWindow(distance, false);
    runWindow(reductionDistance, true);
    f->setAttr(FragmentLDSSchedule, StringAttr::get(f.getContext(), config));
    llvm::errs() << "Fragment LDS prefetch: distance=" << distance
                 << ", reduction-distance=" << reductionDistance
                 << ", unrolled=" << stats.unrolled << ", moved=" << stats.moved
                 << ", anchors=" << stats.anchors << ", function=" << f.getName() << '\n';
  }
};
} // namespace
std::unique_ptr<Pass> createPrefetchFragmentLDSPass(unsigned distance, unsigned reductionDistance) {
  return std::make_unique<PrefetchFragmentLDSPass>(distance, reductionDistance);
}
} // namespace mlir::frisk
