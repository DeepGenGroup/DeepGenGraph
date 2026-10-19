#include "deepgengraph/Conversion/VectorCanonicalization.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_OPTIMIZETENSORLOOPS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
namespace dg = mlir::deepgengraph;
namespace dgt = mlir::deepgengraph::triton;
using Loop = affine::AffineForOp;

// Resolve the pointer's identity, including the isolated device region ABI.
// Different pointer arguments alone do NOT prove non-aliasing.
static Value pointerRoot(Value value) {
  if (auto ptr = value.getDefiningOp<dgt::BlockPointerOfOp>())
    return pointerRoot(ptr.getBasePointer());
  if (auto ptr = value.getDefiningOp<dgt::PointerOfOp>())
    if (auto tensor = ptr.getOperand().getDefiningOp<dgt::TensorFromOp>())
      return pointerRoot(tensor.getOperand());
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto device = dyn_cast<dgt::DeviceKernelOp>(arg.getOwner()->getParentOp())) {
      unsigned offset = device.getGrid().size();
      if (arg.getArgNumber() >= offset)
        return pointerRoot(device.getArgs()[arg.getArgNumber() - offset]);
    }
  }
  return value;
}

// Computed tensors have value semantics, even though the legacy ZeroOp lacks
// a Pure trait to keep its later destination-style buffer initialization local.
static bool tensorValueOp(Operation *op) {
  return isa<dg::ZeroOp>(op) || isMemoryEffectFree(op);
}

static bool canJam(Loop outer, uint64_t maxFactor) {
  auto count = affine::getConstantTripCount(outer);
  if (!count || *count < 2 || maxFactor < 2 || outer.getNumResults())
    return false;
  bool nested = false, useful = false, valid = true;
  SmallVector<Value> reads, writes;
  outer.walk([&](Operation *op) {
    if (auto loop = dyn_cast<Loop>(op)) {
      nested |= loop != outer;
      if (loop != outer) {
        for (Value control : loop.getControlOperands())
          if (!outer.isDefinedOutsideOfLoop(control)) valid = false;
        for (Type type : loop.getResultTypes())
          if (!isa<RankedTensorType>(type)) valid = false;
      }
      return;
    }
    if (auto load = dyn_cast<dgt::BlockLoadOp>(op)) {
      reads.push_back(pointerRoot(load.getSrcPointer()));
      return;
    }
    if (auto store = dyn_cast<dgt::BlockStoreOp>(op)) {
      // Stores must be an epilogue of the outer loop. Their relative order is
      // retained; all moved reads must be proven independent below.
      if (store->getBlock() != outer.getBody()) valid = false;
      for (Operation *next = store->getNextNode(); next; next = next->getNextNode())
        if (isa<Loop>(next)) valid = false;
      writes.push_back(pointerRoot(store.getDstPointer()));
      return;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) return;
    if (!tensorValueOp(op)) valid = false;
    // Require actual tensor work in an inner loop. No shape, kernel name or
    // arithmetic-op name is part of the profitability check.
    useful |= op->getParentOp() != outer &&
              llvm::any_of(op->getResultTypes(), [](Type t) {
                return isa<RankedTensorType>(t);
              });
  });
  for (Value write : writes) {
    auto output = write.getDefiningOp<dgt::PointerOfOp>();
    if (!write.getDefiningOp<dgt::EmptyPointerOp>() &&
        !(output && output->hasAttr("frisk.fresh_output"))) return false;
    for (Value read : reads) {
      auto input = read.getDefiningOp<dgt::PointerOfOp>();
      if (read == write || !input ||
          (output && input.getOperand() == output.getOperand()))
        return false;
    }
  }
  return valid && nested && useful;
}

// Prove equal recurrences by induction: identical initial values and identical
// yielded expressions under the assumed equality of the two carrier args.
// This is structural value equivalence, not reassociation of FP expressions.
class ValueEquivalence {
public:
  Operation *scope = nullptr;
  DenseMap<Value, Value> equal;
  unsigned budget = 10000;
  bool check(Value a, Value b) {
    if (a == b || equal.lookup(a) == b) return true;
    if (!budget || a.getType() != b.getType()) return false;
    --budget;
    auto x = dyn_cast<OpResult>(a), y = dyn_cast<OpResult>(b);
    if (!x || !y || x.getResultNumber() != y.getResultNumber() ||
        !tensorValueOp(x.getOwner()) || !tensorValueOp(y.getOwner()))
      return false;
    if ((isa<dgt::BlockLoadOp>(x.getOwner()) || isa<dgt::BlockLoadOp>(y.getOwner())) &&
        (!scope || !scope->isAncestor(x.getOwner()) || !scope->isAncestor(y.getOwner())))
      return false;
    // Regions may contain external memory or their own recurrence. Their
    // results are shared by ordinary CSE or by a separate inner-loop proof.
    if (x.getOwner()->getNumRegions() || y.getOwner()->getNumRegions()) return false;
    return OperationEquivalence::isEquivalentTo(x.getOwner(), y.getOwner(),
        [&](Value lhs, Value rhs) { return success(check(lhs, rhs)); },
        [&](Value lhs, Value rhs) { equal[lhs] = rhs; },
        OperationEquivalence::IgnoreLocations);
  }
};

static bool mergeCarriers(Loop loop) {
  unsigned count = loop.getNumResults();
  if (count < 2) return false;
  bool valueSemantics = true;
  loop.walk([&](Operation *op) {
    if (!isa<Loop>(op) && !op->hasTrait<OpTrait::IsTerminator>() && !tensorValueOp(op))
      valueSemantics = false;
  });
  if (!valueSemantics) return false;
  auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
  SmallVector<unsigned> representatives;
  bool changed = false;
  for (unsigned i = 0; i < count; ++i) {
    unsigned representative = i;
    if (isa<RankedTensorType>(loop.getResult(i).getType())) {
      for (unsigned j = 0; j < i; ++j) {
        if (representatives[j] != j) continue;
        ValueEquivalence proof;
        proof.scope = loop;
        if (!proof.check(loop.getInits()[i], loop.getInits()[j])) continue;
        proof.equal[loop.getRegionIterArgs()[i]] = loop.getRegionIterArgs()[j];
        if (!proof.check(yield.getOperand(i), yield.getOperand(j))) continue;
        representative = j;
        break;
      }
    }
    representatives.push_back(representative);
    if (representative != i) {
      loop.getRegionIterArgs()[i].replaceAllUsesWith(loop.getRegionIterArgs()[representative]);
      loop.getResult(i).replaceAllUsesWith(loop.getResult(representative));
      changed = true;
    }
  }
  if (!changed) return false;
  SmallVector<Value> inits, yielded;
  for (unsigned i = 0; i < count; ++i)
    if (representatives[i] == i) {
      inits.push_back(loop.getInits()[i]);
      yielded.push_back(yield.getOperand(i));
    }
  OpBuilder b(loop);
  auto replacement = b.create<Loop>(loop.getLoc(), loop.getLowerBoundOperands(),
      loop.getLowerBoundMap(), loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), inits);
  for (NamedAttribute attr : loop->getDiscardableAttrs())
    replacement->setAttr(attr.getName(), attr.getValue());
  IRMapping map;
  map.map(loop.getInductionVar(), replacement.getInductionVar());
  unsigned n = 0;
  for (unsigned i = 0; i < count; ++i)
    if (representatives[i] == i) {
      map.map(loop.getRegionIterArgs()[i], replacement.getRegionIterArgs()[n]);
      loop.getResult(i).replaceAllUsesWith(replacement.getResult(n++));
    }
  b.setInsertionPointToStart(replacement.getBody());
  for (Operation &op : loop.getBody()->without_terminator()) b.clone(op, map);
  SmallVector<Value> values;
  for (Value value : yielded) values.push_back(map.lookupOrDefault(value));
  b.create<affine::AffineYieldOp>(loop.getLoc(), values);
  loop.erase();
  return true;
}

class OptimizeTensorLoops : public impl::OptimizeTensorLoopsBase<OptimizeTensorLoops> {
  // This is a lower bound on register demand, not an occupancy prediction.
  // Nested reductions overlap their carriers with the enclosing reduction.
  // Keep room for operand fragments, addresses and math temporaries (the
  // default permits 192 carrier words/thread for a 128-thread block).
  static uint64_t carrierPeak(Operation *op, uint64_t live = 0) {
    if (auto loop = dyn_cast<Loop>(op)) {
      for (Type type : loop.getResultTypes()) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape() ||
            !tensor.getElementType().isIntOrFloat())
          return UINT64_MAX;
        uint64_t words = llvm::divideCeil(
            uint64_t(tensor.getNumElements()) * tensor.getElementTypeBitWidth(),
            uint64_t(32));
        if (words > UINT64_MAX - live) return UINT64_MAX;
        live += words;
      }
    }
    uint64_t peak = live;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          peak = std::max(peak, carrierPeak(&nested, live));
    return peak;
  }

  LogicalResult cleanup(ModuleOp module) {
    bool changed;
    do {
      PassManager passes(module.getContext());
      passes.addPass(createVectorSafeCanonicalizerPass());
      passes.addPass(createCSEPass());
      if (failed(module == getOperation() ? runPipeline(passes, module)
                                         : passes.run(module))) return failure();
      changed = false;
      module.walk<WalkOrder::PostOrder>([&](Loop loop) {
        changed |= mergeCarriers(loop);
      });
    } while (changed);
    return success();
  }

  uint64_t chooseFactor(Loop loop) {
    auto count = *affine::getConstantTripCount(loop);
    // Trial on an isolated copy: sharing is proven by recurrence CSE before
    // charging carriers. A failed profitability check cannot mutate input IR.
    unsigned ordinal = 0, selected = 0;
    getOperation().walk([&](Loop candidate) {
      if (candidate == loop) selected = ordinal;
      ++ordinal;
    });
    for (uint64_t factor = std::min<uint64_t>(count, maxJamFactor);
         factor >= 2; --factor) {
      if (count % factor) continue;
      if (!maxCarrierWords) return factor;
      OwningOpRef<ModuleOp> trial = cast<ModuleOp>(getOperation()->clone());
      Loop copy;
      ordinal = 0;
      trial->walk([&](Loop candidate) {
        if (ordinal++ == selected) copy = candidate;
      });
      if (failed(affine::loopUnrollJamByFactor(copy, factor)) ||
          failed(cleanup(*trial)))
        continue;
      if (carrierPeak(trial->getOperation()) <= maxCarrierWords)
        return factor;
    }
    return 1;
  }

  void runOnOperation() override {
    SmallVector<Loop> candidates;
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) {
      if (canJam(loop, maxJamFactor)) candidates.push_back(loop);
    });
    // Only outer loops without iter_args are candidates, so nested reduction
    // loop handles are not left dangling by the unroll-and-jam utility.
    for (Loop loop : candidates) {
      uint64_t factor = chooseFactor(loop);
      if (factor == 1) continue;
      if (failed(affine::loopUnrollJamByFactor(loop, factor))) {
        loop.emitError("failed checked tensor loop unroll-and-jam");
        return signalPassFailure();
      }
    }
    if (candidates.empty()) return;
    // CSE exposes shared expressions; carrier merging exposes more CSE. The
    // fixed point terminates because every merge removes at least one carrier.
    if (failed(cleanup(getOperation()))) return signalPassFailure();
    SmallVector<dg::ZeroOp> deadZeros;
    getOperation().walk([&](dg::ZeroOp zero) {
      if (zero->use_empty()) deadZeros.push_back(zero);
    });
    for (auto zero : deadZeros) zero.erase();
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) {
      moveLoopInvariantCode({&loop.getRegion()},
          [&](Value value, Region *) { return loop.isDefinedOutsideOfLoop(value); },
          [&](Operation *op, Region *) {
            // BlockLoadOp is historically marked Pure, but moving a load can
            // speculate an invalid address out of a zero-trip loop. Leave
            // memory motion to an alias/bounds-aware pass; use standard LICM
            // for the value computations whose operands are already available.
            bool containsLoad = false;
            op->walk([&](dgt::BlockLoadOp) { containsLoad = true; });
            return !containsLoad && isMemoryEffectFree(op) && isSpeculatable(op);
          },
          [&](Operation *op, Region *) { op->moveBefore(loop); });
    });
    OpPassManager cleanup(ModuleOp::getOperationName());
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, getOperation()))) signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> createOptimizeTensorLoopsPass() {
  return std::make_unique<OptimizeTensorLoops>();
}
} // namespace mlir::frisk
