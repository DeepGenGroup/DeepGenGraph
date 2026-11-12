// Zero-mask bypass belongs to thread SSA: retain structured branches and
// vector lanes instead of recovering them from LLVM aggregate PHIs.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DECL_THREADMASKADDBYPASS
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"
#define GEN_PASS_DEF_THREADMASKADDBYPASS
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

static bool isProtected(func::FuncOp function) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none",
                         "denormal-fp-math", "denormal-fp-math-f32",
                         "llvm.denormal_fp_math", "llvm.denormal_fp_math_f32"})
    if (function->hasAttr(name)) return true;
  for (StringRef key : {"passthrough", "llvm.passthrough"})
    if (auto attrs = function->getAttrOfType<ArrayAttr>(key))
      for (Attribute attr : attrs) {
        if (auto pair = dyn_cast<ArrayAttr>(attr))
          if (!pair.empty()) attr = pair[0];
        if (auto name = dyn_cast<StringAttr>(attr))
          if (name.getValue() == "strictfp" || name.getValue() == "optnone" ||
              name.getValue() == "denormal-fp-math" ||
              name.getValue() == "denormal-fp-math-f32") return true;
      }
  return false;
}

static bool isPositiveZero(Attribute attr) {
  if (auto scalar = dyn_cast_or_null<FloatAttr>(attr))
    return scalar.getValue().isZero() && !scalar.getValue().isNegative();
  if (auto elements = dyn_cast_or_null<DenseFPElementsAttr>(attr))
    return llvm::all_of(elements.getValues<APFloat>(), [](const APFloat &v) {
      return v.isZero() && !v.isNegative();
    });
  return false;
}

// These are lane views, never GPU wave shuffles, memory, casts or arithmetic.
static bool isView(Operation *op) {
  if (op->getNumResults() != 1) return false;
  if (auto type = dyn_cast<VectorType>(op->getResult(0).getType());
      type && type.isScalable()) return false;
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return extract.getDynamicPosition().empty();
  if (auto shuffle = dyn_cast<vector::ShuffleOp>(op))
    return llvm::all_of(shuffle.getMask(), [](int64_t lane) { return lane >= 0; });
  return isa<vector::ShapeCastOp, vector::BroadcastOp, vector::TransposeOp,
             vector::ExtractStridedSliceOp, vector::FromElementsOp>(op);
}

struct ClearValues {
  DenseMap<Value, Attribute> values;
  Attribute get(Value value, unsigned depth = 0) {
    if (depth > 64 || values.size() > 256) return {};
    if (auto it = values.find(value); it != values.end()) return it->second;
    Attribute constant;
    if (matchPattern(value, m_Constant(&constant))) return constant;
    auto *op = value.getDefiningOp();
    if (!op || !isView(op)) return {};
    SmallVector<Attribute> operands;
    for (Value input : op->getOperands()) {
      auto attr = get(input, depth + 1);
      if (!attr) return {};
      operands.push_back(attr);
    }
    // View fold hooks need not materialize constant tensors. Verified shape
    // changes, permutations and broadcasts preserve an all-zero payload.
    if (isa<vector::TransposeOp, vector::ShapeCastOp, vector::BroadcastOp,
            vector::FromElementsOp>(op) &&
        llvm::all_of(operands, isPositiveZero)) {
      constant = Builder(op->getContext()).getZeroAttr(value.getType());
      return values[value] = constant;
    }
    // Some fold hooks mutate operations; analyze a detached clone only.
    Operation *copy = op->clone();
    SmallVector<OpFoldResult> folded;
    if (succeeded(copy->fold(operands, folded)) && folded.size() == 1) {
      if (auto attr = dyn_cast<Attribute>(folded[0])) constant = attr;
      else if (auto other = dyn_cast<Value>(folded[0]);
               other && other.getDefiningOp() != copy)
        constant = get(other, depth + 1);
    }
    copy->destroy();
    if (constant) values[value] = constant;
    return constant;
  }
};

struct Candidate {
  Operation *source;
  Value root, masked, condition;
  bool clearOnTrue = true;
  ClearValues clear;
  SmallVector<Operation *> views;
  SmallVector<arith::AddFOp> adds;
  llvm::SmallPtrSet<Operation *, 32> seen;
  DenseSet<Value> derived;

  bool available(Value value, DominanceInfo &dominance, unsigned depth = 0) {
    if (depth > 64 || derived.contains(value)) return false;
    if (dominance.dominates(value, source)) return true;
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != source->getBlock()) return false;
    if (isa<arith::ConstantOp>(op)) return true;
    return isView(op) && llvm::all_of(op->getOperands(), [&](Value input) {
      return available(input, dominance, depth + 1);
    });
  }

  Value makeAvailable(OpBuilder &b, Value value, IRMapping &mapping,
                      DominanceInfo &dominance) {
    if (mapping.contains(value)) return mapping.lookup(value);
    if (dominance.dominates(value, source)) return value;
    auto *op = value.getDefiningOp();
    for (Value input : op->getOperands())
      mapping.map(input, makeAvailable(b, input, mapping, dominance));
    return b.clone(*op, mapping)->getResult(0);
  }

  bool match() {
    if (source->getNumResults() != 1) return false;
    root = source->getResult(0);
    if (auto type = dyn_cast<VectorType>(root.getType()); type && type.isScalable()) return false;
    Value yes, no;
    if (auto branch = dyn_cast<scf::IfOp>(source)) {
      if (branch.getElseRegion().empty()) return false;
      yes = branch.thenYield().getOperand(0);
      no = branch.elseYield().getOperand(0);
      condition = branch.getCondition();
    } else if (auto select = dyn_cast<arith::SelectOp>(source)) {
      yes = select.getTrueValue(); no = select.getFalseValue();
      condition = select.getCondition();
    } else return false;
    Attribute zero = clear.get(yes);
    if (!isPositiveZero(zero)) {
      zero = clear.get(no);
      clearOnTrue = false;
    }
    if (!isPositiveZero(zero)) return false;
    masked = clearOnTrue ? no : yes;
    clear.values[root] = zero;
    derived.insert(root);
    SmallVector<Value> pending{root};
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (Operation *user : value.getUsers()) {
        if (user->getBlock() != source->getBlock() ||
            !source->isBeforeInBlock(user)) return false;
        if (!seen.insert(user).second) continue;
        if (seen.size() > 256) return false;
        if (isView(user)) {
          if (!clear.get(user->getResult(0))) return false;
          views.push_back(user);
          derived.insert(user->getResult(0));
          pending.push_back(user->getResult(0));
        } else if (auto add = dyn_cast<arith::AddFOp>(user)) adds.push_back(add);
        else return false;
      }
    }
    if (adds.empty()) return false;
    llvm::sort(adds, [](auto a, auto b) { return a->isBeforeInBlock(b); });
    DominanceInfo dominance;
    for (Operation *view : views)
      for (Value input : view->getOperands())
        if (!derived.contains(input) && !dominance.dominates(input, source)) return false;
    for (auto add : adds) {
      unsigned side = derived.contains(add.getLhs()) ? 0 : 1;
      Value score = add->getOperand(1 - side);
      if (derived.contains(score) || !isPositiveZero(clear.get(add->getOperand(side))) ||
          !available(score, dominance)) return false;
    }
    // Preserve memory/synchronization/call boundaries between mask and adds.
    for (Operation *op = source->getNextNode(); op != adds.back().getOperation();
         op = op->getNextNode())
      if (!op || op->getNumRegions() || isa<CallOpInterface>(op) || !isPure(op)) return false;
    return true;
  }

  Value materialize(OpBuilder &b, Value value, IRMapping &mapping) {
    if (mapping.contains(value)) return mapping.lookup(value);
    if (!derived.contains(value)) return value;
    auto *op = value.getDefiningOp();
    for (Value input : op->getOperands())
      mapping.map(input, materialize(b, input, mapping));
    return b.clone(*op, mapping)->getResult(0);
  }

  static Type predicateType(Type type, OpBuilder &b) {
    if (auto vector = dyn_cast<VectorType>(type))
      return VectorType::get(vector.getShape(), b.getI1Type());
    return b.getI1Type();
  }

  // Distribute vector predicates through precisely the same static lane views.
  Value predicate(OpBuilder &b, Value value, DenseMap<Value, Value> &mapped) {
    if (auto it = mapped.find(value); it != mapped.end()) return it->second;
    if (!derived.contains(value)) {
      Type type = predicateType(value.getType(), b);
      return b.create<arith::ConstantOp>(source->getLoc(), type, b.getOneAttr(type));
    }
    auto *op = value.getDefiningOp();
    IRMapping inputs;
    for (Value input : op->getOperands()) inputs.map(input, predicate(b, input, mapped));
    Operation *copy = b.clone(*op, inputs);
    copy->getResult(0).setType(predicateType(value.getType(), b));
    return mapped[value] = copy->getResult(0);
  }

  unsigned rewrite() {
    OpBuilder b(source);
    IRMapping mapping;
    mapping.map(root, masked);
    // Only duplicate static register views/constants of existing scores.
    // Loads, calls and score arithmetic are never hoisted across the mask.
    DominanceInfo dominance;
    for (auto add : adds) {
      unsigned side = derived.contains(add.getLhs()) ? 0 : 1;
      Value score = add->getOperand(1 - side);
      mapping.map(score, makeAvailable(b, score, mapping, dominance));
    }
    if (auto branch = dyn_cast<scf::IfOp>(source)) {
      SmallVector<Type> types{root.getType()};
      for (auto add : adds) types.push_back(add.getType());
      auto replacement = b.create<scf::IfOp>(source->getLoc(), types, condition, true);
      replacement->setAttrs(source->getAttrs());
      replacement->setAttr("frisk.mask_add_bypass", b.getUnitAttr());
      replacement.getThenRegion().takeBody(branch.getThenRegion());
      replacement.getElseRegion().takeBody(branch.getElseRegion());
      auto clearYield = clearOnTrue ? replacement.thenYield() : replacement.elseYield();
      auto maskedYield = clearOnTrue ? replacement.elseYield() : replacement.thenYield();
      b.setInsertionPoint(maskedYield);
      unsigned index = 1;
      for (auto add : adds) {
        unsigned side = derived.contains(add.getLhs()) ? 0 : 1;
        Value score = mapping.lookupOrDefault(add->getOperand(1 - side));
        materialize(b, add->getOperand(side), mapping);
        Value sum = b.clone(*add, mapping)->getResult(0);
        clearYield->insertOperands(clearYield.getNumOperands(), score);
        maskedYield->insertOperands(maskedYield.getNumOperands(), sum);
        add.replaceAllUsesWith(replacement.getResult(index++));
      }
      root.replaceAllUsesWith(replacement.getResult(0));
    } else {
      // Selects remain selects, including per-lane vector masks: do not
      // introduce divergent branches or claim inactive lanes skip the fadd.
      Value clearPredicate = condition;
      if (!clearOnTrue) {
        auto ones = b.create<arith::ConstantOp>(source->getLoc(), condition.getType(),
                                               b.getOneAttr(condition.getType()));
        clearPredicate = b.create<arith::XOrIOp>(source->getLoc(), condition, ones);
      }
      DenseMap<Value, Value> predicates{{root, clearPredicate}};
      for (auto add : adds) {
        unsigned side = derived.contains(add.getLhs()) ? 0 : 1;
        Value mask = add->getOperand(side);
        Value score = mapping.lookupOrDefault(add->getOperand(1 - side));
        Value cond = condition.getType().isInteger(1) ? clearPredicate : predicate(b, mask, predicates);
        materialize(b, mask, mapping);
        Value sum = b.clone(*add, mapping)->getResult(0);
        auto result = b.create<arith::SelectOp>(add.getLoc(), cond, score, sum);
        result->setAttr("frisk.mask_add_bypass", b.getUnitAttr());
        add.replaceAllUsesWith(result.getResult());
      }
    }
    unsigned count = adds.size();
    for (auto add : adds) add.erase();
    bool changed;
    do {
      changed = false;
      for (auto *&view : views)
        if (view && view->use_empty()) {
          view->erase(); view = nullptr; changed = true;
        }
    } while (changed);
    source->erase();
    return count;
  }
};

class ThreadMaskAddBypassPass : public impl::ThreadMaskAddBypassBase<ThreadMaskAddBypassPass> {
  void runOnOperation() override {
    auto function = getOperation();
    if (mode != "off" && mode != "approx") {
      function.emitError("mask-add-bypass must be off or approx; legal mode has been removed");
      return signalPassFailure();
    }
    if (mode == "off" || isProtected(function)) return;
    unsigned groups = 0, additions = 0;
    // MaskOpTiling can already fuse score additions into the uniform branch.
    // After unrolling, eliminate only its clear-arm +0 operations. Neither
    // unrelated adds nor the original masked-arm arithmetic acquire FP flags.
    function.walk([&](scf::IfOp branch) {
      if (!branch->hasAttr("frisk.uniform_mask")) return;
      SmallVector<arith::AddFOp> clearAdds;
      branch.getThenRegion().walk([&](arith::AddFOp add) { clearAdds.push_back(add); });
      unsigned folded = 0;
      for (auto add : clearAdds) {
        ClearValues clear;
        Value score;
        if (isPositiveZero(clear.get(add.getRhs()))) score = add.getLhs();
        else if (isPositiveZero(clear.get(add.getLhs()))) score = add.getRhs();
        if (!score) continue;
        add.replaceAllUsesWith(score); add.erase(); ++folded;
      }
      if (folded) {
        branch->setAttr("frisk.mask_add_bypass", UnitAttr::get(&getContext()));
        ++groups; additions += folded;
      }
    });
    SmallVector<Operation *> sources;
    function.walk([&](Operation *op) {
      if (isa<scf::IfOp, arith::SelectOp>(op) && !op->hasAttr("frisk.mask_add_bypass"))
        sources.push_back(op);
    });
    for (Operation *source : sources) {
      Candidate candidate{source};
      if (candidate.match()) { additions += candidate.rewrite(); ++groups; }
    }
    llvm::errs() << "MaskAddBypassThread: " << function.getSymName() << ": " << groups
                 << " groups, " << additions << " additions bypassed (approx)\n";
  }
};
} // namespace
std::unique_ptr<Pass> createThreadMaskAddBypassPass() {
  return std::make_unique<ThreadMaskAddBypassPass>();
}
} // namespace mlir::frisk
