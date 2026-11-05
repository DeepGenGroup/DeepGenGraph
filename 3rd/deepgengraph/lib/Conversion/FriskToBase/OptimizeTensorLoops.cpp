// ===========================================================================
// Pass: OptimizeTensorLoops（optimize-tensor-loops）   优化汇总.py 第 1 项，tensor 阶段
//
// 作用：
//   在 tensor 语义的 DeviceKernel IR 上，把“外层循环的多次迭代”jam 到同一个内层
//   reduce 循环里，再把算出来完全相同的 carrier（iter_arg）合并，让不同迭代共享同
//   一份不变计算：结构相同且输入相同的中间结果只算一次。
//
// 流程（runOnOperation）：
//   1. canJam 过滤候选外层循环：常量 trip count、无 iter_args、内层有实际 tensor 计算、
//      读写指针可证明不重叠。
//   2. chooseFactor 在模块的完整克隆上试 unroll-and-jam，选 carrier 占用不超预算的
//      最大因子。试验失败不会污染原 IR。
//   3. 真正做 loopUnrollJamByFactor。
//   4. cleanup 不动点：canonicalize + CSE + mergeCarriers，互相暴露新机会。
//   5. 删除无用 ZeroOp，做不含 load 的 LICM，最后再跑一次 CSE。
//
// 关键选项：
//   maxJamFactor     jam 因子上限，只尝试能整除 trip count 的因子。
//   maxCarrierWords  每个 block 允许的嵌套 carrier 总字数（32-bit word），0 表示不限。
//                    本 case 由 test_src 的 carrier=34816 传入。
//
// 注意：这里说的“合并”依赖结构等价，不做浮点重结合，所以不改变数值。
// ===========================================================================
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

// 返回指针的“根”，穿透 block_pointer_of / pointer_of(tensor_from) 以及 DeviceKernel 的
// 参数映射（region 参数 = grid 坐标 + 外部 args，所以要减去 grid 大小 offset）。
// canJam 用它判断读写是否指向同一块内存。
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

// 判断 op 是否可视为“纯值计算”。canJam、ValueEquivalence 和 mergeCarriers 都靠它
// 确认挪动或合并没有副作用。
// Computed tensors have value semantics, even though the legacy ZeroOp lacks
// a Pure trait to keep its later destination-style buffer initialization local.
static bool tensorValueOp(Operation *op) {
  return isa<dg::ZeroOp>(op) || isMemoryEffectFree(op);
}

// 判断外层循环能否安全且有收益地做 unroll-and-jam。
// 遍历过程中维护：
//   nested  循环内存在嵌套循环（否则没有可 jam 的对象）。
//   useful  内层循环确实产生 tensor 结果（收益判定，不看 kernel 名和具体算子）。
//   valid   所有安全条件都满足，任一违反就置 false。
//   reads / writes  所有 block_load / block_store 的指针根，遍历后做两两不重叠检查。
static bool canJam(Loop outer, uint64_t maxFactor) {
  auto count = affine::getConstantTripCount(outer);
  if (!count || *count < 2 || maxFactor < 2 || outer.getNumResults())
    return false;
  bool nested = false, useful = false, valid = true;
  SmallVector<Value> reads, writes;
  outer.walk([&](Operation *op) {
    if (auto loop = dyn_cast<Loop>(op)) {
      nested |= loop != outer;
      // 内层循环边界不能依赖外层循环内定义的值，否则 jam 后边界不再相同；
      // 内层循环结果只能是 tensor，否则 carrier 的值语义不成立。
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
  // 写目标必须是“新鲜”的输出（EmptyPointer，或带 frisk.fresh_output 的 pointer_of）。
  // 读必须来自 pointer_of 输入，且不能和写是同一块内存，否则 jam 改变读写次序后结果会变。
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
// 证明两个 SSA 值结构等价，mergeCarriers 用它做归纳证明。
//   scope   block_load 只在这个循环内部才允许判等，保证两次读发生在同一个内存状态下。
//   equal   假设等价表，预置“两个 iter_arg 相等”，比较时也把 region 参数对记录进来。
//   budget  递归比较次数上限，防止表达式过大时证明过慢，耗尽后保守返回不等。
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

// 合并同一循环内数值恒等的 carrier（iter_arg）。这是 jam 之后真正省计算的一步。
// 归纳证明：初值相等 + 假设两个 carrier 相等时 yield 也相等 => 每次迭代都相等。
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
  // representatives[i]：第 i 个 carrier 合并到哪个 carrier 上，等于 i 表示它自己是代表。
  // 只和已确认的代表比较（下面 representatives[j] != j 时跳过），避免链式合并。
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
  // affine.for 的 iter_args 不能原地删减，所以用保留下来的 carrier 新建一个循环，把循环体
  // 克隆过去。此时被合并的 carrier 已无使用者。
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
  // live：外层循环已经占用的 carrier 字数。返回沿循环嵌套路径的峰值。
  // 遇到非静态或非 int/float 元素的 tensor 时返回 UINT64_MAX，让预算判定失败。
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

  // 不动点循环：canonicalize + CSE 暴露相同表达式，mergeCarriers 合并后又暴露新的 CSE。
  // trial 克隆和原模块共用这段逻辑：原模块走 runPipeline，克隆是独立模块所以走 passes.run。
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

  // 选 jam 因子：从大到小试能整除 trip count 的因子，用克隆模块实测 carrier 峰值。
  // 返回 1 表示不 jam。
  uint64_t chooseFactor(Loop loop) {
    auto count = *affine::getConstantTripCount(loop);
    // Trial on an isolated copy: sharing is proven by recurrence CSE before
    // charging carriers. A failed profitability check cannot mutate input IR.
    // 克隆没有和原模块的映射，所以靠“遍历序号”定位克隆里的同一个循环。
    // selected 是原模块中这个循环的序号，要求两次 walk 的顺序一致。
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
    // candidates：通过 canJam 的外层循环，用后序收集。
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
    // 自定义 LICM：只外提不含 load 的纯计算。
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
    // 这个局部变量和成员函数 cleanup 同名，这里是单纯的 CSE 管线，只跑一遍。
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
