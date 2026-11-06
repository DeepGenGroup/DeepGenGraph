// ===========================================================================
// Pass: SinkWaitBufferReady（frisk-sink-wait-buffer-ready）   不单独列在 优化汇总.py 中
//
// 作用：在 block 级把 wait_buffer_ready 下沉到第一个依赖它的位置，让更多无关计算和在途 DMA 重叠。
// 由第 6 项 ConvertFriskBaseToFragmentIR 直接调用（sinkWaitBufferReady）。
// 第 11 项 ScheduleAsyncCopies 在 fragment 阶段做类似的事，这里是更早的一次。
//
// 合法性判定见 BufferDependencies::canCross。
// ===========================================================================
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "ScheduleAttrs.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
// LocalAliasAnalysis does not know Frisk logical views or allocations. Resolve
// these and selects explicitly; unknown origins retain MLIR's conservative
// answer. Different slices of the same allocation are deliberately MayAlias.
// 判断 wait 能否越过某个 op。
// mayAlias：穿过视图和 select 后比较两个 memref。同一 alloc 的不同切片也算可能别名。
// aliases：MLIR 的别名分析，上面的规则判断不了时才用。
class BufferDependencies {
public:
  explicit BufferDependencies(func::FuncOp kernel) : aliases(kernel) {}

  bool mayAlias(Value a, Value b, unsigned depth = 0) {
    if (a == b || depth > 32)
      return true;
    auto at = dyn_cast<MemRefType>(a.getType());
    auto bt = dyn_cast<MemRefType>(b.getType());
    if (!at || !bt)
      return true;
    if (Value source = viewSource(a))
      return mayAlias(source, b, depth + 1);
    if (Value source = viewSource(b))
      return mayAlias(a, source, depth + 1);
    if (auto select = a.getDefiningOp<arith::SelectOp>())
      return mayAlias(select.getTrueValue(), b, depth + 1) ||
             mayAlias(select.getFalseValue(), b, depth + 1);
    if (auto select = b.getDefiningOp<arith::SelectOp>())
      return mayAlias(a, select.getTrueValue(), depth + 1) ||
             mayAlias(a, select.getFalseValue(), depth + 1);
    // Strip memory_space_cast before using address spaces to disambiguate.
    if (at.getMemorySpace() != bt.getMemorySpace())
      return false;
    if (isAllocation(a) && isAllocation(b))
      return false;
    return !aliases.alias(a, b).isNo();
  }

  // wait 能否越过 op。不允许越过：控制流、barrier、其它 wait、其它 DMA、桥接、已有手动调度的 op。
  // 其它 op 不能读写 DMA 的目的地，也不能写 DMA 的源。
  bool canCross(Operation *op, AsyncCopyGmToShmOp issue) {
    // A wait is collective. Do not change its control-flow placement or cross
    // another collective/launch, even for disjoint tiles (backend policy).
    if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() ||
        isa<gpu::BarrierOp, SyncThreadsInBlockOp, WaitBufferReadyOp,
            AsyncCopyGmToShmOp, ToThreadTileOp, FromThreadTileOp>(op) ||
        hasSchedule(op))
      return false;
    // Thread-tile bridges above are declared Pure but can materialize memory
    // accesses during finalization. They are not mere descriptor construction.

    // These known ComputedTile operations only read their operands; their
    // legacy unqualified effects do not mean writes to those operands. Do not
    // infer this from the trait alone: it describes results, not all effects.
    // In particular this lets K readiness move past Q scaling, up to GEMM.
    if (isa<AddOp, SubOp, MaxOp, MulOp, DivOp, Exp2Op, CastOp, GemmOp, ZeroOp>(op)) {
      for (Value operand : op->getOperands())
        if (isa<BaseMemRefType>(operand.getType()) &&
            mayAlias(operand, issue.getDst()))
          return false;
      return true;
    }

    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return false;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    if (isa<AllocBufferOp>(op))
      return true; // Fresh owned storage; its legacy Allocate has no value.
    for (const auto &effect : instances) {
      Value value = effect.getValue();
      if (!value)
        return false;
      if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
        if (value.getDefiningOp() != op)
          return false;
        continue;
      }
      if (!isa<MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Free>(
              effect.getEffect()))
        return false;
      if (mayAlias(value, issue.getDst()) ||
          (!isa<MemoryEffects::Read>(effect.getEffect()) &&
           mayAlias(value, issue.getSrc())))
        return false;
    }
    return true;
  }

  // op 带手动或流水调度标注。
  static bool hasSchedule(Operation *op) { return hasAnyScheduleAttr(op); }

private:
  static bool isAllocation(Value value) {
    return isa_and_nonnull<AllocBufferOp, memref::AllocOp, memref::AllocaOp>(
        value.getDefiningOp());
  }

  static Value viewSource(Value value) {
    Operation *def = value.getDefiningOp();
    if (auto view = dyn_cast_or_null<BufferViewOp>(def))
      return view.getSource();
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def))
      return view.getViewSource();
    if (auto cast = dyn_cast_or_null<memref::CastOp>(def))
      return cast.getSource();
    if (auto copy = dyn_cast_or_null<CopyOp>(def))
      return copy.getDst();
    return {};
  }

  AliasAnalysis aliases;
};

class SinkWaitBufferReadyPass
    : public PassWrapper<SinkWaitBufferReadyPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkWaitBufferReadyPass)
  StringRef getArgument() const final { return "frisk-sink-wait-buffer-ready"; }
  StringRef getDescription() const final {
    return "Delay buffer readiness until the first dependent access";
  }
  void runOnOperation() final { sinkWaitBufferReady(getOperation()); }
};
} // namespace

// 对每个 wait：先向上找到同块内 token 相同的 issue；确认 issue 到 wait 之间的 op 都可越过；
// 然后向下移动，直到遇到不能越过的 op。
void mlir::frisk::sinkWaitBufferReady(func::FuncOp kernel) {
  BufferDependencies dependencies(kernel);
  SmallVector<WaitBufferReadyOp> waits;
  kernel.walk([&](WaitBufferReadyOp wait) { waits.push_back(wait); });
  for (auto wait : waits) {
    bool scheduled = false;
    for (Operation *parent = wait; parent; parent = parent->getParentOp())
      scheduled |= BufferDependencies::hasSchedule(parent);
    if (scheduled)
      continue;

    // Require the exact issue token in this block. Loop-carried tasks retain
    // their scheduled completion point; buffer aliases do not identify tasks.
    AsyncCopyGmToShmOp issue;
    for (Operation *prev = wait->getPrevNode(); prev; prev = prev->getPrevNode()) {
      if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(prev)) {
        if (copy.getToken() == wait.getToken() &&
            !BufferDependencies::hasSchedule(copy))
          issue = copy;
        break;
      }
      if (prev->getNumRegions() ||
          isa<gpu::BarrierOp, SyncThreadsInBlockOp, WaitBufferReadyOp>(prev))
        break;
    }
    if (!issue)
      continue;
    // Do not turn an already-invalid issue/consumer/wait sequence into a new
    // schedule; nor cross an intervening unknown effect or source mutation.
    bool valid = true;
    for (Operation *op = issue->getNextNode(); op != wait; op = op->getNextNode())
      valid &= dependencies.canCross(op, issue);
    if (!valid)
      continue;
    while (Operation *next = wait->getNextNode()) {
      if (!dependencies.canCross(next, issue))
        break;
      wait->moveAfter(next);
    }
  }
}

std::unique_ptr<Pass> mlir::frisk::createSinkWaitBufferReadyPass() {
  return std::make_unique<SinkWaitBufferReadyPass>();
}
