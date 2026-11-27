#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskInterfaces.cpp.inc"

namespace mlir::frisk {
namespace {
bool registerValues(Operation *op) {
  auto reg = [](Type type) {
    return isa<VectorType, FragmentType>(type) || type.isIntOrIndexOrFloat();
  };
  return llvm::all_of(op->getOperandTypes(), reg) &&
         llvm::all_of(op->getResultTypes(), reg);
}
} // namespace

ExecutionSemantics inferExecutionSemantics(Operation *op) {
  using S = ExecutionScope;
  using K = ScheduleKind;
  if (isa<SchedBarrierOp>(op))
    return {S::Thread, K::SchedBarrier, false, false, true};
  if (isa<WarpMmaRROp, GemmFragOp>(op))
    return {S::Warp, K::MMA, true, true, true};
  if (isa<WaveAnyOp>(op))
    return {S::Warp, K::Compute, true, true, true};
  if (isa<GemmOp>(op))
    return {S::Workgroup, K::MMA, registerValues(op), true, true};
  if (auto reduce = dyn_cast<ReduceFragOp>(op)) {
    bool lanes = reduce.getLaneExtent() > 1;
    return {lanes ? S::Warp : S::Thread, K::Compute, true, lanes, lanes};
  }
  if (isa<ReduceOp>(op))
    return {S::Workgroup, K::Compute, registerValues(op), true, true};
  if (isa<ConvertLayoutFragOp, ConvertLayoutOp>(op))
    // Even a shuffle plan is collective. Scratch and barriers remain explicit
    // effects until lowering has proved and emitted the selected implementation.
    return {S::Workgroup, K::Layout, false, true, true};
  if (isa<AsyncCopyGmToShmOp>(op))
    return {S::Workgroup, K::AsyncIssue, false, true, true};
  if (isa<WaitBufferReadyOp, GemmWaitOp>(op))
    return {S::Workgroup, K::Wait, false, true, true};
  if (isa<SyncThreadsInBlockOp>(op))
    return {S::Workgroup, K::Barrier, false, true, true};
  if (auto read = dyn_cast<ReadMatrixFromShmOp>(op))
    return {read.isBlockForm() ? S::Workgroup : S::Warp,
            K::Load, false, true, true};
  if (isa<LoadFragOp, LDSLoadOp>(op))
    return {S::Thread, K::Load, false, false, true};
  if (isa<StoreFragOp>(op))
    return {S::Thread, K::Store, false, false, true};
  if (isa<CopyOp, CopyToRegOp>(op))
    return {S::Workgroup, K::Copy, false, false, true};
  if (isa<FillOp>(op))
    return {S::Workgroup, K::Store, false, false, true};
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, Exp2Op, CastOp, ZeroOp>(op)) {
    bool registers = registerValues(op);
    return {S::Workgroup, K::Compute, registers, false, !registers};
  }
  if (isa<ExtractFragOp, InsertFragOp, PackFragOp, UnpackFragOp,
          ElementwiseFragOp, BroadcastFragOp, ConvertPackOp>(op))
    return {S::Thread, K::Compute, true, false, false};
  if (isa<AsyncTaskCompletedOp, BufferViewOp>(op))
    return {S::Workgroup, K::Compute, false, false, false};
  if (op->getNumRegions())
    return {S::Unknown, K::Region, false, false, true};
  return {};
}

ExecutionSemantics getExecutionSemantics(Operation *op) {
  if (!op) return {};
  if (auto interface = dyn_cast<ExecutionOpInterface>(op))
    return interface.getExecutionSemantics();
  return {};
}

bool isOrdinaryRegisterComputation(Operation *op) {
  if (!op || op->getNumRegions() || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (auto interface = dyn_cast<ExecutionOpInterface>(op)) {
    auto semantics = interface.getExecutionSemantics();
    if (semantics.collective || semantics.schedulingBoundary) return false;
  }
  return isMemoryEffectFree(op) && isSpeculatable(op);
}

void getComputedTileReadEffects(Operation *op,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : op->getOpOperands())
    if (isa<BaseMemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
}

#define COMPUTED_READ_EFFECTS(OP) \
  void OP::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) { \
    getComputedTileReadEffects(getOperation(), effects); \
  }
COMPUTED_READ_EFFECTS(AddOp)
COMPUTED_READ_EFFECTS(SubOp)
COMPUTED_READ_EFFECTS(MulOp)
COMPUTED_READ_EFFECTS(DivOp)
COMPUTED_READ_EFFECTS(MaxOp)
COMPUTED_READ_EFFECTS(Exp2Op)
COMPUTED_READ_EFFECTS(CastOp)
#undef COMPUTED_READ_EFFECTS
} // namespace mlir::frisk
