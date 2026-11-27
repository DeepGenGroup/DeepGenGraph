#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "ThreadScheduleUtils.h"

namespace mlir::frisk {
namespace {
struct LowerFriskSchedBarrierPass
    : PassWrapper<LowerFriskSchedBarrierPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFriskSchedBarrierPass)
  explicit LowerFriskSchedBarrierPass(Target target = Target::ROCm)
      : target(target) {}
  StringRef getArgument() const final { return "frisk-lower-sched-barrier"; }
  StringRef getDescription() const final {
    return "Lower Frisk scheduling boundaries to AMDGPU scheduler intrinsics";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<FriskDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    SmallVector<SchedBarrierOp> barriers;
    getOperation().walk([&](SchedBarrierOp op) { barriers.push_back(op); });
    if (barriers.empty()) return;
    if (target != Target::ROCm) {
      barriers.front().emitOpError("lowering requires the ROCm target");
      return signalPassFailure();
    }
    if (failed(thread_schedule::checkGuard(getOperation())))
      return signalPassFailure();

    OpBuilder builder(&getContext());
    auto function = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(
        thread_schedule::Barrier);
    if (!function) {
      builder.setInsertionPointToStart(getOperation().getBody());
      auto type = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(&getContext()), {builder.getI32Type()}, false);
      function = builder.create<LLVM::LLVMFuncOp>(barriers.front().getLoc(),
                                                 thread_schedule::Barrier, type);
      function.setMemoryEffectsAttr(thread_schedule::guardEffects(&getContext()));
    }
    for (SchedBarrierOp op : barriers) {
      builder.setInsertionPoint(op);
      auto mask = builder.create<LLVM::ConstantOp>(
          op.getLoc(), builder.getI32Type(), op.getMaskAttr());
      auto call = builder.create<LLVM::CallOp>(op.getLoc(), function,
                                              ValueRange{mask});
      // Keep origin markers for downstream scheduling/diagnostics, but not
      // the inherent mask attribute (now the intrinsic's immediate argument).
      for (NamedAttribute attr : op->getDiscardableAttrs())
        call->setAttr(attr.getName(), attr.getValue());
      // Retain the compiler side effect through O3 without clobbering LDS.
      // Scheduled MMA anchors use the same inaccessible-memory ordering domain.
      call.setMemoryEffectsAttr(thread_schedule::guardEffects(&getContext()));
      op.erase();
    }
  }
  Target target;
};
} // namespace

std::unique_ptr<Pass> createLowerFriskSchedBarrierPass(Target target) {
  return std::make_unique<LowerFriskSchedBarrierPass>(target);
}
} // namespace mlir::frisk
