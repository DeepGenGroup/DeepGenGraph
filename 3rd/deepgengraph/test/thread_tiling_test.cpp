#include "deepgengraph/Common.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace {
struct TestAsyncToLLVMPass
    : mlir::PassWrapper<TestAsyncToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAsyncToLLVMPass)
  llvm::StringRef getArgument() const final { return "test-async-to-llvm"; }
  llvm::StringRef getDescription() const final { return "Exercise the complete GPU lowering"; }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    mlir::registerAllDialects(registry);
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (!mlir::frisk::firstLowering(module, &getContext()) ||
        !mlir::frisk::secondLowering(module, &getContext(), mlir::frisk::Target::ROCm))
      signalPassFailure();
  }
};
struct TestShmLoopSyncPass
    : mlir::PassWrapper<TestShmLoopSyncPass,
                       mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShmLoopSyncPass)
  llvm::StringRef getArgument() const final { return "test-shm-loop-sync"; }
  llvm::StringRef getDescription() const final {
    return "Protect shared-memory loop backedges for regression tests";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }
  void runOnOperation() override {
    mlir::frisk::synchronizeSharedMemoryLoopBackedges(getOperation());
  }
};

struct TestShmReusePass
    : mlir::PassWrapper<TestShmReusePass,
                       mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShmReusePass)
  llvm::StringRef getArgument() const final { return "test-shm-reuse"; }
  llvm::StringRef getDescription() const final {
    return "Inspect and apply shared-memory reuse for regression tests";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                    mlir::gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto analysis = mlir::frisk::analyzeShmLiveness(getOperation());
    mlir::frisk::dumpShmLiveness(analysis, llvm::errs());
    auto plan = mlir::frisk::planShmReuse(analysis);
    for (auto placement : plan.placements)
      llvm::errs() << "placement " << placement.bufferIndex << " @ "
                   << placement.offsetBytes << '\n';
    mlir::frisk::ShmReuseStats stats;
    mlir::frisk::reuseSharedMemory(getOperation(), &stats);
    llvm::errs() << "shm bytes " << stats.originalBytes << " -> "
                 << stats.pooledBytes << ", barriers " << stats.insertedBarriers
                 << '\n';
  }
};
} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect,
                  mlir::deepgengraph::triton::DeepgengraphTritonDialect,
                  mlir::deepgengraph::DeepgengraphDialect>();
  mlir::registerPass([] { return mlir::frisk::createOptimizeTensorLoopsPass(); });
  mlir::PassRegistration<TestShmReusePass>();
  mlir::PassRegistration<TestAsyncToLLVMPass>();
  mlir::registerPass([] { return mlir::frisk::createLowerFriskAsyncCopyPass(); });
  mlir::registerPass([] { return mlir::pipeline::createPipelineSchedulePass(); });
  mlir::PassRegistration<TestShmLoopSyncPass>();
  mlir::registerPass([] {
    return mlir::frisk::createNormalizeAttentionKLayoutPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createPlanWmmaLayoutConversionsPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createConvertFriskBaseToThreadLevelIRPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createFinalizeThreadTilingPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createFriskFragIRReorderPass();
  });
  mlir::registerPass([] { return mlir::frisk::createLowerFriskFragmentsPass(); });
  mlir::registerPass([] { return mlir::frisk::createPackSharedMemoryPass(); });
  mlir::registerPass([] { return mlir::frisk::createPromoteGlobalToSharedAsyncPass(); });
  mlir::registerPass([] { return mlir::frisk::createLowerFriskAsyncToSyncPass(); });
  mlir::registerPass([] { return mlir::frisk::createScheduleAsyncCopiesPass(); });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Thread tiling tests\n", registry));
}
