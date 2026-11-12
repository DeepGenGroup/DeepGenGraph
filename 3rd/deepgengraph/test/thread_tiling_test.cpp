#include "deepgengraph/Conversion/VectorCanonicalization.h"
#include "deepgengraph/Common.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace {
struct TestExp2ToROCDLPass
    : mlir::PassWrapper<TestExp2ToROCDLPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestExp2ToROCDLPass)
  TestExp2ToROCDLPass() = default;
  TestExp2ToROCDLPass(const TestExp2ToROCDLPass &other) : PassWrapper(other) {}
  Option<bool> nativeExp2{*this, "native-exp2", llvm::cl::init(false)};
  llvm::StringRef getArgument() const final { return "test-exp2-to-rocdl"; }
  llvm::StringRef getDescription() const final { return "Exercise native and OCML exp2 lowering"; }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    mlir::registerAllDialects(registry);
  }
  void runOnOperation() final {
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::frisk::createGPUToROCDLOrNVVMPass(
        mlir::frisk::Target::ROCm, 64, nativeExp2));
    if (failed(pm.run(getOperation()))) signalPassFailure();
  }
};

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
    mlir::PassManager early(&getContext());
    early.addPass(mlir::frisk::createLowerFriskAsyncCopyPass());
    early.addPass(mlir::frisk::createLowerGPUShuffleToROCDLPass());
    early.addPass(mlir::frisk::createVectorSafeCanonicalizerPass());
    early.addPass(mlir::createCSEPass());
    early.addPass(mlir::frisk::createHoistWaveInvariantsPass());
    if (failed(early.run(module))) return signalPassFailure();
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
  mlir::registerPass([] { return mlir::frisk::createFriskOpFusePass(); });
  mlir::registerPass([] { return mlir::frisk::createReorderWorkgroupsPass(); });
  mlir::registerPass([] { return mlir::frisk::createOptimizeTensorLoopsPass(); });
  mlir::PassRegistration<TestShmReusePass>();
  mlir::PassRegistration<TestAsyncToLLVMPass>();
  mlir::PassRegistration<TestExp2ToROCDLPass>();
  mlir::registerPass([] { return mlir::frisk::createVectorSafeCanonicalizerPass(); });
  mlir::registerPass([] { return mlir::createCSEPass(); });
  mlir::registerPass([] { return mlir::frisk::createThreadLevelIRLegalizePass(); });
  mlir::registerPass([] { return mlir::frisk::createApproximateThreadMathPass(); });
  mlir::registerPass([] { return mlir::frisk::createThreadMaskAddBypassPass(); });
  mlir::registerPass([] { return mlir::frisk::createIRDeepOptimizePass(); });
  mlir::registerPass([] { return mlir::frisk::createLowerFriskAsyncCopyPass(); });
  mlir::registerPass([] { return mlir::frisk::createLowerGPUShuffleToROCDLPass(); });
  mlir::registerPass([] { return mlir::pipeline::createPipelineSchedulePass(); });
  mlir::registerPass([] { return mlir::pipeline::createAutoPipelinePass(); });
  mlir::registerPass([] { return mlir::pipeline::createSimplifyLoopControlPass(); });
  mlir::registerPass([] { return mlir::frisk::createCoalesceWaitsAndBarriersPass(); });
  mlir::registerPass([] { return mlir::frisk::createSimplifyBoundedAffinePass(); });
  mlir::registerPass([] { return mlir::frisk::createHoistWaveInvariantsPass(); });
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
  mlir::registerPass([] { return mlir::frisk::createRetainInvariantFragmentsPass(); });
  mlir::registerPass([] { return mlir::frisk::createFuseFragmentAccumulatorPass(); });
  mlir::registerPass([] { return mlir::frisk::createPackSharedMemoryPass(); });
  mlir::registerPass([] { return mlir::frisk::createConvertFriskBaseToFragmentIRPass(); });
  mlir::registerPass([] { return mlir::frisk::createConvertFragmentToThreadIRPass(); });
  mlir::registerPass([] { return mlir::frisk::createPromoteGlobalToSharedAsyncPass(); });
  mlir::registerPass([] { return mlir::frisk::createPromoteGemmMatrixLoadsPass(); });
  mlir::registerPass([] { return mlir::frisk::createLowerFriskAsyncToSyncPass(); });
  mlir::registerPass([] { return mlir::frisk::createScheduleAsyncCopiesPass(); });
  mlir::registerPass([] { return mlir::frisk::createAnalyzeShmLivenessPass(); });
  mlir::registerPass([] { return mlir::frisk::createSinkWaitBufferReadyPass(); });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Thread tiling tests\n", registry));
}
