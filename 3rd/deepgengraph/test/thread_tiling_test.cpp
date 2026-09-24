#include "deepgengraph/Common.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace {
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
                  mlir::deepgengraph::DeepgengraphDialect>();
  mlir::PassRegistration<TestShmReusePass>();
  mlir::registerPass([] {
    return mlir::frisk::createConvertFriskBaseToThreadLevelIRPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createFinalizeThreadTilingPass();
  });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Thread tiling tests\n", registry));
}
