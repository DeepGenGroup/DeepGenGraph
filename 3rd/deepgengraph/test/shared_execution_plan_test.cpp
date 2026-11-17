#include "deepgengraph/Analysis/SharedMemoryPlan.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::frisk;
static void check(bool value, const char *message) {
  if (!value) { llvm::errs() << "FAIL: " << message << '\n'; std::exit(1); }
}
static std::string text(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);
  op->print(os);
  return result;
}
int main(int argc, char **argv) {
  check(argc == 2 || argc == 3, "provide an execution fixture, optionally cross-iteration");
  DialectRegistry registry;
  registry.insert<FriskDialect, func::FuncDialect, arith::ArithDialect,
                  affine::AffineDialect, memref::MemRefDialect, gpu::GPUDialect,
                  scf::SCFDialect, vector::VectorDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  check(bool(module), "parse execution fixture");
  auto function = *module->getOps<func::FuncOp>().begin();
  std::string original = text(function);
  if (argc == 3) {
    auto baseline = planSharedMemoryExecution(function, true);
    auto planned = planSharedMemoryExecution(function, true, true);
    check(text(function) == original, "cross-iteration planning is read-only");
    check(planned.execution->crossIterations.size() == 1, "one cross-iteration plan");
    auto &execution = *planned.execution;
    auto cross = execution.crossIterations.front();
    auto &pipeline = execution.pipelines.front();
    unsigned last = pipeline.stages.back().bufferIndex;
    check(!baseline.execution->lifetimes.interferes(cross.bufferIndex, last) &&
          execution.lifetimes.interferes(cross.bufferIndex, last),
          "circular suffix interferes with final readers");
    check(!execution.lifetimes.interferes(cross.bufferIndex, pipeline.stages[0].bufferIndex),
          "circular middle remains reusable");
    check(execution.storage.pooledBytes == baseline.execution->storage.pooledBytes,
          "cross-iteration reuse preserves the LDS budget");
    OpBuilder b(cross.issueAfter);
    auto extra = b.create<gpu::BarrierOp>(cross.issueAfter.getLoc());
    std::string mutated = text(function);
    {
      ScopedDiagnosticHandler expected(&context, [](Diagnostic &) { return success(); });
      check(failed(planned.materializeExecution()), "stale cross-iteration plan is rejected");
    }
    check(text(function) == mutated, "stale plan leaves IR untouched");
    extra.erase();
    check(succeeded(planned.materializeExecution()) && succeeded(verify(*module)) &&
          succeeded(validateAsyncTaskTokens(function)), "cross-iteration SSA and task validation");
    auto materialized = text(function);
    auto repeated = planSharedMemoryExecution(function, true, true);
    check(repeated.execution->crossIterations.empty() && succeeded(repeated.materializeExecution()) &&
          text(function) == materialized, "cross-iteration materialization is idempotent");
    llvm::outs() << "PASS circular interference, budget, read-only planning, stale-plan rejection, task SSA and idempotence\n";
    return 0;
  }
  auto baseline = planSharedMemoryExecution(function, false);
  auto planned = planSharedMemoryExecution(function, true);
  check(text(function) == original, "planning is read-only");
  check(planned.execution && planned.execution->pipelines.size() == 1, "one planned pipeline");
  auto &execution = *planned.execution;
  auto &pipeline = execution.pipelines.front();
  check(pipeline.stages.size() == 3 && execution.storage.pooledBytes == 2048,
        "three stages use the original two-slot pool budget");
  unsigned a = pipeline.stages[0].bufferIndex, b = pipeline.stages[1].bufferIndex;
  check(!baseline.execution->lifetimes.interferes(a, b) && execution.lifetimes.interferes(a, b),
        "earlier issue extends the lifetime and forbids overlapping slots");
  auto issue = pipeline.stages[1].issue;
  auto oldWidth = issue->getAttr("frisk.dma_bytes");
  issue->setAttr("frisk.dma_bytes", IntegerAttr::get(IntegerType::get(&context, 64), 4));
  auto mutated = text(function);
  {
    ScopedDiagnosticHandler expected(&context, [](Diagnostic &) { return success(); });
    check(failed(planned.materializeExecution()), "stale copy contract is rejected");
  }
  check(text(function) == mutated, "invalidated plan does not partially materialize");
  issue->setAttr("frisk.dma_bytes", oldWidth);
  auto before = pipeline.stages[0].ready;
  OpBuilder builder(before);
  auto extra = builder.create<gpu::BarrierOp>(before.getLoc());
  mutated = text(function);
  {
    ScopedDiagnosticHandler expected(&context, [](Diagnostic &) { return success(); });
    check(failed(planned.materializeExecution()), "changed synchronization invalidates plan");
  }
  check(text(function) == mutated, "changed synchronization is left intact");
  extra.erase();
  check(text(function) == original, "restore original snapshot");
  check(succeeded(planned.materializeExecution()) && succeeded(verify(*module)),
        "materialization consumes the validated plan with valid SSA dominance");
  check(before->isBeforeInBlock(issue), "copy starts after previous readiness");
  check(issue->isBeforeInBlock(pipeline.stages[0].reads.front()), "copy overlaps previous reader");
  auto lowered = text(function);
  auto repeated = planSharedMemoryExecution(function, true);
  check(repeated.execution->pipelines.empty() && succeeded(repeated.materializeExecution()),
        "already materialized pipeline is not planned again");
  check(text(function) == lowered, "idempotent materialization");
  llvm::outs() << "PASS read-only plan, projected interference, stale-plan rejection, SSA and idempotence\n";
}
