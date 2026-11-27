#include "deepgengraph/Pipeline/FragmentScheduling.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Conversion/VectorCanonicalization.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::frisk {
namespace {
struct FragmentSchedulingPipelineOptions
    : PassPipelineOptions<FragmentSchedulingPipelineOptions> {
  Option<bool> heuristicScheduling{
      *this, "heuristic-scheduling", llvm::cl::init(true),
      llvm::cl::desc("Enable fragment fusion, prefetch and DMA/wait scheduling; disable before CP-SAT export")};
  Option<bool> simplifyLoopControl{
      *this, "simplify-loop-control", llvm::cl::init(true),
      llvm::cl::desc("Specialize loop control before invariant retention")};
  Option<unsigned> retainFragmentWords{
      *this, "retain-fragment-words", llvm::cl::init(0),
      llvm::cl::desc("Retained 32-bit words per thread; 0 disables retention")};
  Option<unsigned> ldsPrefetchDistance{*this, "lds-prefetch-distance", llvm::cl::init(0)};
  Option<unsigned> ldsReductionPrefetchDistance{*this, "lds-reduction-prefetch-distance", llvm::cl::init(0)};
  Option<bool> bypassMaskAdds{*this, "bypass-mask-adds", llvm::cl::init(false),
      llvm::cl::desc("Permit approximate +0 mask-add bypass before fragment scheduling")};
  Option<bool> deferLaneSum{*this, "defer-lane-sum", llvm::cl::init(false),
      llvm::cl::desc("Permit approximate f32 lane-sum deferral before scheduling")};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(0)};
  Option<float> lazySoftmaxBaseThreshold{*this, "lazy-softmax-base-threshold", llvm::cl::init(0)};
};
} // namespace

void buildFragmentSchedulingPipeline(
    OpPassManager &pm, const FragmentSchedulingOptions &options) {
  // Specialize while tile coordinates and fragment SSA are available. Keep
  // normalization/CSE at their existing scheduling boundary, independently of
  // this option: early CSE can share broadcast carriers between operand tiles,
  // changing the retention cut and the subsequent shared-memory schedule.
  if (options.bypassMaskAdds)
    pm.addNestedPass<func::FuncOp>(createFragmentMaskAddBypassPass());
  if (options.deferLaneSum)
    pm.addNestedPass<func::FuncOp>(createFragmentDeferLaneSumPass(true, options.waveSize));
  if (options.heuristicScheduling && options.lazySoftmaxBaseThreshold != 0)
    pm.addNestedPass<func::FuncOp>(createFragmentLazySoftmaxBasePass(options.lazySoftmaxBaseThreshold));
  if (options.heuristicScheduling)
    pm.addNestedPass<func::FuncOp>(createFriskFragIRReorderPass());
  pm.addPass(createVectorSafeCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (options.simplifyLoopControl) {
    pm.addNestedPass<func::FuncOp>(pipeline::createSimplifyLoopControlPass());
    pm.addPass(createCSEPass());
  }
  // Retention changes the work graph, so it also precedes CP-SAT export.
  if (options.retainFragmentWords)
    pm.addNestedPass<func::FuncOp>(
        createRetainInvariantFragmentsPass(options.retainFragmentWords));
  if (options.heuristicScheduling &&
      (options.ldsPrefetchDistance || options.ldsReductionPrefetchDistance))
    pm.addNestedPass<func::FuncOp>(createPrefetchFragmentLDSPass(
        options.ldsPrefetchDistance, options.ldsReductionPrefetchDistance));
  if (options.heuristicScheduling)
    pm.addNestedPass<func::FuncOp>(createScheduleAsyncCopiesPass());
}

void registerFragmentSchedulingPipeline() {
  static PassPipelineRegistration<FragmentSchedulingPipelineOptions> registration(
      "frisk-fragment-scheduling",
      "Schedule fragment loops and DMA around loop specialization and invariant retention",
      [](OpPassManager &pm, const FragmentSchedulingPipelineOptions &options) {
        buildFragmentSchedulingPipeline(pm, {options.heuristicScheduling,
                                             options.simplifyLoopControl,
                                             options.retainFragmentWords,
                                             options.ldsPrefetchDistance,
                                             options.ldsReductionPrefetchDistance,
                                             options.bypassMaskAdds,
                                             options.deferLaneSum,
                                             options.waveSize,
                                             options.lazySoftmaxBaseThreshold});
      });
  (void)registration;
}

} // namespace mlir::frisk
