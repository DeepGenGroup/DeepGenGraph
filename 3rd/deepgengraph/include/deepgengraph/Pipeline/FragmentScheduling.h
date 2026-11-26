#ifndef DEEPGENGRAPH_PIPELINE_FRAGMENTSCHEDULING_H
#define DEEPGENGRAPH_PIPELINE_FRAGMENTSCHEDULING_H

namespace mlir {
class OpPassManager;

namespace frisk {

struct FragmentSchedulingOptions {
  // Disable fusion, prefetch and DMA/wait scheduling before CP-SAT export.
  bool heuristicScheduling = true;
  bool simplifyLoopControl = true;
  // Retained 32-bit words per thread; 0 disables retention.
  unsigned retainFragmentWords = 0;
  unsigned ldsPrefetchDistance = 0;
  unsigned ldsReductionPrefetchDistance = 0;
};

// Module pipeline, after accumulator fusion and before fragment lowering (or
// CP-SAT export). Keep cleanup/specialization/retention between fragment loop
// scheduling and DMA motion: they change the reads used to place waits. Optional
// LDS read-ahead follows retention and precedes DMA motion/storage planning.
void buildFragmentSchedulingPipeline(
    OpPassManager &pm, const FragmentSchedulingOptions &options);
void registerFragmentSchedulingPipeline();

} // namespace frisk
} // namespace mlir

#endif
