#ifndef DEEPGENGRAPH_CONVERSION_PREFETCHNEXTK_H
#define DEEPGENGRAPH_CONVERSION_PREFETCHNEXTK_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
bool parseCrossBNK0Option(llvm::StringRef, bool &, bool &consumed);
/// Prime the first K DMA group and issue the next tile's group during final
/// PV, after proving LDS ownership. Anchor exact FP32 denominator additions
/// at their fragments to prevent the observed backend sinking/spill regression.
class PrefetchNextKPass : public llvm::OptionalPassInfoMixin<PrefetchNextKPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
/// Legacy standalone export scheduling. MyTest schedules the DMA through
/// SharedMemoryPlan and calls only anchorNextKSums at LLVM export.
void prefetchNextK(llvm::Module &, bool enabled = false);
/// Preserve denominator lifetimes for SharedMemoryPlan's early cross-iteration
/// schedule. Does not move DMA, select LDS storage or reconstruct the schedule.
void anchorNextKSums(llvm::Module &);
// Erase the barrier ending each iteration once it orders nothing the next
// iteration's opening barrier does not. Requires completed cross-iteration
// scheduling (SharedMemoryPlan or the legacy prefetchNextK transformation).
void elideBackedgeBarrier(llvm::Module &);
}
#endif
