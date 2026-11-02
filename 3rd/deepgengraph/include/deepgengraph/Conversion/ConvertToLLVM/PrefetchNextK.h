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
/// Final export scheduling, after V buffering, K layout and softmax scheduling.
void prefetchNextK(llvm::Module &, bool enabled = false);
}
#endif
