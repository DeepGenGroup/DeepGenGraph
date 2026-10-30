#ifndef DEEPGENGRAPH_CONVERSION_PREFETCHPAIREDLDS_H
#define DEEPGENGRAPH_CONVERSION_PREFETCHPAIREDLDS_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
bool parsePairedLDSOption(llvm::StringRef, unsigned &, bool &consumed);
/// Pair ordinary b64 LDS operands of register MMAC B, then hoist complete
/// pairs within immutable single-block regions. Zero disables the pass.
class PrefetchPairedLDSPass : public llvm::OptionalPassInfoMixin<PrefetchPairedLDSPass> {
  unsigned distance;
public:
  explicit PrefetchPairedLDSPass(unsigned distance = 0) : distance(distance) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
/// Final export pass, after K DMA/layout changes. Enabled runs first perform
/// local CSE/address simplification so repeated MMAC B reads can be paired.
bool prefetchPairedLDS(llvm::Module &, unsigned distance = 0);
}
#endif
