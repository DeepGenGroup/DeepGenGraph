#ifndef DEEPGENGRAPH_CANONICALIZE_LDS_GEPS_H
#define DEEPGENGRAPH_CANONICALIZE_LDS_GEPS_H
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
// Expose provably constant LDS address differences to AMDGPU read2 selection.
// Memory order and widths are retained; alignment is raised only after proof.
class CanonicalizeLDSGEPsPass
    : public llvm::OptionalPassInfoMixin<CanonicalizeLDSGEPsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
}
#endif
