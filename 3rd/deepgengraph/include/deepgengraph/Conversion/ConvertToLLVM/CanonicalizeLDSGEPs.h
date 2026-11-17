#ifndef DEEPGENGRAPH_CANONICALIZE_LDS_GEPS_H
#define DEEPGENGRAPH_CANONICALIZE_LDS_GEPS_H
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
// Legacy standalone .ll experiments only. The production pipeline performs
// these proofs with createCanonicalizeThreadLDSAddressesPass in ThreadIR.
// Expose provably constant LDS address differences to AMDGPU read2 selection.
// Memory order and widths are retained; alignment is raised only after proof.
// Applies to nonvolatile, nonatomic loads from any address-space-3 buffer,
// independently of Q/K/V roles, element types or swizzle selection.
class CanonicalizeLDSGEPsPass
    : public llvm::OptionalPassInfoMixin<CanonicalizeLDSGEPsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
}
#endif
