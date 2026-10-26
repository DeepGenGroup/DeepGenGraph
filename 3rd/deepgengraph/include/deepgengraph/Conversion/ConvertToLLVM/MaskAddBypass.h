#ifndef DEEPGENGRAPH_CONVERSION_MASKADDBYPASS_H
#define DEEPGENGRAPH_CONVERSION_MASKADDBYPASS_H

#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
/// Sink a mask addition into the nonzero arm of an existing diamond. Legal
/// mode uses LLVM's FP simplifier; approximate mode additionally authorizes
/// ignoring signed zero for these additions only. Never changes the CFG,
/// moves memory/calls, or transforms strictfp/constrained FP operations.
class MaskAddBypassPass : public llvm::OptionalPassInfoMixin<MaskAddBypassPass> {
  bool allowSignedZero;
public:
  explicit MaskAddBypassPass(bool allowSignedZero = false)
      : allowSignedZero(allowSignedZero) {}
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &analyses);
};
} // namespace mlir::frisk
#endif
