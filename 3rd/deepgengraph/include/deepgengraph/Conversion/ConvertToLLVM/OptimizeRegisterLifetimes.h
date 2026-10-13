#ifndef DEEPGENGRAPH_CONVERSION_OPTIMIZEREGISTERLIFETIMES_H
#define DEEPGENGRAPH_CONVERSION_OPTIMIZEREGISTERLIFETIMES_H

#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
/// AMDGPU register-pressure scheduling without floating-point reassociation.
/// Simplify bounded workitem expressions, retain a small budget of costly bases,
/// recompute remaining address DAGs near uses, and materialize narrow math
/// consumers early. Empty tied-register asm makes these boundaries survive
/// subsequent LICM and instruction combining; it emits no GPU instruction.
class OptimizeRegisterLifetimesPass
    : public llvm::PassInfoMixin<OptimizeRegisterLifetimesPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &analyses);
};
} // namespace mlir::frisk
#endif
