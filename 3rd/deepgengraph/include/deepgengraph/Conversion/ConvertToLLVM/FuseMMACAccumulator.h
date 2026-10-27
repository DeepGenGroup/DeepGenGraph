#ifndef DEEPGENGRAPH_CONVERSION_FUSEMMACACCUMULATOR_H
#define DEEPGENGRAPH_CONVERSION_FUSEMMACACCUMULATOR_H

#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
/// Opt-in floating-point reassociation: seed a zero-initialized, single-use
/// register MMAC chain with the accumulator from its terminal fadd.
/// Does not approximate exponentials or division. Run SimplifyCFG then
/// InstCombine first to expose chains hidden by constant branches or vector
/// extraction/insertion wrappers. Revisit after thread-bound simplification.
class FuseMMACAccumulatorPass
    : public llvm::OptionalPassInfoMixin<FuseMMACAccumulatorPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &analyses);
};
} // namespace mlir::frisk

#endif
