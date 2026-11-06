#ifndef DEEPGENGRAPH_CONVERSION_SOFTMAXCVTPACK_H
#define DEEPGENGRAPH_CONVERSION_SOFTMAXCVTPACK_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
// RTZ changes rounding relative to fptrunc: require explicit opt-in.
bool parseSoftmaxCvtPackOption(llvm::StringRef arg, bool &enabled, bool &consumed);

// Fuse only compiler-marked four-element softmax conversions feeding MMAC A.
// Run after scheduling and LazySoftmaxBase; no uniform-control-flow requirement.
class SoftmaxCvtPackPass : public llvm::OptionalPassInfoMixin<SoftmaxCvtPackPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};

void softmaxCvtPack(llvm::Module &, bool enabled = false);
} // namespace mlir::frisk
#endif
