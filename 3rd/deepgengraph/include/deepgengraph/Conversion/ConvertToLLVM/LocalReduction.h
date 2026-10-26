#ifndef DEEPGENGRAPH_CONVERSION_LOCALREDUCTION_H
#define DEEPGENGRAPH_CONVERSION_LOCALREDUCTION_H

#include "llvm/IR/PassManager.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
struct LocalReductionOptions {
  bool approximate = false;
  unsigned chains = 4;
  unsigned minInputs = 16;
  enum Operation : unsigned {
    Add = 1, Mul = 2, And = 4, Or = 8, Xor = 16, Min = 32, Max = 64,
    All = 127
  };
  unsigned operations = All;
  bool valid() const;
};

/// Shared CLI parser. Returns false only for malformed local-reduce options;
/// unrelated options set consumed=false. Mode defaults to off at call sites.
bool parseLocalReductionOption(llvm::StringRef argument, llvm::StringRef &mode,
                              LocalReductionOptions &options, bool &consumed);

/// Reassociate long same-block, single-use reduction spines into independent
/// partial accumulators. No target, kernel name, shape or layout assumptions.
class LocalReductionPass : public llvm::OptionalPassInfoMixin<LocalReductionPass> {
  LocalReductionOptions options;
public:
  explicit LocalReductionPass(LocalReductionOptions options = {}) : options(options) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};

void optimizeLocalReductions(llvm::Module &, LocalReductionOptions options = {});
} // namespace mlir::frisk
#endif
