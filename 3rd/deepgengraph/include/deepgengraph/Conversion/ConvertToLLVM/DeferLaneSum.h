#ifndef DEEPGENGRAPH_CONVERSION_DEFERLANESUM_H
#define DEEPGENGRAPH_CONVERSION_DEFERLANESUM_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
struct DeferLaneSumOptions {
  bool approximate = false;
  // 0: require an explicit wavefrontsize target feature. Otherwise the caller
  // promises this launch ABI; conflicts with target features are rejected.
  unsigned waveSize = 0;
  bool valid() const { return waveSize == 0 || waveSize == 32 || waveSize == 64; }
};
bool parseDeferLaneSumOption(llvm::StringRef, llvm::StringRef &mode,
                            DeferLaneSumOptions &, bool &consumed);

/// Defer fixed XOR all-reduces in zero-initialized additive recurrences.
/// The recurrence analysis is independent of kernel names and tensor layouts;
/// the initial communication adapter supports AMDGPU ds.bpermute, f32 and i32.
class DeferLaneSumPass : public llvm::OptionalPassInfoMixin<DeferLaneSumPass> {
  DeferLaneSumOptions options;
public:
  explicit DeferLaneSumPass(DeferLaneSumOptions options = {}) : options(options) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
void deferLaneSums(llvm::Module &, DeferLaneSumOptions options = {});
} // namespace mlir::frisk
#endif
