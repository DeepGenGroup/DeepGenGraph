#ifndef DEEPGENGRAPH_SCHEDULE_SOFTMAX_FRAGMENTS_H
#define DEEPGENGRAPH_SCHEDULE_SOFTMAX_FRAGMENTS_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
// Off by default. Fragment preserves conversion order; PV sinks conversion to
// the first consumer. Neither mode changes arithmetic or memory/sync order.
enum class SoftmaxFragmentMode { Off, Fragment, PV };
bool parseSoftmaxFragmentOption(llvm::StringRef, SoftmaxFragmentMode &, bool &);
class ScheduleSoftmaxFragmentsPass
    : public llvm::OptionalPassInfoMixin<ScheduleSoftmaxFragmentsPass> {
  SoftmaxFragmentMode mode;
public:
  explicit ScheduleSoftmaxFragmentsPass(SoftmaxFragmentMode mode) : mode(mode) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
void scheduleSoftmaxFragments(llvm::Module &, SoftmaxFragmentMode);
} // namespace mlir::frisk
#endif
