#ifndef DEEPGENGRAPH_CONVERSION_SOFTMAXCVTPACK_H
#define DEEPGENGRAPH_CONVERSION_SOFTMAXCVTPACK_H
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::frisk {
// RTZ changes rounding relative to fptrunc: require explicit opt-in.
bool parseSoftmaxCvtPackOption(llvm::StringRef arg, bool &enabled, bool &consumed);

// After ThreadIR scheduling and LazySoftmaxBase, before firstLowering.
// nativeExp2 is the pipeline's exp2 selection, not an additional accuracy opt-in.
// The pass preserves the historical restriction to native-exp2 softmax paths.
std::unique_ptr<Pass> createThreadSoftmaxCvtPackPass(bool enabled = false,
                                                 bool nativeExp2 = false);
} // namespace mlir::frisk
#endif
