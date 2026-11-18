#ifndef DEEPGENGRAPH_CONVERSION_THREADLAZYSOFTMAXBASE_H
#define DEEPGENGRAPH_CONVERSION_THREADLAZYSOFTMAXBASE_H
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
namespace mlir::frisk {
bool parseLazySoftmaxBaseOption(llvm::StringRef arg, bool &enabled,
                                float &threshold, bool &consumed);
// Run after thread scheduling and wave-invariant hoisting, before CFG lowering.
// Explicit wave64 ABI; only non-causal, zero-initialized O/L recurrences.
std::unique_ptr<Pass> createThreadLazySoftmaxBasePass(float threshold = 8.0f,
                                                    unsigned waveSize = 64);
} // namespace mlir::frisk
#endif
