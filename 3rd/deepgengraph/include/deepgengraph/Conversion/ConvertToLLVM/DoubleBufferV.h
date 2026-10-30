#ifndef DEEPGENGRAPH_CONVERSION_DOUBLEBUFFERV_H
#define DEEPGENGRAPH_CONVERSION_DOUBLEBUFFERV_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
bool parseVDoubleBufferOption(llvm::StringRef argument, bool &enabled, bool &consumed);
/// Late, opt-in LDS reuse pass for an unrolled matrix-read/DMA stream. Proves
/// byte footprints and ownership barriers before changing any instruction.
/// Does not assume SSA names, tile dimensions, packet count or PV chain order.
class DoubleBufferVPass : public llvm::OptionalPassInfoMixin<DoubleBufferVPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};
void doubleBufferV(llvm::Module &, bool enabled = false);
} // namespace mlir::frisk
#endif
