#ifndef DEEPGENGRAPH_CONVERSION_INTERLEAVEPV_H
#define DEEPGENGRAPH_CONVERSION_INTERLEAVEPV_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace mlir::frisk {
/// 0 disables; 1 adds the same scheduling guards without interleaving;
/// 2/4 alternate that many independent accumulator chains. Default is off.
bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width,
                             bool &consumed);

/// Late LLVM scheduling pass. Matches register MMAC chains consuming the same
/// ordered matrix-read packets, not SSA names or a fixed attention tile size.
/// Requires all packet reads to precede the region. Keeps memory/sync operations
/// at their original MMAC ordinals, preserves each accumulator's SSA edges, and
/// rejects unknown effects, cross-chain dependencies and unsupported shapes.
class InterleavePVPass : public llvm::OptionalPassInfoMixin<InterleavePVPass> {
  unsigned width;
public:
  explicit InterleavePVPass(unsigned width = 0) : width(width) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};

/// Run after final MMA fusion and DMA export; does not insert reader handoffs,
/// hoist reads, alter math flags or change the workgroup/layout configuration.
void interleavePVAccumulators(llvm::Module &, unsigned width = 0);
} // namespace mlir::frisk
#endif
