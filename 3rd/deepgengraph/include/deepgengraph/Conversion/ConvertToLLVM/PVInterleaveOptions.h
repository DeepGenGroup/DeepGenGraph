#ifndef DEEPGENGRAPH_CONVERSION_PVINTERLEAVEOPTIONS_H
#define DEEPGENGRAPH_CONVERSION_PVINTERLEAVEOPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
/// MyTest Thread IR scheduling: 0 disables, 1 guards serial chains, 2/4 interleave.
bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width,
                             bool &consumed);
} // namespace mlir::frisk
#endif
