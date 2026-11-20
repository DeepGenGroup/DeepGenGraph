#ifndef DEEPGENGRAPH_CONVERSION_PVINTERLEAVEOPTIONS_H
#define DEEPGENGRAPH_CONVERSION_PVINTERLEAVEOPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
struct MMAChainScheduleOptions {
  unsigned qkWidth = 0;
  int pvGuardGroup = -1;
  int qkGuardGroup = 2;
  unsigned qkPrefetchWords = 32;
};
bool parseMMAChainScheduleOption(llvm::StringRef argument,
                                MMAChainScheduleOptions &options, bool &consumed);
/// MyTest Thread IR scheduling: 0 disables, 1 guards serial chains, 2/4 interleave.
bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width,
                             bool &consumed);
} // namespace mlir::frisk
#endif
