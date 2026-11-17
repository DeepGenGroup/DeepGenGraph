#ifndef DEEPGENGRAPH_CONVERSION_PREFETCHPAIREDLDS_H
#define DEEPGENGRAPH_CONVERSION_PREFETCHPAIREDLDS_H
#include "llvm/ADT/StringRef.h"
namespace mlir::frisk {
// MyTest CLI parsing only; there is no LLVM scheduling entry point.
bool parsePairedLDSOption(llvm::StringRef, unsigned &, bool &consumed);
bool parsePairedLDSTailOption(llvm::StringRef, unsigned &, bool &consumed);
}
#endif
