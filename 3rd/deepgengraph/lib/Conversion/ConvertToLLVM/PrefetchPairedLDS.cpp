// CLI compatibility only. Scheduling is implemented exclusively in ThreadIR.
#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchPairedLDS.h"

namespace mlir::frisk {
bool parsePairedLDSOption(llvm::StringRef arg,unsigned &distance,bool &consumed) {
  consumed=arg.starts_with("--k-lds-prefetch");
  if (!consumed) return true;
  return arg.consume_front("--k-lds-prefetch=") && !arg.getAsInteger(10,distance) && distance<=8;
}

bool parsePairedLDSTailOption(llvm::StringRef arg, unsigned &distance, bool &consumed) {
  consumed = arg.starts_with("--k-tail-prefetch");
  if (!consumed) return true;
  return arg.consume_front("--k-tail-prefetch=") &&
         !arg.getAsInteger(10, distance) && distance <= 8;
}
} // namespace mlir::frisk
