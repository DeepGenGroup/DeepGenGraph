#ifndef DEEPGENGRAPH_CONVERSION_NARROWKDMA_H
#define DEEPGENGRAPH_CONVERSION_NARROWKDMA_H
#include "llvm/ADT/StringRef.h"
namespace llvm { class Module; }
namespace mlir::frisk {
struct KDMAOptions {
  llvm::StringRef mode = "128";
  unsigned argument = 1; // Explicit lowered ABI input index; Q,K,V,O by default.
};
bool parseKDMAOption(llvm::StringRef, KDMAOptions &, bool &consumed);
/// Final export transformation, after PV scheduling and V double buffering.
/// Explicit unsupported requests return false, without exporting partial IR.
bool narrowKDMA(llvm::Module &, const KDMAOptions &);
}
#endif
