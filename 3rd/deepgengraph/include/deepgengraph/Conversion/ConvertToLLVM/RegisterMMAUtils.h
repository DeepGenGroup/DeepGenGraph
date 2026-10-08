#ifndef DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_REGISTERMMAUTILS_H
#define DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_REGISTERMMAUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
// Match the complete register-only instruction with NOP padding. Padding count
// and delay are target tuning choices, independent of the asm's memory effects.
// This recognizer does not validate hardware latency or change the padding.
// Never infer memory(none) from a mnemonic substring or an untrusted marker.
inline bool isPaddedRegisterMMA(llvm::StringRef assembly,
                                llvm::StringRef constraints) {
  if (constraints != "=v,v,v,0")
    return false;
  llvm::SmallVector<llvm::StringRef> lines;
  assembly.split(lines, '\n', -1, false);
  llvm::erase_if(lines, [](llvm::StringRef line) { return line.trim().empty(); });
  if (lines.size() < 3)
    return false;
  bool foundMMA = false;
  for (unsigned i = 0; i < lines.size(); ++i) {
    llvm::StringRef line = lines[i].trim();
    if (line == "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3") {
      if (foundMMA || i == 0 || i + 1 == lines.size())
        return false;
      foundMMA = true;
      continue;
    }
    if (!line.consume_front("s_nop") || line.empty() ||
        (line.front() != ' ' && line.front() != '\t'))
      return false;
    line = line.trim();
    unsigned delay;
    if (line.empty() ||
        !llvm::all_of(line, [](char c) { return c >= '0' && c <= '9'; }) ||
        line.getAsInteger(10, delay) || delay > 15)
      return false;
  }
  return foundMMA;
}
} // namespace mlir::frisk
#endif
