// CLI compatibility only. Scheduling is implemented exclusively in ThreadIR.
#include "deepgengraph/Conversion/ConvertToLLVM/ScheduleSoftmaxFragments.h"

namespace mlir::frisk {
bool parseSoftmaxFragmentOption(llvm::StringRef arg, SoftmaxFragmentMode &mode,
                               bool &consumed) {
  consumed = arg.starts_with("--softmax-fragments");
  if (!consumed) return true;
  if (!arg.consume_front("--softmax-fragments=")) return false;
  if (arg == "off") mode = SoftmaxFragmentMode::Off;
  else if (arg == "fragment") mode = SoftmaxFragmentMode::Fragment;
  else if (arg == "pv") mode = SoftmaxFragmentMode::PV;
  else return false;
  return true;
}
} // namespace mlir::frisk
