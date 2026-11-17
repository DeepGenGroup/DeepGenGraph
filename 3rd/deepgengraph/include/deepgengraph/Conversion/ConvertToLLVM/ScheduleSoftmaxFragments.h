#ifndef DEEPGENGRAPH_SCHEDULE_SOFTMAX_FRAGMENTS_H
#define DEEPGENGRAPH_SCHEDULE_SOFTMAX_FRAGMENTS_H
#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
// Off by default. Fragment preserves conversion order; PV sinks conversion to
// the first consumer. Neither mode changes arithmetic or memory/sync order.
enum class SoftmaxFragmentMode { Off, Fragment, PV };
bool parseSoftmaxFragmentOption(llvm::StringRef, SoftmaxFragmentMode &, bool &);
} // namespace mlir::frisk
#endif
