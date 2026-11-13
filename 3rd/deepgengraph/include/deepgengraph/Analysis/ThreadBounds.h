#ifndef DEEPGENGRAPH_ANALYSIS_THREADBOUNDS_H
#define DEEPGENGRAPH_ANALYSIS_THREADBOUNDS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <cstdint>
#include <optional>

namespace mlir::frisk {
struct ThreadIdRange {
  int64_t lower;
  int64_t upper; // Exclusive.
};

/// Query the enclosing function's one-dimensional launch contract: thread_num
/// means [N, 1, 1], so x is in [0, N) and y/z are in [0, 1). Accepts a function
/// or an operation nested in it, both before and after func-to-LLVM conversion.
/// Missing/malformed thread_num or a size outside [1, 1024] yields no range.
std::optional<ThreadIdRange> getThreadIdRange(Operation *scope,
                                             gpu::Dimension dimension);
} // namespace mlir::frisk

#endif
