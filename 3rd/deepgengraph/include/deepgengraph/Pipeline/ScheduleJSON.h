#ifndef DEEPGENGRAPH_PIPELINE_SCHEDULEJSON_H
#define DEEPGENGRAPH_PIPELINE_SCHEDULEJSON_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::pipeline {
// Empty selector: unique marked loop, otherwise the outermost compute/copy
// loop of a single kernel. Explicit selectors are "function:affine-preorder-id".
struct ScheduleJSONOptions {
  std::string exportPath, importPath, baselinePath, loopSelector, dumpIRPath;
  bool enabled() const {
    return !exportPath.empty() || !importPath.empty() || !baselinePath.empty();
  }
};

// Reconstruct and validate the problem before applying a solution. The
// fingerprint binds a solution to the entire function at this compiler phase.
// Import is transactional: transformations are verified on a cloned module.
LogicalResult processScheduleJSON(ModuleOp module,
                                  const ScheduleJSONOptions &options);
} // namespace mlir::pipeline
#endif
