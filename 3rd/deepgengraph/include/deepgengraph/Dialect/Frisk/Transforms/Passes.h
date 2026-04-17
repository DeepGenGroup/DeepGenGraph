#ifndef FRISK_TRANSFORMS_PASSES_H
#define FRISK_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::frisk {

#define GEN_PASS_DECL
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDeepgenGraphSimplifyPass();

std::unique_ptr<Pass> createConvertDeepgenGraphToFriskPass();

std::unique_ptr<Pass> createOverlapPass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

} // namespace mlir::frisk

#endif // FRISK_TRANSFORMS_PASSES_H
