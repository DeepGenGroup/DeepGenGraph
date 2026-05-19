#ifndef FRISK_CONVERSION_FRISKTOBASE_FRISKTOBASE_PASS_H
#define FRISK_CONVERSION_FRISKTOBASE_FRISKTOBASE_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::frisk {

std::unique_ptr<mlir::Pass> createConvertFriskToBasePass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

} // namespace mlir::deepgengraph

#endif