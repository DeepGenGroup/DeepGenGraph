#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.h"

using namespace mlir;
using namespace mlir::frisk;

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.cpp.inc"

void FriskDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.cpp.inc"
      >();
}
