#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"
// #include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.h"

using namespace mlir;
using namespace mlir::threadimp;

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.cpp.inc"

void ThreadImpDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.cpp.inc"
      >();
}
