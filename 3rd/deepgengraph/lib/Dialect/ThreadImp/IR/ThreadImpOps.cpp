#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"

#include "dbg.h"

#define GET_OP_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpOps.cpp.inc"
// #include "deepgengraph/Dialect/ThreadImp/IR/DeepgengraphTritonOps.cpp.inc"

#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.cpp.inc"
// #include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.cpp.inc"
namespace mlir {
namespace threadimp {
void ThreadImpDialect::initialize() {
  // registerTypes();
  addOperations<
#define GET_OP_LIST
#include "deepgengraph/Dialect/ThreadImp/IR//ThreadImpOps.cpp.inc"
// #include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonOps.cpp.inc"
      >();
}

} // namespace deepgengraph::triton
} // namespace mlir

namespace mlir {
namespace threadimp {


} // namespace threadimp
} // namespace mlir