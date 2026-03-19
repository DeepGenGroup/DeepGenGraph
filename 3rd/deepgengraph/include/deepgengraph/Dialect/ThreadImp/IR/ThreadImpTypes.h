#ifndef THREADIMP_TYPES_H_
#define THREADIMP_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.h.inc"

#endif // THREADIMP_TYPES_H_