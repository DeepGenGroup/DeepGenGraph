#ifndef THREADIMP_DIALECT_H_
#define THREADIMP_DIALECT_H_

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// // Triton
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
// #include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"

#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h.inc"

// #include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphTypes.h"

namespace mlir::threadimp {
// #include "deepgengraph/Dialect/TL/IR/TilelangOpInterfaces.h.inc"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpOpInterfaces.h.inc"
}
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpEnums.h.inc"
// #include "deepgengraph/Dialect/TL/IR/TilelangEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpAttrs.h.inc"
// #include "deepgengraph/Dialect/TL/IR/TilelangAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.h.inc"
// #include "deepgengraph/Dialect/TL/IR/TilelangTypes.h.inc"

#define GET_OP_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpOps.h.inc"
// #include "deepgengraph/Dialect/TL/IR/TilelangOps.h.inc"


#endif