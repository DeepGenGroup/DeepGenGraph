#ifndef TILELANG_DIALECT
#define TILELANG_DIALECT

include "mlir/IR/OpBase.td"

def Tilelang_Dialect : Dialect {
  let name = "tilelang";

  let cppNamespace = "::mlir::tilelang";

  let summary = "Tilelang Dialact in MLIR";

  let description = [{
    Tilelang Dialect.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
    "memref::MemRefDialect"
  ];
  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;

}

#endif // TILELANG_DIALECT