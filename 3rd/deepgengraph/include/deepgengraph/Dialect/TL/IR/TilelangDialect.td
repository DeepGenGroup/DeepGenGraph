#ifndef TILELANG_DIALECT
#define TILELANG_DIALECT

include "mlir/IR/OpBase.td"

def Tilelang_Dialect : Dialect {
  let name = "tilelang";

  let cppNamespace = "::mlir::tilelang";

  // Preserve the existing IR syntax with inherent attributes in attr-dict.
  let useStrictPropertiesInAssemblyFormat = 0;

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
    "::mlir::memref::MemRefDialect"
  ];
  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;

}

#endif // TILELANG_DIALECT
