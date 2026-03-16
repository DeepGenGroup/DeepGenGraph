#ifndef THREADIMP_DIALECT
#define THREADIMP_DIALECT

include "mlir/IR/OpBase.td"

def ThreadImp_Dialect : Dialect {
  let name = "threadimp";

  let cppNamespace = "::mlir::threadimp";

  let summary = "Threadimp Dialact in MLIR";

  let description = [{
    ThreadImp Dialect.
  }];

  let dependentDialects = [
    "::mlir::deepgengraph::DeepgengraphDialect",
    "::mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect"
  ];
  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;

}

#endif // TILELANG_DIALECT