#ifndef THREADIR_TO_LLVM_CONVERSION_PASSES
#define THREADIR_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ThreadLevelIRLegalize: Pass<"thread-ir-legalize", "::mlir::ModuleOp"> {
  let summary = "对threadLevelIR进行合法化变换 准备将其lower to LLVM";
  let description = [{}];

  let constructor = "mlir::frisk::createThreadLevelIRLegalizePass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def RegMemrefVectorize: Pass<"reg-memref-vectorize", "func::FuncOp"> {
  let summary = "legalize后，将表达reg的memref改为 vector类型";
  let description = [{}];

  let constructor = "mlir::frisk::createRegMemrefVectorizePass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

#endif // THREADIR_TO_LLVM_CONVERSION_PASSES
