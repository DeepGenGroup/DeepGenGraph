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

#endif // THREADIR_TO_LLVM_CONVERSION_PASSES
