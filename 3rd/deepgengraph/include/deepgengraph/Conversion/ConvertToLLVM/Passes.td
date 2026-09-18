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

def IRDeepOptimize: Pass<"ir-deep-optimize", "func::FuncOp"> {
  let summary = "Staticize register-vector indices by selective affine unrolling";
  let description = [{}];

  let constructor = "mlir::frisk::createIRDeepOptimizePass()";

  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

def FuseFragmentAccumulator: Pass<"fuse-fragment-accumulator", "func::FuncOp"> {
  let summary = "Fuse a full-tile add into independent MMA fragment updates";
  let constructor = "mlir::frisk::createFuseFragmentAccumulatorPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::vector::VectorDialect"];
}

#endif // THREADIR_TO_LLVM_CONVERSION_PASSES
