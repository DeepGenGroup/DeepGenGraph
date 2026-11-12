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
  let summary = "Staticize vector indices, then interleave independent memory and compute ops";
  let options = [Option<"reorderAfterUnroll", "reorder-after-unroll", "bool", "true",
                       "Schedule straight-line memory/compute operations after unrolling">];
  let description = [{}];

  let constructor = "mlir::frisk::createIRDeepOptimizePass()";

  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

def BarrierOptimize: Pass<"barrier-optimize", "func::FuncOp"> {
  let summary = "Remove workgroup barriers without cross-thread memory dependencies";
  let constructor = "mlir::frisk::createBarrierOptimizePass()";
}

def LowerFriskAsyncCopy: Pass<"frisk-lower-async-copy", "::mlir::ModuleOp"> {
  let summary = "Lower block tile copies and buffer readiness for gfx936";
  let constructor = "mlir::frisk::createLowerFriskAsyncCopyPass()";
}

def FuseFragmentAccumulator: Pass<"fuse-fragment-accumulator", "func::FuncOp"> {
  let summary = "Fuse a full-tile add into independent MMA fragment updates";
  let constructor = "mlir::frisk::createFuseFragmentAccumulatorPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::vector::VectorDialect"];
}

def ThreadMaskAddBypass: Pass<"frisk-bypass-mask-adds", "func::FuncOp"> {
  let summary = "Bypass zero-mask additions in thread SSA before LLVM lowering";
  let constructor = "mlir::frisk::createThreadMaskAddBypassPass()";
  let options = [Option<"mode", "mode", "std::string", "\"approx\"",
      "off or approx; approx explicitly permits signed-zero differences in matched mask additions">];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::vector::VectorDialect", "mlir::scf::SCFDialect"];
}

def ApproximateThreadMath: Pass<"frisk-approximate-thread-math", "func::FuncOp"> {
  let summary = "Share FP32 reciprocals in thread IR (explicit approximate mode only)";
  let constructor = "mlir::frisk::createApproximateThreadMathPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // THREADIR_TO_LLVM_CONVERSION_PASSES
