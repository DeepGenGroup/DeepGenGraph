#ifndef FRISKTOBASE_CONVERSION_PASSES
#define FRISKTOBASE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFriskToBase: Pass<"convert-frisk-to-base", "::mlir::ModuleOp"> {
  let summary = "convert frisk to base";
  let description = [{}];

  let constructor = "mlir::frisk::createConvertFriskToBasePass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def ConvertFriskBaseToThreadLevelIR: Pass<"convert-friskbase-to-thread", "::mlir::func::FuncOp"> {
  let summary = "frisk base 转换到thread级别的实现";
  let description = [{}];

  let constructor = "mlir::frisk::createConvertFriskBaseToThreadLevelIRPass()";
  let options = [Option<"packSharedOperands", "pack-shared-operands", "bool", "true",
      "Plan packed LDS layouts for owned GEMM B operands">];

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def PackSharedMemory: Pass<"pack-shared-memory", "::mlir::func::FuncOp"> {
  let summary = "Pack owned shared buffers and coalesce proven contiguous accesses";
  let constructor = "mlir::frisk::createPackSharedMemoryPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect", "mlir::vector::VectorDialect"];
}

def FinalizeThreadTiling: Pass<"finalize-thread-tiling", "::mlir::func::FuncOp"> {
  let summary = "基于thread级别的实现 将其转为 mlir built-in IR 表达的形式。完全去除 to_threadtile 和 from_threadtile";
  let description = [{}];

  let constructor = "mlir::frisk::createFinalizeThreadTilingPass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}



def FriskFragIRReorder: Pass<"frisk-frag-ir-reorder", "::mlir::func::FuncOp"> {
  let summary = "Fuse fragment loops and pipeline read-only fragment loads";
  let constructor = "mlir::frisk::createFriskFragIRReorderPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::frisk::FriskDialect"];
}

def LowerFriskFragments: Pass<"lower-frisk-fragments", "::mlir::func::FuncOp"> {
  let summary = "Lower scheduled fragment regions, plan storage and unroll marked loops";
  let constructor = "mlir::frisk::createLowerFriskFragmentsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect", "mlir::vector::VectorDialect", "mlir::gpu::GPUDialect", "mlir::frisk::FriskDialect"];
}

#endif // FRISKTOBASE_CONVERSION_PASSES
