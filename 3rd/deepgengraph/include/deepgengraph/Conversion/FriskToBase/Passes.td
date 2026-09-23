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

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def FinalizeThreadTiling: Pass<"finalize-thread-tiling", "::mlir::func::FuncOp"> {
  let summary = "基于thread级别的实现 将其转为 mlir built-in IR 表达的形式。完全去除 to_threadtile 和 from_threadtile";
  let description = [{}];

  let constructor = "mlir::frisk::createFinalizeThreadTilingPass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}



#endif // FRISKTOBASE_CONVERSION_PASSES