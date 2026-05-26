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



#endif // FRISKTOBASE_CONVERSION_PASSES