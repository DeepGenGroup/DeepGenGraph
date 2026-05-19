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



#endif // FRISKTOBASE_CONVERSION_PASSES