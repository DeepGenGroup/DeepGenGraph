#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def FriskLayoutInfer: InterfacePass<"frisk-layoutInfer", "FunctionOpInterface">{
    let summary = "Frisk layoutInfer";
    let description = [{
        Infer layout op
    }];

    let constructor = "mlir::frisk::createFriskLayoutInferPass";

    let dependentDialects = [];

}

def DeepgengraphToFrisk: Pass<"deepgengraph-convert-to-frisk", "::mlir::ModuleOp"> {
  let summary = "Deepgengraph convert to Frisk";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createConvertToFrisk()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::frisk::FriskDialect"
  ];
}

#endif // FRISK_PASSES