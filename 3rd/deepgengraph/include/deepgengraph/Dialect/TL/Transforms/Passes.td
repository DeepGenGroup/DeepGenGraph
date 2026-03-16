#ifndef DEEPGENGRAPH_PASSES
#define DEEPGENGRAPH_PASSES

include "mlir/Pass/PassBase.td"

def DeepgengraphToTilelang: Pass<"deepgengraph-convert-to-tilelang", "::mlir::ModuleOp"> {
  let summary = "Deepgengraph convert to tilelang";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createConvertGraphToTilelang()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::tilelang::TilelangDialect",
  ];
}


#endif // DEEPGENGRAPH_PASSES