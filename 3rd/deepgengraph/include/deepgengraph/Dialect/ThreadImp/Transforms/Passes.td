#ifndef THREADIMP_PASSES
#define THREADIMP_PASSES

include "mlir/Pass/PassBase.td"

def DeepgengraphToThreadImp: Pass<"deepgengraph-convert-to-threadimp", "::mlir::ModuleOp"> {
  let summary = "Deepgengraph convert to ThreadImp";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createConvertr2ThreadImp()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::threadimp::ThreadImpDialect",
  ];
}


#endif // DEEPGENGRAPH_PASSES