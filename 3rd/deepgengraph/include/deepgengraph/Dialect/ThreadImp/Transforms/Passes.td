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

def ConvertBlockCalculateOpToThreadLevelImp: Pass<"convert-to-threadicalc", "::mlir::deepgengraph::KernelOp"> {
  let summary = "convert block level calculate to thread-level imp";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createConvertBlockCalcToThreadImp()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::threadimp::ThreadImpDialect",
  ];
}

def DeepgengraphInlineDevicekernel: Pass<"deepgengraph-inline-devicekernel", "::mlir::deepgengraph::KernelOp"> {
  let summary = "Inline devicekernel op into KernelOp";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createInlineDevicekernelOpPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::threadimp::ThreadImpDialect",
  ];
}


#endif // DEEPGENGRAPH_PASSES