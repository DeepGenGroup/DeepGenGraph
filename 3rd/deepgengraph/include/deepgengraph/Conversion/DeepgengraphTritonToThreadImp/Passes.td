#ifndef DEEPGENGRAPHTRITON_TO_THREAD_IMP_CONVERSION_PASSES
#define DEEPGENGRAPHTRITON_TO_THREAD_IMP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDeepgengraphTritonToThreadLevelImp: Pass<"convert-deepgengraphtriton-to-thread-imp", "::mlir::ModuleOp"> {
  let summary = "convert deepgengraphtriton to thread-level kernel implementation";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createConvertDeepgengraphTritonToThreadImpPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}



#endif // DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES