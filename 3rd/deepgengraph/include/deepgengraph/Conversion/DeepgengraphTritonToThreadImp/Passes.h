#ifndef DEEPGENGRAPH_CONVERSION_DEEPGENGRAPHTRITON_TO_THREAD_IMP_PASS_H
#define DEEPGENGRAPH_CONVERSION_DEEPGENGRAPHTRITON_TO_THREAD_IMP_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::deepgengraph {

std::unique_ptr<mlir::Pass> createConvertDeepgengraphTritonToThreadImpPass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/DeepgengraphTritonToThreadImp/Passes.h.inc"

} // namespace mlir::deepgengraph

#endif