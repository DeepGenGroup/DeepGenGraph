#ifndef DEEPGENGRAPH_TRANSFORMS_PASSES_H_
#define DEEPGENGRAPH_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"

namespace mlir::deepgengraph {

// FIXME: remove all manual constructors in tablegen and use the default one?
#define GEN_PASS_DECL
#include "deepgengraph/Dialect/TL/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createConvertDeepgenGraphToTilelangPass();
// std::unique_ptr<Pass> createReplaceExpAndLogPass();

// std::unique_ptr<Pass> createLowerComplexReducePass();

// std::unique_ptr<Pass> createEraseTypeInKernelPass();

// std::unique_ptr<Pass> createMulScalarHoistingPass();
// std::unique_ptr<Pass> createBroadcastTransformPass();

// std::unique_ptr<Pass> createEquivalentTransformPass();

// std::unique_ptr<Pass> createPermuteHoistingPass();

// std::unique_ptr<Pass> createAnnotateParallelismPass();
// std::unique_ptr<Pass> createAnnotateParallelismPass(const DeepgengraphAnnotateParallelismOptions &);

// std::unique_ptr<Pass> createToMaskPass();

// std::unique_ptr<Pass> createParallelizePass();
// std::unique_ptr<Pass> createParallelizePass(const DeepgengraphParallelizeOptions &);

// std::unique_ptr<Pass> createTilingPass();

// std::unique_ptr<Pass> createDynamicForPass();

// std::unique_ptr<Pass> createRecoverTypeInKernelPass();

// std::unique_ptr<Pass> createUserReplicatePass();

} // namespace mlir::deepgengraph

#endif