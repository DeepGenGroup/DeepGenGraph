#ifndef THREADIMP_TRANSFORMS_PASSES_H_
#define THREADIMP_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"


namespace mlir::threadimp {

// FIXME: remove all manual constructors in tablegen and use the default one?
#define GEN_PASS_DECL
#include "deepgengraph/Dialect/ThreadImp/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createConvertMemOpPass();
std::unique_ptr<Pass> createInlineDevicekernelOpPass();
std::unique_ptr<Pass> createConvertBlockCalcOpToThreadImpPass();


std::unique_ptr<Pass> createLayoutAnalyzePass();




} // namespace mlir::deepgengraph

#endif