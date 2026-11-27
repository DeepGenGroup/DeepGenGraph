#ifndef DEEPGENGRAPH_CONVERSION_THREADREDUCTIONS_H
#define DEEPGENGRAPH_CONVERSION_THREADREDUCTIONS_H

#include "deepgengraph/Conversion/ConvertToLLVM/DeferLaneSum.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
namespace func {
class FuncOp;
}
namespace frisk {

/// Run enabled ThreadIR reductions in order: split local chains, then defer
/// lane sums across loop iterations. Each stage has independent FP permission.
std::unique_ptr<Pass> createThreadReductionsPass(
    std::optional<LocalReductionOptions> local,
    std::optional<DeferLaneSumOptions> lane);

/// Apply lane-sum deferral to one function with validated options.
void deferThreadLaneSums(func::FuncOp function, DeferLaneSumOptions options);

} // namespace frisk
} // namespace mlir

#endif
