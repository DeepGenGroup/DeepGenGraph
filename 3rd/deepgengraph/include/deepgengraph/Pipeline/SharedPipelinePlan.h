#ifndef DEEPGENGRAPH_PIPELINE_SHARED_PIPELINE_PLAN_H
#define DEEPGENGRAPH_PIPELINE_SHARED_PIPELINE_PLAN_H

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::pipeline {
// AutoPipeline's provisional straight-line schedule, carried by each DMA:
// [stage, stage count, logical slot, issue-after stage (-1 = ordinary safe
// hoisting, with no preceding stream readiness constraining the first issue)].
// Stage IDs are local to a block. Completion is identified by the DMA's task
// SSA, not by proximity to an unrelated wait. Keep the serial IR until physical
// validation succeeds, so rejecting a candidate needs no speculative rollback.
inline constexpr StringLiteral sharedPipelinePlanAttr = "frisk.shared_pipeline_plan";
} // namespace mlir::pipeline

#endif
