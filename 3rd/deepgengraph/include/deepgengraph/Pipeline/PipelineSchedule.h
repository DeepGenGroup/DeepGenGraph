#ifndef DEEPGENGRAPH_PIPELINE_PIPELINESCHEDULE_H
#define DEEPGENGRAPH_PIPELINE_PIPELINESCHEDULE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::pipeline {

#define GEN_PASS_DECL
#include "deepgengraph/Pipeline/Passes.h.inc"

/// Options controlling how an annotated frisk loop is turned into a software
/// pipeline.
struct PipelineScheduleOptions {
  /// Number of trailing tiles that are peeled off the steady loop.  A tile
  /// whose stage offset is smaller than this number is emitted in the
  /// epilogue (FA3 peels the last tile so that the epilogue can rescale O).
  int64_t peeledTiles = 1;

  /// Maximum number of copies a stage-crossing buffer is split into.  v1
  /// supports 2, i.e. plain double buffering.
  int64_t maxSlots = 2;
};

/// Rewrites the annotated `affine.for` of `func` into a software pipeline.
///
/// Returns success (and leaves the IR untouched) when `func` does not contain
/// a loop annotated with `pipeline.stage`/`pipeline.order`, which makes the
/// transform idempotent.
LogicalResult applyPipelineSchedule(
    func::FuncOp func, const PipelineScheduleOptions &options = {});

std::unique_ptr<::mlir::Pass> createPipelineSchedulePass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Pipeline/Passes.h.inc"

} // namespace mlir::pipeline

#endif // DEEPGENGRAPH_PIPELINE_PIPELINESCHEDULE_H
