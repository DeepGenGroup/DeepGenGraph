#ifndef DEEPGENGRAPH_PIPELINE_PASSES
#define DEEPGENGRAPH_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def SimplifyLoopControl : Pass<"frisk-simplify-loop-control", "::mlir::func::FuncOp"> {
  let summary = "Specialize bounded static loops and fold their control/address expressions";
  let constructor = "mlir::pipeline::createSimplifyLoopControlPass()";
  let description = [{
    Fully unroll small affine/scf loops with proven constant signed bounds,
    then canonicalize guards, selectors and index calculations. By default
    only pipeline.scheduled or frisk.simplify_loop loops are considered.
    This has no attention-specific dimensions or buffer names. Dynamic,
    unsigned, wrapping and over-budget loops are left rolled.
    Run after thread mapping and before physical packing/shared-storage reuse
    in the Frisk pipeline, so layout inference still sees the rolled carriers.
  }];
  let options = [
    Option<"maxTripCount", "max-trip-count", "uint64_t", "8",
           "Maximum static trip count to fully unroll">,
    Option<"maxExpandedOps", "max-expanded-ops", "uint64_t", "2048",
           "Per-function budget for expanded operations">,
    Option<"onlyMarked", "only-marked", "bool", "true",
           "Only specialize explicitly marked or scheduled loops">
  ];
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect", "::mlir::memref::MemRefDialect"
  ];
}

def AutoPipeline : Pass<"frisk-auto-pipeline", "::mlir::func::FuncOp"> {
  let summary = "Prefetch global tiles into rotating shared buffers on hinted loops";
  let description = [{
    Only attempt loops carrying frisk.pipeline_hint = N : i64 (unit means 2,
    1 disables). At Frisk Base level, prove full-tile global/shared transfers
    independent of loop-carried computations and external writes. Allocate N
    slots, guard N-1 prologue loads, then wait for tile i, issue tile i+N-1 and
    compute tile i. Unproven loops remain unchanged with a diagnostic remark.
  }];
  let constructor = "mlir::pipeline::createAutoPipelinePass()";
  let dependentDialects = [
    "::mlir::frisk::FriskDialect", "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def PipelineSchedule : Pass<"frisk-pipeline-schedule", "::mlir::func::FuncOp"> {
  let summary = "把带 pipeline.stage/pipeline.order 标注的 frisk 循环排成软件流水线";
  let description = [{
    参考 tilelang 的 T.Pipelined(stage = [...], order = [...]) 思路，按 FA3 的
    prologue / steady / epilogue 结构，对 func 中唯一一个被标注的 affine.for
    做软件流水变换：

      * pipeline.stage  : 语句所属流水级（0 最超前，数字越大越靠近当前迭代）
      * pipeline.order  : 稳态循环体内的发射顺序（数值小的先发射）

    两个属性都是 array<i64: ...>，每个顶层 op 一项（同一条语句的 op 重复
    同一个 (stage, order) 对）。

    变换后：跨流水级的 buffer 被拆成多槽、prologue 被显式展开、
    稳态循环用 scf.if 按 tile 奇偶选择槽位、最后一块被剥到 epilogue。
  }];

  let constructor = "mlir::pipeline::createPipelineSchedulePass()";

  let dependentDialects = [
    "::mlir::frisk::FriskDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // DEEPGENGRAPH_PIPELINE_PASSES
