#ifndef DEEPGENGRAPH_PIPELINE_PASSES
#define DEEPGENGRAPH_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

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
