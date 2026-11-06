// op 上表示“调度已经确定”的属性。调度类 Pass 遇到这些属性就不再移动 op。
#ifndef DEEPGENGRAPH_FRISKTOBASE_SCHEDULEATTRS_H
#define DEEPGENGRAPH_FRISKTOBASE_SCHEDULEATTRS_H

#include "mlir/IR/Operation.h"

namespace mlir::frisk {

// 手动指定或软流水生成的调度。
inline bool hasPipelineScheduleAttr(Operation *op) {
  for (StringRef name : {"pipeline.stage", "pipeline.order", "pipeline.scheduled", "pipeline_slot"})
    if (op->hasAttr(name)) return true;
  return false;
}

// 上面的调度，或 FriskFragIRReorder 生成的 fragment 流水。
inline bool hasAnyScheduleAttr(Operation *op) {
  return hasPipelineScheduleAttr(op) || op->hasAttr("frisk.pipelined") ||
         op->hasAttr("frisk.pipeline_stage");
}

} // namespace mlir::frisk

#endif // DEEPGENGRAPH_FRISKTOBASE_SCHEDULEATTRS_H
