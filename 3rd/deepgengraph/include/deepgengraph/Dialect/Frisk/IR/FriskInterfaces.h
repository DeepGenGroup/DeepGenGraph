#ifndef DEEPGENGRAPH_FRISK_INTERFACES_H
#define DEEPGENGRAPH_FRISK_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::frisk {
enum class ExecutionScope { Unknown, Thread, Warp, Workgroup };
enum class ScheduleKind {
  Unknown, Compute, MMA, Load, Store, Copy, Layout, AsyncIssue, Wait, Barrier, Region
};
struct ExecutionSemantics {
  ExecutionScope scope = ExecutionScope::Unknown;
  ScheduleKind kind = ScheduleKind::Unknown;
  // SSA register computation, not a promise about physical allocation/spills.
  bool registerOnly = false;
  bool collective = false;
  // A boundary for generic register scheduling. Specialized schedulers must
  // still prove memory, task and participation dependencies before crossing it.
  bool schedulingBoundary = true;
};
ExecutionSemantics inferExecutionSemantics(Operation *op);
ExecutionSemantics getExecutionSemantics(Operation *op);
// Legality for ordinary local computation motion, not a scheduling decision.
// Also accepts other dialects through their standard effect/speculation APIs.
bool isOrdinaryRegisterComputation(Operation *op);

// Computed tiles are fresh logical values. Only memref operands are read;
// scalar/vector operands and results do not introduce memory effects.
void getComputedTileReadEffects(Operation *op,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
} // namespace mlir::frisk

#include "deepgengraph/Dialect/Frisk/IR/FriskInterfaces.h.inc"
#endif
