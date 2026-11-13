#include "deepgengraph/Analysis/ThreadBounds.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

std::optional<mlir::frisk::ThreadIdRange>
mlir::frisk::getThreadIdRange(Operation *scope, gpu::Dimension dimension) {
  while (scope && !isa<FunctionOpInterface>(scope))
    scope = scope->getParentOp();
  if (!scope) return std::nullopt;
  auto threads = scope->getAttrOfType<IntegerAttr>("thread_num");
  if (!threads || !threads.getValue().isStrictlyPositive() ||
      threads.getValue().ugt(1024))
    return std::nullopt;
  switch (dimension) {
  case gpu::Dimension::x:
    return ThreadIdRange{0, int64_t(threads.getValue().getZExtValue())};
  case gpu::Dimension::y:
  case gpu::Dimension::z:
    return ThreadIdRange{0, 1};
  }
  return std::nullopt;
}
