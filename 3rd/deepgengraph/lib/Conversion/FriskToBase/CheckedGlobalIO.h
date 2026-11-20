#ifndef DEEPGENGRAPH_CONVERSION_CHECKEDGLOBALIO_H
#define DEEPGENGRAPH_CONVERSION_CHECKEDGLOBALIO_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::frisk {
// Lower the opt-in frisk.checked_global_io kernel contract after buffer_view
// composition, while the original tensor axes still exist: OOB reads return
// zero, OOB writes do nothing. No allocation padding is required.
LogicalResult lowerCheckedGlobalIO(func::FuncOp kernel);
}

#endif
