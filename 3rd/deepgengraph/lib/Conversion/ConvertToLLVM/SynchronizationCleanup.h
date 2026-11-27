#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::frisk {

std::unique_ptr<Pass> createSynchronizationCleanupPass(bool coalesceWaits,
                                                       bool analyzeDependencies);
void removeRedundantWorkgroupBarriers(func::FuncOp function);

} // namespace mlir::frisk
