#ifndef FRISK_CONVERSION_FRISKTOBASE_FRISKTOBASE_PASS_H
#define FRISK_CONVERSION_FRISKTOBASE_FRISKTOBASE_PASS_H

#include "mlir/Pass/Pass.h"
namespace mlir::func { class FuncOp; }

namespace mlir::frisk {

std::unique_ptr<mlir::Pass> createConvertFriskToBasePass();
std::unique_ptr<mlir::Pass> createOptimizeTensorLoopsPass();
std::unique_ptr<mlir::Pass> createNormalizeAttentionKLayoutPass();
std::unique_ptr<mlir::Pass> createPlanWmmaLayoutConversionsPass();
std::unique_ptr<mlir::Pass> createConvertFriskBaseToThreadLevelIRPass();
std::unique_ptr<mlir::Pass> createFinalizeThreadTilingPass();
std::unique_ptr<mlir::Pass> createFriskFragIRReorderPass();
std::unique_ptr<mlir::Pass> createLowerFriskFragmentsPass();
std::unique_ptr<mlir::Pass> createPackSharedMemoryPass();
std::unique_ptr<mlir::Pass> createPromoteGlobalToSharedAsyncPass();
std::unique_ptr<mlir::Pass> createLowerFriskAsyncToSyncPass();
std::unique_ptr<mlir::Pass> createScheduleAsyncCopiesPass();
void packSharedMemory(mlir::func::FuncOp kernel);

#define GEN_PASS_DECL_CONVERTFRISKBASETOTHREADLEVELIR
#define GEN_PASS_DECL_OPTIMIZETENSORLOOPS
#define GEN_PASS_DECL_NORMALIZEATTENTIONKLAYOUT
#define GEN_PASS_DECL_PLANWMMALAYOUTCONVERSIONS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

} // namespace mlir::deepgengraph

#endif
