#ifndef CONVERT_THREAD_TO_LLVM_PASS_H
#define CONVERT_THREAD_TO_LLVM_PASS_H

#include "mlir/Pass/Pass.h"
#include "deepgengraph/Common.h"

namespace mlir::frisk {

std::unique_ptr<Pass> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth,
                                              bool nativeExp2 = false);
std::unique_ptr<Pass> createLowerGPUShuffleToROCDLPass();
std::unique_ptr<mlir::Pass> createThreadLevelIRLegalizePass();

std::unique_ptr<OperationPass<ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target);
std::unique_ptr<OperationPass<ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target);
std::unique_ptr<mlir::Pass>  createVectorOpLoopUnrollPass();
std::unique_ptr<mlir::Pass> createIRDeepOptimizePass();
std::unique_ptr<mlir::Pass> createBarrierOptimizePass();
std::unique_ptr<mlir::Pass> createCoalesceWaitsAndBarriersPass();
std::unique_ptr<mlir::Pass> createLowerFriskAsyncCopyPass(bool useAsyncCopy = true);
std::unique_ptr<mlir::Pass> createFuseFragmentAccumulatorPass();
std::unique_ptr<mlir::Pass> createApproximateThreadMathPass();
std::unique_ptr<mlir::Pass> createSimplifyBoundedAffinePass();
std::unique_ptr<mlir::Pass> createHoistWaveInvariantsPass(bool ldsBases = true, bool laneShuffles = true);

// -------- Lower Pipeline ----------
bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context) ;
bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context, Target target,
                    bool coalesceWaits = true, bool nativeExp2 = false);
std::string translate(mlir::ModuleOp& mod, mlir::frisk::Target target) ;

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

} // namespace mlir::frisk

#endif
