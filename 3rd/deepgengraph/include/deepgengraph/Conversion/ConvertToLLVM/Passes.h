#ifndef CONVERT_THREAD_TO_LLVM_PASS_H
#define CONVERT_THREAD_TO_LLVM_PASS_H

#include "mlir/Pass/Pass.h"
#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h"
#include "deepgengraph/Conversion/ConvertToLLVM/DeferLaneSum.h"
#include "deepgengraph/Conversion/ConvertToLLVM/ScheduleSoftmaxFragments.h"
#include "deepgengraph/Conversion/ConvertToLLVM/ThreadLazySoftmaxBase.h"
#include "deepgengraph/Conversion/ConvertToLLVM/SoftmaxCvtPack.h"
#include "deepgengraph/Conversion/ConvertToLLVM/ThreadScheduling.h"

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
// Keep scheduling boundaries explicit until the final target lowering.
std::unique_ptr<mlir::Pass> createLowerFriskSchedBarrierPass(Target target = Target::ROCm);
std::unique_ptr<mlir::Pass> createFuseFragmentAccumulatorPass();
std::unique_ptr<mlir::Pass> createApproximateThreadMathPass();
std::unique_ptr<mlir::Pass> createThreadMaskAddBypassPass();
// After thread loop unrolling and reduction rewrites; before async/GPU lowering.
// guardGroup: -1 preserves per-MMA before/after guards, 0 disables guards,
// positive values emit one guard after each group (including a partial tail).
std::unique_ptr<mlir::Pass> createThreadInterleavePVPass(unsigned width = 0,
                                                     int guardGroup = -1);
std::unique_ptr<mlir::Pass> createThreadInterleaveQKPass(unsigned width = 0,
    int guardGroup = 2, unsigned prefetchWords = 32);
// After physical LDS address canonicalization, unrolling and PV interleaving.
// Keep this order: paired loads, softmax fragments, reduction-aware tail loads.
// Compatibility C++ wrappers; canonical options are declared in ThreadScheduling.h.
std::unique_ptr<mlir::Pass> createThreadPrefetchPairedLDSPass(unsigned distance);
std::unique_ptr<mlir::Pass> createThreadScheduleSoftmaxFragmentsPass(
    SoftmaxFragmentMode mode = SoftmaxFragmentMode::Off);
std::unique_ptr<mlir::Pass> createThreadPrefetchPairedLDSTailPass(unsigned distance = 0);
// After every ThreadIR memory schedule, before lowering async readiness.
std::unique_ptr<mlir::Pass> createThreadElideBackedgeBarrierPass();
// Only tagged scalar FP reduction chains; run after thread loop unrolling.
std::unique_ptr<mlir::Pass> createThreadLocalReductionPass(
    LocalReductionOptions options = {});
// Run on structured thread SSA before lowering gpu.shuffle.
std::unique_ptr<mlir::Pass> createThreadDeferLaneSumPass(
    DeferLaneSumOptions options = {});
std::unique_ptr<mlir::Pass> createSimplifyBoundedAffinePass();
// After physical shared storage and thread unrolling, before LLVM lowering.
std::unique_ptr<mlir::Pass> createCanonicalizeThreadLDSAddressesPass();
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
