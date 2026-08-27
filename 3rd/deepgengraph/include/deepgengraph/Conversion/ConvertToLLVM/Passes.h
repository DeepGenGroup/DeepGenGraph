#ifndef CONVERT_THREAD_TO_LLVM_PASS_H
#define CONVERT_THREAD_TO_LLVM_PASS_H

#include "mlir/Pass/Pass.h"
#include "deepgengraph/Common.h"

namespace mlir::frisk {

std::unique_ptr<Pass> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth);
std::unique_ptr<mlir::Pass> createThreadLevelIRLegalizePass();
std::unique_ptr<OperationPass<ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target);

// -------- Lower Pipeline ----------
bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context) ;
bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context, Target target) ;
std::string translate(mlir::ModuleOp& mod, mlir::frisk::Target target) ;

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

} // namespace mlir::frisk

#endif
