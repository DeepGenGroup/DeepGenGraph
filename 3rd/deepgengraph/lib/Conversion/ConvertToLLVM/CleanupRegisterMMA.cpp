// Preserve the register MMA preparation and cleanup used by LLVM export.
// Accumulator reassociation is handled by FriskOpFusePass before tiling.
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

namespace mlir::frisk {
void cleanupRegisterMMA(llvm::Module &module, bool simplifyCFG) {
  prepareRegisterMMAForLLVM(module);
  llvm::FunctionAnalysisManager analyses;
  llvm::ModuleAnalysisManager modules;
  llvm::LoopAnalysisManager loops;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(modules);
  pb.registerLoopAnalyses(loops);
  pb.registerCGSCCAnalyses(cgscc);
  pb.registerFunctionAnalyses(analyses);
  pb.crossRegisterProxies(loops, analyses, cgscc, modules);
  llvm::FunctionPassManager pipeline;
  // Preserve the late constant-guarded LDS writeback CFG cleanup.
  if (simplifyCFG) pipeline.addPass(llvm::SimplifyCFGPass());
  pipeline.addPass(llvm::InstCombinePass());
  pipeline.addPass(llvm::InstCombinePass());
  for (llvm::Function &function : module)
    if (!function.isDeclaration()) pipeline.run(function, analyses);
}
} // namespace mlir::frisk
