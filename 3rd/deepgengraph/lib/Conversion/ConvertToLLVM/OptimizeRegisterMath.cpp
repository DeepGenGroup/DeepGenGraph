// ===========================================================================
// Pass: optimizeApproximateRegisterMath   优化汇总.py 第 19 项，LLVM 阶段，math=approx
//
// 作用：
//   1. 先调用 fuseMMACAccumulators（第 25 项的函数，这里是第一次运行）。
//   2. nativeExp2：把 __ocml_exp2_f32 换成原生 llvm.amdgcn.exp2。
//   3. reciprocalMode：同一基本块内相同分母的 fdiv 共用一次 1/x，变成 a * (1/x)。
//
// 这是近似优化：原生 exp2 和“乘倒数”都不保证与原实现逐位相同，所以只在 approx 模式打开。
// 跳过 strictfp 和 optnone 的函数。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

namespace mlir::frisk {
void optimizeRegisterMath(llvm::Module &module, bool nativeExp2, bool reciprocalMode) {
  fuseMMACAccumulators(module);
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
  llvm::FunctionPassManager canonicalize;
  canonicalize.addPass(llvm::InstCombinePass());
  // exponentials / divisions：替换次数，只用于日志。
  unsigned exponentials = 0, divisions = 0;
  for (llvm::Function &function : module) {
    if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP) ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone)) continue;
    // Collapse the extract/insert wrappers emitted by vector lowering before
    // examining actual register chains. InstCombine itself preserves FP order.
    canonicalize.run(function, analyses);
    for (llvm::BasicBlock &block : function) {
      // reciprocals：分母 -> 已生成的 1/分母。按基本块重建，所以不会跨块共用。
      llvm::DenseMap<llvm::Value *, llvm::Value *> reciprocals;
      for (llvm::Instruction &inst : llvm::make_early_inc_range(block)) {
        llvm::IRBuilder<> b(&inst);
        b.SetCurrentDebugLocation(inst.getDebugLoc());
        if (reciprocalMode && inst.getOpcode() == llvm::Instruction::FDiv &&
                   inst.getType()->getScalarType()->isFloatTy()) {
          llvm::Value *denominator = inst.getOperand(1);
          if (llvm::isa<llvm::Constant>(denominator)) continue;
          llvm::Value *&reciprocal = reciprocals[denominator];
          if (!reciprocal) {
            auto *divide = llvm::cast<llvm::Instruction>(b.CreateFDiv(
                llvm::ConstantFP::get(inst.getType(), 1.0), denominator, "row.reciprocal"));
            llvm::FastMathFlags flags;
            flags.setAllowReciprocal();
            divide->setFastMathFlags(flags);
            reciprocal = divide;
          }
          llvm::Value *product = b.CreateFMul(inst.getOperand(0), reciprocal, "normalized");
          inst.replaceAllUsesWith(product);
          inst.eraseFromParent();
          ++divisions;
        } else if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          auto *callee = call->getCalledFunction();
          if (!nativeExp2 || !callee || callee->getName() != "__ocml_exp2_f32" ||
              call->arg_size() != 1 || !call->getType()->isFloatTy() ||
              call->getNumOperandBundles()) continue;
          auto *native = llvm::Intrinsic::getOrInsertDeclaration(
              &module, llvm::Intrinsic::amdgcn_exp2, {b.getFloatTy()});
          call->replaceAllUsesWith(b.CreateCall(native, {call->getArgOperand(0)}));
          call->eraseFromParent();
          ++exponentials;
        }
      }
    }
    analyses.invalidate(function, llvm::PreservedAnalyses::none());
    canonicalize.run(function, analyses);
  }
  llvm::errs() << "Approximate register math: " << exponentials << " native exp2, " << divisions << " reciprocal divisions\n";
}
void optimizeApproximateRegisterMath(llvm::Module &module) {
  optimizeRegisterMath(module, true, true);
}
} // namespace mlir::frisk
