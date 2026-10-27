#include "deepgengraph/Conversion/ConvertToLLVM/FuseMMACAccumulator.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

namespace mlir::frisk {
namespace {
llvm::CallInst *getMMAC(llvm::Value *value) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(value);
  if (!call || (call->arg_size() != 3 && call->arg_size() != 5) ||
      call->getNumOperandBundles())
    return nullptr;
  auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand());
  auto *half4 = llvm::FixedVectorType::get(llvm::Type::getHalfTy(call->getContext()), 4);
  auto *float4 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(call->getContext()), 4);
  if (call->getType() != float4 || call->getArgOperand(2)->getType() != float4 ||
      call->getArgOperand(0)->getType() != half4 ||
      call->getArgOperand(1)->getType() != half4)
    return nullptr;
  if (assembly && call->arg_size() == 3 && !assembly->isAlignStack() &&
      !assembly->canThrow() &&
      isRegisterMMA(assembly->getAsmString(), assembly->getConstraintString()))
    return call;
  auto *callee = call->getCalledFunction();
  if (!callee || !callee->isDeclaration() || call->arg_size() != 5 ||
      callee->getName() != "llvm.hcu.mmac.f32.16x16x16.f16")
    return nullptr;
  // Only the unmodified form emitted by the Frisk lowering is covered by
  // accumulator reassociation. Keep both controls and the C input intact.
  for (unsigned i : {3u, 4u}) {
    auto *control = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(i));
    if (!control || !control->getType()->isIntegerTy(1) || !control->isZero())
      return nullptr;
  }
  return call;
}

bool collectSeedComputations(llvm::Value *value, llvm::Instruction *before,
                             llvm::DominatorTree &dominance,
                             llvm::SmallVectorImpl<llvm::Instruction *> &moves) {
  if (dominance.dominates(value, before)) return true;
  auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
  if (!inst || inst->getParent() != before->getParent() ||
      (!llvm::isa<llvm::ExtractValueInst, llvm::ExtractElementInst,
                  llvm::InsertValueInst, llvm::InsertElementInst,
                  llvm::ShuffleVectorInst>(inst) &&
       inst->getOpcode() != llvm::Instruction::FMul)) return false;
  if (llvm::is_contained(moves, inst)) return true;
  for (llvm::Value *operand : inst->operands())
    if (!collectSeedComputations(operand, before, dominance, moves)) return false;
  moves.push_back(inst);
  return true;
}
} // namespace

llvm::PreservedAnalyses FuseMMACAccumulatorPass::run(
    llvm::Function &function, llvm::FunctionAnalysisManager &analyses) {
  if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP))
    return llvm::PreservedAnalyses::all();
  auto &dominance = analyses.getResult<llvm::DominatorTreeAnalysis>(function);
  unsigned fused = 0;
  for (llvm::BasicBlock &block : function) {
    for (llvm::Instruction &inst : llvm::make_early_inc_range(block)) {
      if (inst.getOpcode() != llvm::Instruction::FAdd) continue;
      for (unsigned side : {0u, 1u}) {
        auto *last = getMMAC(inst.getOperand(side));
        if (!last || !last->hasOneUse() || last->getParent() != &block) continue;
        auto *first = last;
        while (auto *previous = getMMAC(first->getArgOperand(2))) {
          if (!previous->hasOneUse() || previous->getParent() != &block) break;
          first = previous;
        }
        auto *zero = llvm::dyn_cast<llvm::Constant>(first->getArgOperand(2));
        llvm::Value *seed = inst.getOperand(1 - side);
        llvm::SmallVector<llvm::Instruction *> extracts;
        if (!zero || !zero->isNullValue() ||
            !collectSeedComputations(seed, first, dominance, extracts)) continue;
        // Move only aggregate/vector construction and unconstrained multiplies
        // of already available values (e.g. broadcast(alpha) * O). Never move
        // a load or call, or move instructions across a block boundary.
        // StrictFP functions are excluded above. Keep all MMAC
        // calls, their order, padding, side effects and convergence intact.
        for (auto *extract : extracts) extract->moveBefore(first->getIterator());
        first->setArgOperand(2, seed);
        inst.replaceAllUsesWith(last);
        inst.eraseFromParent();
        ++fused;
        break;
      }
    }
  }
  if (!fused) return llvm::PreservedAnalyses::all();
  llvm::errs() << "FuseMMACAccumulator: " << function.getName() << ": "
               << fused << " accumulator chains fused\n";
  auto preserved = llvm::PreservedAnalyses::none();
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

void fuseMMACAccumulators(llvm::Module &module, bool simplifyCFG) {
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
  // Merge constant-guarded LDS writeback blocks before looking for the
  // terminal fadd. Fusion deliberately requires a single basic block.
  if (simplifyCFG) pipeline.addPass(llvm::SimplifyCFGPass());
  pipeline.addPass(llvm::InstCombinePass());
  pipeline.addPass(FuseMMACAccumulatorPass());
  pipeline.addPass(llvm::InstCombinePass());
  for (llvm::Function &function : module)
    if (!function.isDeclaration()) pipeline.run(function, analyses);
}
} // namespace mlir::frisk
