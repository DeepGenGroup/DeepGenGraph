// ===========================================================================
// Pass: fuseMMACAccumulators（FuseMMACAccumulatorPass）   优化汇总.py 第 25 项，LLVM 阶段，math != strict
//
// 作用：把 MMAC 链末尾的那次 fadd 融入链的初值，省一次向量加法和一个临时值。
//   before:  c = 0; c = mmac(a0,b0,c); ...; c = mmac(an,bn,c); r = seed + c
//   after:   c = seed; c = mmac(a0,b0,c); ...; c = mmac(an,bn,c); r = c
// 这是浮点重结合：(0 + 各项) + seed 变成 seed + 各项，所以 strict 模式不开启。
//
// 会运行两次：
//   1. 早期由 optimizeRegisterMath 调用，simplifyCFG=false，保留 CFG 给后面的归约和生命周期匹配。
//   2. printLLVM22CompatibleIR 里调用，simplifyCFG=true。生命周期优化折叠了常量分支后，
//      融合需要的“同一基本块”才成立。
//
// sinkLoopLayoutConversions（第 6 项中调用）把布局转换移出循环后，O 和 C 的布局才一致，这个融合才会命中。
// ===========================================================================
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
// 识别 MMAC 调用，返回 nullptr 表示不是。接受两种形式：
//   1. 3 个参数的 inline asm（A, B, C），由 isRegisterMMA 校验字符串。
//   2. 5 个参数的 llvm.hcu.mmac.f32.16x16x16.f16，后两个控制位必须为 0。
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

// seed 必须在链头之前可用。不支配时，把它的计算前移到链头之前。
// moves：需要前移的指令，依赖顺序排列。只允许向量拼装和 fmul（如 broadcast(alpha) * O）。
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
    // inst 是链尾的 fadd。
    //   side   哪个操作数是 MMAC 链尾。
    //   last   链尾 MMAC。
    //   first  沿 C 输入往上追到的链头。中间节点必须单使用、在同一基本块。
    //   seed   fadd 的另一个操作数。
    //   zero   链头的 C 初值，必须是全 0。
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

// InstCombine 去掉 extract/insert 包装，才能看到连续的 MMAC 链；融合后再清理一次。
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
