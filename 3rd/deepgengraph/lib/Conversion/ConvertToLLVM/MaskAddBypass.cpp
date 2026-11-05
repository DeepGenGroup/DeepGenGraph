// ===========================================================================
// Pass: bypassMaskAdds（MaskAddBypassPass）   优化汇总.py 第 20 项，LLVM 阶段，本 case 无 mask，无效果
//
// 作用：
//   mask 在“完全可见”路径上是 0，此时 S + 0 白白计算一次。把加法复制到有 mask 的路径，
//   完全可见的路径直接用 S。
//
//   before:  mask = phi [0, clear], [m, masked]
//            result = score + view(mask)
//   after:   masked 分支内算 score + view(m)
//            result = phi [score, clear], [score + view(m), masked]
//
// allowSignedZero：
//   x + 0 对 x = -0 不严格相等。true（approx）时允许忽略；false（legal）时要求指令本身有 nsz。
//   实际判断交给 LLVM 的 simplifyFAddInst。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/MaskAddBypass.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk {
namespace {
// Deliberately exclude casts, arithmetic, loads, calls and dynamic indexing.
// Constant folding below also rejects shuffles that select poison/undef lanes.
// “视图”操作：只取或重排元素，不做运算，可以在 clear 路径上常量折叠。
bool isView(llvm::Instruction *inst) {
  return llvm::isa<llvm::ExtractValueInst, llvm::ExtractElementInst,
                   llvm::ShuffleVectorInst>(inst);
}

// 一个候选 PHI 的匹配状态。
//   root     PHI：一路是全 0 常量，另一路是真实 mask。
//   clear / masked  两个前驱块：mask 全 0 的路径和真实 mask 的路径。
//   incoming masked 路径上的 mask 值。
//   zeros    root 及其视图 -> 在 clear 路径上的常量值。
//   views    root 派生的视图指令。
//   adds     需要改写的 fadd。
//   corridor 允许出现使用者的块：merge 块及其后面的直线块，最多 4 个。
//   seen     已访问的使用者。
struct Candidate {
  llvm::PHINode *root;
  llvm::DominatorTree &dt;
  bool allowSignedZero;
  llvm::BasicBlock *masked = nullptr, *clear = nullptr;
  llvm::Value *incoming = nullptr;
  llvm::DenseMap<llvm::Value *, llvm::Constant *> zeros;
  llvm::SmallVector<llvm::Instruction *> views;
  llvm::SmallVector<llvm::BinaryOperator *> adds;
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> corridor;
  llvm::SmallPtrSet<llvm::Instruction *, 32> seen;

  // 计算 value 在 clear 路径上的常量值，算不出就返回 nullptr。结果缓存到 zeros。
  llvm::Constant *onClear(llvm::Value *value, unsigned depth = 0) {
    if (depth > 64 || zeros.size() > 256) return nullptr;
    if (auto *c = llvm::dyn_cast<llvm::Constant>(value)) return c;
    if (auto it = zeros.find(value); it != zeros.end()) return it->second;
    auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
    if (!inst || !isView(inst) || !corridor.contains(inst->getParent()))
      return nullptr;
    llvm::SmallVector<llvm::Constant *> operands;
    for (auto &operand : inst->operands()) {
      auto *constant = onClear(operand, depth + 1);
      if (!constant) return nullptr;
      operands.push_back(constant);
    }
    auto *result = llvm::ConstantFoldInstOperands(
        inst, operands, root->getModule()->getDataLayout(), nullptr, false);
    if (result) zeros[value] = result;
    return result;
  }

  // 匹配分三步：找到“0 + mask”形状的 PHI；划定 corridor；遍历 root 的使用者，把它们分成视图和加法，
  // 其它使用者一律导致失败。
  bool match() {
    if (root->getNumIncomingValues() != 2) return false;
    unsigned zeroIndex = 0;
    auto *zero = llvm::dyn_cast<llvm::Constant>(root->getIncomingValue(0));
    if (!zero || !zero->isNullValue()) {
      zeroIndex = 1;
      zero = llvm::dyn_cast<llvm::Constant>(root->getIncomingValue(1));
    }
    if (!zero || !zero->isNullValue()) return false;
    clear = root->getIncomingBlock(zeroIndex);
    masked = root->getIncomingBlock(1 - zeroIndex);
    incoming = root->getIncomingValue(1 - zeroIndex);
    auto *merge = root->getParent();
    // No backedges, duplicate edges, invokes or speculation into other exits.
    if (clear == masked || !dt.isReachableFromEntry(merge) ||
        dt.dominates(merge, clear) || dt.dominates(merge, masked)) return false;
    auto *branch = llvm::dyn_cast<llvm::UncondBrInst>(masked->getTerminator());
    if (!branch || branch->getSuccessor(0) != merge)
      return false;
    if (!llvm::isa<llvm::UncondBrInst, llvm::CondBrInst>(clear->getTerminator())) return false;
    // Accept both immediate consumers and the empty join block emitted by
    // Frisk. Do not walk into loops or unrelated control-flow regions.
    auto *block = merge;
    for (unsigned n = 0; n != 4; ++n) {
      if (!corridor.insert(block).second) break;
      auto *next = llvm::dyn_cast<llvm::UncondBrInst>(block->getTerminator());
      if (!next ||
          next->getSuccessor(0)->getSinglePredecessor() != block) break;
      block = next->getSuccessor(0);
    }
    zeros[root] = zero;
    llvm::SmallVector<llvm::Value *> pending{root};
    while (!pending.empty()) {
      auto *value = pending.pop_back_val();
      for (auto *user : value->users()) {
        auto *inst = llvm::dyn_cast<llvm::Instruction>(user);
        if (!inst || !corridor.contains(inst->getParent())) return false;
        if (!seen.insert(inst).second) continue;
        if (seen.size() > 256) return false;
        if (isView(inst)) {
          if (!onClear(inst)) return false;
          views.push_back(inst);
          pending.push_back(inst);
          continue;
        }
        auto *add = llvm::dyn_cast<llvm::BinaryOperator>(inst);
        if (!add || add->getOpcode() != llvm::Instruction::FAdd ||
            (add->getType()->isVectorTy() &&
             !llvm::isa<llvm::FixedVectorType>(add->getType()))) return false;
        adds.push_back(add);
      }
    }
    if (adds.empty()) return false;
    // Preserve the original score-fragment order rather than LLVM use-list
    // order. This keeps the standalone experiment's scheduling input stable.
    llvm::sort(adds, [&](auto *lhs, auto *rhs) {
      return lhs->getParent() == rhs->getParent()
          ? lhs->comesBefore(rhs)
          : dt.dominates(lhs->getParent(), rhs->getParent());
    });
    // side：哪个操作数来自 mask。score：另一个操作数。
    // score 必须支配两条路径的跳转，才能作为 PHI 的入值；同时必须确认 LLVM 判定 score + 0 可以化简为 score。
    for (auto *add : adds) {
      unsigned side = zeros.count(add->getOperand(0)) ? 0 : 1;
      auto *constant = onClear(add->getOperand(side));
      auto *score = add->getOperand(1 - side);
      if (!constant || !constant->isNullValue() || zeros.count(score) ||
          !dt.dominates(score, masked->getTerminator()) ||
          !dt.dominates(score, clear->getTerminator())) return false;
      auto flags = add->getFastMathFlags();
      if (allowSignedZero) flags.setNoSignedZeros();
      llvm::SimplifyQuery query(root->getModule()->getDataLayout(), &dt,
                                nullptr, add, true, false);
      // Use LLVM's legality rules, including the function's FP environment.
      auto *folded = side == 0
          ? llvm::simplifyFAddInst(constant, score, flags, query)
          : llvm::simplifyFAddInst(score, constant, flags, query);
      if (folded != score) return false;
    }
    return true;
  }

  unsigned rewrite() {
    // mapped：原值 -> masked 路径上的对应值。
    // materialize：把视图链克隆到 masked 块末尾。
    llvm::DenseMap<llvm::Value *, llvm::Value *> mapped;
    mapped[root] = incoming;
    auto materialize = [&](auto &&self, llvm::Value *value) -> llvm::Value * {
      if (auto it = mapped.find(value); it != mapped.end()) return it->second;
      if (!zeros.count(value)) return value;
      auto *inst = llvm::cast<llvm::Instruction>(value);
      auto *clone = inst->clone();
      for (unsigned i = 0; i != clone->getNumOperands(); ++i)
        clone->setOperand(i, self(self, inst->getOperand(i)));
      clone->setName(inst->getName() + ".masked");
      clone->insertBefore(masked->getTerminator()->getIterator());
      mapped[value] = clone;
      return clone;
    };
    // 每条加法：克隆到 masked 块，在 merge 块建 PHI，替换原加法。
    for (auto *add : adds) {
      unsigned side = zeros.count(add->getOperand(0)) ? 0 : 1;
      auto *score = add->getOperand(1 - side);
      auto *clone = llvm::cast<llvm::BinaryOperator>(add->clone());
      clone->setOperand(side, materialize(materialize, add->getOperand(side)));
      clone->setName(add->getName() + ".masked");
      clone->insertBefore(masked->getTerminator()->getIterator());
      auto *phi = llvm::PHINode::Create(add->getType(), 2, "mask.bypass",
                                        root->getIterator());
      phi->setDebugLoc(add->getDebugLoc());
      phi->addIncoming(clone, masked);
      phi->addIncoming(score, clear);
      add->replaceAllUsesWith(phi);
      phi->takeName(add);
      add->eraseFromParent();
    }
    // Only delete the matched views/PHI, never recursively delete calls or
    // alter arithmetic elsewhere in the kernel as a side effect of cleanup.
    bool changed;
    do {
      changed = false;
      for (auto *&view : views)
        if (view && view->use_empty()) {
          view->eraseFromParent();
          view = nullptr;
          changed = true;
        }
    } while (changed);
    if (root->use_empty()) root->eraseFromParent();
    return adds.size();
  }
};
} // namespace

llvm::PreservedAnalyses MaskAddBypassPass::run(
    llvm::Function &function, llvm::FunctionAnalysisManager &analyses) {
  if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP) ||
      function.hasFnAttribute(llvm::Attribute::OptimizeNone))
    return llvm::PreservedAnalyses::all();
  auto &dt = analyses.getResult<llvm::DominatorTreeAnalysis>(function);
  llvm::SmallVector<llvm::WeakTrackingVH> roots;
  for (auto &block : function)
    for (auto &phi : block.phis()) roots.emplace_back(&phi);
  unsigned groups = 0, adds = 0;
  for (auto &handle : roots) {
    auto *root = llvm::dyn_cast_or_null<llvm::PHINode>(handle);
    if (!root) continue;
    Candidate candidate{root, dt, allowSignedZero};
    if (candidate.match()) {
      adds += candidate.rewrite();
      ++groups;
    }
  }
  llvm::errs() << "MaskAddBypass: " << function.getName() << ": " << groups
               << " groups, " << adds << " additions bypassed ("
               << (allowSignedZero ? "approx" : "legal") << ")\n";
  if (!groups) return llvm::PreservedAnalyses::all();
  auto preserved = llvm::PreservedAnalyses::none();
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

void bypassMaskAdds(llvm::Module &module, bool allowSignedZero) {
  llvm::FunctionAnalysisManager functions;
  llvm::ModuleAnalysisManager modules;
  llvm::LoopAnalysisManager loops;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(modules);
  pb.registerLoopAnalyses(loops);
  pb.registerCGSCCAnalyses(cgscc);
  pb.registerFunctionAnalyses(functions);
  pb.crossRegisterProxies(loops, functions, cgscc, modules);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(MaskAddBypassPass(allowSignedZero));
  for (auto &function : module)
    if (!function.isDeclaration()) pipeline.run(function, functions);
}
} // namespace mlir::frisk
