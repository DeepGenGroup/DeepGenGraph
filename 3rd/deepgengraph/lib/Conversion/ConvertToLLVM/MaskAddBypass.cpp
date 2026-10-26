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
bool isView(llvm::Instruction *inst) {
  return llvm::isa<llvm::ExtractValueInst, llvm::ExtractElementInst,
                   llvm::ShuffleVectorInst>(inst);
}

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
