#include "deepgengraph/Conversion/ConvertToLLVM/InterleavePV.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

namespace mlir::frisk {
namespace {
using Inst = llvm::Instruction;
using Call = llvm::CallInst;
using Insts = llvm::SmallVector<Inst *, 32>;
constexpr llvm::StringLiteral Marker = "frisk.pv_interleave";

bool isMMAC(Inst *i) {
  auto *c = llvm::dyn_cast_or_null<Call>(i);
  if (!c || c->arg_size() != 3 || c->hasOperandBundles() ||
      c->isMustTailCall() || !c->doesNotAccessMemory()) return false;
  auto *a = llvm::dyn_cast<llvm::InlineAsm>(c->getCalledOperand());
  auto &ctx = c->getContext();
  auto *ab = llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4);
  auto *cd = llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
  return a && !a->hasSideEffects() && !a->isAlignStack() && !a->canThrow() &&
      isRegisterMMA(a->getAsmString(), a->getConstraintString()) &&
      c->getType() == cd && c->getArgOperand(0)->getType() == ab &&
      c->getArgOperand(1)->getType() == ab && c->getArgOperand(2)->getType() == cd;
}

bool namedCall(Inst *i, llvm::StringRef name) {
  auto *c = llvm::dyn_cast<Call>(i);
  return c && !c->hasOperandBundles() && !c->isMustTailCall() &&
      c->getCalledFunction() && c->getCalledFunction()->isDeclaration() &&
      c->getCalledFunction()->getName() == name;
}

bool isHint(Inst *i) {
  if (!namedCall(i, "llvm.amdgcn.sched.barrier")) return false;
  auto *c = llvm::cast<Call>(i);
  auto *mask = c->arg_size() == 1 ? llvm::dyn_cast<llvm::ConstantInt>(c->getArgOperand(0)) : nullptr;
  return mask && mask->isZero();
}

// Only the known low/high half of one matrix packet is accepted. No arbitrary
// arithmetic, selects or multi-packet shuffles can disguise the producer.
Call *matrixPacket(llvm::Value *v) {
  auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
  if (!shuffle || !llvm::isa<llvm::PoisonValue>(shuffle->getOperand(1))) return nullptr;
  auto mask = shuffle->getShuffleMask();
  if (mask.size() != 4 || (mask[0] != 0 && mask[0] != 4)) return nullptr;
  for (unsigned j = 1; j < 4; ++j)
    if (mask[j] != mask[0]+int(j)) return nullptr;
  auto *cast = llvm::dyn_cast<llvm::BitCastInst>(shuffle->getOperand(0));
  auto *read = cast ? llvm::dyn_cast<Call>(cast->getOperand(0)) : nullptr;
  if (!read || !namedCall(read, "llvm.hcu.ds.read.m32x16.b16.v4i32.p3") ||
      read->arg_size() != 2) return nullptr;
  auto *pointer = llvm::dyn_cast<llvm::PointerType>(read->getArgOperand(0)->getType());
  return pointer && pointer->getAddressSpace() == 3 &&
      read->getType() == llvm::FixedVectorType::get(llvm::Type::getInt32Ty(read->getContext()), 4) &&
      read->getArgOperand(1)->getType()->isIntegerTy(32) ? read : nullptr;
}

bool isPure(Inst *i) {
  return !llvm::isa<llvm::CallBase, llvm::PHINode, llvm::AllocaInst>(i) &&
      !i->isTerminator() && !i->mayReadOrWriteMemory() &&
      !i->mayHaveSideEffects() && llvm::isSafeToSpeculativelyExecute(i);
}

// These operations stay in order and at the same MMAC ordinal. Unknown calls,
// stores, atomics, reads and nonzero scheduling masks reject the whole region.
bool isFixed(Inst *i) {
  if (auto *f = llvm::dyn_cast<llvm::FenceInst>(i))
    return f->getSyncScopeID() == f->getContext().getOrInsertSyncScopeID("workgroup") &&
        (f->getOrdering() == llvm::AtomicOrdering::Acquire ||
         f->getOrdering() == llvm::AtomicOrdering::Release);
  if (namedCall(i, "llvm.amdgcn.s.barrier")) return llvm::cast<Call>(i)->arg_empty();
  // Fresh MyTest export still contains DMA-address readfirstlane calls that
  // O3 folds/hoists in the archived baseline. Keep these wave operations fixed;
  // never treat them as speculative scalar arithmetic or move across a fence.
  if (namedCall(i, "llvm.amdgcn.readfirstlane.i32")) {
    auto *c = llvm::cast<Call>(i);
    return c->arg_size() == 1 && c->getType()->isIntegerTy(32) &&
        c->getArgOperand(0)->getType()->isIntegerTy(32);
  }
  auto *c = llvm::dyn_cast<Call>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall()) return false;
  auto *a = llvm::dyn_cast<llvm::InlineAsm>(c->getCalledOperand());
  return a && !a->canThrow() && !a->isAlignStack() && a->hasSideEffects() &&
      a->getAsmString() == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds" &&
      a->getConstraintString() == "={m0},s,v,s,~{memory}" && c->arg_size() == 3;
}

struct Chain {
  llvm::SmallVector<Call *, 16> mmas, packets;
};

Chain getChain(Call *root) {
  Chain chain;
  for (Call *c = root; c;) {
    if (chain.mmas.size() == 32 || c->getMetadata(Marker)) return {};
    auto *packet = matrixPacket(c->getArgOperand(1));
    if (!packet || packet->getParent() != root->getParent() ||
        llvm::is_contained(chain.packets, packet)) return {};
    chain.mmas.push_back(c);
    chain.packets.push_back(packet);
    Call *next = nullptr;
    for (auto *u : c->users()) {
      auto *candidate = llvm::dyn_cast<Call>(u);
      if (isMMAC(candidate) && candidate->getArgOperand(2) == c) {
        if (next || !c->hasOneUse() || candidate->getParent() != root->getParent()) return {};
        next = candidate;
      }
    }
    c = next;
  }
  return chain;
}

bool schedule(llvm::ArrayRef<Chain *> chains, unsigned width) {
  if (chains.size() < 2 || chains.size() > 16 || chains.size() % width) return false;
  auto *block = chains.front()->mmas.front()->getParent();
  llvm::DenseMap<Inst *, unsigned> positions;
  unsigned pos = 0;
  for (auto &i : *block) positions[&i] = pos++;
  llvm::SmallPtrSet<Inst *, 32> mmas;
  Inst *lastRead = nullptr, *lastMMA = nullptr;
  unsigned firstMMA = positions.size();
  for (auto *read : chains.front()->packets)
    if (!lastRead || positions[read] > positions[lastRead]) lastRead = read;
  for (auto *chain : chains)
    for (auto *c : chain->mmas) {
      mmas.insert(c);
      firstMMA = std::min(firstMMA, positions[c]);
      if (!lastMMA || positions[c] > positions[lastMMA]) lastMMA = c;
    }
  if (!lastRead || positions[lastRead] >= firstMMA) return false;
  // Restrict planning to a small, single basic-block window.
  if (positions[lastMMA]-positions[lastRead] > 2048) return false;
  Inst *end = lastMMA->getNextNode();
  // Include an existing trailing hint, so a second run cannot multiply guards.
  while (isHint(end)) end = end->getNextNode();
  Insts region, hints;
  llvm::SmallPtrSet<Inst *, 32> members;
  llvm::SmallVector<Insts, 32> fixed(mmas.size()+1);
  unsigned count = 0;
  for (auto *i = lastRead->getNextNode(); i != end; i = i->getNextNode()) {
    region.push_back(i);
    members.insert(i);
    if (mmas.contains(i)) ++count;
    else if (isHint(i)) hints.push_back(i);
    else if (!isPure(i)) {
      if (!isFixed(i)) return false;
      fixed[count].push_back(i);
    }
  }
  // Fully validate a topological schedule BEFORE mutating the IR. A requested
  // MMAC cannot pull a different MMAC, memory operation or synchronization
  // forward as a dependency; an unsupported dependency rejects the region.
  Insts desired, plan;
  for (unsigned group = 0; group < chains.size(); group += width)
    for (unsigned k = 0; k < chains.front()->mmas.size(); ++k)
      for (unsigned c = group; c < group+width; ++c)
        desired.push_back(chains[c]->mmas[k]);
  llvm::SmallPtrSet<Inst *, 32> emitted, visiting;
  std::function<bool(Inst *, bool)> emit = [&](Inst *i, bool explicitOp) {
    if (!members.contains(i) || emitted.contains(i)) return true;
    if ((!explicitOp && !isPure(i)) || !visiting.insert(i).second) return false;
    for (auto &operand : i->operands())
      if (auto *dep = llvm::dyn_cast<Inst>(operand.get()))
        if (!emit(dep, false)) return false;
    visiting.erase(i);
    emitted.insert(i);
    plan.push_back(i);
    return true;
  };
  for (unsigned slot = 0; slot <= desired.size(); ++slot) {
    for (auto *i : fixed[slot]) if (!emit(i, true)) return false;
    if (slot < desired.size() && !emit(desired[slot], true)) return false;
  }
  for (auto *i : region)
    if (!isHint(i) && !emit(i, false)) return false;
  if (plan.size()+hints.size() != region.size()) return false;

  for (auto *i : plan) i->moveBefore(end->getIterator());
  for (auto *i : hints) i->eraseFromParent();
  auto *hint = llvm::Intrinsic::getOrInsertDeclaration(block->getModule(), llvm::Intrinsic::amdgcn_sched_barrier);
  auto *marker = llvm::MDNode::get(block->getContext(), llvm::ConstantAsMetadata::get(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(block->getContext()), width)));
  for (auto *i : desired) {
    llvm::IRBuilder<> before(i), after(i->getNextNode());
    for (auto *guard : {before.CreateCall(hint, {before.getInt32(0)}),
                        after.CreateCall(hint, {after.getInt32(0)})})
      guard->setMemoryEffects(llvm::MemoryEffects::inaccessibleMemOnly());
    i->setMetadata(Marker, marker);
  }
  return true;
}
} // namespace

bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width, bool &consumed) {
  consumed = argument.starts_with("--pv-interleave");
  if (!consumed) return true;
  unsigned parsed;
  if (!argument.consume_front("--pv-interleave=") || argument.getAsInteger(10, parsed) ||
      (parsed != 0 && parsed != 1 && parsed != 2 && parsed != 4)) return false;
  width = parsed;
  return true;
}

llvm::PreservedAnalyses InterleavePVPass::run(llvm::Function &function, llvm::FunctionAnalysisManager &) {
  if (width != 1 && width != 2 && width != 4) return llvm::PreservedAnalyses::all();
  unsigned changed = 0;
  for (auto &block : function) {
    llvm::SmallVector<Chain, 16> candidates;
    for (auto &i : block) {
      if (!isMMAC(&i)) continue;
      auto *call = llvm::cast<Call>(&i);
      if (isMMAC(llvm::dyn_cast<Inst>(call->getArgOperand(2)))) continue;
      auto chain = getChain(call);
      if (chain.mmas.size() >= 2) candidates.push_back(std::move(chain));
    }
    llvm::SmallPtrSet<Chain *, 16> visited;
    for (auto &seed : candidates) {
      if (!visited.insert(&seed).second) continue;
      llvm::SmallVector<Chain *, 8> group{&seed};
      for (auto &other : candidates)
        if (!visited.contains(&other) && seed.packets == other.packets) {
          group.push_back(&other);
          visited.insert(&other);
        }
      changed += schedule(group, width);
    }
  }
  llvm::errs() << "PV interleave: " << changed << " regions, width=" << width
               << ", function=" << function.getName() << "\n";
  if (!changed) return llvm::PreservedAnalyses::all();
  auto preserved = llvm::PreservedAnalyses::none();
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

void interleavePVAccumulators(llvm::Module &module, unsigned width) {
  if (!width) return;
  llvm::FunctionAnalysisManager analyses;
  llvm::PassBuilder pb;
  pb.registerFunctionAnalyses(analyses);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(InterleavePVPass(width));
  for (auto &function : module)
    if (!function.isDeclaration()) pipeline.run(function, analyses);
}
} // namespace mlir::frisk
