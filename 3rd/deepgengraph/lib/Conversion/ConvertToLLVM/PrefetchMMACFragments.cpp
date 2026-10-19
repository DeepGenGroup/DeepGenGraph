#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk {
namespace {
using Inst = llvm::Instruction;
using Insts = llvm::SmallVector<Inst *, 32>;

bool isMMAC(Inst *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  if (!call || call->getNumOperandBundles() || call->isMustTailCall() ||
      !call->doesNotAccessMemory() || call->arg_size() < 3)
    return false;
  auto &ctx = call->getContext();
  auto *ab = llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4);
  auto *cd = llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
  if (call->getType() != cd || call->getArgOperand(0)->getType() != ab ||
      call->getArgOperand(1)->getType() != ab || call->getArgOperand(2)->getType() != cd)
    return false;
  if (auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand()))
    return call->arg_size() == 3 && !assembly->isAlignStack() && !assembly->canThrow() &&
           isRegisterMMA(assembly->getAsmString(), assembly->getConstraintString());
  auto *callee = call->getCalledFunction();
  if (!callee || !callee->isDeclaration() || call->arg_size() != 5 ||
      callee->getName() != "llvm.hcu.mmac.f32.16x16x16.f16")
    return false;
  for (unsigned i : {3u, 4u})
    if (!llvm::isa<llvm::ConstantInt>(call->getArgOperand(i)) ||
        !call->getArgOperand(i)->getType()->isIntegerTy(1))
      return false;
  return true;
}

// Recognize an exact target operation, never an arbitrary readonly call. A
// matrix read is one indivisible packet, including halves reused by later MMAs.
bool isMatrixRead(Inst *inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
  if (!call || call->getNumOperandBundles() || call->isMustTailCall() || call->arg_size() != 2)
    return false;
  auto *callee = call->getCalledFunction();
  auto *pointer = llvm::dyn_cast<llvm::PointerType>(call->getArgOperand(0)->getType());
  return callee && callee->isDeclaration() &&
         callee->getName() == "llvm.hcu.ds.read.m32x16.b16.v4i32.p3" &&
         pointer && pointer->getAddressSpace() == 3 &&
         call->getArgOperand(1)->getType()->isIntegerTy(32) &&
         call->getType() == llvm::FixedVectorType::get(llvm::Type::getInt32Ty(call->getContext()), 4);
}

bool isLDSRead(Inst *inst) {
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(inst))
    return load->isSimple() && load->getPointerAddressSpace() == 3 &&
           !load->getType()->isScalableTy();
  return isMatrixRead(inst);
}

bool isPure(Inst *inst) {
  return !llvm::isa<llvm::CallBase, llvm::PHINode, llvm::AllocaInst>(inst) &&
         !inst->isTerminator() && !inst->mayReadOrWriteMemory() &&
         !inst->mayHaveSideEffects() && llvm::isSafeToSpeculativelyExecute(inst);
}

uint64_t words(llvm::Value *value, const llvm::DataLayout &layout) {
  auto *type = value->getType();
  if (!type->isSized() || type->isScalableTy()) return 0;
  return (layout.getTypeSizeInBits(type).getFixedValue() + 31) / 32;
}

// Conservatively keep external inputs throughout this window. Internal values
// are counted from definition to last use; live-outs remain live at the end.
// Count pointers, indices and packing temporaries as well as loaded data. Even
// a bitcast is charged separately (the machine allocator may coalesce it).
uint64_t peakWords(llvm::ArrayRef<Inst *> order, const llvm::DataLayout &layout) {
  llvm::SmallPtrSet<Inst *, 32> members(order.begin(), order.end());
  llvm::SmallPtrSet<llvm::Value *, 32> live;
  for (auto *inst : order) {
    for (auto &operand : inst->operands()) {
      auto *def = llvm::dyn_cast<Inst>(operand.get());
      if (!llvm::isa<llvm::Constant>(operand) && (!def || !members.contains(def)))
        live.insert(operand);
    }
    if (llvm::any_of(inst->users(), [&](llvm::User *user) {
          return !members.contains(llvm::dyn_cast<Inst>(user));
        }))
      live.insert(inst);
  }
  uint64_t current = 0;
  for (auto *value : live) current += words(value, layout);
  uint64_t peak = current;
  for (auto *inst : llvm::reverse(order)) {
    if (live.erase(inst)) current -= words(inst, layout);
    for (auto &operand : inst->operands())
      if (!llvm::isa<llvm::Constant>(operand) && live.insert(operand).second)
        current += words(operand, layout);
    peak = std::max(peak, current);
  }
  return peak;
}

struct Packet {
  Inst *current;
  llvm::SmallPtrSet<Inst *, 32> visited, moving;
  llvm::SmallPtrSet<Inst *, 16> reads;

  bool available(Inst *inst) const {
    return inst->getParent() != current->getParent() || inst->comesBefore(current);
  }

  // Hoist only the raw reads and their pure address DAG, not FP preparation of
  // the next operand: doing that now would wait for the prefetched result before
  // the current MMA. Do not clone anything or traverse an indirect-memory chain.
  bool address(llvm::Value *value) {
    auto *inst = llvm::dyn_cast<Inst>(value);
    if (!inst || available(inst)) return true;
    if (inst == current || !isPure(inst)) return false;
    if (moving.contains(inst)) return true;
    if (moving.size() >= 256) return false;
    moving.insert(inst);
    return llvm::all_of(inst->operands(), [&](auto &op) { return address(op); });
  }

  bool operand(llvm::Value *value) {
    auto *inst = llvm::dyn_cast<Inst>(value);
    if (!inst || available(inst)) return true; // already available A/B reuse
    if (inst == current) return false; // next A/B depends on current accumulator
    if (!visited.insert(inst).second) return true;
    if (visited.size() > 256) return false;
    if (isLDSRead(inst)) {
      reads.insert(inst);
      moving.insert(inst);
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(inst))
        return address(load->getPointerOperand());
      auto *call = llvm::cast<llvm::CallInst>(inst);
      return llvm::all_of(call->args(), [&](auto &arg) { return address(arg); });
    }
    return isPure(inst) &&
           llvm::all_of(inst->operands(), [&](auto &op) { return operand(op); });
  }
};

void scheduleBoundary(Inst *before) {
  llvm::IRBuilder<> builder(before);
  builder.SetCurrentDebugLocation(before->getDebugLoc());
  auto *hint = llvm::Intrinsic::getOrInsertDeclaration(
      before->getModule(), llvm::Intrinsic::amdgcn_sched_barrier);
  // A scheduling hint, NOT an s_barrier/fence/wait. No guessed lgkmcnt and no
  // fake dependency on the next load's value. The backend inserts operand waits
  // after lowering, combining/splitting reads, and assigning real registers.
  auto *call = builder.CreateCall(hint, {builder.getInt32(0)});
  // Retain the scheduling side effect without claiming a write to program
  // memory. The intrinsic's default effects inhibit LDS CSE across the hint;
  // memory(none)/readonly instead lets DCE delete the hint altogether. Model
  // only compiler-private state, at this call site, not other barriers/calls.
  call->setMemoryEffects(llvm::MemoryEffects::inaccessibleMemOnly());
}

bool prefetchPair(llvm::CallInst *current, llvm::CallInst *next,
                  unsigned budget, unsigned &budgetSkips,
                  unsigned &groupSkips) {
  Packet packet{current};
  if (!packet.operand(next->getArgOperand(0)) ||
      !packet.operand(next->getArgOperand(1)) || packet.reads.empty())
    return false;
  auto &layout = current->getModule()->getDataLayout();
  uint64_t rawWords = 0;
  for (auto *load : packet.reads) rawWords += words(load, layout);
  if (rawWords > budget) { ++budgetSkips; return false; }

  Insts original, moves, proposed;
  for (Inst *inst = current;; inst = inst->getNextNode()) {
    original.push_back(inst);
    if (packet.moving.contains(inst)) moves.push_back(inst);
    // Existing scheduling may have placed two or more reads together for DS
    // combining or future operand reuse. Do not pull just one of them across
    // an MMAC and insert a hard boundary through that group. Including the
    // others would exceed the one-MMAC window, so leave this pair unchanged.
    // Deliberately conservative: no target-specific offset/shape assumptions.
    else if (isLDSRead(inst)) { ++groupSkips; return false; }
    if (inst == next) break;
    if (original.size() > 4096) return false;
  }
  if (moves.size() != packet.moving.size()) return false;
  proposed.append(moves);
  for (auto *inst : original)
    if (!packet.moving.contains(inst)) proposed.push_back(inst);
  if (peakWords(proposed, layout) > peakWords(original, layout) + budget) {
    ++budgetSkips;
    return false;
  }

  // Complete current operands -> issue complete next packet -> current MMAC.
  // Keep current and next reads in the same scheduling region so the backend
  // can combine them. Fence only the MMAC: next reads stay before it, while a
  // third packet cannot advance across it. Wait counts remain backend-owned.
  for (auto *inst : moves) inst->moveBefore(current->getIterator());
  scheduleBoundary(current);
  scheduleBoundary(current->getNextNode());
  return true;
}
} // namespace

void prefetchMMACFragments(llvm::Module &module, unsigned maxExtraWords) {
  if (!maxExtraWords) return;
  prepareRegisterMMAForLLVM(module);
  unsigned prefetched = 0, budgetSkips = 0, groupSkips = 0;
  for (auto &function : module) {
    const auto &triple = module.getTargetTriple();
    if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
        function.hasFnAttribute("frisk.frag_prefetch") ||
        (!triple.isAMDGPU() && !(triple.str().empty() &&
          function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL))) continue;
    for (auto &block : function) {
      // Collect pairs before inserting hints. All writes (even disjoint ones),
      // existing waits, barriers, shuffle, opaque calls/asm, volatile/atomic
      // accesses and control flow cut the phase. No cross-block speculation.
      llvm::SmallVector<std::pair<llvm::CallInst *, llvm::CallInst *>> pairs;
      llvm::CallInst *previous = nullptr;
      for (auto &inst : block) {
        if (isMMAC(&inst)) {
          auto *mma = llvm::cast<llvm::CallInst>(&inst);
          if (previous) pairs.emplace_back(previous, mma);
          previous = mma;
        } else if (!isPure(&inst) && !isLDSRead(&inst)) {
          previous = nullptr;
        }
      }
      for (auto [current, next] : pairs)
        prefetched += prefetchPair(current, next, maxExtraWords, budgetSkips,
                                   groupSkips);
    }
    function.addFnAttr("frisk.frag_prefetch", std::to_string(maxExtraWords));
  }
  llvm::errs() << "FragPrefetch: " << prefetched << " complete A/B packets advanced, "
               << budgetSkips << " budget skips, " << groupSkips
               << " read-group skips; one-MMAC lookahead, "
               << maxExtraWords << " extra SSA words/thread (not allocated VGPRs)\n";
}
} // namespace mlir::frisk
