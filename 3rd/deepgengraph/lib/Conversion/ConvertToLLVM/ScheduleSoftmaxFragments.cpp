#include "deepgengraph/Conversion/ConvertToLLVM/ScheduleSoftmaxFragments.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

namespace mlir::frisk {
namespace {
using Inst = llvm::Instruction;
using Insts = llvm::SmallVector<Inst *, 32>;
constexpr llvm::StringLiteral marker = "frisk.softmax_fragments";

// These exact, compiler-owned tied identities are scheduling/packing markers,
// not arbitrary side-effecting inline asm. Keep both identities and operands.
bool identity(Inst *inst) {
  auto *c = llvm::dyn_cast_or_null<llvm::CallInst>(inst);
  if (!c || c->arg_size() != 1 || c->getNumOperandBundles() ||
      !c->getMetadata("frisk.register_lifetimes") || !c->doesNotAccessMemory())
    return false;
  auto *a = llvm::dyn_cast<llvm::InlineAsm>(c->getCalledOperand());
  return a && a->getAsmString().empty() && a->getConstraintString() == "=v,0" &&
         !a->canThrow() && !a->isAlignStack() &&
         c->getType() == c->getArgOperand(0)->getType();
}
bool mma(Inst *inst) {
  auto *c = llvm::dyn_cast<llvm::CallInst>(inst);
  if (!c || c->arg_size() != 3 || !c->doesNotAccessMemory() ||
      c->getNumOperandBundles()) return false;
  auto *a = llvm::dyn_cast<llvm::InlineAsm>(c->getCalledOperand());
  return a && !a->hasSideEffects() && !a->canThrow() && !a->isAlignStack() &&
         isRegisterMMA(a->getAsmString(), a->getConstraintString());
}
struct Packet {
  Inst *packed = nullptr;
  Insts nodes, exps;
};

// Match one actual four-FP32 -> four-half MMAC operand, independent of names,
// tile dimensions and local-reduction chain count. Max/alpha stay outside.
bool matchPacket(llvm::FPTruncInst *cast, Packet &p) {
  auto &ctx = cast->getContext();
  if (cast->getSrcTy() != llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4) ||
      cast->getDestTy() != llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4))
    return false;
  auto *result = llvm::dyn_cast<Inst>(cast->getOperand(0));
  if (!identity(result) || !cast->hasOneUse()) return false;
  auto *packed = llvm::dyn_cast<Inst>(*cast->user_begin());
  if (!identity(packed) || packed->use_empty()) return false;
  for (auto *u : packed->users()) {
    auto *i = llvm::dyn_cast<Inst>(u);
    if (!i || i->getParent() != cast->getParent() || !mma(i) ||
        llvm::cast<llvm::CallInst>(i)->getArgOperand(0) != packed) return false;
  }
  p.packed = packed;
  p.nodes.append({cast, result, packed});
  llvm::Value *v = llvm::cast<llvm::CallInst>(result)->getArgOperand(0);
  Inst *subtract = nullptr;
  for (int lane = 3; lane >= 0; --lane) {
    auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(v);
    if (!insert) return false;
    auto *index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
    auto *exp = llvm::dyn_cast<llvm::IntrinsicInst>(insert->getOperand(1));
    if (!index || index->getZExtValue() != unsigned(lane) || !exp ||
        exp->getIntrinsicID() != llvm::Intrinsic::amdgcn_exp2 ||
        !exp->getType()->isFloatTy()) return false;
    auto *extract = llvm::dyn_cast<llvm::ExtractElementInst>(exp->getArgOperand(0));
    if (!extract || extract->getIndexOperand() != index) return false;
    auto *sub = llvm::dyn_cast<llvm::BinaryOperator>(extract->getVectorOperand());
    if (!sub || sub->getOpcode() != llvm::Instruction::FSub ||
        (subtract && subtract != sub)) return false;
    subtract = sub;
    p.nodes.append({insert, exp, extract});
    p.exps.push_back(exp);
    v = insert->getOperand(0);
  }
  if (!llvm::isa<llvm::UndefValue>(v)) return false; // includes poison
  p.nodes.push_back(subtract);
  return llvm::all_of(p.nodes, [&](Inst *i) { return i->getParent() == cast->getParent(); });
}

bool reductionNode(Inst *i) {
  return (i->getOpcode() == llvm::Instruction::FAdd && i->getType()->isFloatTy()) ||
         llvm::isa<llvm::InsertElementInst, llvm::InsertValueInst>(i);
}

// The movable set contains no memory access. Reject unknown calls/effects in
// its movement interval, even readnone calls that might throw or not return.
// Known DMA/wait/fence/read operations remain in exactly their original order;
// only register math may sink across them, never a load or a PV MMAC.
bool knownFixed(Inst *i) {
  if (llvm::isa<llvm::PHINode>(i) || i->isTerminator()) return true;
  if (auto *f = llvm::dyn_cast<llvm::FenceInst>(i))
    return f->getSyncScopeID() != llvm::SyncScope::System;
  if (auto *l = llvm::dyn_cast<llvm::LoadInst>(i))
    return l->isSimple() && l->getPointerAddressSpace() == 3;
  if (auto *c = llvm::dyn_cast<llvm::CallInst>(i)) {
    if (c->getNumOperandBundles() || c->isMustTailCall()) return false;
    if (identity(i) || mma(i)) return true;
    if (auto *f = c->getCalledFunction()) {
      auto n = f->getName();
      return n == "llvm.amdgcn.s.barrier" || n == "llvm.amdgcn.sched.barrier" ||
             n == "llvm.hcu.ds.read.m32x16.b16.v4i32.p3" ||
             n == "llvm.amdgcn.exp2.f32";
    }
    auto *a = llvm::dyn_cast<llvm::InlineAsm>(c->getCalledOperand());
    if (!a || !a->hasSideEffects() || a->canThrow() || a->isAlignStack()) return false;
    return (a->getAsmString() == "s_waitcnt vmcnt(0)" &&
            a->getConstraintString() == "~{memory}" && c->arg_empty()) ||
           (a->getAsmString() == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds" &&
            a->getConstraintString() == "={m0},s,v,s,~{memory}" && c->arg_size() == 3);
  }
  return !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects();
}

unsigned schedule(llvm::BasicBlock &block, SoftmaxFragmentMode mode) {
  llvm::SmallVector<Packet, 4> packets;
  Insts original;
  llvm::DenseMap<Inst *, unsigned> index;
  for (auto &i : block) {
    index[&i] = original.size();
    original.push_back(&i);
    if (auto *cast = llvm::dyn_cast<llvm::FPTruncInst>(&i)) {
      Packet p;
      if (matchPacket(cast, p)) packets.push_back(std::move(p));
    }
  }
  if (packets.size() < 2) return 0;
  llvm::SmallPtrSet<Inst *, 32> moving, reductions, anchors;
  Insts work;
  unsigned first = original.size(), last = 0;
  for (auto &p : packets) {
    anchors.insert(p.packed);
    for (auto *i : p.nodes) {
      if (!moving.insert(i).second) return 0; // shared/overlapping packet
      first = std::min(first, index[i]);
    }
    work.append(p.exps);
    for (auto *u : p.packed->users()) last = std::max(last, index[llvm::cast<Inst>(u)]);
  }
  // Only existing forward sum DAG and its aggregate wrappers. Never reassociate,
  // introduce +0, convert the denominator to half, or traverse a loop PHI.
  for (unsigned n = 0; n < work.size(); ++n)
    for (auto *u : work[n]->users()) {
      auto *i = llvm::dyn_cast<Inst>(u);
      if (i && i->getParent() == &block && !moving.contains(i) &&
          reductionNode(i) && reductions.insert(i).second) work.push_back(i);
    }
  for (auto *i : reductions) moving.insert(i);
  for (unsigned n = first; n <= last; ++n)
    if (!moving.contains(original[n]) && !knownFixed(original[n])) return 0;

  Insts order;
  llvm::SmallPtrSet<Inst *, 32> emitted, visiting;
  auto ready = [&](Inst *i) {
    return llvm::all_of(i->operand_values(), [&](llvm::Value *v) {
      auto *d = llvm::dyn_cast<Inst>(v);
      return !d || d->getParent() != &block || emitted.contains(d);
    });
  };
  // Deterministic ready-list ordering follows original instruction order.
  // The first fragment's exp values may seed the existing sum chains and must
  // remain alive until the next fragment. Do not replace those seeds with +0.
  auto drainSums = [&]() {
    for (auto *i : original)
      if (reductions.contains(i) && !emitted.contains(i) && ready(i)) {
        order.push_back(i);
        emitted.insert(i);
      }
  };
  std::function<bool(Inst *)> emit = [&](Inst *i) {
    if (emitted.contains(i)) return true;
    if (!moving.contains(i) || !visiting.insert(i).second) return false;
    for (auto *v : i->operand_values())
      if (auto *d = llvm::dyn_cast<Inst>(v))
        if (d->getParent() == &block && !emit(d)) return false;
    visiting.erase(i);
    // Exp-dependent sum updates that are ready precede the FP32 identity and
    // half packing. Missing inputs are never pulled from a future fragment.
    drainSums();
    if (!emitted.contains(i)) { order.push_back(i); emitted.insert(i); }
    drainSums();
    return true;
  };
  for (auto *i : original) {
    if (moving.contains(i)) {
      if (mode == SoftmaxFragmentMode::Fragment && anchors.contains(i) && !emit(i)) return 0;
      continue;
    }
    if (i->isTerminator()) {
      // Include loop-carried/live-out denominator wrappers before terminator.
      for (auto *m : original) if (moving.contains(m) && !emit(m)) return 0;
    }
    if (!llvm::isa<llvm::PHINode>(i))
      for (auto *v : i->operand_values())
        if (auto *d = llvm::dyn_cast<Inst>(v))
          if (d->getParent() == &block && !emit(d)) return 0;
    order.push_back(i);
    emitted.insert(i);
  }
  if (order.size() != original.size() || order == original) return 0;
  // Validate before mutation, including users outside the packet and block.
  llvm::DenseMap<Inst *, unsigned> proposed;
  for (auto *i : order) proposed[i] = proposed.size();
  for (auto *i : order) {
    if (llvm::isa<llvm::PHINode>(i)) continue;
    for (auto *v : i->operand_values())
      if (auto *d = llvm::dyn_cast<Inst>(v))
        if (d->getParent() == &block && proposed[d] >= proposed[i]) return 0;
  }
  auto *end = block.getTerminator();
  for (auto *i : order) if (i != end) i->moveBefore(end->getIterator());
  for (auto &p : packets) {
    llvm::IRBuilder<> b(p.packed->getNextNode());
    auto *fn = llvm::Intrinsic::getOrInsertDeclaration(block.getModule(),
                                                     llvm::Intrinsic::amdgcn_sched_barrier);
    auto *hint = b.CreateCall(fn, {b.getInt32(0)});
    hint->setMemoryEffects(llvm::MemoryEffects::inaccessibleMemOnly());
    hint->setMetadata(marker, llvm::MDNode::get(block.getContext(), {}));
  }
  return packets.size();
}
} // namespace

bool parseSoftmaxFragmentOption(llvm::StringRef arg, SoftmaxFragmentMode &mode,
                               bool &consumed) {
  consumed = arg.starts_with("--softmax-fragments");
  if (!consumed) return true;
  if (!arg.consume_front("--softmax-fragments=")) return false;
  if (arg == "off") mode = SoftmaxFragmentMode::Off;
  else if (arg == "fragment") mode = SoftmaxFragmentMode::Fragment;
  else if (arg == "pv") mode = SoftmaxFragmentMode::PV;
  else return false;
  return true;
}
llvm::PreservedAnalyses ScheduleSoftmaxFragmentsPass::run(
    llvm::Function &f, llvm::FunctionAnalysisManager &) {
  if (mode == SoftmaxFragmentMode::Off || f.isDeclaration() ||
      f.hasFnAttribute(marker) || f.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
      f.hasFnAttribute(llvm::Attribute::StrictFP)) return llvm::PreservedAnalyses::all();
  unsigned count = 0;
  for (auto &b : f) count += schedule(b, mode);
  llvm::errs() << "SoftmaxFragments: " << f.getName() << ": " << count
               << " fragments, mode=" << (mode == SoftmaxFragmentMode::PV ? "pv" : "fragment")
               << "; original FP32 sum DAG and memory/MMAC order retained\n";
  if (!count) return llvm::PreservedAnalyses::all();
  f.addFnAttr(marker, mode == SoftmaxFragmentMode::PV ? "pv" : "fragment");
  return llvm::PreservedAnalyses::none();
}
void scheduleSoftmaxFragments(llvm::Module &m, SoftmaxFragmentMode mode) {
  if (mode == SoftmaxFragmentMode::Off) return;
  llvm::FunctionAnalysisManager am;
  ScheduleSoftmaxFragmentsPass pass(mode);
  for (auto &f : m) pass.run(f, am);
}
} // namespace mlir::frisk
