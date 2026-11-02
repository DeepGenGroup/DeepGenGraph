#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchNextK.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "LDSAddressAnalysis.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

namespace mlir::frisk {
namespace {
using namespace llvm;
using Inst = Instruction;
using Call = CallInst;
using detail::LDSAddress;
using detail::addressNamed;
using detail::addressAssembly;
constexpr StringLiteral Marker = "frisk.cross_bn_k0";
constexpr StringLiteral SumMarker = "frisk.cross_bn_sum_anchor";
constexpr StringLiteral MatrixRead = "llvm.hcu.ds.read.m32x16.b16.v4i32.p3";

unsigned dmaWidth(Inst *i) {
  auto *a = addressAssembly(i);
  auto *c = dyn_cast<Call>(i);
  if (!a || !a->hasSideEffects() || a->getConstraintString() != "={m0},s,v,s,~{memory}" ||
      c->arg_size() != 3 || !c->use_empty() || !c->getType()->isIntegerTy(32) ||
      !c->getArgOperand(0)->getType()->isIntegerTy(32) || !c->getArgOperand(1)->getType()->isIntegerTy(32) ||
      c->getArgOperand(2)->getType() != FixedVectorType::get(Type::getInt32Ty(c->getContext()),4)) return 0;
  auto s = a->getAsmString();
  if (s == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx2 $2, $3, 0 offen lds") return 8;
  if (s == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds") return 16;
  return 0;
}
bool wait(Inst *i) {
  auto *a = addressAssembly(i);
  return a && a->hasSideEffects() && a->getAsmString() == "s_waitcnt vmcnt(0)" &&
      a->getConstraintString() == "~{memory}" && cast<Call>(i)->arg_empty();
}
bool identity(Inst *i) {
  auto *a = addressAssembly(i);
  if (!a || !a->hasSideEffects() || !a->getAsmString().empty()) return false;
  auto *c = cast<Call>(i);
  bool tied = (a->getConstraintString() == "=v,0" && c->arg_size() == 1) ||
      (a->getConstraintString() == "=v,0,i" && c->arg_size() == 2 && isa<ConstantInt>(c->getArgOperand(1)));
  return tied && c->getArgOperand(0)->getType() == i->getType() && c->doesNotAccessMemory();
}
bool pure(Inst *i) {
  return !isa<CallBase, PHINode, AllocaInst>(i) && !i->isTerminator() &&
      !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects() && isSafeToSpeculativelyExecute(i);
}
bool addressOp(Inst *i) {
  if (identity(i)) return i->getType()->isIntegerTy();
  if (addressNamed(i,"llvm.amdgcn.readfirstlane.i32")) return cast<Call>(i)->arg_size() == 1;
  return pure(i) && (isa<GetElementPtrInst, CastInst, InsertElementInst, ExtractElementInst, ShuffleVectorInst>(i) ||
      (isa<BinaryOperator>(i) && i->getType()->isIntOrIntVectorTy()));
}
bool mma(Inst *i) {
  auto *a = addressAssembly(i);
  return a && !a->hasSideEffects() && cast<Call>(i)->doesNotAccessMemory() &&
      isRegisterMMA(a->getAsmString(),a->getConstraintString());
}
bool fence(Inst *i, AtomicOrdering order) {
  auto *f = dyn_cast_or_null<FenceInst>(i);
  return f && f->getOrdering() == order &&
      f->getSyncScopeID() == f->getContext().getOrInsertSyncScopeID("workgroup");
}
struct Sync { Inst *release, *barrier, *acquire, *wait; };
std::optional<Sync> sync(Inst *i) {
  if (!addressNamed(i,"llvm.amdgcn.s.barrier") || !cast<Call>(i)->arg_empty() ||
      !fence(i->getPrevNode(),AtomicOrdering::Release) || !fence(i->getNextNode(),AtomicOrdering::Acquire)) return {};
  auto *r = i->getPrevNode();
  return Sync{r,i,i->getNextNode(),r->getPrevNode() && wait(r->getPrevNode()) ? r->getPrevNode() : nullptr};
}
bool uniform(Value *v, SmallPtrSetImpl<Value *> &seen) {
  if (isa<Constant,Argument>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i) return false;
  if (isa<Call>(i)) return addressNamed(i,"llvm.amdgcn.workgroup.id.x") ||
      addressNamed(i,"llvm.amdgcn.workgroup.id.y") || addressNamed(i,"llvm.amdgcn.workgroup.id.z");
  if (!isa<PHINode>(i) && !pure(i)) return false;
  return llvm::all_of(i->operands(),[&](Value *op) { return uniform(op,seen); });
}

struct Plan {
  BasicBlock *entry = nullptr, *header = nullptr, *body = nullptr;
  PHINode *iv = nullptr;
  BinaryOperator *next = nullptr;
  Value *limit = nullptr;
  ICmpInst::Predicate continuePredicate = ICmpInst::ICMP_ULT;
  Inst *anchor = nullptr;
  SmallVector<Call *> copies;
  SmallVector<Inst *> primeDeps, nextDeps, sums;
  unsigned overlap = 0, bytes = 0;
};

bool dmaBytes(Call *c, LDSAddress &addresses, unsigned threads, unsigned capacity, BitVector &bytes) {
  unsigned width = dmaWidth(c);
  if (!width) return false;
  bytes = BitVector(capacity);
  for (unsigned t = 0; t < threads; ++t) {
    auto base = addresses.get(c->getArgOperand(0),t);
    auto leader = addresses.get(c->getArgOperand(0),t/64*64);
    if (!base || base != leader || *base > capacity) return false;
    uint64_t start = *base + t%64*width;
    if (start > capacity || capacity-start < width) return false;
    for (unsigned b = start; b < start+width; ++b) {
      if (bytes[b]) return false;
      bytes.set(b);
    }
  }
  return true;
}
bool readBytes(Inst *i, LDSAddress &addresses, unsigned threads, unsigned capacity, BitVector &bytes) {
  Value *ptr = nullptr, *offset = nullptr;
  unsigned width = 0;
  if (auto *l = dyn_cast<LoadInst>(i)) {
    if (!l->isSimple() || l->getPointerAddressSpace() != 3) return false;
    auto size = i->getModule()->getDataLayout().getTypeStoreSize(l->getType());
    if (size.isScalable() || size.getFixedValue() > capacity) return false;
    width = size.getFixedValue(); ptr = l->getPointerOperand();
  } else if (addressNamed(i,MatrixRead)) {
    auto *c = cast<Call>(i);
    if (c->arg_size() != 2 || !c->getArgOperand(0)->getType()->isPointerTy() ||
        c->getArgOperand(0)->getType()->getPointerAddressSpace() != 3 ||
        c->getType() != FixedVectorType::get(Type::getInt32Ty(c->getContext()),4)) return false;
    ptr = c->getArgOperand(0); offset = c->getArgOperand(1); width = 16;
  } else return false;
  bytes = BitVector(capacity);
  for (unsigned t = 0; t < threads; ++t) {
    auto a = addresses.get(ptr,t), o = offset ? addresses.get(offset,t) : std::optional<uint64_t>(0);
    if (!a || !o || *a > capacity || *o > capacity-*a || width > capacity-*a-*o) return false;
    for (unsigned b = *a+*o; b < *a+*o+width; ++b) bytes.set(b);
  }
  return true;
}
bool knownCall(Inst *i) {
  if (mma(i) || identity(i)) return true;
  if (addressNamed(i,"llvm.amdgcn.readfirstlane.i32")) return true;
  auto *c = dyn_cast<IntrinsicInst>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall() || !c->doesNotAccessMemory()) return false;
  auto name = c->getCalledFunction()->getName();
  return name.starts_with("llvm.amdgcn.exp2.") || name.starts_with("llvm.maxnum.") ||
      name.starts_with("llvm.maximum.") || name.starts_with("llvm.fabs.") ||
      name == "llvm.amdgcn.ds.bpermute" || name == "llvm.amdgcn.mbcnt.lo" ||
      name == "llvm.amdgcn.mbcnt.hi";
}
bool hint(Inst *i) {
  if (!addressNamed(i,"llvm.amdgcn.sched.barrier")) return false;
  auto *c = cast<Call>(i);
  auto *v = c->arg_size() == 1 ? dyn_cast<ConstantInt>(c->getArgOperand(0)) : nullptr;
  return v && v->isZero();
}

bool plan(Function &f, Loop &loop, DominatorTree &dt, GlobalVariable *storage,
          unsigned threads, unsigned capacity, Plan &p, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  // Match a two-block loop with a proven nonempty, nonwrapping trip range.
  // Causal exports use IV <= queryTile (often inverted to an exit test).
  // Reuse that exact bound for next-K0; no speculative read past the diagonal.
  p.header = loop.getHeader(); p.body = loop.getLoopLatch(); p.entry = loop.getLoopPreheader();
  if (loop.getParentLoop() || loop.getNumBlocks() != 2 || !p.body || p.body == p.header || !p.entry ||
      p.entry != &f.getEntryBlock()) return fail("requires an entry preheader and one compute/latch block");
  auto *branch = dyn_cast<CondBrInst>(p.header->getTerminator());
  auto *back = dyn_cast<UncondBrInst>(p.body->getTerminator());
  auto *cmp = branch ? dyn_cast<ICmpInst>(branch->getCondition()) : nullptr;
  if (!cmp || !back || back->getSuccessor(0) != p.header)
    return fail("requires canonical unsigned loop condition");
  bool continueOnTrue = branch->getSuccessor(0) == p.body;
  if (branch->getSuccessor(continueOnTrue ? 0 : 1) != p.body ||
      loop.contains(branch->getSuccessor(continueOnTrue ? 1 : 0)))
    return fail("requires one loop exit");
  auto predicate = continueOnTrue ? cmp->getPredicate() : cmp->getInversePredicate();
  unsigned ivOperand = isa<PHINode>(cmp->getOperand(0)) ? 0 : 1;
  if (ivOperand) predicate = ICmpInst::getSwappedPredicate(predicate);
  if (predicate != ICmpInst::ICMP_ULT && predicate != ICmpInst::ICMP_ULE)
    return fail("requires canonical unsigned loop condition");
  p.continuePredicate = predicate;
  p.iv = dyn_cast<PHINode>(cmp->getOperand(ivOperand));
  p.limit = cmp->getOperand(1-ivOperand);
  if (!p.iv || p.iv->getParent() != p.header || p.iv->getNumIncomingValues() != 2 ||
      !loop.isLoopInvariant(p.limit)) return fail("requires invariant loop bound");
  if (auto *limitInst = dyn_cast<Inst>(p.limit))
    if (!dt.dominates(limitInst,p.entry->getTerminator())) return fail("loop bound unavailable at entry");
  auto bits = computeKnownBits(p.limit,f.getParent()->getDataLayout(),nullptr,p.entry->getTerminator(),&dt);
  if (!bits.isNonNegative() || (predicate == ICmpInst::ICMP_ULT && !bits.isNonZero()))
    return fail("requires proven nonempty nonwrapping trip range");
  auto *initial = dyn_cast<ConstantInt>(p.iv->getIncomingValueForBlock(p.entry));
  p.next = dyn_cast<BinaryOperator>(p.iv->getIncomingValueForBlock(p.body));
  if (!initial || !initial->isZero() || !p.next || p.next->getParent() != p.body ||
      p.next->getOpcode() != Inst::Add || p.next->getOperand(0) != p.iv ||
      !isa<ConstantInt>(p.next->getOperand(1)) || !cast<ConstantInt>(p.next->getOperand(1))->isOne())
    return fail("requires zero-based unit-stride induction");
  for (auto &i : *p.header)
    if (!isa<PHINode,ICmpInst,CondBrInst>(i)) return fail("noncanonical loop header");
  for (auto *user : p.next->users())
    if (user != p.iv) return fail("induction advance has other uses");

  SmallVector<Sync> syncs;
  SmallPtrSet<Inst *,32> syncOps;
  for (auto &i : *p.body) if (auto s = sync(&i)) {
    syncs.push_back(*s);
    for (auto *op : {s->release,s->barrier,s->acquire,s->wait}) if (op) syncOps.insert(op);
  }
  if (syncs.size() < 3 || !syncs.front().wait || syncs.back().wait) return fail("missing K0/final synchronization");
  for (auto &i : *p.body) {
    if (&i == syncs.front().wait) break;
    if (dmaWidth(&i)) p.copies.push_back(cast<Call>(&i));
    else if (!addressOp(&i)) return fail("unknown operation before K0 ready wait");
  }
  if (p.copies.empty()) return fail("missing initial K DMA group");
  Value *resource = p.copies.front()->getArgOperand(2);
  SmallPtrSet<Value *,32> uniformSeen;
  if (!uniform(resource,uniformSeen)) return fail("nonuniform K resource");
  for (auto *c : p.copies) if (c->getArgOperand(2) != resource) return fail("mixed initial DMA resources");
  // The final matrix-read stage starts after the last wait/publication. Keep
  // the final read handoff, and keep the next iteration's original K0 wait.
  auto finalReady = llvm::find_if(llvm::reverse(syncs),[](const Sync &s) { return s.wait; });
  if (finalReady == syncs.rend() || finalReady->barrier == syncs.front().barrier)
    return fail("missing final PV ready synchronization");
  p.anchor = finalReady->acquire->getNextNode();
  if (!p.anchor || !p.anchor->comesBefore(syncs.back().release))
    return fail("final ready barrier is not followed by a matrix-read stage");
  LDSAddress addresses(f.getParent()->getDataLayout(),storage);
  BitVector k0(capacity);
  for (auto *c : p.copies) {
    BitVector bytes;
    if (!dmaBytes(c,addresses,threads,capacity,bytes) || k0.anyCommon(bytes)) return fail("unknown/overlapping K0 DMA footprint");
    k0 |= bytes;
  }
  p.bytes = k0.count();
  bool haveFinalRead = false;
  for (auto *i = p.anchor; i != p.body->getTerminator(); i = i->getNextNode()) {
    if (mma(i)) { ++p.overlap; continue; }
    if (isa<LoadInst>(i) || addressNamed(i,MatrixRead)) {
      BitVector bytes;
      if (!readBytes(i,addresses,threads,capacity,bytes) || k0.anyCommon(bytes))
        return fail("final PV reads overlap K0 or have unknown addresses");
      if (!i->comesBefore(syncs.back().release)) return fail("reader after final handoff");
      haveFinalRead |= addressNamed(i,MatrixRead);
      continue;
    }
    if (syncOps.contains(i) || pure(i) || hint(i) || identity(i)) continue;
    return fail("unknown operation/memory after prefetch point");
  }
  if (!haveFinalRead || !p.overlap) return fail("no final PV overlap window");
  // All moved global reads must remain ahead of global writes, and no unknown
  // LDS writer/async producer may survive into the final stage.
  for (auto &i : *p.body) {
    if (i.isTerminator() || dmaWidth(&i) || syncOps.contains(&i) || pure(&i) || hint(&i) || knownCall(&i)) continue;
    if (isa<LoadInst>(&i) || addressNamed(&i,MatrixRead)) {
      BitVector bytes;
      if (readBytes(&i,addresses,threads,capacity,bytes)) continue;
    }
    std::string detail; raw_string_ostream stream(detail); i.print(stream);
    return fail("unknown memory effect or unsupported compute operation: " + stream.str());
  }
  // Priming may reuse only LDS whose prologue readers have been released.
  std::optional<Sync> entryHandoff;
  for (auto &i : *p.entry) if (auto s = sync(&i)) entryHandoff = s;
  if (!entryHandoff) return fail("missing prologue reader handoff");
  bool pending = false;
  for (auto &i : *p.entry) {
    if (dmaWidth(&i)) { pending = true; continue; }
    if (wait(&i)) { pending = false; continue; }
    if (&i == entryHandoff->barrier && pending) return fail("outstanding prologue DMA at reader handoff");
    if (isa<LoadInst,StoreInst,FenceInst>(&i) || i.isTerminator() || pure(&i) || knownCall(&i) ||
        addressNamed(&i,"llvm.amdgcn.s.barrier") || addressNamed(&i,"llvm.amdgcn.workitem.id.x") ||
        addressNamed(&i,"llvm.amdgcn.workgroup.id.x") || addressNamed(&i,"llvm.amdgcn.workgroup.id.y") ||
        addressNamed(&i,"llvm.amdgcn.workgroup.id.z")) continue;
    return fail("unknown prologue operation");
  }
  for (auto *i = entryHandoff->acquire->getNextNode(); i != p.entry->getTerminator(); i = i->getNextNode())
    if (!addressOp(i)) return fail("operation after prologue handoff");
  for (auto &block : f) if (&block != p.entry && !loop.contains(&block))
    for (auto &i : block) {
      if (isa<LoadInst>(&i) && cast<LoadInst>(&i)->getPointerAddressSpace() != 1)
        return fail("LDS reader outside loop");
      if (dmaWidth(&i) || addressNamed(&i,MatrixRead)) return fail("LDS user outside loop");
    }

  // Preflight both clone DAGs, without mutating IR on a failed match. Prime
  // clones loop-local address dependencies. Future clones only IV-dependent
  // expressions, reproducing the measured sum-anchor variant's lifetimes.
  DenseMap<Value *,bool> dependent;
  dependent[p.iv] = true;
  std::function<bool(Value *)> depends = [&](Value *v) {
    auto it = dependent.find(v); if (it != dependent.end()) return it->second;
    auto *i = dyn_cast<Inst>(v);
    if (!i || !loop.contains(i)) return false;
    if (isa<PHINode>(i)) return true;
    bool result = llvm::any_of(i->operands(),[&](Value *op) { return depends(op); });
    dependent[v] = result; return result;
  };
  for (bool prime : {true,false}) {
    SmallPtrSet<Inst *,32> seen;
    auto &deps = prime ? p.primeDeps : p.nextDeps;
    std::function<bool(Value *)> visit = [&](Value *v) {
      if (v == p.iv) return true;
      auto *i = dyn_cast<Inst>(v);
      if (!i) return true;
      if (!loop.contains(i)) return dt.dominates(i,p.entry->getTerminator());
      if (i->getParent() != p.body) return false;
      if (!prime && !depends(i)) return dt.dominates(i,p.anchor) && i != p.anchor;
      if (seen.contains(i)) return true;
      if (!addressOp(i)) return false;
      for (Value *op : i->operands()) if (!visit(op)) return false;
      seen.insert(i); deps.push_back(i); return true;
    };
    for (auto *c : p.copies) for (Value *op : c->args())
      if (!visit(op)) return fail("K0 address depends on an unavailable value/effect");
  }
  bool varies = false;
  for (auto *c : p.copies) varies |= depends(c->getArgOperand(1));
  if (!varies) return fail("K DMA source is not induction-dependent");

  // Find denominator additions through the SSA recurrence, not temporary
  // names/metadata. Stop at MMAC, exp/max calls and all loop PHIs. Output
  // accumulator recurrences therefore cannot pull QK arithmetic into this set.
  SmallPtrSet<Inst *,32> seen, sums;
  std::function<void(Value *)> collect = [&](Value *v) {
    auto *i = dyn_cast<Inst>(v);
    if (!i || i->getParent() != p.body || !seen.insert(i).second) return;
    if (identity(i)) { collect(cast<Call>(i)->getArgOperand(0)); return; }
    if (isa<CallBase,PHINode>(i) || !pure(i)) return;
    if (i->getOpcode() == Inst::FAdd && i->getType()->isFloatTy() && i->comesBefore(p.anchor)) sums.insert(i);
    for (Value *op : i->operands()) collect(op);
  };
  for (auto &i : *p.header) if (auto *phi = dyn_cast<PHINode>(&i))
    if (phi != p.iv) collect(phi->getIncomingValueForBlock(p.body));
  for (auto &i : *p.body) if (sums.contains(&i)) p.sums.push_back(&i);
  if (p.sums.empty()) return fail("missing scalar FP32 sum recurrence");
  return true;
}

void apply(Plan &p) {
  auto &ctx = p.body->getContext();
  auto *tag = MDNode::get(ctx,{});
  for (auto *add : p.sums) {
    auto *ft = FunctionType::get(add->getType(),{add->getType()},false);
    auto *a = InlineAsm::get(ft,"","=v,0",true);
    IRBuilder<> builder(add->getNextNode());
    auto *anchor = builder.CreateCall(a,{add},"k0.sum.anchor");
    anchor->setDoesNotThrow(); anchor->setMemoryEffects(MemoryEffects::none());
    anchor->addFnAttr(Attribute::Convergent); anchor->addFnAttr(Attribute::WillReturn);
    anchor->setMetadata(SumMarker,tag); anchor->setDebugLoc(add->getDebugLoc());
    add->replaceUsesWithIf(anchor,[&](Use &u) { return u.getUser() != anchor; });
  }
  auto clone = [&](ArrayRef<Inst *> deps, Value *index, Inst *before, StringRef prefix) {
    DenseMap<Value *,Value *> map; map[p.iv] = index;
    SmallPtrSet<Inst *,32> allowed(deps.begin(),deps.end());
    for (auto *c : p.copies) allowed.insert(c);
    std::function<Value *(Value *)> copy = [&](Value *v) -> Value * {
      if (auto *mapped = map.lookup(v)) return mapped;
      auto *i = dyn_cast<Inst>(v);
      if (!i || !allowed.contains(i)) return v;
      auto *c = i->clone();
      for (auto &op : c->operands()) op.set(copy(op.get()));
      c->setName(prefix + i->getName()); c->insertBefore(before->getIterator()); map[i] = c;
      return c;
    };
    for (auto *c : p.copies) cast<Inst>(copy(c))->setMetadata(Marker,tag);
  };
  clone(p.primeDeps,ConstantInt::get(p.iv->getType(),0),p.entry->getTerminator(),"k0.prime.");
  p.next->moveBefore(p.anchor->getIterator());
  auto *join = p.body->splitBasicBlock(p.anchor,"k0.join");
  auto *issue = BasicBlock::Create(ctx,"k0.issue",p.body->getParent(),join);
  auto *exit = IRBuilder<>(issue).CreateBr(join);
  clone(p.nextDeps,p.next,exit,"k0.next.");
  auto *oldBr = p.body->getTerminator();
  IRBuilder<> builder(oldBr);
  auto *hasNext = builder.CreateICmp(p.continuePredicate,p.next,p.limit,"k0.has.next");
  builder.CreateCondBr(hasNext,issue,join); oldBr->eraseFromParent();
  for (auto *c : p.copies) c->eraseFromParent();
}
} // namespace

bool parseCrossBNK0Option(llvm::StringRef arg, bool &enabled, bool &consumed) {
  consumed = arg.starts_with("--cross-bn-k0");
  if (!consumed) return true;
  if (!arg.consume_front("--cross-bn-k0=") || (arg != "0" && arg != "1")) return false;
  enabled = arg == "1"; return true;
}
llvm::PreservedAnalyses PrefetchNextKPass::run(llvm::Function &f, llvm::FunctionAnalysisManager &) {
  using namespace llvm;
  auto skip = [&](StringRef why) {
    errs() << "Cross BN K0: skipped, function=" << f.getName() << ": " << why << "\n";
    return PreservedAnalyses::all();
  };
  if (f.isDeclaration() || f.hasFnAttribute(Marker)) return PreservedAnalyses::all();
  if (f.getCallingConv() != CallingConv::AMDGPU_KERNEL || f.hasFnAttribute(Attribute::OptimizeNone) ||
      f.hasFnAttribute(Attribute::StrictFP)) return skip("requires an optimizable AMDGPU kernel");
  auto *dims = f.getMetadata("reqd_work_group_size");
  if (!dims || dims->getNumOperands() != 3) return skip("requires fixed 1D wave64 workgroup");
  unsigned threads = 0;
  for (unsigned n = 0; n < 3; ++n) {
    auto *v = mdconst::dyn_extract<ConstantInt>(dims->getOperand(n));
    if (!v || v->getValue().getActiveBits() > 32 || (n && !v->isOne())) return skip("invalid workgroup dimensions");
    if (!n) threads = v->getZExtValue();
  }
  if (!threads || threads > 1024 || threads%64 ||
      f.getFnAttribute("target-features").getValueAsString().contains("+wavefrontsize32")) return skip("requires wave64");
  GlobalVariable *storage = nullptr;
  for (auto &g : f.getParent()->globals()) if (g.getAddressSpace() == 3) {
    if (storage || !g.hasInitializer()) return skip("requires one static LDS allocation");
    storage = &g;
  }
  if (!storage) return skip("missing LDS allocation");
  auto size = f.getParent()->getDataLayout().getTypeAllocSize(storage->getValueType());
  if (size.isScalable() || !size.getFixedValue() || size.getFixedValue() > 65536) return skip("unsupported LDS capacity");
  for (auto &b : f) {
    if (auto *br = dyn_cast<CondBrInst>(b.getTerminator())) {
      SmallPtrSet<Value *,32> seen;
      if (!uniform(br->getCondition(),seen)) return skip("nonuniform control flow");
    } else if (!isa<UncondBrInst,ReturnInst>(b.getTerminator())) return skip("unsupported control flow");
  }
  DominatorTree dt(f); LoopInfo loops(dt);
  if (std::distance(loops.begin(),loops.end()) != 1) return skip("requires one top-level loop");
  Plan p; std::string reason;
  if (!plan(f,**loops.begin(),dt,storage,threads,size.getFixedValue(),p,reason)) return skip(reason);
  apply(p);
  f.addFnAttr(Marker,"sum-anchor-v1");
  errs() << "Cross BN K0: packets=" << p.copies.size() << ", slot-bytes=" << p.bytes
         << ", overlap-mmac=" << p.overlap << ", sum-anchors=" << p.sums.size()
         << ", function=" << f.getName() << "\n";
  return PreservedAnalyses::none();
}
void prefetchNextK(llvm::Module &module, bool enabled) {
  if (!enabled) return;
  llvm::FunctionAnalysisManager analyses;
  PrefetchNextKPass pass;
  for (auto &f : module) pass.run(f,analyses);
}
} // namespace mlir::frisk
