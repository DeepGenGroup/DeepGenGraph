#include "deepgengraph/Conversion/ConvertToLLVM/NarrowKDMA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <map>
#include <optional>
#include <tuple>

namespace mlir::frisk {
namespace {
using namespace llvm;
using Inst = Instruction;
using Call = CallInst;
constexpr StringLiteral Marker = "frisk.k_dma";
bool named(Inst *i, StringRef name) {
  auto *c = dyn_cast_or_null<Call>(i);
  return c && !c->hasOperandBundles() && !c->isMustTailCall() &&
      c->getCalledFunction() && c->getCalledFunction()->getName() == name;
}
InlineAsm *assembly(Inst *i) {
  auto *c = dyn_cast<Call>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall()) return nullptr;
  auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
  return a && !a->canThrow() && !a->isAlignStack() ? a : nullptr;
}
bool dma(Inst *i) {
  auto *a = assembly(i);
  auto *c = dyn_cast<Call>(i);
  return a && a->hasSideEffects() && c->arg_size() == 3 && c->use_empty() &&
      c->getType()->isIntegerTy(32) && c->getArgOperand(0)->getType()->isIntegerTy(32) &&
      c->getArgOperand(1)->getType()->isIntegerTy(32) &&
      c->getArgOperand(2)->getType() == FixedVectorType::get(Type::getInt32Ty(c->getContext()),4) &&
      a->getAsmString() == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds" &&
      a->getConstraintString() == "={m0},s,v,s,~{memory}";
}
bool pure(Inst *i) {
  return !isa<CallBase, PHINode, AllocaInst>(i) && !i->isTerminator() &&
      !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects() && isSafeToSpeculativelyExecute(i);
}
bool identity(Call *c) {
  auto *a = assembly(c);
  return a && a->getAsmString().empty() && a->getConstraintString() == "=v,0" &&
      c->arg_size() == 1 && c->getArgOperand(0)->getType() == c->getType() &&
      c->getMetadata("frisk.register_lifetimes");
}
bool uniform(Value *v, SmallPtrSetImpl<Value *> &seen) {
  if (isa<Constant, Argument>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i) return false;
  if (auto *c = dyn_cast<Call>(i))
    return named(c,"llvm.amdgcn.workgroup.id.x") || named(c,"llvm.amdgcn.workgroup.id.y") ||
           named(c,"llvm.amdgcn.workgroup.id.z");
  if (!isa<PHINode>(i) && !pure(i)) return false;
  return llvm::all_of(i->operands(), [&](Value *op) { return uniform(op,seen); });
}

// Exact restricted address evaluator (same admitted operations as V buffering).
class LDSAddress {
  const DataLayout &dl;
  GlobalVariable *storage;
  std::map<std::tuple<Value *, unsigned, unsigned>, std::optional<uint64_t>> cache;
  std::optional<uint64_t> compute(Value *v, unsigned lane, unsigned element, unsigned depth) {
    if (depth > 128) return {};
    if (v == storage) return 0;
    if (auto *c = dyn_cast<ConstantInt>(v)) return c->getValue().getLimitedValue();
    if (auto *c = dyn_cast<Constant>(v)) {
      if (v->getType()->isVectorTy() && !isa<ConstantExpr>(v)) {
        auto *e = c->getAggregateElement(element);
        return e ? get(e, lane, 0, depth+1) : std::nullopt;
      }
    }
    if (auto *g = dyn_cast<GEPOperator>(v)) {
      auto base = get(g->getPointerOperand(), lane, element, depth+1);
      if (!base) return {};
      uint64_t offset = *base;
      for (auto it = gep_type_begin(g), end = gep_type_end(g); it != end; ++it) {
        auto ix = get(it.getOperand(), lane, element, depth+1);
        if (!ix) return {};
        if (auto *st = it.getStructTypeOrNull()) {
          if (*ix >= st->getNumElements()) return {};
          offset += dl.getStructLayout(st)->getElementOffset(*ix);
        } else {
          auto size = dl.getTypeAllocSize(it.getIndexedType());
          if (size.isScalable()) return {};
          unsigned bits = it.getOperand()->getType()->getScalarSizeInBits();
          if (!bits || bits > 64) return {};
          offset += APInt(bits, *ix).getSExtValue() * size.getFixedValue();
        }
      }
      return offset;
    }
    if (auto *e = dyn_cast<ExtractElementInst>(v)) {
      auto ix = get(e->getIndexOperand(), lane, 0, depth+1);
      if (!ix || *ix >= cast<FixedVectorType>(e->getVectorOperandType())->getNumElements()) return {};
      return get(e->getVectorOperand(), lane, *ix, depth+1);
    }
    if (auto *e = dyn_cast<InsertElementInst>(v)) {
      auto ix = get(e->getOperand(2), lane, 0, depth+1);
      if (!ix) return {};
      return get(e->getOperand(*ix == element ? 1 : 0), lane, *ix == element ? 0 : element, depth+1);
    }
    if (auto *s = dyn_cast<ShuffleVectorInst>(v)) {
      if (element >= s->getShuffleMask().size()) return {};
      int ix = s->getMaskValue(element);
      unsigned n = cast<FixedVectorType>(s->getOperand(0)->getType())->getNumElements();
      if (ix < 0) return {};
      return get(s->getOperand(ix >= int(n)), lane, ix % n, depth+1);
    }
    auto *op = dyn_cast<Operator>(v);
    if (op && Instruction::isCast(op->getOpcode())) {
      auto a = get(op->getOperand(0), lane, element, depth+1);
      if (!a) return {};
      unsigned bits = op->getOperand(0)->getType()->getScalarSizeInBits();
      switch (op->getOpcode()) {
      case Instruction::SExt: return bits && bits <= 64 ? std::optional<uint64_t>(APInt(bits,*a).getSExtValue()) : std::nullopt;
      case Instruction::ZExt: case Instruction::Trunc:
      case Instruction::PtrToInt: case Instruction::IntToPtr: return a;
      case Instruction::BitCast:
        if (dl.getTypeSizeInBits(op->getOperand(0)->getType()->getScalarType()) ==
            dl.getTypeSizeInBits(v->getType()->getScalarType())) return a;
        return {};
      default: return {};
      }
    }
    if (op && Instruction::isBinaryOp(op->getOpcode())) {
      auto a = get(op->getOperand(0), lane, element, depth+1);
      auto b = get(op->getOperand(1), lane, element, depth+1);
      unsigned bits = v->getType()->getScalarSizeInBits();
      if (!a || !b || !bits || bits > 64) return {};
      switch (op->getOpcode()) {
      case Instruction::Add: return *a + *b;
      case Instruction::Sub: return *a - *b;
      case Instruction::Mul: return *a * *b;
      case Instruction::And: return *a & *b;
      case Instruction::Or: return *a | *b;
      case Instruction::Xor: return *a ^ *b;
      case Instruction::Shl: return *b < bits ? std::optional<uint64_t>(*a << *b) : std::nullopt;
      case Instruction::LShr: return *b < bits ? std::optional<uint64_t>(*a >> *b) : std::nullopt;
      case Instruction::AShr: return *b < bits ? std::optional<uint64_t>(APInt(bits,*a).ashr(*b).getZExtValue()) : std::nullopt;
      case Instruction::UDiv: return *b ? std::optional<uint64_t>(*a / *b) : std::nullopt;
      case Instruction::URem: return *b ? std::optional<uint64_t>(*a % *b) : std::nullopt;
      case Instruction::SDiv: case Instruction::SRem: {
        APInt left(bits,*a), right(bits,*b);
        if (right.isZero() || (left.isMinSignedValue() && right.isAllOnes())) return {};
        return (op->getOpcode() == Instruction::SDiv ? left.sdiv(right) : left.srem(right)).getZExtValue();
      }
      default: return {};
      }
    }
    auto *c = dyn_cast<Call>(v);
    if (!c) return {};
    if (named(c, "llvm.amdgcn.workitem.id.x") && c->arg_empty()) return lane;
    if ((named(c, "llvm.amdgcn.workitem.id.y") || named(c, "llvm.amdgcn.workitem.id.z")) && c->arg_empty()) return 0;
    if (named(c, "llvm.amdgcn.readfirstlane.i32") && c->arg_size() == 1)
      return get(c->getArgOperand(0), lane/64*64, 0, depth+1);
    auto *a = assembly(c);
    if (a && (a->getConstraintString() == "=v,0" || a->getConstraintString() == "=v,0,i") &&
        a->getAsmString().empty() && c->arg_size() >= 1)
      return get(c->getArgOperand(0), lane, element, depth+1);
    return {};
  }
public:
  LDSAddress(const DataLayout &dl, GlobalVariable *storage) : dl(dl), storage(storage) {}
  std::optional<uint64_t> get(Value *v, unsigned lane, unsigned element = 0, unsigned depth = 0) {
    if (isa<ScalableVectorType>(v->getType()) || v->getType()->getScalarSizeInBits() > 64) return {};
    auto key = std::make_tuple(v,lane,element);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    cache[key] = {}; // Cyclic PHIs/unknown address loops fail closed.
    auto result = compute(v,lane,element,depth);
    unsigned bits = v->getType()->getScalarSizeInBits();
    if (result && bits && bits < 64) *result &= (uint64_t(1)<<bits)-1;
    cache[key] = result;
    return result;
  }
};


// Recompute a source-offset expression for another lane in the SAME wave.
// Uniform leaves, including key-loop PHIs, stay unchanged. No lane shuffle is
// emitted: clone only pure thread-dependent integer expressions.
class SourceMap {
  std::map<Value *, bool> varying;
public:
  bool valid = true;
  bool depends(Value *v) {
    if (auto it = varying.find(v); it != varying.end()) return it->second;
    SmallPtrSet<Value *,32> seen;
    if (uniform(v,seen)) return varying[v] = false;
    varying[v] = false;
    auto *i = dyn_cast<Inst>(v);
    if (!i) { valid = false; return false; }
    if (auto *c = dyn_cast<Call>(i)) {
      if (named(c,"llvm.amdgcn.workitem.id.x") && c->arg_empty()) return varying[v] = true;
      if (identity(c)) return varying[v] = depends(c->getArgOperand(0));
      // A leader read is unchanged by a permutation confined to its wave.
      if (named(c,"llvm.amdgcn.readfirstlane.i32") && c->arg_size()==1) return false;
      valid = false; return false;
    }
    if (!pure(i) || !i->getType()->isIntegerTy()) { valid = false; return false; }
    bool result = false;
    for (auto &op : i->operands()) result |= depends(op);
    return varying[v] = result;
  }
  Value *clone(Value *v, Value *owner, IRBuilder<> &b, DenseMap<Value *,Value *> &cache) {
    if (!depends(v)) return v;
    if (auto it = cache.find(v); it != cache.end()) return it->second;
    auto *i = cast<Inst>(v);
    if (named(i,"llvm.amdgcn.workitem.id.x")) return owner;
    if (auto *c = dyn_cast<Call>(i); c && identity(c)) return clone(c->getArgOperand(0),owner,b,cache);
    auto *n = i->clone();
    for (unsigned k=0;k<i->getNumOperands();++k) n->setOperand(k,clone(i->getOperand(k),owner,b,cache));
    // Owner is one of the original legal workitems; all integer contracts are
    // retained. Drop stale metadata about the old SSA value's range/lifetime.
    n->setMetadata(LLVMContext::MD_range,nullptr);
    b.Insert(n,"kdma.source");
    return cache[v] = n;
  }
};

bool resourceFrom(Value *v, Argument *arg, SmallPtrSetImpl<Value *> &seen, bool &found) {
  if (v == arg) { found = true; return true; }
  if (isa<Constant>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i || !pure(i)) return false;
  return llvm::all_of(i->operands(),[&](Value *op) { return resourceFrom(op,arg,seen,found); });
}
struct ReadPlan { LoadInst *load; SmallVector<uint64_t> addresses; };
struct Plan {
  Function *function;
  GlobalVariable *storage;
  unsigned threads;
  SmallVector<Call *> copies;
  SmallVector<ReadPlan> reads;
};

bool plan(Function &f, const KDMAOptions &options, Plan &p, std::string &why) {
  auto fail = [&](StringRef s) { why = s.str(); return false; };
  if (options.argument >= f.arg_size()) return fail("K ABI argument is out of range");
  auto *arg = f.getArg(options.argument);
  if (!arg->getType()->isPointerTy() || arg->getType()->getPointerAddressSpace()!=1)
    return fail("K ABI argument must be a global pointer");
  auto *dims = f.getMetadata("reqd_work_group_size");
  if (!dims || dims->getNumOperands()!=3) return fail("requires fixed 1D workgroup size");
  unsigned threads = 0;
  for (unsigned k=0;k<3;++k) {
    auto *n = mdconst::dyn_extract<ConstantInt>(dims->getOperand(k));
    if (!n || n->getValue().getActiveBits()>32 || (k && !n->isOne())) return fail("requires fixed 1D workgroup size");
    if (!k) threads = n->getZExtValue();
  }
  if (!threads || threads>1024 || threads%64 || f.getFnAttribute("target-features").getValueAsString().contains("+wavefrontsize32"))
    return fail("requires wave64 workgroups");
  GlobalVariable *storage = nullptr;
  for (auto &g : f.getParent()->globals()) if (g.getAddressSpace()==3) {
    if (storage || !g.hasInitializer()) return fail("requires one static LDS allocation");
    storage = &g;
  }
  if (!storage) return fail("missing LDS allocation");
  const auto &dl = f.getDataLayout();
  auto bytes = dl.getTypeAllocSize(storage->getValueType());
  if (bytes.isScalable() || bytes.getFixedValue()>65536) return fail("unsupported LDS allocation size");
  LDSAddress eval(dl,storage);
  DominatorTree dt(f);
  LoopInfo loops(dt);
  p.function=&f; p.storage=storage; p.threads=threads;
  // Require one compute block containing all DMA for K and all its consumers.
  // This permits a self loop or a separate PHI/condition-only loop header.
  BasicBlock *compute = nullptr;
  for (auto &i : instructions(f)) if (dma(&i)) {
    auto *c = cast<Call>(&i);
    SmallPtrSet<Value *,32> seen; bool found = false;
    if (resourceFrom(c->getArgOperand(2),arg,seen,found) && found) {
      if (compute && compute!=c->getParent()) return fail("K DMA spans multiple compute blocks");
      compute = c->getParent(); p.copies.push_back(c);
      SourceMap sources; sources.depends(c->getArgOperand(1));
      if (!sources.valid) return fail("unsupported K source-address expression");
    }
  }
  if (!compute) return fail("no asm128 DMA rooted in the selected K ABI argument");
  for (auto &b : f) {
    if (auto *br = dyn_cast<CondBrInst>(b.getTerminator())) {
      SmallPtrSet<Value *,32> seen;
      if (!uniform(br->getCondition(),seen)) return fail("requires workgroup-uniform control flow");
    } else if (!isa<UncondBrInst,ReturnInst>(b.getTerminator())) return fail("unsupported control flow");
  }
  auto *loop = loops.getLoopFor(compute);
  if (loop) {
    if (loop->getParentLoop() || loop->getNumBlocks()>2 || loop->getLoopLatch()!=compute)
      return fail("requires a single compute/latch block");
    for (auto *other : loop->blocks()) if (other!=compute)
      for (auto &i : *other) if (!pure(&i) && !isa<PHINode,CondBrInst,UncondBrInst>(&i))
        return fail("loop header is not pure");
  }
  // Prove K ownership at every memory access, without depending on debug
  // locations, SSA names, tensor dimensions or a fixed number of K slices.
  // Reject K values live across the compute block boundary.
  SmallVector<uint8_t> owner(bytes.getFixedValue(),0);
  for (auto &i : *compute) {
    if (dma(&i)) {
      auto *c = cast<Call>(&i);
      bool selected = llvm::is_contained(p.copies,c);
      BitVector covered(owner.size());
      for (unsigned t=0;t<threads;t+=64) {
        auto base = eval.get(c->getArgOperand(0),t);
        if (!base || *base%1024 || *base+1024>owner.size()) return fail("DMA wave destination must be an in-bounds aligned 1024B packet");
        for (unsigned l=1;l<64;++l)
          if (eval.get(c->getArgOperand(0),t+l)!=base) return fail("DMA m0 is not wave-uniform");
        for (unsigned k=0;k<1024;++k) {
          if (covered.test(*base+k)) return fail("overlapping DMA waves");
          covered.set(*base+k);
          owner[*base+k]=selected;
        }
      }
      continue;
    }
    auto *load = dyn_cast<LoadInst>(&i);
    if (load && load->getPointerAddressSpace()==3) {
      auto size = dl.getTypeStoreSize(load->getType());
      if (load->isVolatile() || load->isAtomic() || size.isScalable()) return fail("unsupported shared load");
      bool any=false, all=true;
      ReadPlan read{load,{}};
      for (unsigned t=0;t<threads;++t) {
        auto a=eval.get(load->getPointerOperand(),t);
        if (!a || *a+size.getFixedValue()>owner.size()) return fail("unknown shared read footprint");
        read.addresses.push_back(*a);
        for (unsigned k=0;k<size.getFixedValue();++k) { any |= owner[*a+k]; all &= owner[*a+k]; }
        if (options.mode=="64-swizzle" && all && (*a%8 || size.getFixedValue()!=8)) return fail("K swizzle requires aligned b64 consumers");
      }
      if (any && !all) return fail("mixed K/non-K ownership in a shared read");
      if (all) p.reads.push_back(std::move(read));
      continue;
    }
    if (auto *c=dyn_cast<Call>(&i); c && named(c,"llvm.hcu.ds.read.m32x16.b16.v4i32.p3")) {
      // Matrix V has a collective footprint. Be conservative: all addresses
      // in its aligned 8KiB region must have been released by K already.
      for (unsigned t=0;t<threads;++t) {
        auto a=eval.get(c->getArgOperand(0),t);
        if (!a || *a>=owner.size()) return fail("unknown matrix footprint");
        uint64_t base=*a/8192*8192, end=std::min<uint64_t>(base+8192,owner.size());
        for (auto k=base;k<end;++k) if (owner[k]) return fail("K ownership escapes to a matrix reader");
      }
      continue;
    }
    if (auto *s=dyn_cast<StoreInst>(&i); s && s->getPointerAddressSpace()==3) return fail("shared stores in the K compute block are unsupported");
    if (auto *c=dyn_cast<Call>(&i); c && c->mayReadOrWriteMemory()) {
      auto *a=assembly(c);
      if (identity(c) || named(c,"llvm.amdgcn.s.barrier") || named(c,"llvm.amdgcn.sched.barrier") ||
          (a && a->getAsmString()=="s_waitcnt vmcnt(0)")) continue;
      if (llvm::any_of(c->args(),[](Value *v) { return v->getType()->isPointerTy() && v->getType()->getPointerAddressSpace()==3; }) ||
          (a && a->hasSideEffects())) return fail("unknown shared-memory side effect");
    }
  }
  if (p.reads.empty()) return fail("no ordinary K LDS consumers");
  if (llvm::any_of(owner,[](uint8_t x) { return x!=0; })) return fail("K lifetime extends beyond compute block");
  return true;
}

void apply(Plan &p, const KDMAOptions &options) {
  auto &f=*p.function;
  bool swizzle=options.mode=="64-swizzle";
  // One lane remapping in the entry block; full-wave owner never leaves its
  // original wave. SourceMap handles arbitrary supported source strides.
  IRBuilder<> entry(&*f.getEntryBlock().getFirstInsertionPt());
  auto *tid=entry.CreateCall(Intrinsic::getOrInsertDeclaration(f.getParent(),Intrinsic::amdgcn_workitem_id_x),{},"kdma.tid");
  auto *lane=entry.CreateAnd(tid,63,"kdma.lane");
  auto *wave=entry.CreateAnd(tid,~63u,"kdma.wave");
  auto *owner=entry.CreateOr(wave,entry.CreateLShr(lane,1),"kdma.owner");
  Value *owners[2]={owner,entry.CreateAdd(owner,entry.getInt32(32),"kdma.owner.hi")};
  auto *intra=entry.CreateShl(entry.CreateAnd(lane,1),3,"kdma.intra");
  Value *intras[2]={intra,swizzle?entry.CreateXor(intra,8,"kdma.intra.hi"):intra};
  SourceMap sources;
  for (auto *c:p.copies) {
    IRBuilder<> b(c); b.SetCurrentDebugLocation(c->getDebugLoc());
    auto *old=assembly(c);
    auto *a=InlineAsm::get(c->getFunctionType(),
        "s_mov_b32 $0, $1\n\tbuffer_load_dwordx2 $2, $3, 0 offen lds",old->getConstraintString(),true);
    for (unsigned h=0;h<2;++h) {
      DenseMap<Value *,Value *> cache;
      Value *off=sources.clone(c->getArgOperand(1),owners[h],b,cache);
      off=b.CreateAdd(off,intras[h],"kdma.offset");
      auto *dst=b.CreateAdd(c->getArgOperand(0),b.getInt32(h*512),"kdma.lds");
      auto *n=b.CreateCall(c->getFunctionType(),a,{dst,off,c->getArgOperand(2)});
      n->setAttributes(c->getAttributes()); n->setTailCallKind(c->getTailCallKind());
    }
    c->eraseFromParent();
  }
  unsigned bases=0;
  if (swizzle) {
    struct Anchor { Value *pointer; SmallVector<uint64_t> offsets; };
    SmallVector<Anchor> anchors;
    DominatorTree dt(f);
    for (auto &r:p.reads) {
      for (auto &a:r.addresses) a ^= (a>>6)&8;
      IRBuilder<> b(r.load); b.SetCurrentDebugLocation(r.load->getDebugLoc());
      Value *ptr=nullptr;
      for (auto &a:anchors) {
        auto delta=int64_t(r.addresses[0])-int64_t(a.offsets[0]);
        bool match=true;
        for (unsigned t=0;t<p.threads;++t) match &= int64_t(r.addresses[t])-int64_t(a.offsets[t])==delta;
        if (match && dt.dominates(a.pointer,r.load)) { ptr=b.CreateGEP(b.getInt8Ty(),a.pointer,b.getInt64(delta),"kdma.gep"); break; }
      }
      if (!ptr) {
        auto *old=r.load->getPointerOperand();
        auto *offset=b.CreateSub(b.CreatePtrToInt(old,b.getInt64Ty()),b.CreatePtrToInt(p.storage,b.getInt64Ty()));
        auto *mask=b.CreateAnd(b.CreateLShr(offset,6),8);
        auto *delta=b.CreateSub(b.CreateXor(offset,mask),offset);
        ptr=b.CreateGEP(b.getInt8Ty(),old,delta,"kdma.swizzled");
        anchors.push_back({ptr,r.addresses}); ++bases;
      }
      r.load->setOperand(0,ptr);
    }
  }
  // Leave scheduling and FP arithmetic alone. Remove dead source-offset DAGs
  // exposed by packet replacement; ordinary O3 will CSE the new expressions.
  bool changed;
  do { changed=false; for (auto &b:f) for (auto it=b.begin();it!=b.end();) {
    auto *i=&*it++; if (isInstructionTriviallyDead(i)) { i->eraseFromParent(); changed=true; }
  }} while (changed);
  f.addFnAttr(Marker,(options.mode+":"+Twine(options.argument)).str());
  errs()<<"K DMA: mode="<<options.mode<<", argument="<<options.argument<<", packets="<<p.copies.size()
        <<"->"<<p.copies.size()*2<<", reads="<<p.reads.size()<<", bases="<<bases<<", function="<<f.getName()<<"\n";
}
} // namespace

bool parseKDMAOption(llvm::StringRef arg,KDMAOptions &o,bool &consumed) {
  consumed=arg.starts_with("--k-dma");
  if (!consumed) return true;
  if (arg.consume_front("--k-dma-arg=")) return !arg.getAsInteger(10,o.argument);
  if (!arg.consume_front("--k-dma=") || (arg!="128" && arg!="64" && arg!="64-swizzle")) return false;
  o.mode=arg; return true;
}
bool narrowKDMA(llvm::Module &module,const KDMAOptions &options) {
  if (options.mode=="128") return true;
  SmallVector<Plan> plans;
  for (auto &f:module) if (!f.isDeclaration() && f.getCallingConv()==llvm::CallingConv::AMDGPU_KERNEL) {
    std::string setting=(options.mode+":"+llvm::Twine(options.argument)).str();
    if (f.hasFnAttribute(Marker)) {
      if (f.getFnAttribute(Marker).getValueAsString()==setting) continue;
      llvm::errs()<<"K DMA: conflicting prior transformation in "<<f.getName()<<"\n"; return false;
    }
    Plan p; std::string why;
    if (!plan(f,options,p,why)) { llvm::errs()<<"K DMA: rejected, function="<<f.getName()<<": "<<why<<"\n"; return false; }
    plans.push_back(std::move(p));
  }
  for (auto &p:plans) apply(p,options);
  return true;
}
} // namespace mlir::frisk
