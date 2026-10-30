#include "deepgengraph/Conversion/ConvertToLLVM/DoubleBufferV.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <optional>
#include <tuple>

namespace mlir::frisk {
namespace {
using namespace llvm;
using Inst = Instruction;
using Call = CallInst;
constexpr StringLiteral Marker = "frisk.v_double_buffer";
constexpr StringLiteral MatrixRead = "llvm.hcu.ds.read.m32x16.b16.v4i32.p3";

bool named(Inst *i, StringRef name) {
  auto *c = dyn_cast_or_null<Call>(i);
  return c && !c->hasOperandBundles() && !c->isMustTailCall() &&
      c->getCalledFunction() && c->getCalledFunction()->isDeclaration() &&
      c->getCalledFunction()->getName() == name;
}
InlineAsm *assembly(Inst *i) {
  auto *c = dyn_cast<Call>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall()) return nullptr;
  auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
  return a && !a->canThrow() && !a->isAlignStack() ? a : nullptr;
}
bool dma(Inst *i) {
  auto *a = assembly(i);
  if (!a || !a->hasSideEffects() ||
      a->getAsmString() != "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds" ||
      a->getConstraintString() != "={m0},s,v,s,~{memory}") return false;
  auto *c = cast<Call>(i);
  return c->arg_size() == 3 && c->use_empty() && c->getType()->isIntegerTy(32) &&
      c->getArgOperand(0)->getType()->isIntegerTy(32) &&
      c->getArgOperand(1)->getType()->isIntegerTy(32) &&
      c->getArgOperand(2)->getType() == FixedVectorType::get(Type::getInt32Ty(c->getContext()), 4);
}
bool wait(Inst *i) {
  auto *a = assembly(i);
  return a && a->hasSideEffects() && a->getAsmString() == "s_waitcnt vmcnt(0)" &&
      a->getConstraintString() == "~{memory}" && cast<Call>(i)->arg_empty();
}
bool mmac(Inst *i) {
  auto *a = assembly(i);
  return a && !a->hasSideEffects() && cast<Call>(i)->doesNotAccessMemory() &&
      isRegisterMMA(a->getAsmString(), a->getConstraintString());
}
bool hint(Inst *i) {
  if (!named(i, "llvm.amdgcn.sched.barrier")) return false;
  auto *c = cast<Call>(i);
  auto *v = c->arg_size() == 1 ? dyn_cast<ConstantInt>(c->getArgOperand(0)) : nullptr;
  return v && v->isZero();
}
bool pure(Inst *i) {
  return !isa<CallBase, PHINode, AllocaInst>(i) && !i->isTerminator() &&
      !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects() && isSafeToSpeculativelyExecute(i);
}
// Export may construct the same resource descriptor at every DMA site, before
// CSE. Compare its pure DAG, including opcode flags/shuffle masks/constants.
bool sameValue(Value *a, Value *b, unsigned depth = 0) {
  if (a == b) return true;
  if (depth > 32 || a->getType() != b->getType()) return false;
  auto *ia = dyn_cast<Inst>(a), *ib = dyn_cast<Inst>(b);
  if (!ia || !ib || !pure(ia) || !pure(ib) || !ia->isSameOperationAs(ib)) return false;
  for (unsigned k = 0; k < ia->getNumOperands(); ++k)
    if (!sameValue(ia->getOperand(k),ib->getOperand(k),depth+1)) return false;
  return true;
}
bool workgroupFence(Inst *i, AtomicOrdering order) {
  auto *f = dyn_cast_or_null<FenceInst>(i);
  return f && f->getOrdering() == order &&
      f->getSyncScopeID() == f->getContext().getOrInsertSyncScopeID("workgroup");
}
struct Sync {
  Inst *release = nullptr, *barrier = nullptr, *acquire = nullptr, *wait = nullptr;
};
std::optional<Sync> sync(Inst *i) {
  if (!named(i, "llvm.amdgcn.s.barrier") || !cast<Call>(i)->arg_empty() ||
      !workgroupFence(i->getPrevNode(), AtomicOrdering::Release) ||
      !workgroupFence(i->getNextNode(), AtomicOrdering::Acquire)) return {};
  Sync s{i->getPrevNode(), i, i->getNextNode(), nullptr};
  if (s.release->getPrevNode() && wait(s.release->getPrevNode())) s.wait = s.release->getPrevNode();
  return s;
}

// Exact evaluation of the restricted integer/thread-address DAG. Unknown
// values fail closed. A single LDS allocation is assigned relative base zero;
// actual rewrites add a byte displacement to the original pointer/base.
// This naturally handles arbitrary tile sizes, lane permutations and swizzles.
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

bool uniform(Value *v, SmallPtrSetImpl<Value *> &seen) {
  if (isa<Constant, Argument>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i) return false;
  if (auto *c = dyn_cast<Call>(i))
    return named(c,"llvm.amdgcn.workgroup.id.x") || named(c,"llvm.amdgcn.workgroup.id.y") || named(c,"llvm.amdgcn.workgroup.id.z");
  if (!isa<PHINode>(i) && !pure(i)) return false;
  return llvm::all_of(i->operands(), [&](Value *op) { return uniform(op,seen); });
}

struct Stage {
  Sync ready, handoff;
  SmallVector<Call *, 16> reads, copies;
  SmallVector<Inst *, 16> dependencies;
};
struct Plan {
  SmallVector<Stage, 8> stages;
  Sync finish;
  int64_t displacement = 0;
  unsigned sliceBytes = 0;
};

// All writes cover a full contiguous slice, with no duplicate lane writes.
bool footprint(ArrayRef<Call *> copies, LDSAddress &addresses, unsigned threads,
               unsigned capacity, BitVector &bytes) {
  bytes = BitVector(capacity);
  for (auto *c : copies)
    for (unsigned t = 0; t < threads; ++t) {
      auto base = addresses.get(c->getArgOperand(0),t);
      auto leader = addresses.get(c->getArgOperand(0),t/64*64);
      if (!base || base != leader) return false;
      uint64_t start = *base + (t%64)*16;
      if (start > capacity || capacity-start < 16) return false;
      for (unsigned b = start; b < start+16; ++b) {
        if (bytes[b]) return false;
        bytes.set(b);
      }
    }
  return bytes.any();
}

bool plan(BasicBlock &block, GlobalVariable *storage, unsigned capacity,
          unsigned threads, DominatorTree &dt, Plan &p, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  auto &dl = block.getModule()->getDataLayout();
  LDSAddress addresses(dl,storage);
  SmallVector<Sync,16> syncs;
  for (auto &i : block) {
    if (i.getMetadata(Marker)) return fail("already transformed");
    if (auto s = sync(&i)) syncs.push_back(*s);
    if (!named(&i,MatrixRead)) continue;
    auto *c = cast<Call>(&i);
    if (c->arg_size() != 2 || !c->getArgOperand(0)->getType()->isPointerTy() ||
        c->getArgOperand(0)->getType()->getPointerAddressSpace() != 3 ||
        c->getType() != FixedVectorType::get(Type::getInt32Ty(c->getContext()),4)) return fail("unsupported matrix packet");
    auto it = llvm::find_if(llvm::reverse(syncs), [](const Sync &s) { return s.wait; });
    if (it == syncs.rend()) return fail("missing ready wait/barrier");
    if (p.stages.empty() || p.stages.back().ready.barrier != it->barrier)
      p.stages.push_back(Stage{*it, {}, {}, {}, {}});
    p.stages.back().reads.push_back(c);
  }
  if (p.stages.size() < 2 || p.stages.size() > 32) return fail("requires 2..32 unrolled V slices");
  for (unsigned s = 1; s < p.stages.size(); ++s) {
    auto &previous = p.stages[s-1];
    for (auto *i = previous.ready.acquire->getNextNode(); i != p.stages[s].ready.wait; i = i->getNextNode()) {
      if (dma(i)) p.stages[s].copies.push_back(cast<Call>(i));
      if (auto b = sync(i)) {
        if (previous.handoff.barrier || b->wait) return fail("ambiguous reader handoff");
        previous.handoff = *b;
      }
    }
    if (!previous.handoff.barrier || p.stages[s].copies.empty()) return fail("missing reader handoff/DMA");
    if (!previous.reads.back()->comesBefore(previous.handoff.release) ||
        !previous.handoff.acquire->comesBefore(p.stages[s].copies.front())) return fail("unsupported read/handoff order");
  }
  Value *resource = p.stages[1].copies.front()->getArgOperand(2);
  for (auto &i : block) {
    if (&i == p.stages.front().ready.wait) break;
    if (dma(&i) && sameValue(cast<Call>(&i)->getArgOperand(2),resource))
      p.stages.front().copies.push_back(cast<Call>(&i));
  }
  BitVector original;
  if (!footprint(p.stages.front().copies,addresses,threads,capacity,original)) return fail("unknown initial DMA footprint");
  unsigned begin = original.find_first(), size = original.count();
  if (begin+size > capacity || original.find_last() != int(begin+size-1)) return fail("non-contiguous DMA slice");
  int64_t spare = begin+size;
  if (spare+size > capacity) spare = int64_t(begin)-size;
  if (spare < 0) return fail("no spare LDS slice");
  p.displacement = spare-begin;
  p.sliceBytes = size;
  SmallPtrSet<Inst *,32> allowedMemory;
  for (auto &s : p.stages) {
    BitVector bytes;
    if (!footprint(s.copies,addresses,threads,capacity,bytes) || bytes != original)
      return fail("DMA slices differ or are already buffered");
    for (auto *c : s.copies) {
      if (!sameValue(c->getArgOperand(2),resource)) return fail("mixed DMA sources");
      allowedMemory.insert(c);
    }
    for (auto *r : s.reads) {
      for (unsigned t = 0; t < threads; ++t) {
        auto a = addresses.get(r->getArgOperand(0),t), imm = addresses.get(r->getArgOperand(1),t);
        if (!a || !imm || *a+*imm < begin || *a+*imm > begin+size-16) return fail("matrix read outside DMA slice");
      }
      allowedMemory.insert(r);
    }
  }
  auto *lastRead = p.stages.back().reads.back();
  for (auto *i = lastRead->getNextNode(); i; i = i->getNextNode())
    if (auto s = sync(i)) { p.finish = *s; break; }
  if (!p.finish.barrier || p.finish.wait) return fail("missing final reader barrier");
  SmallPtrSet<Inst *,32> syncOps;
  for (auto &s : syncs)
    for (auto *i : {s.release,s.barrier,s.acquire,s.wait}) if (i) syncOps.insert(i);
  for (auto *i = p.stages.front().ready.acquire->getNextNode(); i != p.finish.acquire; i = i->getNextNode()) {
    if (allowedMemory.contains(i) || syncOps.contains(i) || mmac(i) || hint(i) || pure(i) ||
        named(i,"llvm.amdgcn.readfirstlane.i32")) continue;
    return fail("unknown operation or other live LDS user in V region");
  }

  // Prove the alternate slot's old contents are dead, including the loop
  // backedge: any earlier same-block read of that slot must be preceded by a
  // write in this iteration. Other LDS readers must strictly dominate this
  // block (prologue Q reads); readers in successors/other loops are rejected.
  BitVector written(capacity);
  for (auto &i : instructions(*block.getParent())) {
    bool before = i.getParent() == &block && i.comesBefore(p.stages.front().ready.wait);
    if (dma(&i)) {
      if (before) {
        BitVector bytes;
        if (!footprint({cast<Call>(&i)},addresses,threads,capacity,bytes)) return fail("unknown earlier DMA footprint");
        written |= bytes;
      }
      continue;
    }
    if (allowedMemory.contains(&i)) continue;
    auto *load = dyn_cast<LoadInst>(&i);
    if (load && load->getPointerAddressSpace() == 3) {
      if (load->isVolatile() || load->isAtomic()) return fail("volatile/atomic LDS reader");
      if (!before) {
        if (i.getParent() != &block && dt.dominates(&i,p.stages.front().ready.wait) &&
            !dt.dominates(&block,i.getParent())) continue;
        return fail("LDS reader survives V region");
      }
      auto width = dl.getTypeStoreSize(load->getType());
      if (width.isScalable()) return fail("scalable LDS read");
      for (unsigned t = 0; t < threads; ++t) {
        auto a = addresses.get(load->getPointerOperand(),t);
        if (!a || *a > capacity || width.getFixedValue() > capacity-*a) return fail("unknown LDS read address");
        for (unsigned b = *a; b < *a+width.getFixedValue(); ++b)
          if (b >= spare && b < spare+size && !written[b]) return fail("alternate slot live across iteration");
      }
      continue;
    }
    if (auto *store = dyn_cast<StoreInst>(&i)) {
      if (store->getPointerAddressSpace() != 1) return fail("unknown/LDS store");
      continue;
    }
    if (load && load->getPointerAddressSpace() == 1) continue;
    if (i.mayReadOrWriteMemory() && !syncOps.contains(&i) && !wait(&i) && !hint(&i) &&
        !named(&i,"llvm.amdgcn.s.barrier") && !isa<FenceInst>(&i))
      return fail("unknown memory effect outside V region");
  }
  // Preflight the dependencies of every moved DMA. Clone only pure address
  // computations and known lane broadcasts; never hoist reads or another DMA.
  for (unsigned s = 1; s < p.stages.size(); ++s) {
    auto *anchor = p.stages[s-1].ready.acquire->getNextNode();
    SmallPtrSet<Inst *,32> seen;
    std::function<bool(Value *)> visit = [&](Value *v) {
      auto *i = dyn_cast<Inst>(v);
      if (!i || (dt.dominates(i,anchor) && i != anchor) || seen.contains(i)) return true;
      if (i->getParent() != &block || (!pure(i) && !named(i,"llvm.amdgcn.readfirstlane.i32"))) return false;
      if (named(i,"llvm.amdgcn.readfirstlane.i32") && !addresses.get(i,0)) return false;
      for (Value *op : i->operands()) if (!visit(op)) return false;
      seen.insert(i);
      p.stages[s].dependencies.push_back(i);
      return true;
    };
    for (auto *c : p.stages[s].copies)
      for (Value *arg : c->args()) if (!visit(arg)) return fail("DMA address depends on unavailable effect");
  }
  return true;
}

void apply(Plan &p) {
  auto &ctx = p.finish.barrier->getContext();
  auto *marker = MDNode::get(ctx, ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(ctx),p.sliceBytes)));
  for (unsigned s = 0; s < p.stages.size(); ++s) {
    auto &stage = p.stages[s];
    if (s) {
      auto *anchor = p.stages[s-1].ready.acquire->getNextNode();
      DenseMap<Value *,Value *> mapped;
      for (auto *i : stage.dependencies) {
        auto *copy = i->clone();
        for (auto &op : copy->operands()) if (auto *v = mapped.lookup(op.get())) op.set(v);
        copy->setName("vdb.address");
        copy->insertBefore(anchor->getIterator());
        mapped[i] = copy;
      }
      for (auto *c : stage.copies) {
        for (auto &op : c->args()) if (auto *v = mapped.lookup(op.get())) op.set(v);
        if (s%2) {
          IRBuilder<> builder(anchor);
          c->setArgOperand(0,builder.CreateAdd(c->getArgOperand(0),
              builder.getInt32(p.displacement),"vdb.dma.base"));
        }
        c->moveBefore(anchor->getIterator());
      }
    }
    for (auto *r : stage.reads) {
      if (s%2) {
        IRBuilder<> builder(r);
        r->setArgOperand(0,builder.CreateGEP(builder.getInt8Ty(),r->getArgOperand(0),
            builder.getInt64(p.displacement),"vdb.read"));
      }
      r->setMetadata(Marker,marker);
    }
  }
  for (auto &s : p.stages) if (s.handoff.barrier) {
    s.handoff.release->eraseFromParent();
    s.handoff.barrier->eraseFromParent();
    s.handoff.acquire->eraseFromParent();
  }
}
} // namespace

bool parseVDoubleBufferOption(llvm::StringRef argument, bool &enabled, bool &consumed) {
  consumed = argument.starts_with("--v-double-buffer");
  if (!consumed) return true;
  if (!argument.consume_front("--v-double-buffer=") || (argument != "0" && argument != "1")) return false;
  enabled = argument == "1";
  return true;
}

llvm::PreservedAnalyses DoubleBufferVPass::run(llvm::Function &f, llvm::FunctionAnalysisManager &) {
  auto skip = [&](llvm::StringRef why) {
    llvm::errs() << "V double buffer: skipped, function=" << f.getName() << ": " << why << "\n";
    return llvm::PreservedAnalyses::all();
  };
  if (f.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) return skip("not an AMDGPU kernel");
  auto *dims = f.getMetadata("reqd_work_group_size");
  if (!dims || dims->getNumOperands() != 3) return skip("requires fixed 1D workgroup size");
  unsigned threads = 0;
  for (unsigned i = 0; i < 3; ++i) {
    auto *v = llvm::mdconst::dyn_extract<llvm::ConstantInt>(dims->getOperand(i));
    if (!v || v->getValue().getActiveBits() > 32 || (i && !v->isOne())) return skip("requires fixed 1D workgroup size");
    if (!i) threads = v->getZExtValue();
  }
  if (!threads || threads > 1024 || threads%64 ||
      f.getFnAttribute("target-features").getValueAsString().contains("+wavefrontsize32")) return skip("requires wave64 workgroups");
  llvm::GlobalVariable *storage = nullptr;
  for (auto &g : f.getParent()->globals()) if (g.getAddressSpace() == 3) {
    if (storage || !g.hasInitializer()) return skip("requires one statically allocated LDS object");
    storage = &g;
  }
  if (!storage) return skip("no LDS allocation");
  auto capacity = f.getParent()->getDataLayout().getTypeAllocSize(storage->getValueType());
  if (capacity.isScalable() || capacity.getFixedValue() > 65536) return skip("unsupported LDS allocation size");
  for (auto &b : f) {
    auto *br = llvm::dyn_cast<llvm::CondBrInst>(b.getTerminator());
    if (br) {
      llvm::SmallPtrSet<llvm::Value *,32> seen;
      if (!uniform(br->getCondition(),seen)) return skip("cannot prove workgroup-uniform control flow");
    } else if (!llvm::isa<llvm::UncondBrInst,llvm::ReturnInst>(b.getTerminator())) return skip("unsupported control flow");
  }
  llvm::DominatorTree dt(f);
  llvm::LoopInfo loops(dt);
  unsigned changed = 0;
  bool candidate = false;
  for (auto &b : f) {
    if (!llvm::any_of(b,[](Inst &i) { return named(&i,MatrixRead); })) continue;
    candidate = true;
    auto *loop = loops.getLoopFor(&b);
    if (loop) {
      // Fresh MLIR export keeps a PHI/condition-only header separate from the
      // compute/latch block; O3 folds it to the archived single-block loop.
      bool simple = !loop->getParentLoop() && (loop->getNumBlocks() == 1 ||
          (loop->getNumBlocks() == 2 && loop->getLoopLatch() == &b && loop->getHeader() != &b));
      for (auto *other : loop->blocks()) if (other != &b)
        for (auto &i : *other)
          simple &= pure(&i) || llvm::isa<llvm::PHINode,llvm::CondBrInst,llvm::UncondBrInst>(&i);
      if (!simple) { skip("requires one compute block with an optional pure loop header"); continue; }
    }
    Plan p;
    std::string reason;
    if (!plan(b,storage,capacity.getFixedValue(),threads,dt,p,reason)) { skip(reason); continue; }
    apply(p);
    ++changed;
    llvm::errs() << "V double buffer: " << p.stages.size() << " slices, slot-bytes=" << p.sliceBytes
                 << ", displacement=" << p.displacement << ", removed-barriers=" << p.stages.size()-1
                 << ", LDS=" << capacity.getFixedValue() << ", function=" << f.getName() << "\n";
  }
  if (!candidate) return skip("no supported matrix-read stream");
  return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

void doubleBufferV(llvm::Module &module, bool enabled) {
  if (!enabled) return;
  llvm::FunctionAnalysisManager analyses;
  llvm::PassBuilder pb;
  pb.registerFunctionAnalyses(analyses);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(DoubleBufferVPass());
  for (auto &f : module) if (!f.isDeclaration()) pipeline.run(f,analyses);
}
} // namespace mlir::frisk
