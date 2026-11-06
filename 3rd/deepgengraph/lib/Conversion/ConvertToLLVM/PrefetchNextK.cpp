// ===========================================================================
// 本文件包含两个顺序相关的 Pass。详细说明见仓库根目录 SafeSoftmax最后两项优化说明.md。
//
// prefetchNextK（PrefetchNextKPass）   优化汇总.py 第 32 项，LLVM 阶段，--cross-bn-k0=1
//   下一轮的 K0 DMA 原本在下一轮开头才发，只能干等。改成：
//     第一轮的 K0 在进入循环前发（prime）。
//     每轮的最后一片 PV 开始时，发下一轮的 K0（V3 只读 slot1，此时 slot0 已空），用 PV 的计算掩盖延迟。
//   分母锚定（sum anchor）：改写新增了一个 if 分支，LLVM 会把分母加法全沉到分支之后，输入一直存活，
//   导致 VGPR 256、scratch 20，反而变慢。所以每条分母加法后加一个空 asm，把它钉在原地。
//   这两件事必须同时做。
//
// elideBackedgeBarrier   第 33 项，--elide-backedge-barrier=1，未实测
//   删除循环尾的 reader barrier。必须在 prefetchNextK 之后运行，见函数前的说明。
//
// 流程（prefetchNextK）：
//   run    函数级检查：kernel、1D wave64、唯一静态 LDS、控制流 uniform、只有一个顶层循环。
//   plan   识别循环结构和 K0，证明合法。只分析，不改 IR。
//   apply  加锚定 -> 在入口发 K0 -> 把循环体切开，插入“有下一轮才发”的分支 -> 删除原来的 K0。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchNextK.h"
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
using detail::MatrixReadM32x16;
using detail::Sync;
using detail::analyzeWave64Kernel;
using detail::dmaBytesPerLane;
using detail::getInlineAsm;
using detail::isCallTo;
using detail::isPureInst;
using detail::isRegisterMMACall;
using detail::isSchedHint;
using detail::isWaitCall;
using detail::isWorkgroupUniform;
using detail::matchSync;
using detail::LDSAddress;
// Marker：函数属性，也标在新 DMA 上。elideBackedgeBarrier 用它确认前置 Pass 已运行。
// SumMarker：标在分母锚定的空 asm 上。
constexpr StringLiteral Marker = "frisk.cross_bn_k0";
constexpr StringLiteral SumMarker = "frisk.cross_bn_sum_anchor";

bool identity(Inst *i) {
  auto *a = getInlineAsm(i);
  if (!a || !a->hasSideEffects() || !a->getAsmString().empty()) return false;
  auto *c = cast<Call>(i);
  bool tied = (a->getConstraintString() == "=v,0" && c->arg_size() == 1) ||
      (a->getConstraintString() == "=v,0,i" && c->arg_size() == 2 && isa<ConstantInt>(c->getArgOperand(1)));
  return tied && c->getArgOperand(0)->getType() == i->getType() && c->doesNotAccessMemory();
}
// 可以跟随 DMA 克隆的地址计算，包括 identity 和 readfirstlane。
bool addressOp(Inst *i) {
  if (identity(i)) return i->getType()->isIntegerTy();
  if (isCallTo(i,"llvm.amdgcn.readfirstlane.i32")) return cast<Call>(i)->arg_size() == 1;
  return isPureInst(i) && (isa<GetElementPtrInst, CastInst, InsertElementInst, ExtractElementInst, ShuffleVectorInst>(i) ||
      (isa<BinaryOperator>(i) && i->getType()->isIntOrIntVectorTy()));
}

//   entry / header / body  入口块、循环头、计算块（兼 latch）。
//   iv / next / limit      循环变量、iv + 1、循环上界。
//   continuePredicate      继续循环的比较谓词（ULT 或 ULE）。
//   anchor                 最后一片 PV 开始的位置，下一轮 K0 在这里发。
//   copies                 循环开头的 K0 DMA。
//   primeDeps              在入口发 K0 时要克隆的地址计算。
//   nextDeps               在 anchor 发下一轮 K0 时要克隆的、依赖 iv 的计算。
//   sums                   要加锚定的分母 FP32 加法。
//   overlap                K0 在途时能执行的 MMAC 数。
//   bytes                  K0 占用的 LDS 字节数。
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

// 枚举线程，得到 DMA 写入的字节集合。这里写死了 wave64。
// readBytes：同样地得到一次 LDS 读的字节集合。
bool dmaBytes(Call *c, LDSAddress &addresses, unsigned threads, unsigned capacity, BitVector &bytes) {
  unsigned width = dmaBytesPerLane(c);
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
  } else if (isCallTo(i,MatrixReadM32x16)) {
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
  if (isRegisterMMACall(i) || identity(i)) return true;
  if (isCallTo(i,"llvm.amdgcn.readfirstlane.i32")) return true;
  auto *c = dyn_cast<IntrinsicInst>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall() || !c->doesNotAccessMemory()) return false;
  auto name = c->getCalledFunction()->getName();
  return name.starts_with("llvm.amdgcn.exp2.") || name.starts_with("llvm.maxnum.") ||
      name.starts_with("llvm.maximum.") || name.starts_with("llvm.fabs.") ||
      name == "llvm.amdgcn.ds.bpermute" || name == "llvm.amdgcn.mbcnt.lo" ||
      name == "llvm.amdgcn.mbcnt.hi";
}

// 合法性证明，分 8 步，下面逐段标注。失败时 reason 写原因。
// 第 1 步：循环必须是 entry -> header -> body -> header 的计数循环，iv 从 0 开始步长 1，
// 并能证明至少执行一次（否则在入口发 K0 就是越界多读）。
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

  // 第 2 步：收集 body 中的同步序列，开头的 wait 之前必须只有 K0 DMA 和地址计算。
  // syncs：所有同步序列。syncOps：它们包含的指令。
  SmallVector<Sync> syncs;
  SmallPtrSet<Inst *,32> syncOps;
  for (auto &i : *p.body) if (auto s = matchSync(&i)) {
    syncs.push_back(*s);
    for (auto *op : {s->release,s->barrier,s->acquire,s->wait}) if (op) syncOps.insert(op);
  }
  if (syncs.size() < 3 || !syncs.front().wait || syncs.back().wait) return fail("missing K0/final synchronization");
  for (auto &i : *p.body) {
    if (&i == syncs.front().wait) break;
    if (dmaBytesPerLane(&i)) p.copies.push_back(cast<Call>(&i));
    else if (!addressOp(&i)) return fail("unknown operation before K0 ready wait");
  }
  if (p.copies.empty()) return fail("missing initial K DMA group");
  Value *resource = p.copies.front()->getArgOperand(2);
  SmallPtrSet<Value *,32> uniformSeen;
  if (!isWorkgroupUniform(resource,uniformSeen)) return fail("nonuniform K resource");
  for (auto *c : p.copies) if (c->getArgOperand(2) != resource) return fail("mixed initial DMA resources");
  // The final matrix-read stage starts after the last wait/publication. Keep
  // the final read handoff, and keep the next iteration's original K0 wait.
  // 第 3 步：最后一个带 wait 的同步就是最后一片 V 的 ready，它之后就是 anchor。
  auto finalReady = llvm::find_if(llvm::reverse(syncs),[](const Sync &s) { return s.wait; });
  if (finalReady == syncs.rend() || finalReady->barrier == syncs.front().barrier)
    return fail("missing final PV ready synchronization");
  p.anchor = finalReady->acquire->getNextNode();
  if (!p.anchor || !p.anchor->comesBefore(syncs.back().release))
    return fail("final ready barrier is not followed by a matrix-read stage");
  LDSAddress addresses(f.getParent()->getDataLayout(),storage);
  // 第 4 步：k0 = K0 写入的字节。anchor 之后的所有读都不能碰这段字节，否则 DMA 会覆盖正在读的数据。
  // haveFinalRead：anchor 之后必须有矩阵读，这才是最后一片 PV。
  BitVector k0(capacity);
  for (auto *c : p.copies) {
    BitVector bytes;
    if (!dmaBytes(c,addresses,threads,capacity,bytes) || k0.anyCommon(bytes)) return fail("unknown/overlapping K0 DMA footprint");
    k0 |= bytes;
  }
  p.bytes = k0.count();
  bool haveFinalRead = false;
  for (auto *i = p.anchor; i != p.body->getTerminator(); i = i->getNextNode()) {
    if (isRegisterMMACall(i)) { ++p.overlap; continue; }
    if (isa<LoadInst>(i) || isCallTo(i,MatrixReadM32x16)) {
      BitVector bytes;
      if (!readBytes(i,addresses,threads,capacity,bytes) || k0.anyCommon(bytes))
        return fail("final PV reads overlap K0 or have unknown addresses");
      if (!i->comesBefore(syncs.back().release)) return fail("reader after final handoff");
      haveFinalRead |= isCallTo(i,MatrixReadM32x16);
      continue;
    }
    if (syncOps.contains(i) || isPureInst(i) || isSchedHint(i) || identity(i)) continue;
    return fail("unknown operation/memory after prefetch point");
  }
  if (!haveFinalRead || !p.overlap) return fail("no final PV overlap window");
  // All moved global reads must remain ahead of global writes, and no unknown
  // LDS writer/async producer may survive into the final stage.
  // 第 5 步：body 中不能有未知的内存操作。
  for (auto &i : *p.body) {
    if (i.isTerminator() || dmaBytesPerLane(&i) || syncOps.contains(&i) || isPureInst(&i) || isSchedHint(&i) || knownCall(&i)) continue;
    if (isa<LoadInst>(&i) || isCallTo(&i,MatrixReadM32x16)) {
      BitVector bytes;
      if (readBytes(&i,addresses,threads,capacity,bytes)) continue;
    }
    std::string detail; raw_string_ostream stream(detail); i.print(stream);
    return fail("unknown memory effect or unsupported compute operation: " + stream.str());
  }
  // Priming may reuse only LDS whose prologue readers have been released.
  // 第 6 步：入口块的 prologue（读 Q）。在入口发 K0 之前，必须已经有 reader barrier，
  // 且这个 barrier 处没有未完成的 DMA，也就是 Q 的 LDS 读者已经释放。
  // 循环外不能有其它 LDS 使用者。
  std::optional<Sync> entryHandoff;
  for (auto &i : *p.entry) if (auto s = matchSync(&i)) entryHandoff = s;
  if (!entryHandoff) return fail("missing prologue reader handoff");
  bool pending = false;
  for (auto &i : *p.entry) {
    if (dmaBytesPerLane(&i)) { pending = true; continue; }
    if (isWaitCall(&i)) { pending = false; continue; }
    if (&i == entryHandoff->barrier && pending) return fail("outstanding prologue DMA at reader handoff");
    if (isa<LoadInst,StoreInst,FenceInst>(&i) || i.isTerminator() || isPureInst(&i) || knownCall(&i) ||
        isCallTo(&i,"llvm.amdgcn.s.barrier") || isCallTo(&i,"llvm.amdgcn.workitem.id.x") ||
        isCallTo(&i,"llvm.amdgcn.workgroup.id.x") || isCallTo(&i,"llvm.amdgcn.workgroup.id.y") ||
        isCallTo(&i,"llvm.amdgcn.workgroup.id.z")) continue;
    return fail("unknown prologue operation");
  }
  for (auto *i = entryHandoff->acquire->getNextNode(); i != p.entry->getTerminator(); i = i->getNextNode())
    if (!addressOp(i)) return fail("operation after prologue handoff");
  for (auto &block : f) if (&block != p.entry && !loop.contains(&block))
    for (auto &i : block) {
      if (isa<LoadInst>(&i) && cast<LoadInst>(&i)->getPointerAddressSpace() != 1)
        return fail("LDS reader outside loop");
      if (dmaBytesPerLane(&i) || isCallTo(&i,MatrixReadM32x16)) return fail("LDS user outside loop");
    }

  // Preflight both clone DAGs, without mutating IR on a failed match. Prime
  // clones loop-local address dependencies. Future clones only IV-dependent
  // expressions, reproducing the measured sum-anchor variant's lifetimes.
  // 第 7 步：预检两种克隆。
  // dependent / depends：值是否依赖 iv。
  // prime = true：在入口克隆，所有循环内依赖都要克隆。
  // prime = false：在 anchor 克隆，只克隆依赖 iv 的部分，其它直接复用。
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
  // 第 8 步：从循环 PHI 的回边值反向追 FP32 加法，这就是分母递推。
  // 遇到 MMAC、exp/max 和 PHI 就停止，所以输出累加器的 QK 运算不会被算进来。
  // 只锚定 anchor 之前的加法。
  SmallPtrSet<Inst *,32> seen, sums;
  std::function<void(Value *)> collect = [&](Value *v) {
    auto *i = dyn_cast<Inst>(v);
    if (!i || i->getParent() != p.body || !seen.insert(i).second) return;
    if (identity(i)) { collect(cast<Call>(i)->getArgOperand(0)); return; }
    if (isa<CallBase,PHINode>(i) || !isPureInst(i)) return;
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
  // 分母锚定：add 后面加空 asm，其余使用者改用 asm 的结果。
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
  // 把 K0 DMA 及 deps 克隆到 before 之前，iv 替换成 index。
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
  // 入口处发第一轮的 K0（iv = 0）。
  clone(p.primeDeps,ConstantInt::get(p.iv->getType(),0),p.entry->getTerminator(),"k0.prime.");
  // 在 anchor 处切开 body：
  //   body:    ...; if (next < limit) goto k0.issue; else goto k0.join
  //   k0.issue 发 iv = next 的 K0
  //   k0.join  最后一片 PV 和后面的指令
  // 最后一轮不发，不会越界多读。
  p.next->moveBefore(p.anchor->getIterator());
  auto *join = p.body->splitBasicBlock(p.anchor,"k0.join");
  auto *issue = BasicBlock::Create(ctx,"k0.issue",p.body->getParent(),join);
  auto *exit = IRBuilder<>(issue).CreateBr(join);
  clone(p.nextDeps,p.next,exit,"k0.next.");
  auto *oldBr = p.body->getTerminator();
  IRBuilder<> builder(oldBr);
  auto *hasNext = builder.CreateICmp(p.continuePredicate,p.next,p.limit,"k0.has.next");
  builder.CreateCondBr(hasNext,issue,join); oldBr->eraseFromParent();
  // 删除循环开头原来的 K0 DMA。原来的 wait + barrier 保留，下一轮照常等待。
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
  std::string why;
  auto kernel = analyzeWave64Kernel(f, why);
  if (!kernel) return skip(why);
  if (!kernel->ldsBytes) return skip("unsupported LDS allocation size");
  DominatorTree dt(f); LoopInfo loops(dt);
  if (std::distance(loops.begin(),loops.end()) != 1) return skip("requires one top-level loop");
  Plan p; std::string reason;
  if (!plan(f,**loops.begin(),dt,kernel->storage,kernel->threads,kernel->ldsBytes,p,reason)) return skip(reason);
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

// After next-K0 prefetch, the barrier ending an iteration releases V3 readers,
// but nothing touches LDS before the wait+barrier opening the next iteration.
// That second barrier therefore orders the same accesses. On the last iteration
// the loop exits without any later LDS user, so no ordering is needed at all.
//
// This cannot be done on MLIR. There, V is not yet double buffered and K0 has not
// moved, so the K0 DMA sits between the two barriers and writes the slot V3 reads,
// and this barrier is required. prefetchNextK also anchors on it, so it must stay
// until that pass has run.
//
// prefetchNextK's marker already proves uniform control flow, one two-way loop,
// and no LDS user outside it. Only the adjacency is checked here.
void elideBackedgeBarrier(llvm::Module &module) {
  using namespace llvm;
  for (auto &f : module) {
    if (f.isDeclaration() || !f.hasFnAttribute(Marker)) continue;
    auto skip = [&](StringRef why) {
      errs() << "Elide backedge barrier: skipped, function=" << f.getName() << ": " << why << "\n";
    };
    DominatorTree dt(f); LoopInfo loops(dt);
    if (std::distance(loops.begin(),loops.end()) != 1) { skip("requires one top-level loop"); continue; }
    Loop &loop = **loops.begin();
    auto *latch = loop.getLoopLatch(), *header = loop.getHeader();
    if (!latch || latch == header || !isa<UncondBrInst>(latch->getTerminator())) { skip("noncanonical latch"); continue; }

    // The last synchronization in the latch must have nothing but pure address
    // math after it, and must not complete any DMA.
    // tail：latch 里的最后一个同步，必须不带 wait，也就是 reader barrier。
    // onlyAddressMath：两点之间只有地址计算。
    std::optional<Sync> tail;
    for (auto &i : *latch) if (auto s = matchSync(&i)) tail = s;
    if (!tail || tail->wait) { skip("latch does not end in a reader barrier"); continue; }
    auto onlyAddressMath = [](Inst *from, Inst *to) {
      for (auto *i = from; i != to; i = i->getNextNode())
        if (!addressOp(i)) return false;
      return true;
    };
    if (!onlyAddressMath(tail->acquire->getNextNode(),latch->getTerminator())) { skip("work after latch barrier"); continue; }

    // The next barrier on the backedge path must complete pending DMA and be
    // reached through pure control only.
    auto *branch = dyn_cast<CondBrInst>(header->getTerminator());
    auto *body = branch ? branch->getSuccessor(loop.contains(branch->getSuccessor(0)) ? 0 : 1) : nullptr;
    if (!body || !loop.contains(body) || body == latch) { skip("noncanonical loop header"); continue; }
    if (llvm::any_of(*header,[&](Inst &i) { return !isa<PHINode,ICmpInst,CondBrInst>(i); })) { skip("noncanonical loop header"); continue; }
    // head：下一轮开头的同步，必须带 wait。它之前只有地址计算，所以能代替 tail 保证同样的顺序。
    std::optional<Sync> head;
    for (auto &i : *body) if (auto s = matchSync(&i)) { head = s; break; }
    if (!head || !head->wait) { skip("next iteration does not open with wait+barrier"); continue; }
    if (!onlyAddressMath(&body->front(),head->wait)) { skip("LDS or DMA before next barrier"); continue; }

    tail->acquire->eraseFromParent();
    tail->barrier->eraseFromParent();
    tail->release->eraseFromParent();
    errs() << "Elide backedge barrier: removed 1 per iteration, function=" << f.getName() << "\n";
  }
}
} // namespace mlir::frisk
