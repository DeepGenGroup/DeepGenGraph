// ===========================================================================
// Pass: narrowKDMA   优化汇总.py 第 28 项，LLVM 阶段，--k-dma=64-swizzle
// 详细设计见同目录 KDMA.md。
//
// 作用：
//   解决 K 的 LDS bank 冲突（本 case 15.90% -> 0.125%）。
//   1. 把 K 的一条 128-bit DMA（16 B/lane）拆成两条 64-bit DMA（8 B/lane）。
//      16 B 的包内部不能做 bit3 的置换，8 B 的包才可以。
//   2. 64-swizzle 模式下，LDS 地址再做 a ^= ((a >> 9) & 1) << 3，DMA 写和 LDS 读都走这个映射。
//
// 模式（KDMAOptions.mode）：
//   128         不改变。
//   64          只拆分。
//   64-swizzle  拆分并置换。
//
// 地址换算（一个 wave，原 lane L 写 16 B，新 lane l 写 8 B）：
//   h = 0/1     第几条 DMA，目的基址 m0 + 512 * h。
//   owner(l,h)  原来负责这段数据的 lane = wave | (l >> 1) | (h * 32)。
//   intra(l,h)  在原 16 B 内的偏移 = (l & 1) * 8；swizzle 且 h = 1 时再异或 8。
//   新源地址 = 原源地址表达式在 owner 上重新求值 + intra。
//
// 失败策略：用户显式请求的模式不能应用时，narrowKDMA 返回 false，调用方 report_fatal_error。
// 所有函数都通过 plan 后才统一 apply。
//
// 注意：这里的 1024、512、8192 都是写死的数字，对应 wave64 下 16 B/lane 的包和本 case 的 8KiB 分片。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/NarrowKDMA.h"
#include "LDSAddressAnalysis.h"
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
using detail::DMA64Asm;
using detail::LDSAddress;
using detail::MatrixReadM32x16;
using detail::RegisterLifetimesMarker;
using detail::WaitAsm;
using detail::analyzeWave64Kernel;
using detail::getInlineAsm;
using detail::isCallTo;
using detail::isDMA128;
using detail::isPureInst;
using detail::isWorkgroupUniform;
// Marker：函数属性，值为 "模式:参数号"。重复运行时配置相同就跳过，不同就报错。
constexpr StringLiteral Marker = "frisk.k_dma";
// OptimizeRegisterLifetimes 生成的空 asm，语义上是恒等。
bool identity(Call *c) {
  auto *a = getInlineAsm(c);
  return a && a->getAsmString().empty() && a->getConstraintString() == "=v,0" &&
      c->arg_size() == 1 && c->getArgOperand(0)->getType() == c->getType() &&
      c->getMetadata(RegisterLifetimesMarker);
}

// Recompute a source-offset expression for another lane in the SAME wave.
// Uniform leaves, including key-loop PHIs, stay unchanged. No lane shuffle is
// emitted: clone only pure thread-dependent integer expressions.
// 把 DMA 的源偏移表达式重建到另一个 lane（owner）上。
//   varying  值 -> 是否依赖线程号的缓存。
//   valid    遇到不支持的表达式时置 false。
// depends：判断 v 是否依赖线程号，同时校验表达式可重建。
// clone：把线程号替换成 owner，重新生成依赖线程号的部分。
class SourceMap {
  std::map<Value *, bool> varying;
public:
  bool valid = true;
  bool depends(Value *v) {
    if (auto it = varying.find(v); it != varying.end()) return it->second;
    SmallPtrSet<Value *,32> seen;
    if (isWorkgroupUniform(v,seen)) return varying[v] = false;
    varying[v] = false;
    auto *i = dyn_cast<Inst>(v);
    if (!i) { valid = false; return false; }
    if (auto *c = dyn_cast<Call>(i)) {
      if (isCallTo(c,"llvm.amdgcn.workitem.id.x") && c->arg_empty()) return varying[v] = true;
      if (identity(c)) return varying[v] = depends(c->getArgOperand(0));
      // A leader read is unchanged by a permutation confined to its wave.
      if (isCallTo(c,"llvm.amdgcn.readfirstlane.i32") && c->arg_size()==1) return false;
      valid = false; return false;
    }
    if (!isPureInst(i) || !i->getType()->isIntegerTy()) { valid = false; return false; }
    bool result = false;
    for (auto &op : i->operands()) result |= depends(op);
    return varying[v] = result;
  }
  Value *clone(Value *v, Value *owner, IRBuilder<> &b, DenseMap<Value *,Value *> &cache) {
    if (!depends(v)) return v;
    if (auto it = cache.find(v); it != cache.end()) return it->second;
    auto *i = cast<Inst>(v);
    if (isCallTo(i,"llvm.amdgcn.workitem.id.x")) return owner;
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

// 判断 DMA 的资源描述符是否由 arg 构造，也就是这条 DMA 是否在读 K。
bool resourceFrom(Value *v, Argument *arg, SmallPtrSetImpl<Value *> &seen, bool &found) {
  if (v == arg) { found = true; return true; }
  if (isa<Constant>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i || !isPureInst(i)) return false;
  return llvm::all_of(i->operands(),[&](Value *op) { return resourceFrom(op,arg,seen,found); });
}
// ReadPlan：一个 K 的 LDS 读，以及每个线程读的字节地址（已枚举）。
// Plan：一个函数的改写计划。copies = K 的 DMA，reads = K 的读，storage = LDS 全局变量。
struct ReadPlan { LoadInst *load; SmallVector<uint64_t> addresses; };
struct Plan {
  Function *function;
  GlobalVariable *storage;
  unsigned threads;
  SmallVector<Call *> copies;
  SmallVector<ReadPlan> reads;
};

// 合法性证明，只分析不改 IR，失败时 why 写原因。
// options.argument 是 K 在 kernel 参数中的下标，默认 1（Q,K,V,O）。
bool plan(Function &f, const KDMAOptions &options, Plan &p, std::string &why) {
  auto fail = [&](StringRef s) { why = s.str(); return false; };
  if (options.argument >= f.arg_size()) return fail("K ABI argument is out of range");
  auto *arg = f.getArg(options.argument);
  if (!arg->getType()->isPointerTy() || arg->getType()->getPointerAddressSpace()!=1)
    return fail("K ABI argument must be a global pointer");
  auto kernel = analyzeWave64Kernel(f, why);
  if (!kernel) return false;
  auto *storage = kernel->storage;
  unsigned threads = kernel->threads;
  const auto &dl = f.getDataLayout();
  LDSAddress eval(dl,storage);
  DominatorTree dt(f);
  LoopInfo loops(dt);
  p.function=&f; p.storage=storage; p.threads=threads;
  // Require one compute block containing all DMA for K and all its consumers.
  // This permits a self loop or a separate PHI/condition-only loop header.
  // compute：K 的所有 DMA 和读所在的基本块。
  BasicBlock *compute = nullptr;
  for (auto &i : instructions(f)) if (isDMA128(&i)) {
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
  auto *loop = loops.getLoopFor(compute);
  if (loop) {
    if (loop->getParentLoop() || loop->getNumBlocks()>2 || loop->getLoopLatch()!=compute)
      return fail("requires a single compute/latch block");
    for (auto *other : loop->blocks()) if (other!=compute)
      for (auto &i : *other) if (!isPureInst(&i) && !isa<PHINode,CondBrInst,UncondBrInst>(&i))
        return fail("loop header is not pure");
  }
  // Prove K ownership at every memory access, without depending on debug
  // locations, SSA names, tensor dimensions or a fixed number of K slices.
  // Reject K values live across the compute block boundary.
  // owner[byte]：这个字节当前是否存放 K。按指令顺序模拟：K 的 DMA 置 1，其它 DMA 覆盖后清 0。
  // 读 K 必须完全落在 owner = 1 的范围内，这就区分出了 K 的读和其它读。
  // 循环结束时 owner 必须全 0，也就是 K 的内容不跨基本块存活。
  SmallVector<uint8_t> owner(kernel->ldsBytes,0);
  for (auto &i : *compute) {
    if (isDMA128(&i)) {
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
    // 矩阵读的实际范围不易枚举。保守假设它覆盖对齐的 8KiB 区域，要求区域内没有 K。
    if (auto *c=dyn_cast<Call>(&i); c && isCallTo(c,MatrixReadM32x16)) {
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
      auto *a=getInlineAsm(c);
      if (identity(c) || isCallTo(c,"llvm.amdgcn.s.barrier") || isCallTo(c,"llvm.amdgcn.sched.barrier") ||
          (a && a->getAsmString()==WaitAsm)) continue;
      if (llvm::any_of(c->args(),[](Value *v) { return v->getType()->isPointerTy() && v->getType()->getPointerAddressSpace()==3; }) ||
          (a && a->hasSideEffects())) return fail("unknown shared-memory side effect");
    }
  }
  if (p.reads.empty()) return fail("no ordinary K LDS consumers");
  if (llvm::any_of(owner,[](uint8_t x) { return x!=0; })) return fail("K lifetime extends beyond compute block");
  return true;
}

// 执行改写，变量名对应文件头的换算公式。
void apply(Plan &p, const KDMAOptions &options) {
  auto &f=*p.function;
  bool swizzle=options.mode=="64-swizzle";
  // One lane remapping in the entry block; full-wave owner never leaves its
  // original wave. SourceMap handles arbitrary supported source strides.
  // 在入口块算一次：tid、lane（wave 内）、wave（wave 基地址）、owners[h]、intras[h]。
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
    auto *old=getInlineAsm(c);
    auto *a=InlineAsm::get(c->getFunctionType(),
        DMA64Asm,old->getConstraintString(),true);
    // 一条旧 DMA 变成两条：h = 0 写前 512 B，h = 1 写后 512 B。
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
  // 改写 K 的读地址：a ^= (a >> 6) & 8，把 bit 9 异或到 bit 3。
  // anchors：已生成的读指针及其逐线程地址。与某个 anchor 差值恒定时，用“anchor + 常量 GEP”，
  // 这样后面 ds_read2 的合并机会不会丢失。否则才生成新的 XOR 计算。
  // bases：生成新基址的个数。
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
