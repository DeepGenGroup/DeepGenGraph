// ===========================================================================
// 本文件包含两个共用 schedule() 的 Pass。详细设计见同目录 PairedLDS.md。
//
// prefetchPairedLDS（PrefetchPairedLDSPass）   优化汇总.py 第 29 项，LLVM 阶段，--k-lds-prefetch=8
//   QK 的 B 操作数（K）来自 LDS 读。把成对的 b64 读整对提前，越过若干条无关的 MMAC，
//   数据在路上时 MMAC 照常执行，用计算掩盖 LDS 延迟。
//   两个 load 紧贴在一起，后端才能合并成 read2_b64。
//
// prefetchPairedLDSTail（PrefetchPairedLDSTailPass）   第 31 项，--k-tail-prefetch=4
//   第 29 项把局部 max/sum 归约当作区域边界，QK 尾部够不到。这个 Pass 允许区域包含归约，
//   补上这一段。归约自身不动，已被第 29 项保护的读也不再处理。
//
// distance：一对读最多越过多少条 MMAC，上限 8。不是预取多少个 tile。
//
// MMAC 的相对顺序始终不变，只移动 LDS 读及其地址计算。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchPairedLDS.h"
#include "LateIRPatterns.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include <functional>
#include <limits>

namespace mlir::frisk {
namespace {
using namespace llvm;
using Inst = Instruction;
using detail::isTypedRegisterMMACall;
// Order：一段指令的顺序。规划都在这个“虚拟顺序”上做，校验通过后才移动真实指令。
// Marker：函数属性，第 29 项已运行，值为 distance。
// TailMarker：函数属性，第 31 项已运行。
// TailPair：第 31 项给配对读打的元数据。第 29 项用的是 frisk.k_lds_pair。
using Order = SmallVector<Inst *, 64>;
constexpr StringLiteral Marker = "frisk.k_lds_prefetch";
constexpr StringLiteral TailMarker = "frisk.k_tail_prefetch";
constexpr StringLiteral TailPair = "frisk.k_tail_pair";

// 可以跟随 load 一起移动的纯地址计算。
// b64：简单的 8 字节 LDS 读，类型 <4 x half>，对齐不小于 8。
// candidate：b64 读，且所有使用者都是同一基本块内 MMAC 的 B 操作数。
bool address(Inst *i) {
  return (isa<GetElementPtrInst,CastInst>(i) ||
          (isa<BinaryOperator>(i) && i->getType()->isIntOrIntVectorTy())) &&
      !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects() &&
      isSafeToSpeculativelyExecute(i);
}
bool b64(LoadInst *l) {
  return l && l->isSimple() && l->getPointerAddressSpace()==3 &&
      l->getAlign().value()>=8 &&
      l->getType()==FixedVectorType::get(Type::getHalfTy(l->getContext()),4);
}
bool candidate(LoadInst *l) {
  if (!b64(l) || l->use_empty()) return false;
  for (auto &u:l->uses()) {
    auto *c=dyn_cast<CallInst>(u.getUser());
    if (!isTypedRegisterMMACall(c) || u.getOperandNo()!=1 || c->getParent()!=l->getParent()) return false;
  }
  return true;
}
// Pair：一对读。first 是这对读里更早被用到的 MMAC 序号。
// Stats：日志计数。anchors 是被越过的 MMAC 数；unpaired 是没配上对的读。
struct Pair { LoadInst *a; LoadInst *b; unsigned first; };
struct Stats { unsigned regions=0, pairs=0, moved=0, anchors=0, unpaired=0; };

// Plan against a virtual order. Only hoisting is allowed, never delaying a
// read past a use. A dependency on another load/call rejects the movement.
// 在虚拟顺序里把 roots 及其地址依赖移到 before 之前。
// 只能提前。依赖另一个 load 或 call 时失败，此时 order 不变。
//   pos      指令当前位置。
//   moving   要移动的指令集合。
//   visiting 递归路径，用来防环。
bool hoist(Order &order, ArrayRef<Inst *> roots, Inst *before) {
  DenseMap<Inst *,unsigned> pos;
  for (unsigned n=0;n<order.size();++n) pos[order[n]]=n;
  if (!pos.contains(before)) return false;
  unsigned at=pos[before];
  SmallPtrSet<Inst *,32> moving, visiting;
  std::function<bool(Inst *,bool)> collect=[&](Inst *i,bool root) {
    if (!pos.contains(i) || pos[i]<at || moving.contains(i)) return true;
    if (i==before) return root;
    if ((!root && !address(i)) || !visiting.insert(i).second) return false;
    for (auto &op:i->operands())
      if (auto *dep=dyn_cast<Inst>(op.get())) if (!collect(dep,false)) return false;
    visiting.erase(i); moving.insert(i); return true;
  };
  for (auto *i:roots) if (!collect(i,true)) return false;
  Order next;
  for (auto *i:order) {
    if (i==before) for (auto *j:order) if (moving.contains(j)) next.push_back(j);
    if (!moving.contains(i)) next.push_back(i);
  }
  order=std::move(next);
  return true;
}

// 核心调度，对一个区域依次做：
//   1. 配对 load。
//   2. 把每对的两个 load 贴在一起（所有配对都做，包括后面提前不了的）。
//   3. 把每对尽量提前，最多越过 distance 条 MMAC。
//   4. 校验没有拆开配对、没有反转 SSA 依赖，然后提交。
//   5. 在被越过的 MMAC 前后加 sched.barrier(0)，防止调度器把读移回去。
// marker / pairMetadata 由两个 Pass 分别传入。
void schedule(Order region, unsigned distance, Stats &stats,
              StringRef marker = Marker, StringRef pairMetadata = "frisk.k_lds_pair") {
  if (region.empty()) return;
  // original：原顺序，用来确定提交时的插入位置。
  // loads / mmas：区域内的 load 和 MMAC，mmas 的下标就是 MMAC 序号。
  // pos：区域内位置。
  // firstUse[l]：第一条把 l 作为 B 操作数的 MMAC 序号，等于 mmas.size() 表示没有。
  Order original=region;
  SmallVector<LoadInst *> loads;
  SmallVector<CallInst *> mmas;
  DenseMap<Inst *,unsigned> pos, firstUse;
  for (unsigned n=0;n<region.size();++n) {
    auto *i=region[n]; pos[i]=n;
    if (auto *l=dyn_cast<LoadInst>(i)) loads.push_back(l);
    if (isTypedRegisterMMACall(i)) mmas.push_back(cast<CallInst>(i));
  }
  if (loads.size()<2 || mmas.size()<2) { stats.unpaired+=loads.size(); return; }
  for (auto *l:loads) {
    unsigned first=mmas.size();
    for (unsigned n=0;n<mmas.size();++n)
      if (mmas[n]->getArgOperand(1)==l) { first=n; break; }
    firstUse[l]=first;
  }
  auto &dl=region.front()->getModule()->getDataLayout();
  // paired：已配对的 load。pairs：配对结果。
  SmallPtrSet<LoadInst *,32> paired;
  SmallVector<Pair> pairs;
  // Prefer normal read2 offsets, then read2st64. Pairing is deterministic and
  // uses only a common pointer base plus a constant b64-encodable difference.
  // No table of old assembly registers, SSA names, or attention tile sizes.
  for (auto *a:loads) {
    if (paired.contains(a) || firstUse[a]==mmas.size()) continue;
    int64_t oa=0; auto *ba=GetPointerBaseWithConstantOffset(a->getPointerOperand(),oa,dl);
    LoadInst *best=nullptr;
    unsigned bestClass=3, bestUse=std::numeric_limits<unsigned>::max();
    for (auto *b:loads) {
      if (b==a || paired.contains(b) || pos[b]<pos[a] || firstUse[b]==mmas.size()) continue;
      int64_t ob=0; auto *bb=GetPointerBaseWithConstantOffset(b->getPointerOperand(),ob,dl);
      if (ba!=bb) continue;
      // Unsigned subtraction avoids signed overflow for adversarial offsets.
      uint64_t delta=oa<ob ? uint64_t(ob)-uint64_t(oa) : uint64_t(oa)-uint64_t(ob);
      // kind：地址差能否编码。
      //   0  read2_b64，差是 8 的倍数且不超过 255*8。
      //   1  read2st64_b64，差是 512 的倍数且不超过 255*512。
      //   2  无法编码。
      // 优先选 kind 小的，其次选 firstUse 早的。
      unsigned kind=delta && delta%8==0 && delta<=255*8 ? 0 :
                    delta && delta%512==0 && delta<=255*512 ? 1 : 2;
      if (kind==2 || kind>bestClass || (kind==bestClass && firstUse[b]>=bestUse)) continue;
      // 在副本上试移动，确认 b 能紧贴到 a 之后。
      Order trial=region;
      // Close the pair at its earlier read; include the later read's address
      // dependencies, without crossing the earlier read itself.
      auto it=llvm::find(trial,static_cast<Inst *>(a));
      if (std::next(it)==trial.end() || !hoist(trial,{b},*std::next(it))) continue;
      best=b; bestClass=kind; bestUse=firstUse[b];
    }
    if (best) {
      paired.insert(a); paired.insert(best);
      pairs.push_back({a,best,std::min(firstUse[a],firstUse[best])});
    }
  }
  // First close ALL pairs, including those which cannot be prefetched. This
  // prevents a new sched.barrier from splitting a later pair's two halves.
  for (auto &p:pairs) {
    auto it=llvm::find(region,static_cast<Inst *>(p.a));
    if (std::next(it)==region.end() || !hoist(region,{p.b},*std::next(it))) return;
  }
  // anchors：被读越过的 MMAC，也就是要加调度保护的位置。
  // moved：成功提前的配对数。
  SmallPtrSet<Inst *,32> anchors;
  unsigned moved=0;
  for (auto &p:pairs) {
    // 从最早的目标开始试，能提多早提多早。
    unsigned start=p.first>distance ? p.first-distance : 0;
    for (unsigned n=start;n<p.first;++n) {
      auto *anchor=mmas[n];
      auto a=llvm::find(region,static_cast<Inst *>(p.a));
      auto b=llvm::find(region,static_cast<Inst *>(p.b));
      auto target=llvm::find(region,static_cast<Inst *>(anchor));
      // Both reads must move together from after the independent MMAC.
      if (a<target || b<target) continue;
      if (hoist(region,{p.a,p.b},anchor)) { anchors.insert(anchor); ++moved; break; }
    }
  }
  if (anchors.empty()) return;
  // Prove no hint splits a pair and no SSA edge was reversed before committing.
  pos.clear();
  for (unsigned n=0;n<region.size();++n) pos[region[n]]=n;
  for (auto &p:pairs) for (auto *anchor:anchors)
    if (pos[p.a]<pos[anchor] && pos[anchor]<pos[p.b]) return;
  for (auto *i:region) for (auto &op:i->operands())
    if (auto *dep=dyn_cast<Inst>(op.get()); dep && pos.contains(dep) && pos[dep]>=pos[i]) return;
  Inst *end=original.back()->getNextNode();
  for (auto *i:region) i->moveBefore(end->getIterator());
  // pairID：配对编号，写入元数据。后续 Pass 据此识别已保护的配对。
  unsigned pairID=stats.pairs;
  for (auto &p:pairs) {
    auto *tag=MDNode::get(end->getContext(),ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(end->getContext()),pairID++)));
    p.a->setMetadata(pairMetadata,tag);
    p.b->setMetadata(pairMetadata,tag);
  }
  auto *hint=Intrinsic::getOrInsertDeclaration(end->getModule(),Intrinsic::amdgcn_sched_barrier);
  auto *mark=MDNode::get(end->getContext(),{});
  for (auto *anchor:anchors) {
    IRBuilder<> before(anchor), after(anchor->getNextNode());
    for (auto *guard:{before.CreateCall(hint,{before.getInt32(0)}),after.CreateCall(hint,{after.getInt32(0)})}) {
      guard->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
      guard->setMetadata(marker,mark);
    }
  }
  stats.regions++; stats.pairs+=pairs.size(); stats.moved+=moved;
  stats.anchors+=anchors.size(); stats.unpaired+=loads.size()-2*pairs.size();
}

// These instructions stay in their original order. Only LDS reads and their
// pure integer address dependencies move across them. In particular, no FP
// reassociation, speculation of calls, or relaxation of FP flags is needed.
// 第 31 项允许出现在区域内的局部归约指令。它们保持原有顺序，只有 LDS 读和地址计算越过它们。
// reductionWrapper：归约周围的向量/聚合包装及纯寄存器 mask 比较/选择。
// unprotectedCandidate：还没被任何配对保护的候选读。
bool localReduction(Inst *i) {
  if (i->getOpcode() == Instruction::FAdd) return true;
  auto *c = dyn_cast<IntrinsicInst>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall() ||
      !c->doesNotAccessMemory() || c->isConvergent()) return false;
  switch (c->getIntrinsicID()) {
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
  case Intrinsic::maximum:
  case Intrinsic::minimum:
    return true;
  default:
    return false;
  }
}
bool reductionWrapper(Inst *i) {
  // Causal masks lowered directly to score/-inf selects must not split the
  // QK-tail region. These instructions stay in place, like the reductions;
  // only independent LDS reads and their checked address dependencies move.
  return isa<ExtractElementInst, InsertElementInst, ShuffleVectorInst,
             ExtractValueInst, InsertValueInst, ICmpInst, SelectInst>(i);
}
bool unprotectedCandidate(Inst *i) {
  auto *l = dyn_cast<LoadInst>(i);
  return candidate(l) && !l->getMetadata("frisk.k_lds_pair") &&
         !l->getMetadata(TailPair);
}
} // namespace

bool parsePairedLDSOption(llvm::StringRef arg,unsigned &distance,bool &consumed) {
  consumed=arg.starts_with("--k-lds-prefetch");
  if (!consumed) return true;
  return arg.consume_front("--k-lds-prefetch=") && !arg.getAsInteger(10,distance) && distance<=8;
}

llvm::PreservedAnalyses PrefetchPairedLDSPass::run(llvm::Function &f,llvm::FunctionAnalysisManager &) {
  if (!distance || distance>8 || f.hasFnAttribute(Marker)) return PreservedAnalyses::all();
  Stats stats;
  // Regions end at ALL operations other than approved address calculations,
  // simple B-operand LDS reads, and pure register MMAC. In particular we never
  // cross any DMA, wait, fence, store, wave operation, hint or unknown call.
  // 区域由连续的“地址计算、候选读、MMAC”组成。遇到其它任何指令（DMA、wait、fence、hint 等）就结束。
  SmallVector<Order,4> regions;
  for (auto &block:f) {
    Order region;
    for (auto &i:block) {
      if (address(&i) || candidate(dyn_cast<LoadInst>(&i)) || isTypedRegisterMMACall(&i)) region.push_back(&i);
      else if (!region.empty()) { regions.push_back(std::move(region)); region.clear(); }
    }
    if (!region.empty()) regions.push_back(std::move(region));
  }
  for (auto &region:regions) schedule(std::move(region),distance,stats);
  f.addFnAttr(Marker,Twine(distance).str());
  errs()<<"K LDS prefetch: distance="<<distance<<", regions="<<stats.regions<<", pairs="<<stats.pairs
        <<", moved="<<stats.moved<<", anchors="<<stats.anchors<<", unpaired="<<stats.unpaired
        <<", function="<<f.getName()<<"\n";
  auto preserved=PreservedAnalyses::none(); preserved.preserveSet<CFGAnalyses>(); return preserved;
}

bool prefetchPairedLDS(llvm::Module &module,unsigned distance) {
  if (!distance) return true;
  if (distance>8) return false;
  for (auto &f:module) if (!f.isDeclaration() && f.hasFnAttribute(Marker) &&
      f.getFnAttribute(Marker).getValueAsString()!=Twine(distance).str()) {
    errs()<<"K LDS prefetch: conflicting prior distance in "<<f.getName()<<"\n"; return false;
  }
  LoopAnalysisManager loops;
  FunctionAnalysisManager analyses;
  CGSCCAnalysisManager cgscc;
  ModuleAnalysisManager modules;
  PassBuilder pb;
  pb.registerModuleAnalyses(modules); pb.registerCGSCCAnalyses(cgscc);
  pb.registerFunctionAnalyses(analyses); pb.registerLoopAnalyses(loops);
  pb.crossRegisterProxies(loops,analyses,cgscc,modules);
  // 先跑 CSE 和 InstCombine，合并重复的读，暴露更多可配对的读。
  FunctionPassManager pipeline;
  pipeline.addPass(EarlyCSEPass());
  pipeline.addPass(InstCombinePass());
  pipeline.addPass(EarlyCSEPass());
  pipeline.addPass(PrefetchPairedLDSPass(distance));
  for (auto &f:module) if (!f.isDeclaration() && !f.hasFnAttribute(Marker)) pipeline.run(f,analyses);
  return true;
}

bool parsePairedLDSTailOption(StringRef arg, unsigned &distance, bool &consumed) {
  consumed = arg.starts_with("--k-tail-prefetch");
  if (!consumed) return true;
  return arg.consume_front("--k-tail-prefetch=") &&
         !arg.getAsInteger(10, distance) && distance <= 8;
}

PreservedAnalyses PrefetchPairedLDSTailPass::run(Function &f, FunctionAnalysisManager &) {
  if (!distance || distance > 8 || f.isDeclaration() || f.hasFnAttribute(TailMarker) ||
      f.hasFnAttribute(Attribute::OptimizeNone) || f.hasFnAttribute(Attribute::StrictFP))
    return PreservedAnalyses::all();
  Stats stats;
  unsigned reductionRegions = 0;
  // 只保留含归约的区域，因为不含归约的区域第 29 项已经处理过。
  // reductionRegions：这种区域的数量，只用于日志。
  SmallVector<Order, 4> regions;
  for (auto &block : f) {
    Order region;
    bool hasReduction = false;
    auto flush = [&]() {
      if (hasReduction) {
        regions.push_back(std::move(region));
        ++reductionRegions;
      }
      region.clear();
      hasReduction = false;
    };
    for (auto &i : block) {
      bool reduction = localReduction(&i);
      if (address(&i) || unprotectedCandidate(&i) || isTypedRegisterMMACall(&i) ||
          reduction || reductionWrapper(&i)) {
        region.push_back(&i);
        hasReduction |= reduction;
      } else {
        // Also includes existing hints, protected pairs, other memory reads,
        // constrained FP, wave operations and unknown readnone calls.
        flush();
      }
    }
    flush();
  }
  for (auto &region : regions)
    schedule(std::move(region), distance, stats, TailMarker, TailPair);
  errs() << "K tail LDS prefetch: distance=" << distance << ", regions=" << stats.regions
         << ", pairs=" << stats.pairs << ", moved=" << stats.moved
         << ", anchors=" << stats.anchors << ", unpaired=" << stats.unpaired
         << ", reduction-regions=" << reductionRegions << ", function=" << f.getName() << "\n";
  if (!stats.regions) return PreservedAnalyses::all();
  f.addFnAttr(TailMarker, Twine(distance).str());
  auto preserved = PreservedAnalyses::none();
  preserved.preserveSet<CFGAnalyses>();
  return preserved;
}

bool prefetchPairedLDSTail(Module &module, unsigned distance) {
  if (!distance) return true;
  if (distance > 8) return false;
  for (auto &f : module)
    if (!f.isDeclaration() && f.hasFnAttribute(TailMarker) &&
        f.getFnAttribute(TailMarker).getValueAsString() != Twine(distance).str()) {
      errs() << "K tail LDS prefetch: conflicting prior distance in " << f.getName() << "\n";
      return false;
    }
  FunctionAnalysisManager analyses;
  PrefetchPairedLDSTailPass pass(distance);
  for (auto &f : module) pass.run(f, analyses);
  return true;
}
} // namespace mlir::frisk
