#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchPairedLDS.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
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
using Order = SmallVector<Inst *, 64>;
constexpr StringLiteral Marker = "frisk.k_lds_prefetch";
constexpr StringLiteral TailMarker = "frisk.k_tail_prefetch";
constexpr StringLiteral TailPair = "frisk.k_tail_pair";

bool mma(Inst *i) {
  auto *c = dyn_cast_or_null<CallInst>(i);
  if (!c || c->arg_size()!=3 || c->hasOperandBundles() || c->isMustTailCall() ||
      !c->doesNotAccessMemory()) return false;
  auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
  auto &ctx=c->getContext();
  auto *ab=FixedVectorType::get(Type::getHalfTy(ctx),4);
  auto *cd=FixedVectorType::get(Type::getFloatTy(ctx),4);
  return a && !a->hasSideEffects() && !a->canThrow() && !a->isAlignStack() &&
      isRegisterMMA(a->getAsmString(),a->getConstraintString()) &&
      c->getType()==cd && c->getArgOperand(0)->getType()==ab &&
      c->getArgOperand(1)->getType()==ab && c->getArgOperand(2)->getType()==cd;
}
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
    if (!mma(c) || u.getOperandNo()!=1 || c->getParent()!=l->getParent()) return false;
  }
  return true;
}
struct Pair { LoadInst *a; LoadInst *b; unsigned first; };
struct Stats { unsigned regions=0, pairs=0, moved=0, anchors=0, unpaired=0; };

// Plan against a virtual order. Only hoisting is allowed, never delaying a
// read past a use. A dependency on another load/call rejects the movement.
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

void schedule(Order region, unsigned distance, Stats &stats,
              StringRef marker = Marker, StringRef pairMetadata = "frisk.k_lds_pair") {
  if (region.empty()) return;
  Order original=region;
  SmallVector<LoadInst *> loads;
  SmallVector<CallInst *> mmas;
  DenseMap<Inst *,unsigned> pos, firstUse;
  for (unsigned n=0;n<region.size();++n) {
    auto *i=region[n]; pos[i]=n;
    if (auto *l=dyn_cast<LoadInst>(i)) loads.push_back(l);
    if (mma(i)) mmas.push_back(cast<CallInst>(i));
  }
  if (loads.size()<2 || mmas.size()<2) { stats.unpaired+=loads.size(); return; }
  for (auto *l:loads) {
    unsigned first=mmas.size();
    for (unsigned n=0;n<mmas.size();++n)
      if (mmas[n]->getArgOperand(1)==l) { first=n; break; }
    firstUse[l]=first;
  }
  auto &dl=region.front()->getModule()->getDataLayout();
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
      unsigned kind=delta && delta%8==0 && delta<=255*8 ? 0 :
                    delta && delta%512==0 && delta<=255*512 ? 1 : 2;
      if (kind==2 || kind>bestClass || (kind==bestClass && firstUse[b]>=bestUse)) continue;
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
  SmallPtrSet<Inst *,32> anchors;
  unsigned moved=0;
  for (auto &p:pairs) {
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
  return isa<ExtractElementInst, InsertElementInst, ShuffleVectorInst,
             ExtractValueInst, InsertValueInst>(i);
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
  SmallVector<Order,4> regions;
  for (auto &block:f) {
    Order region;
    for (auto &i:block) {
      if (address(&i) || candidate(dyn_cast<LoadInst>(&i)) || mma(&i)) region.push_back(&i);
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
      if (address(&i) || unprotectedCandidate(&i) || mma(&i) ||
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
