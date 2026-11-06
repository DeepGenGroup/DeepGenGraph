// ===========================================================================
// Pass: interleavePVAccumulators（InterleavePVPass）   优化汇总.py 第 26 项，LLVM 阶段，--pv-interleave=4
// 详细设计见同目录 PVInterleave.md。
//
// 作用：
//   把“一条累加链做完再做下一条”改成多条链交错，相邻 MMAC 没有 C 依赖，可以连续发射。每条链内部
//   顺序不变，所以结果逐位相同。
//     before:  a a a ... b b b ... c c c ... d d d ...
//     after:   a b c d a b c d ...
//
// width 选项：
//   0     关闭。
//   1     不交错，只加调度保护，作为对照。
//   2/4   每组交错的链数。链的总数必须能被 width 整除。
//
// 流程：
//   1. 在每个基本块里找 MMAC 链（getChain）。B 操作数来自同一组矩阵读包的链归为一组。
//   2. schedule：先完整生成计划并校验，通过后才改 IR。
//   3. 在每条 MMAC 前后加 sched.barrier(0)，防止后端调度器把顺序改回去。
//
// 实测收益接近噪声（优化汇总.py 第 26 项）。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/InterleavePV.h"
#include "LateIRPatterns.h"
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
using detail::MatrixReadM32x16;
using detail::getInlineAsm;
using detail::isCallTo;
using detail::isDMA128Asm;
using detail::isPureInst;
using detail::isSchedHint;
using detail::isTypedRegisterMMACall;
// Marker：标在已处理的 MMAC 上，值为 width。getChain 遇到它就放弃，防止重复处理。
constexpr llvm::StringLiteral Marker = "frisk.pv_interleave";

// Only the known low/high half of one matrix packet is accepted. No arbitrary
// arithmetic, selects or multi-packet shuffles can disguise the producer.
// B 操作数必须是一次矩阵读（ds.read.m32x16）的低或高 4 个 half。返回对应的读调用。
Call *matrixPacket(llvm::Value *v) {
  auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
  if (!shuffle || !llvm::isa<llvm::PoisonValue>(shuffle->getOperand(1))) return nullptr;
  auto mask = shuffle->getShuffleMask();
  if (mask.size() != 4 || (mask[0] != 0 && mask[0] != 4)) return nullptr;
  for (unsigned j = 1; j < 4; ++j)
    if (mask[j] != mask[0]+int(j)) return nullptr;
  auto *cast = llvm::dyn_cast<llvm::BitCastInst>(shuffle->getOperand(0));
  auto *read = cast ? llvm::dyn_cast<Call>(cast->getOperand(0)) : nullptr;
  if (!read || !isCallTo(read, MatrixReadM32x16) ||
      read->arg_size() != 2) return nullptr;
  auto *pointer = llvm::dyn_cast<llvm::PointerType>(read->getArgOperand(0)->getType());
  return pointer && pointer->getAddressSpace() == 3 &&
      read->getType() == llvm::FixedVectorType::get(llvm::Type::getInt32Ty(read->getContext()), 4) &&
      read->getArgOperand(1)->getType()->isIntegerTy(32) ? read : nullptr;
}

// These operations stay in order and at the same MMAC ordinal. Unknown calls,
// stores, atomics, reads and nonzero scheduling masks reject the whole region.
// 可以出现在区域内、但必须保持在原 MMAC 序号处的 op：fence、barrier、readfirstlane、DMA。
// 其它有副作用的 op 会让整个区域放弃。
bool isFixed(Inst *i) {
  if (auto *f = llvm::dyn_cast<llvm::FenceInst>(i))
    return f->getSyncScopeID() == f->getContext().getOrInsertSyncScopeID("workgroup") &&
        (f->getOrdering() == llvm::AtomicOrdering::Acquire ||
         f->getOrdering() == llvm::AtomicOrdering::Release);
  if (isCallTo(i, "llvm.amdgcn.s.barrier")) return llvm::cast<Call>(i)->arg_empty();
  // Fresh MyTest export still contains DMA-address readfirstlane calls that
  // O3 folds/hoists in the archived baseline. Keep these wave operations fixed;
  // never treat them as speculative scalar arithmetic or move across a fence.
  if (isCallTo(i, "llvm.amdgcn.readfirstlane.i32")) {
    auto *c = llvm::cast<Call>(i);
    return c->arg_size() == 1 && c->getType()->isIntegerTy(32) &&
        c->getArgOperand(0)->getType()->isIntegerTy(32);
  }
  auto *a = getInlineAsm(i);
  return a && a->hasSideEffects() && isDMA128Asm(a) && llvm::cast<Call>(i)->arg_size() == 3;
}

// 一条累加链。mmas[k] 是第 k 条 MMAC，packets[k] 是它的 B 操作数来自哪次矩阵读。
struct Chain {
  llvm::SmallVector<Call *, 16> mmas, packets;
};

// 从链头沿 C 输入追到链尾。任何不符合的情况都返回空链。
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
      if (isTypedRegisterMMACall(candidate) && candidate->getArgOperand(2) == c) {
        if (next || !c->hasOneUse() || candidate->getParent() != root->getParent()) return {};
        next = candidate;
      }
    }
    c = next;
  }
  return chain;
}

// 核心调度。区域定义为：最后一次矩阵读之后，到最后一条 MMAC（含其后的调度保护）之间。
bool schedule(llvm::ArrayRef<Chain *> chains, unsigned width) {
  if (chains.size() < 2 || chains.size() > 16 || chains.size() % width) return false;
  auto *block = chains.front()->mmas.front()->getParent();
  // positions：指令在块内的位置。
  // mmas：所有链的 MMAC。
  // lastRead / lastMMA / firstMMA：用于界定区域，并确认读都在 MMAC 之前。
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
  while (isSchedHint(end)) end = end->getNextNode();
  // region：区域内的全部指令。
  // hints：区域内已有的调度保护，改写时删除后重新生成。
  // members：region 的集合形式。
  // fixed[k]：原本位于第 k 条 MMAC 之前的不可移动 op，新顺序里同样放在第 k 条 MMAC 之前。
  Insts region, hints;
  llvm::SmallPtrSet<Inst *, 32> members;
  llvm::SmallVector<Insts, 32> fixed(mmas.size()+1);
  unsigned count = 0;
  for (auto *i = lastRead->getNextNode(); i != end; i = i->getNextNode()) {
    region.push_back(i);
    members.insert(i);
    if (mmas.contains(i)) ++count;
    else if (isSchedHint(i)) hints.push_back(i);
    else if (!isPureInst(i)) {
      if (!isFixed(i)) return false;
      fixed[count].push_back(i);
    }
  }
  // Fully validate a topological schedule BEFORE mutating the IR. A requested
  // MMAC cannot pull a different MMAC, memory operation or synchronization
  // forward as a dependency; an unsupported dependency rejects the region.
  // desired：期望的 MMAC 顺序，也就是交错后的序列。
  // plan：最终的完整指令顺序，由 emit 做拓扑展开生成。
  // emit：发射 i 并先发射它的依赖。依赖只能是纯计算；要求连带发射其它 MMAC 或有副作用的 op 时失败。
  Insts desired, plan;
  for (unsigned group = 0; group < chains.size(); group += width)
    for (unsigned k = 0; k < chains.front()->mmas.size(); ++k)
      for (unsigned c = group; c < group+width; ++c)
        desired.push_back(chains[c]->mmas[k]);
  llvm::SmallPtrSet<Inst *, 32> emitted, visiting;
  std::function<bool(Inst *, bool)> emit = [&](Inst *i, bool explicitOp) {
    if (!members.contains(i) || emitted.contains(i)) return true;
    if ((!explicitOp && !isPureInst(i)) || !visiting.insert(i).second) return false;
    for (auto &operand : i->operands())
      if (auto *dep = llvm::dyn_cast<Inst>(operand.get()))
        if (!emit(dep, false)) return false;
    visiting.erase(i);
    emitted.insert(i);
    plan.push_back(i);
    return true;
  };
  // 按 slot 交替放置 fixed 和 MMAC，最后补上未被依赖的纯计算。
  for (unsigned slot = 0; slot <= desired.size(); ++slot) {
    for (auto *i : fixed[slot]) if (!emit(i, true)) return false;
    if (slot < desired.size() && !emit(desired[slot], true)) return false;
  }
  for (auto *i : region)
    if (!isSchedHint(i) && !emit(i, false)) return false;
  if (plan.size()+hints.size() != region.size()) return false;

  // 走到这里计划已通过校验，开始改 IR。
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
      if (!isTypedRegisterMMACall(&i)) continue;
      auto *call = llvm::cast<Call>(&i);
      if (isTypedRegisterMMACall(llvm::dyn_cast<Inst>(call->getArgOperand(2)))) continue;
      auto chain = getChain(call);
      if (chain.mmas.size() >= 2) candidates.push_back(std::move(chain));
    }
    llvm::SmallPtrSet<Chain *, 16> visited;
    // 把 packets 完全相同的链归为一组，它们才是同一片 V 上的多条 PV 链。
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
