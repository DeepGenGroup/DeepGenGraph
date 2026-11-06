// ===========================================================================
// Pass: doubleBufferV（DoubleBufferVPass）   优化汇总.py 第 27 项，LLVM 阶段，--v-double-buffer=1
// 详细设计见同目录 VDoubleBuffer.md。
//
// 作用：
//   V 的 4 片原本都放在同一块 LDS，下一片的 DMA 必须等当前片读完（要一个 reader barrier）。
//   改成 V0/V2 用 slot0、V1/V3 用 slot1，下一片不再覆盖当前片：
//     1. 删除 reader barrier（stage 数 - 1 个）。
//     2. DMA 提前到“本片 ready 之后立刻发”。
//   不增加 LDS 总量，复用 K/Q 已经用完的另一半。
//
// 流程：
//   run   函数级检查：kernel、固定 1D workgroup、wave64、唯一的静态 LDS、控制流 uniform。
//   plan  在一个基本块上识别 stage，并证明合法性。只分析，不改 IR；失败时打印原因。
//   apply 移动 DMA，奇数片偏移一个 slot，删除 handoff barrier。
//
// 识别基于指令结构和地址证明，不依赖 SSA 名字或 tile 尺寸。IR 形状的定义见 LateIRPatterns.h。
// ===========================================================================
#include "LDSAddressAnalysis.h"
#include "deepgengraph/Conversion/ConvertToLLVM/DoubleBufferV.h"
#include "LateIRPatterns.h"
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
using detail::LDSAddress;
using detail::Sync;
using detail::analyzeWave64Kernel;
using detail::isCallTo;
using detail::isDMA128;
using detail::isPureInst;
using detail::isRegisterMMACall;
using detail::isSchedHint;
using detail::isWaitCall;
using detail::matchSync;
using detail::MatrixReadM32x16;
// Marker：标在已改写的矩阵读上，值为 slot 字节数，防止重复处理。
constexpr StringLiteral Marker = "frisk.v_double_buffer";

// Export may construct the same resource descriptor at every DMA site, before
// CSE. Compare its pure DAG, including opcode flags/shuffle masks/constants.
// 两个值是否由结构相同的纯计算产生。用来判断两条 DMA 的资源描述符是否同一个（是否都在读 V）。
bool sameValue(Value *a, Value *b, unsigned depth = 0) {
  if (a == b) return true;
  if (depth > 32 || a->getType() != b->getType()) return false;
  auto *ia = dyn_cast<Inst>(a), *ib = dyn_cast<Inst>(b);
  if (!ia || !ib || !isPureInst(ia) || !isPureInst(ib) || !ia->isSameOperationAs(ib)) return false;
  for (unsigned k = 0; k < ia->getNumOperands(); ++k)
    if (!sameValue(ia->getOperand(k),ib->getOperand(k),depth+1)) return false;
  return true;
}

// V 的一片。
//   ready         这一片的数据就绪 barrier（带 wait）。
//   handoff       这一片读完后的读者交接 barrier，apply 时删除。
//   reads         这一片的矩阵读。
//   copies        把这一片搬进 LDS 的 DMA。
//   dependencies  这些 DMA 依赖的、要跟着提前克隆的地址计算。
struct Stage {
  Sync ready, handoff;
  SmallVector<Call *, 16> reads, copies;
  SmallVector<Inst *, 16> dependencies;
};
//   stages        所有片。
//   finish        最后一片读完后的 barrier，保留。
//   displacement  奇数片相对原地址的字节偏移。
//   sliceBytes    一个 slot 的字节数。
struct Plan {
  SmallVector<Stage, 8> stages;
  Sync finish;
  int64_t displacement = 0;
  unsigned sliceBytes = 0;
};

// All writes cover a full contiguous slice, with no duplicate lane writes.
// 枚举所有线程，算出这组 DMA 写入的字节集合 bytes。
// 要求 m0 在 wave 内一致，没有重复写入。这里写死了 wave64 和每 lane 16 字节。
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

// 合法性证明。步骤：
//   1. 按矩阵读归属的 ready barrier 划分 stage。
//   2. 每两个 stage 之间找到 DMA 和 handoff，顺序必须是 读 -> handoff -> DMA。
//   3. 所有片的 DMA 写入范围必须完全相同，从而确定当前 slot 和备用 slot。
//   4. 所有读都在 DMA 范围内；区域内没有未知操作。
//   5. 备用 slot 的旧内容已死，包括跨回边的情况。
//   6. DMA 依赖的地址计算可以提前。
bool plan(BasicBlock &block, GlobalVariable *storage, unsigned capacity,
          unsigned threads, DominatorTree &dt, Plan &p, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  auto &dl = block.getModule()->getDataLayout();
  LDSAddress addresses(dl,storage);
  // syncs：到当前位置为止遇到的所有同步序列。
  SmallVector<Sync,16> syncs;
  for (auto &i : block) {
    if (i.getMetadata(Marker)) return fail("already transformed");
    if (auto s = matchSync(&i)) syncs.push_back(*s);
    if (!isCallTo(&i,MatrixReadM32x16)) continue;
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
      if (isDMA128(i)) p.stages[s].copies.push_back(cast<Call>(i));
      if (auto b = matchSync(i)) {
        if (previous.handoff.barrier || b->wait) return fail("ambiguous reader handoff");
        previous.handoff = *b;
      }
    }
    if (!previous.handoff.barrier || p.stages[s].copies.empty()) return fail("missing reader handoff/DMA");
    if (!previous.reads.back()->comesBefore(previous.handoff.release) ||
        !previous.handoff.acquire->comesBefore(p.stages[s].copies.front())) return fail("unsupported read/handoff order");
  }
  // resource：V 的资源描述符。第 0 片的 DMA 在第一个 wait 之前，用 resource 从块头找出来。
  Value *resource = p.stages[1].copies.front()->getArgOperand(2);
  for (auto &i : block) {
    if (&i == p.stages.front().ready.wait) break;
    if (isDMA128(&i) && sameValue(cast<Call>(&i)->getArgOperand(2),resource))
      p.stages.front().copies.push_back(cast<Call>(&i));
  }
  // original：原来所有片共用的字节范围，必须连续。
  // begin / size：这段范围的起点和长度。
  // spare：备用 slot 起点，优先放在后面，放不下就放在前面。
  BitVector original;
  if (!footprint(p.stages.front().copies,addresses,threads,capacity,original)) return fail("unknown initial DMA footprint");
  unsigned begin = original.find_first(), size = original.count();
  if (begin+size > capacity || original.find_last() != int(begin+size-1)) return fail("non-contiguous DMA slice");
  int64_t spare = begin+size;
  if (spare+size > capacity) spare = int64_t(begin)-size;
  if (spare < 0) return fail("no spare LDS slice");
  p.displacement = spare-begin;
  p.sliceBytes = size;
  // allowedMemory：已确认属于 V 的 DMA 和矩阵读。其它碰内存的指令都要单独证明。
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
    if (auto s = matchSync(i)) { p.finish = *s; break; }
  if (!p.finish.barrier || p.finish.wait) return fail("missing final reader barrier");
  SmallPtrSet<Inst *,32> syncOps;
  for (auto &s : syncs)
    for (auto *i : {s.release,s.barrier,s.acquire,s.wait}) if (i) syncOps.insert(i);
  for (auto *i = p.stages.front().ready.acquire->getNextNode(); i != p.finish.acquire; i = i->getNextNode()) {
    if (allowedMemory.contains(i) || syncOps.contains(i) || isRegisterMMACall(i) || isSchedHint(i) || isPureInst(i) ||
        isCallTo(i,"llvm.amdgcn.readfirstlane.i32")) continue;
    return fail("unknown operation or other live LDS user in V region");
  }

  // Prove the alternate slot's old contents are dead, including the loop
  // backedge: any earlier same-block read of that slot must be preceded by a
  // write in this iteration. Other LDS readers must strictly dominate this
  // block (prologue Q reads); readers in successors/other loops are rejected.
  // written：V 区域之前，同一块内已被 DMA 写入的字节。
  // 读备用 slot 的字节时，如果它们本轮还没写过，说明旧内容跨迭代存活，不能占用。
  BitVector written(capacity);
  for (auto &i : instructions(*block.getParent())) {
    bool before = i.getParent() == &block && i.comesBefore(p.stages.front().ready.wait);
    if (isDMA128(&i)) {
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
    if (i.mayReadOrWriteMemory() && !syncOps.contains(&i) && !isWaitCall(&i) && !isSchedHint(&i) &&
        !isCallTo(&i,"llvm.amdgcn.s.barrier") && !isa<FenceInst>(&i))
      return fail("unknown memory effect outside V region");
  }
  // Preflight the dependencies of every moved DMA. Clone only pure address
  // computations and known lane broadcasts; never hoist reads or another DMA.
  // anchor：上一片 ready barrier 之后的位置，DMA 要提前到这里。
  // visit：收集 DMA 参数依赖的、在 anchor 之后定义的指令。只允许纯计算和 readfirstlane。
  for (unsigned s = 1; s < p.stages.size(); ++s) {
    auto *anchor = p.stages[s-1].ready.acquire->getNextNode();
    SmallPtrSet<Inst *,32> seen;
    std::function<bool(Value *)> visit = [&](Value *v) {
      auto *i = dyn_cast<Inst>(v);
      if (!i || (dt.dominates(i,anchor) && i != anchor) || seen.contains(i)) return true;
      if (i->getParent() != &block || (!isPureInst(i) && !isCallTo(i,"llvm.amdgcn.readfirstlane.i32"))) return false;
      if (isCallTo(i,"llvm.amdgcn.readfirstlane.i32") && !addresses.get(i,0)) return false;
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

// 对每个 stage：克隆依赖到 anchor，把 DMA 移到 anchor；s 为奇数时 DMA 和读都加 displacement。
// 最后删除所有 handoff barrier。
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
  std::string why;
  auto kernel = analyzeWave64Kernel(f, why);
  if (!kernel) return skip(why);
  llvm::DominatorTree dt(f);
  llvm::LoopInfo loops(dt);
  unsigned changed = 0;
  bool candidate = false;
  for (auto &b : f) {
    if (!llvm::any_of(b,[](Inst &i) { return isCallTo(&i,MatrixReadM32x16); })) continue;
    candidate = true;
    auto *loop = loops.getLoopFor(&b);
    if (loop) {
      // Fresh MLIR export keeps a PHI/condition-only header separate from the
      // compute/latch block; O3 folds it to the archived single-block loop.
      // 只接受一个计算块，可以带一个纯的循环头。
      bool simple = !loop->getParentLoop() && (loop->getNumBlocks() == 1 ||
          (loop->getNumBlocks() == 2 && loop->getLoopLatch() == &b && loop->getHeader() != &b));
      for (auto *other : loop->blocks()) if (other != &b)
        for (auto &i : *other)
          simple &= isPureInst(&i) || llvm::isa<llvm::PHINode,llvm::CondBrInst,llvm::UncondBrInst>(&i);
      if (!simple) { skip("requires one compute block with an optional pure loop header"); continue; }
    }
    Plan p;
    std::string reason;
    if (!plan(b,kernel->storage,kernel->ldsBytes,kernel->threads,dt,p,reason)) { skip(reason); continue; }
    apply(p);
    ++changed;
    llvm::errs() << "V double buffer: " << p.stages.size() << " slices, slot-bytes=" << p.sliceBytes
                 << ", displacement=" << p.displacement << ", removed-barriers=" << p.stages.size()-1
                 << ", LDS=" << kernel->ldsBytes << ", function=" << f.getName() << "\n";
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
