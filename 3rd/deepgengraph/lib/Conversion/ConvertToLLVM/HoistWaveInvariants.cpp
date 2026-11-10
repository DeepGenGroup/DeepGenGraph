// ===========================================================================
// Pass: HoistWaveInvariants (frisk-hoist-wave-invariants).
//
// 作用：把每轮重算的 LDS 基址和 lane 索引移到满足不变性条件的循环外。
//   hoistBases         每条 DMA 的 LDS 基址（readfirstlane 链）。
//   hoistShuffleSetup  mbcnt 的 lane 号和由它算出的 bpermute 索引。
//
// 运行时机：DMA/shuffle 局部 lowering 之后、firstLowering 之前。
//
// 下游依赖：lowerWideAsyncDMAToBufferAsm 的 reuseLeaderRead 依赖这里提到入口的基址。
// 下面是原文件头的合法性说明。
// ===========================================================================
// Hoist wave-invariant setup out of structured affine/scf loops. A legacy CFG
// path remains available for already-lowered LLVM functions.
//
// Generic LICM does not prove execution-mask equivalence for readfirstlane,
// whose missing memory-effect declaration also prevents ordinary CSE.
//
//   lds-bases:     the readfirstlane chain that selects a DMA's LDS wave base.
//   lane-shuffles: the mbcnt lane ID and the bpermute index computed from it.
//
// readfirstlane depends on the execution mask. Structured loops must be in
// provably block-uniform control flow; slices never cross a conditional region.
// The legacy CFG path only accepts a two-block counted loop after entry.
// Mbcnt with a fixed all-ones mask computes a physical lane ID; arbitrary mbcnt
// operands must still pass the ordinary loop-invariance dependency checks.
//
// Lane-shuffle hoisting keeps the bpermute indices in VGPRs across the loop.
// That trades a few registers for fewer ALU instructions per iteration, so it
// has its own option for A/B measurement.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
bool isConstant(Value value) {
  return matchPattern(value, m_Constant());
}

// Side-effect free and unable to trap. Division needs a nonzero constant
// divisor, because a zero-trip loop must not gain a trapping instruction.
// readfirstlane additionally requires the control-flow proof at the caller.
bool isHoistablePure(Operation *op) {
  if (isa<LLVM::UDivOp, LLVM::SDivOp, LLVM::URemOp, LLVM::SRemOp,
          arith::DivUIOp, arith::DivSIOp, arith::RemUIOp, arith::RemSIOp>(op)) {
    APInt divisor;
    return matchPattern(op->getOperand(1), m_ConstantInt(&divisor)) &&
           !divisor.isZero() && !divisor.isAllOnes();
  }
  if (isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
          arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ExtUIOp, arith::ExtSIOp,
          arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp,
          arith::CmpIOp, arith::SelectOp, affine::AffineApplyOp,
          UnrealizedConversionCastOp, memref::CastOp, memref::SubViewOp,
          memref::ReinterpretCastOp, gpu::ThreadIdOp, gpu::BlockIdOp>(op))
    return true;
  return isa<LLVM::ConstantOp, LLVM::AddressOfOp, LLVM::ZeroOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
             LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp,
             LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp,
             LLVM::GEPOp, LLVM::ICmpOp, LLVM::SelectOp, LLVM::ExtractValueOp, LLVM::InsertValueOp,
             LLVM::UndefOp, LLVM::PoisonOp,
             ROCDL::ThreadIdXOp, ROCDL::MbcntLoOp, ROCDL::MbcntHiOp, ROCDL::ReadfirstlaneOp>(op);
}

// Match the loop shape that the header comment's exec proof relies on.
// 要求的形状：entry -> header（条件跳转，比较 iv 和常量）-> latch（body，iv + 常量）-> header，共两块。
// header 里的比较结果在所有 lane 上相同，所以是 uniform 分支，exec 不变。
bool isUniformCountedLoop(CFGLoop *loop, Block *entry) {
  Block *header = loop->getHeader(), *latch = loop->getLoopLatch();
  // getLoopPreheader() would also consult Block::isLegalToHoistInto; the branch
  // match below already proves that the entry block's only successor is the header.
  if (loop->getLoopPredecessor() != entry || loop->getBlocks().size() != 2 || !latch || latch == header)
    return false;
  auto exitTest = dyn_cast<LLVM::CondBrOp>(header->getTerminator());
  auto back = dyn_cast<LLVM::BrOp>(latch->getTerminator());
  auto enter = dyn_cast<LLVM::BrOp>(entry->getTerminator());
  if (!exitTest || !back || !enter || back.getDest() != header) return false;
  auto compare = exitTest.getCondition().getDefiningOp<LLVM::ICmpOp>();
  if (!compare || compare->getBlock() != header) return false;
  auto iv = dyn_cast<BlockArgument>(compare.getLhs());
  if (!iv || iv.getOwner() != header || !isConstant(compare.getRhs())) return false;
  unsigned position = iv.getArgNumber();
  if (!isConstant(enter.getDestOperands()[position])) return false;
  auto step = back.getDestOperands()[position].getDefiningOp<LLVM::AddOp>();
  return step && step.getLhs() == iv && isConstant(step.getRhs());
}

// Collect `root` and its in-loop dependencies. Fails on the induction variable,
// other loop block arguments, and anything not known to be pure.
// 反向收集 root 依赖的循环内计算。遇到 iv、循环内 block 参数或非白名单 op 就失败。
bool collectSlice(Value root, CFGLoop *loop, Block *body, llvm::SetVector<Operation *> &slice) {
  Operation *op = root.getDefiningOp();
  if (!op) {
    auto argument = cast<BlockArgument>(root);
    return !loop->contains(argument.getOwner());
  }
  if (!loop->contains(op->getBlock()) || slice.contains(op)) return true;
  if (op->getBlock() != body || !isHoistablePure(op)) return false;
  for (Value operand : op->getOperands())
    if (!collectSlice(operand, loop, body, slice)) return false;
  slice.insert(op);
  return true;
}

// Move in original program order so that defs stay ahead of uses.
void moveInOrder(Block *body, const llvm::SetVector<Operation *> &slice, Operation *before) {
  for (Operation &op : llvm::make_early_inc_range(*body))
    if (slice.contains(&op)) op.moveBefore(before);
}

// Reject induction/iter arguments, nested regions, memory reads, and partial
// slices. Values defined outside the loop already dominate the insertion point.
bool collectStructuredSlice(Value root, Operation *loop, Block *body,
                            llvm::SetVector<Operation *> &slice) {
  Operation *op = root.getDefiningOp();
  if (!op) {
    auto arg = cast<BlockArgument>(root);
    Operation *owner = arg.getOwner()->getParentOp();
    return owner != loop && !loop->isAncestor(owner);
  }
  if (!loop->isAncestor(op) || slice.contains(op)) return true;
  if (op->getBlock() != body || !isHoistablePure(op)) return false;
  for (Value operand : op->getOperands())
    if (!collectStructuredSlice(operand, loop, body, slice)) return false;
  slice.insert(op);
  return true;
}

// Constant/lane-only setup can live at the very beginning of the entry block.
// This also keeps setup icmp/select ahead of the late K0 prologue handoff.
// No arguments, thread IDs, pointers, or readfirstlane may be pulled here.
bool collectEntrySetup(Value value, Block *entry,
                       llvm::SetVector<Operation *> &slice) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != entry) return false;
  if (slice.contains(op)) return true;
  if (!isConstant(value) &&
      (!isHoistablePure(op) || op->getNumOperands() == 0 ||
       isa<ROCDL::ReadfirstlaneOp>(op))) return false;
  for (Value operand : op->getOperands())
    if (!collectEntrySetup(operand, entry, slice)) return false;
  slice.insert(op);
  return true;
}

void moveEntryShuffleSetup(Block *entry) {
  llvm::SetVector<Operation *> slice;
  llvm::SmallPtrSet<Operation *, 16> derived;
  for (Operation &op : *entry) {
    bool seed = isa<ROCDL::MbcntLoOp, ROCDL::MbcntHiOp>(op);
    if (op.getNumResults() != 1 ||
        (!seed && !llvm::any_of(op.getOperands(), [&](Value v) {
          return derived.contains(v.getDefiningOp());
        }))) continue;
    llvm::SetVector<Operation *> candidate;
    if (!collectEntrySetup(op.getResult(0), entry, candidate)) continue;
    derived.insert(&op);
    slice.insert(candidate.begin(), candidate.end());
  }
  if (slice.empty()) return;
  OpBuilder builder(entry, entry->begin());
  Operation *anchor = builder.create<LLVM::UndefOp>(entry->front().getLoc(), builder.getI32Type());
  moveInOrder(entry, slice, anchor);
  anchor->erase();
}

unsigned mergeEntryReads(Operation *function, Block *entry) {
  IRRewriter rewriter(function->getContext());
  DominanceInfo dominance(function);
  eliminateCommonSubExpressions(rewriter, dominance, function);
  DenseMap<Value, Value> leaders;
  unsigned merged = 0;
  for (Operation &op : llvm::make_early_inc_range(*entry)) {
    // Do not infer execution-mask equivalence across structured control flow
    // or opaque calls. Reconverging normal loop/if regions are deliberately
    // treated conservatively here.
    if (op.getNumRegions() || isa<func::CallOp, LLVM::CallOp, LLVM::InlineAsmOp>(op))
      leaders.clear();
    auto read = dyn_cast<ROCDL::ReadfirstlaneOp>(op);
    if (!read) continue;
    auto [known, inserted] = leaders.try_emplace(read.getOperand(), read.getResult());
    if (inserted) continue;
    read.getResult().replaceAllUsesWith(known->second);
    read.erase();
    ++merged;
  }
  if (merged) eliminateCommonSubExpressions(rewriter, dominance, function);
  return merged;
}

struct HoistWaveInvariantsPass
    : PassWrapper<HoistWaveInvariantsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistWaveInvariantsPass)
  HoistWaveInvariantsPass() = default;
  HoistWaveInvariantsPass(bool bases, bool shuffles) { ldsBases = bases; laneShuffles = shuffles; }
  HoistWaveInvariantsPass(const HoistWaveInvariantsPass &other) : PassWrapper(other) {}
  Option<bool> ldsBases{*this, "lds-bases",
      llvm::cl::desc("Hoist the readfirstlane chain selecting each DMA's LDS wave base"),
      llvm::cl::init(true)};
  Option<bool> laneShuffles{*this, "lane-shuffles",
      llvm::cl::desc("Hoist mbcnt lane IDs and bpermute indices; keeps the indices in VGPRs"),
      llvm::cl::init(true)};
  StringRef getArgument() const final { return "frisk-hoist-wave-invariants"; }
  StringRef getDescription() const final {
    return "Hoist LDS wave bases and lane-shuffle setup out of uniform kernel loops";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect, ROCDL::ROCDLDialect,
                    affine::AffineDialect, arith::ArithDialect>();
  }

  void hoistStructured(func::FuncOp function) {
    if (function.isExternal()) return;
    unsigned bases = 0, setup = 0;
    // Inner loops first: a chain may move across several uniform loop levels,
    // but stops at the first induction-variable or control-flow dependency.
    function.walk<WalkOrder::PostOrder>([&](Operation *loop) {
      Block *body = nullptr;
      if (auto affineLoop = dyn_cast<affine::AffineForOp>(loop))
        body = affineLoop.getBody();
      else if (auto scfLoop = dyn_cast<scf::ForOp>(loop))
        body = scfLoop.getBody();
      if (!body || !frisk::isInBlockUniformControlFlow(body->getTerminator())) return;
      llvm::SetVector<Operation *> slice;
      if (ldsBases) {
        // Include copies in inner loops: their base may already have moved
        // into this body during the preceding postorder visit.
        SmallVector<ROCDL::GlobalLoadAsyncLDSOp> copies;
        body->walk([&](ROCDL::GlobalLoadAsyncLDSOp copy) { copies.push_back(copy); });
        for (auto copy : copies) {
          auto pointer = copy.getLdsPtr().getDefiningOp<LLVM::IntToPtrOp>();
          auto read = pointer ? pointer.getOperand().getDefiningOp<ROCDL::ReadfirstlaneOp>()
                              : ROCDL::ReadfirstlaneOp{};
          if (!read || read->getBlock() != body) continue;
          llvm::SetVector<Operation *> candidate;
          if (!collectStructuredSlice(pointer.getResult(), loop, body, candidate)) continue;
          slice.insert(candidate.begin(), candidate.end());
          ++bases;
        }
      }
      if (laneShuffles) {
        llvm::SmallPtrSet<Operation *, 16> derived;
        for (Operation &op : *body) {
          bool seed = isa<ROCDL::MbcntLoOp, ROCDL::MbcntHiOp>(op);
          if (op.getNumResults() != 1 ||
              (!seed && !llvm::any_of(op.getOperands(), [&](Value v) {
                return derived.contains(v.getDefiningOp());
              }))) continue;
          llvm::SetVector<Operation *> candidate;
          if (!collectStructuredSlice(op.getResult(0), loop, body, candidate)) continue;
          derived.insert(&op);
          slice.insert(candidate.begin(), candidate.end());
          ++setup;
        }
      }
      moveInOrder(body, slice, loop);
    });
    if (!bases && !setup) return;
    Block *entry = &function.getBody().front();
    if (laneShuffles) moveEntryShuffleSetup(entry);
    unsigned merged = mergeEntryReads(function, entry);
    llvm::errs() << "HoistWaveInvariants: " << function.getName() << ": " << bases
                 << " LDS bases (" << merged << " merged), " << setup
                 << " shuffle setup ops (structured)\n";
  }

  // 匹配 DMA 的 lds 指针 = inttoptr(readfirstlane(...))，把整条链移到 entry 末尾。
  // slice：所有成功候选的并集。candidate：单条 DMA 的切片，失败时不污染 slice。
  unsigned hoistBases(CFGLoop *loop, Block *entry, Block *body) {
    llvm::SetVector<Operation *> slice;
    unsigned count = 0;
    SmallVector<ROCDL::GlobalLoadAsyncLDSOp> copies;
    body->walk([&](ROCDL::GlobalLoadAsyncLDSOp copy) { copies.push_back(copy); });
    for (auto copy : copies) {
      auto pointer = copy.getLdsPtr().getDefiningOp<LLVM::IntToPtrOp>();
      auto base = pointer ? pointer.getOperand().getDefiningOp<ROCDL::ReadfirstlaneOp>() : ROCDL::ReadfirstlaneOp{};
      if (!base || base->getBlock() != body) continue;
      llvm::SetVector<Operation *> candidate;
      // A failed candidate leaves `slice` untouched.
      if (!collectSlice(pointer.getResult(), loop, body, candidate)) continue;
      slice.insert(candidate.begin(), candidate.end());
      ++count;
    }
    moveInOrder(body, slice, entry->getTerminator());
    return count;
  }

  // 正向闭包：从 mbcnt 出发，只收“输入全部来自 mbcnt 或常量”的 op，不会顺带拉入普通的不变计算。
  unsigned hoistShuffleSetup(CFGLoop *loop, Block *entry, Block *body) {
    // Forward closure from mbcnt over values computed only from mbcnt and
    // constants, so it cannot pull in ordinary loop-invariant work. Lowering
    // leaves constant expressions such as `sub 0, width`, which count as constants.
    // slice：要外提的 op。
    // constantExpressions：只由常量算出的表达式（如 sub 0, width）。只有被 slice 真正用到时才一起移动。
    // require：把用到的常量表达式及其依赖加入 slice。
    llvm::SetVector<Operation *> slice;
    llvm::SmallPtrSet<Operation *, 8> constantExpressions;
    auto isConstantLike = [&](Value value) {
      return isConstant(value) || constantExpressions.contains(value.getDefiningOp());
    };
    auto require = [&](auto &&self, Value value) -> void {
      Operation *op = value.getDefiningOp();
      if (!constantExpressions.contains(op) || !slice.insert(op)) return;
      for (Value operand : op->getOperands()) self(self, operand);
    };
    for (Operation &op : *body) {
      if (!isHoistablePure(&op) || isa<LLVM::ConstantOp>(op) || op.getOperands().empty()) continue;
      if (isa<ROCDL::MbcntLoOp, ROCDL::MbcntHiOp>(op)) {
        slice.insert(&op);
        continue;
      }
      bool anyDerived = false, allKnown = true;
      for (Value operand : op.getOperands()) {
        bool derived = slice.contains(operand.getDefiningOp());
        anyDerived |= derived;
        allKnown &= derived || isConstantLike(operand);
      }
      if (!allKnown) continue;
      if (!anyDerived) {
        constantExpressions.insert(&op);
        continue;
      }
      slice.insert(&op);
      for (Value operand : op.getOperands()) require(require, operand);
    }
    if (slice.empty()) return 0;
    // Place the chain first in the entry block. It uses only mbcnt and
    // constants, and K0 priming rejects icmp/select after the prologue handoff.
    OpBuilder builder(entry, entry->begin());
    // anchor：入口块最前面的临时占位。所有外提的 op 都插到它之前，结束后删除。
    // 放在最前面是因为 prefetchNextK 的 K0 前置逻辑会拒绝 prologue 交接点之后的 icmp/select。
    Operation *anchor = builder.create<LLVM::UndefOp>(entry->front().getLoc(), builder.getI32Type());
    for (Operation &op : llvm::make_early_inc_range(*body)) {
      if (!slice.contains(&op)) continue;
      for (Value operand : op.getOperands()) {
        Operation *constant = operand.getDefiningOp();
        if (slice.contains(constant)) continue;
        // Constants may sit later in the entry block, or in the loop body.
        if (constant->getBlock() != entry || anchor->isBeforeInBlock(constant))
          constant->moveBefore(anchor);
      }
      op.moveBefore(anchor);
    }
    anchor->erase();
    return slice.size();
  }

  void runOnOperation() final {
    getOperation().walk([&](func::FuncOp function) { hoistStructured(function); });
    getOperation().walk([&](LLVM::LLVMFuncOp function) {
      if (function.isExternal()) return;
      Block *entry = &function.getBody().front();
      unsigned bases = 0, setup = 0;
      {
        DominanceInfo dominance(function);
        CFGLoopInfo loops(dominance.getDomTree(&function.getBody()));
        for (CFGLoop *loop : loops.getTopLevelLoops()) {
          if (!isUniformCountedLoop(loop, entry)) {
            llvm::errs() << "HoistWaveInvariants: " << function.getName()
                         << ": skipped a loop that is not a uniform counted loop after entry\n";
            continue;
          }
          Block *body = loop->getLoopLatch();
          if (ldsBases) bases += hoistBases(loop, entry, body);
          if (laneShuffles) setup += hoistShuffleSetup(loop, entry, body);
        }
      }
      if (!bases && !setup) return;
      // 外提后再做 CSE，合并等价值。
      // Hoisted values now dominate equivalent ones in and after the loop.
      IRRewriter rewriter(function.getContext());
      DominanceInfo dominance(function);
      eliminateCommonSubExpressions(rewriter, dominance, function);
      // readfirstlane has no effect declaration, so CSE skips it. Same block
      // and same operand means the same exec mask and therefore the same value.
      // leaders：readfirstlane 操作数 -> 第一个读结果。同一块、同一操作数的 readfirstlane 只留一个。
      // merged：被合并的数量。
      DenseMap<Value, Value> leaders;
      unsigned merged = 0;
      for (Operation &op : llvm::make_early_inc_range(*entry)) {
        auto read = dyn_cast<ROCDL::ReadfirstlaneOp>(op);
        if (!read) continue;
        auto [known, inserted] = leaders.try_emplace(read.getOperand(), read.getResult());
        if (inserted) continue;
        read.getResult().replaceAllUsesWith(known->second);
        read.erase();
        ++merged;
      }
      if (merged) eliminateCommonSubExpressions(rewriter, dominance, function);
      llvm::errs() << "HoistWaveInvariants: " << function.getName() << ": " << bases
                   << " LDS bases (" << merged << " merged), " << setup << " shuffle setup ops\n";
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createHoistWaveInvariantsPass(bool ldsBases, bool laneShuffles) {
  return std::make_unique<HoistWaveInvariantsPass>(ldsBases, laneShuffles);
}
