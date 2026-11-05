// ===========================================================================
// Pass: HoistWaveInvariants（frisk-hoist-wave-invariants）   优化汇总.py 第 18 项，LLVM dialect 阶段，未实测
//
// 作用：把循环里每轮重算、但在一个 wave 内其实不变的计算提到函数入口块。
//   hoistBases         每条 DMA 的 LDS 基址（readfirstlane 链）。
//   hoistShuffleSetup  mbcnt 的 lane 号和由它算出的 bpermute 索引。
//
// 运行时机：secondLowering 之后，翻译到 LLVM IR 之前。
//
// 下游依赖：lowerWideAsyncDMAToBufferAsm 的 reuseLeaderRead 依赖这里提到入口的基址。
// 下面是原文件头的合法性说明。
// ===========================================================================
// Hoist wave-invariant setup out of a uniform kernel loop on LLVM-dialect IR.
//
// Upstream LICM cannot do this. The loop is already a CFG here, and
// rocdl.readfirstlane declares no memory effects, so it is treated as unknown.
//
//   lds-bases:     the readfirstlane chain that selects a DMA's LDS wave base.
//   lane-shuffles: the mbcnt lane ID and the bpermute index computed from it.
//
// readfirstlane depends on the execution mask, so moving it needs a proof that
// the mask cannot change. Only a two-block loop directly after the entry block,
// with a constant-start, constant-step counter against a constant bound, is
// accepted. Uniform branches leave exec untouched, so the entry block and loop
// body run under the same mask. Anything else is left unchanged.
//
// Lane-shuffle hoisting keeps the bpermute indices in VGPRs across the loop.
// That trades a few registers for fewer ALU instructions per iteration, so it
// has its own option for A/B measurement.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
bool isConstant(Value value) {
  return value.getDefiningOp<LLVM::ConstantOp>() != nullptr;
}

// Side-effect free and unable to trap. Division needs a nonzero constant
// divisor, because a zero-trip loop must not gain a trapping instruction.
// 可外提的 op 白名单。这里包含 readfirstlane 和 mbcnt，它们依赖 exec，合法性由 isUniformCountedLoop 保证。
bool isHoistablePure(Operation *op) {
  if (isa<LLVM::UDivOp, LLVM::SDivOp, LLVM::URemOp, LLVM::SRemOp>(op)) {
    auto divisor = op->getOperand(1).getDefiningOp<LLVM::ConstantOp>();
    auto value = divisor ? dyn_cast<IntegerAttr>(divisor.getValue()) : IntegerAttr{};
    return value && !value.getValue().isZero() && !value.getValue().isAllOnes();
  }
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
    registry.insert<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
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
