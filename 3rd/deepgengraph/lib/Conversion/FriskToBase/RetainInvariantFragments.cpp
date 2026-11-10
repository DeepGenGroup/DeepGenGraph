// Retain immutable, thread-distributed operand values across a collective loop.
// Run on typed fragments after loop specialization, before thread lowering.
// The legacy scalar/vector form remains supported for standalone callers.
// This deliberately matches dataflow and effects, never argument roles or names.
// ===========================================================================
// Pass: RetainInvariantFragments（frisk-retain-invariant-fragments）   优化汇总.py 第 10 项，fragment 阶段
//
// 作用：
//   如果循环里每轮都把同一份 GM 数据搬到 LDS、读进寄存器、做同样的计算（典型例子是 Q），就把
//   “DMA + wait + 读 + 不变计算”整条链移到循环外，结果常驻寄存器。循环里少了 DMA、wait、
//   barrier 和 LDS 读，LDS 也释放给 K/V。用寄存器换搬运：本 case VGPR 228 -> 244。
//
// 判定完全基于数据流和副作用，不看参数角色或名字。
//
// 流程（runOnOperation）：
//   对函数入口层的每个循环：
//     1. 必须确定至少执行一次（nonempty）、trip count > 1、控制流 uniform。
//     2. 按顺序取循环体里每个 async_copy：
//        immutableSource 证明源在循环内不会被写；
//        Candidate::analyze 算出可外提的切片和常驻字数；
//        未超预算就 apply。
//     3. 把已用字数记到 frisk.retained_words，避免重复处理。
//
// 选项：
//   maxWords  每线程可常驻的 32-bit 字数上限，0 表示关闭。不是物理 VGPR 上限。
//             本 case 为 --retain-fragment-words=32。
//
// 只处理入口层循环：嵌套的保留范围生存期会重叠，预算不能简单相加。
// ===========================================================================
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace mlir::frisk {
#define GEN_PASS_DECL_RETAININVARIANTFRAGMENTS
#define GEN_PASS_DEF_RETAININVARIANTFRAGMENTS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
namespace {
using Loop = affine::AffineForOp;
// Range 是闭区间 [lo, hi]。Bounds 为空表示范围未知。
// 下面的 combine/expression/range 是一个小型区间分析，只用于 nonempty。
struct Range { int64_t lo, hi; };
using Bounds = std::optional<Range>;

// Small, conservative index-range proof. Refuse arithmetic outside signed
// 64-bit range, unknown values, and division whose sign is not established.
// kind：'+' '-' '*' '/' '%'。用 __int128 计算防溢出；除法和取模只接受非负被除数和正除数。
static Bounds combine(Bounds a, Bounds b, char kind) {
  if (!a || !b) return {};
  __int128 lo, hi;
  if (kind == '+') { lo = (__int128)a->lo + b->lo; hi = (__int128)a->hi + b->hi; }
  else if (kind == '-') { lo = (__int128)a->lo - b->hi; hi = (__int128)a->hi - b->lo; }
  else if (kind == '*') {
    __int128 v[] = {(__int128)a->lo*b->lo, (__int128)a->lo*b->hi,
                    (__int128)a->hi*b->lo, (__int128)a->hi*b->hi};
    lo = *std::min_element(v, v+4); hi = *std::max_element(v, v+4);
  } else {
    if (a->lo < 0 || b->lo <= 0) return {};
    if (kind == '/') { lo = a->lo / b->hi; hi = a->hi / b->lo; }
    else { lo = 0; hi = std::min(a->hi, b->hi - 1); }
  }
  if (lo < std::numeric_limits<int64_t>::min() ||
      hi > std::numeric_limits<int64_t>::max()) return {};
  return Range{int64_t(lo), int64_t(hi)};
}
static Bounds range(Value v, unsigned depth = 0);
static Bounds expression(AffineExpr e, AffineMap map, ValueRange values,
                         unsigned depth) {
  if (depth > 64) return {};
  if (auto c = dyn_cast<AffineConstantExpr>(e)) return Range{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e)) return range(values[d.getPosition()], depth+1);
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return range(values[map.getNumDims()+s.getPosition()], depth+1);
  auto b = dyn_cast<AffineBinaryOpExpr>(e);
  if (!b) return {};
  char kind;
  switch (e.getKind()) {
  case AffineExprKind::Add: kind = '+'; break;
  case AffineExprKind::Mul: kind = '*'; break;
  case AffineExprKind::FloorDiv: kind = '/'; break;
  case AffineExprKind::Mod: kind = '%'; break;
  default: return {};
  }
  return combine(expression(b.getLHS(), map, values, depth+1),
                 expression(b.getRHS(), map, values, depth+1), kind);
}
static Bounds range(Value v, unsigned depth) {
  if (depth > 64 || !v.getType().isIndex()) return {};
  APInt constant;
  if (matchPattern(v, m_ConstantInt(&constant))) {
    if (constant.getBitWidth() > 64) return {};
    return Range{constant.getSExtValue(), constant.getSExtValue()};
  }
  Operation *op = v.getDefiningOp();
  if (!op) return {};
  if (isa<gpu::BlockIdOp>(op)) {
    auto n = op->getAttrOfType<IntegerAttr>("range");
    if (n && n.getInt() > 0) return Range{0, n.getInt()-1};
  }
  if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
    return expression(apply.getAffineMap().getResult(0), apply.getAffineMap(),
                      apply.getMapOperands(), depth+1);
  char kind;
  if (isa<arith::AddIOp>(op)) kind = '+';
  else if (isa<arith::SubIOp>(op)) kind = '-';
  else if (isa<arith::MulIOp>(op)) kind = '*';
  else if (isa<arith::DivUIOp>(op)) kind = '/';
  else if (isa<arith::RemUIOp>(op)) kind = '%';
  else return {};
  return combine(range(op->getOperand(0), depth+1),
                 range(op->getOperand(1), depth+1), kind);
}
// 证明循环至少执行一次：下界的最大可能值 < 上界的最小可能值。
// 必须证明，否则把 DMA 外提到零次循环之外就是投机执行，可能读非法地址。
static bool nonempty(Loop loop) {
  int64_t lower = std::numeric_limits<int64_t>::min();
  int64_t upper = std::numeric_limits<int64_t>::max();
  for (AffineExpr e : loop.getLowerBoundMap().getResults()) {
    auto r = expression(e, loop.getLowerBoundMap(), loop.getLowerBoundOperands(), 0);
    if (!r) return false;
    lower = std::max(lower, r->hi);
  }
  for (AffineExpr e : loop.getUpperBoundMap().getResults()) {
    auto r = expression(e, loop.getUpperBoundMap(), loop.getUpperBoundOperands(), 0);
    if (!r) return false;
    upper = std::min(upper, r->lo);
  }
  return upper > lower;
}

// 返回 op 在 loop 循环体顶层的祖先（op 自身或包含它的那个顶层 op）。
static Operation *topLevel(Operation *op, Loop loop) {
  while (op && op->getBlock() != loop.getBody()) op = op->getParentOp();
  return op;
}
static bool inside(Value value, Operation *root) {
  Operation *owner = value.getDefiningOp();
  if (auto arg = dyn_cast<BlockArgument>(value)) owner = arg.getOwner()->getParentOp();
  return owner && (owner == root || root->isAncestor(owner));
}
// root（含嵌套 region）用到的、定义在 root 外部的所有值。
static SmallVector<Value> captures(Operation *root) {
  SmallVector<Value> result;
  root->walk([&](Operation *op) {
    for (Value v : op->getOperands())
      if (!inside(v, root) && !llvm::is_contained(result, v)) result.push_back(v);
  });
  return result;
}
// 判断 memref 的地址空间：1 = GM，3 = LDS。
static bool space(Value v, unsigned n) {
  auto type = dyn_cast<MemRefType>(v.getType());
  return type && type.getMemorySpaceAsInt() == n;
}

// No unknown effects or potentially aliasing global writes may change the
// producer between iterations (or before its original first execution).
// 能否证明循环执行期间，source 指向的 GM（全局内存）数据不会被修改。
// 这是把 DMA 外提到循环外的前提：只有源数据不变，才可以只搬运一次，在后续迭代中复用。
static bool immutableSource(Loop loop, Value source, AliasAnalysis &aa) {
  bool safe = true;
  // barrier、wait、写 LDS 的 DMA 都不会改变 GM 源，可以放行。
  // 其它 op：无法获知副作用就判不安全；写操作必须能证明与 source 不别名（LDS 目标除外）。
  loop.walk([&](Operation *op) {
    if (!safe || op == loop.getOperation()) return;
    if (isa<gpu::BarrierOp, WaitBufferReadyOp>(op)) return;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) {
      if (!space(copy.getDst(), 3)) safe = false;
      return;
    }
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return;
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) { safe = false; return; }
    SmallVector<MemoryEffects::EffectInstance> list;
    effects.getEffects(list);
    for (auto effect : list) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
      Value target = effect.getValue();
      if (!target || (!space(target, 3) && !aa.alias(source, target).isNo()))
        safe = false;
    }
  });
  return safe;
}

// 一个可外提的候选：循环 loop 中的一条 async_copy 及其相关计算。
//   loop / copy  输入。
//   storage      copy 的 LDS 目的 alloc，必须是循环内专用的一块。
//   wait         copy 对应的唯一 wait。
//   movable      已证明可移到循环外的 op（记忆化）。
//   rejected     已判定不可移的 op（记忆化），兼作递归防环。
//   selected     最终要移动的集合 = data + 它们的反向依赖切片。
//   data         依赖这次读取结果、且可移动的 op，也就是外提后要常驻的数据链。
struct Candidate {
  Loop loop;
  AsyncCopyGmToShmOp copy;
  memref::AllocOp storage;
  WaitBufferReadyOp wait;
  llvm::SmallPtrSet<Operation *, 32> movable;
  llvm::SmallPtrSet<Operation *, 32> rejected;
  llvm::SmallPtrSet<Operation *, 32> selected;
  llvm::SmallPtrSet<Operation *, 32> data;

  // Region internals may use their own IVs/carriers, but may only read this
  // dedicated allocation. Never promote MMA, shuffle, conditional control flow
  // or another memory source as an incidental part of an operand's slice.
  // 白名单：只放行读本 storage 的 load、常量边界循环和纯计算（arith/vector/affine.apply/视图/
  // fragment 打包拆包等）。MMA、shuffle、其它内存源不能被顺带移出去。
  bool allowed(Operation *root) {
    bool safe = true;
    root->walk([&](Operation *op) {
      if (!safe) return;
      if (op == storage || op == copy || op == wait) return;
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        safe = load.getMemref() == storage.getResult(); return;
      }
      if (auto load = dyn_cast<LoadFragOp>(op)) {
        safe = load.getSource() == storage.getResult(); return;
      }
      if (auto forOp = dyn_cast<Loop>(op)) {
        safe = forOp.hasConstantBounds() &&
               forOp.getConstantUpperBound() >= forOp.getConstantLowerBound();
        return;
      }
      if (isa<affine::AffineYieldOp>(op)) return;
      StringRef dialect = op->getName().getDialectNamespace();
      safe = !op->getNumRegions() && isMemoryEffectFree(op) &&
          (dialect == "arith" || dialect == "vector" ||
           isa<affine::AffineApplyOp, gpu::ThreadIdOp, gpu::BlockIdOp,
               memref::SubViewOp, memref::CastOp, memref::TransposeOp,
               BufferViewOp, ExtractFragOp, InsertFragOp, PackFragOp,
               UnpackFragOp, ElementwiseFragOp, BroadcastFragOp>(op));
    });
    return safe;
  }
  // op 能否移到循环外：自身在白名单内，且它用到的循环内定义都能移（递归）。
  // 先加入 rejected 再递归，SSA 成环时会失败；成功后才从 rejected 移到 movable。
  bool canMove(Operation *op) {
    if (movable.contains(op)) return true;
    if (rejected.contains(op) || op->getBlock() != loop.getBody()) return false;
    rejected.insert(op); // Bound recursion, including unexpected SSA cycles.
    if (!allowed(op)) return false;
    for (Value value : captures(op)) {
      if (!inside(value, loop)) continue;
      auto *def = value.getDefiningOp();
      if (!def || def->getBlock() != loop.getBody() || !canMove(def)) return false;
    }
    rejected.erase(op);
    movable.insert(op);
    return true;
  }
  void select(Operation *op) {
    if (!selected.insert(op).second) return;
    for (Value v : captures(op))
      if (auto *def = v.getDefiningOp(); def && def->getBlock() == loop.getBody())
        select(def);
  }
  // 返回外提后需要常驻的字数；不可外提时返回空。只分析，不改 IR。
  std::optional<unsigned> analyze() {
    storage = copy.getDst().getDefiningOp<memref::AllocOp>();
    if (!storage || storage->getBlock() != loop.getBody() ||
        !storage.getType().hasStaticShape() || !space(copy.getSrc(), 1) ||
        !space(copy.getDst(), 3)) return {};
    // readers：循环内读 storage 的顶层 op。storage 只要有其它用途（别名逃逸）就放弃。
    SmallVector<Operation *> readers;
    for (Operation *user : storage->getUsers()) {
      if (user == copy) continue;
      if (!isa<memref::LoadOp, LoadFragOp>(user) || !loop->isAncestor(user)) return {}; // No escaping alias.
      readers.push_back(topLevel(user, loop));
    }
    for (Operation *user : copy.getToken().getUsers()) {
      auto ready = dyn_cast<WaitBufferReadyOp>(user);
      if (!ready || wait || ready->getBlock() != loop.getBody()) return {};
      wait = ready;
    }
    if (!wait || readers.empty() || !copy->isBeforeInBlock(wait)) return {};
    for (auto *reader : readers)
      if (!wait->isBeforeInBlock(reader) || !canMove(reader)) return {};
    if (!canMove(copy) || !canMove(wait)) return {};
    select(copy); select(wait);
    // Grow through invariant register computations, stopping at mixed/variant
    // consumers. The final cut, not every intermediate, determines the budget.
    // 按顺序向前传播：依赖 readers 或已有 data 的 op，如果可移就纳入 data。
    // 遇到依赖循环变量的消费者时停止，这就是外提的切割线。
    for (Operation &op : *loop.getBody()) {
      bool depends = llvm::is_contained(readers, &op);
      for (Value v : captures(&op))
        if (data.contains(v.getDefiningOp())) depends = true;
      if (depends && canMove(&op)) { data.insert(&op); select(&op); }
    }
    // 只统计切割线上的值，也就是被“未移动的 op”使用的 data 结果，它们才需要跨循环常驻。
    // 切片内部的中间值用完就释放，不计入预算。
    uint64_t bits = 0;
    for (Operation *op : data) {
      for (Value value : op->getResults()) {
        bool live = llvm::any_of(value.getUsers(), [&](Operation *user) {
          return !selected.contains(topLevel(user, loop));
        });
        if (!live) continue;
        Type type = value.getType();
        if (auto fragment = dyn_cast<FragmentType>(type)) type = fragment.getRegisterType();
        if (auto vector = dyn_cast<VectorType>(type)) {
          if (vector.isScalable()) return {};
          bits += vector.getNumElements() * vector.getElementTypeBitWidth();
        } else if (type.isIntOrFloat()) bits += type.getIntOrFloatBitWidth();
        else return {};
      }
    }
    if (!bits || bits > unsigned(-1)) return {};
    return (bits + 31) / 32;
  }
  // 按循环体内的原始顺序移到循环之前，因此依赖顺序不变。
  void apply() {
    // Moving, rather than cloning, preserves FP precision and eliminates the
    // repeated DMA/read/compute chain. Subsequent packing and liveness analysis
    // see all reads in their final scope and protect the storage handoff.
    for (Operation &op : llvm::make_early_inc_range(*loop.getBody()))
      if (selected.contains(&op)) op.moveBefore(loop);
  }
};

class RetainInvariantFragments
    : public impl::RetainInvariantFragmentsBase<RetainInvariantFragments> {
  void runOnOperation() override {
    if (!maxWords || getOperation().isExternal() ||
        !getOperation()->hasAttr("thread_num")) return;
    if (failed(normalizeTypedFragmentIR(getOperation()))) return signalPassFailure();
    // Only entry-level collective loops in v1: nested retention scopes would
    // require an overlapping-live-range budget, rather than independent sums.
    SmallVector<Loop> loops;
    for (Operation &op : getOperation().front())
      if (auto loop = dyn_cast<Loop>(op)) loops.push_back(loop);
    unsigned count = 0;
    for (Loop loop : loops) {
      if (loop->hasAttr("frisk.retained_words") || !nonempty(loop) ||
          !isInBlockUniformControlFlow(loop.getBody()->getTerminator())) continue;
      // trip count <= 1 时外提没有收益。
      if (loop.hasConstantBounds() &&
          (__int128)loop.getConstantUpperBound() - loop.getConstantLowerBound() <= loop.getStepAsInt()) continue;
      // used：本循环已用的常驻字数。copies：循环体顶层的所有 async_copy，按出现顺序尝试。
      unsigned used = 0;
      SmallVector<AsyncCopyGmToShmOp> copies;
      for (Operation &op : *loop.getBody())
        if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) copies.push_back(copy);
      for (auto copy : copies) {
        if (used == maxWords) break;
        AliasAnalysis aa(getOperation());
        // 能否证明循环执行期间，source 指向的 GM（全局内存）数据不会被修改。
        // 这是把 DMA 外提到循环外的前提：只有源数据不变，才可以只搬运一次，在后续迭代中复用。
        if (!immutableSource(loop, copy.getSrc(), aa)) continue;
        Candidate candidate{loop, copy};
        auto words = candidate.analyze();
        if (!words || *words > maxWords - used) continue;
        candidate.apply(); used += *words; ++count;
      }
      if (used) loop->setAttr("frisk.retained_words", IntegerAttr::get(
          IntegerType::get(&getContext(), 32), used));
    }
    llvm::errs() << "RetainInvariantFragments: " << count
                 << " operand tiles retained; budget " << maxWords << " words/thread\n";
  }
};
} // namespace
std::unique_ptr<Pass> createRetainInvariantFragmentsPass() {
  return std::make_unique<RetainInvariantFragments>();
}
} // namespace mlir::frisk
