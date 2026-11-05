// ===========================================================================
// Pass: FuseFragmentAccumulator（fuse-fragment-accumulator）   优化汇总.py 第 7 项，fragment 阶段
//
// 作用：
//   把“先算完整个 part tile，再用一个循环逐 fragment 加到 S 上”合并成一个循环，每算完一个
//   fragment 立刻加。part 不用整块保存，中间结果生命周期更短。
//
//   before:  part = for f { insert(f, mmac 链) }
//            S    = for f { insert(f, extract(S, f) + extract(part, f)) }
//   after:   S    = for f { insert(f, extract(S, f) + mmac 链) }
//
// 有两套匹配器：
//   fuseTypedAccumulator  当前主路径，处理 fragment 类型 IR（InsertFrag/ExtractFrag/ElementwiseFrag）。
//   fuseAccumulator       旧路径，处理 vector.insert_strided_slice 链 + arith.addf。
//
// 注意：不把 S 作为 MMAC 的初值（C 输入）。K reduce 仍从 0 开始，保持精度和操作数顺序。
// 真正把旧值喂给 MMAC 是 LLVM 阶段 fuseMMACAccumulators 在 approx 约定下做的事。
//
// 产物属性：
//   frisk.fragment_accumulate  标在融合后的 add 上。
//   frisk.accumulator_fused    标在 producer 循环上。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>
#include <set>

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_FUSEFRAGMENTACCUMULATOR
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

using Loop = affine::AffineForOp;

// 把 value 在循环变量 = iv 时求值成常量。支持 iv、整型常量、affine.apply。求值不了返回空。
// FriskFragIRReorder.cpp 的 evaluate 是同样的逻辑。
static std::optional<int64_t> coordinate(Value value, Loop loop, int64_t iv,
                                         unsigned depth = 0) {
  if (depth > 64) return {};
  if (value == loop.getInductionVar()) return iv;
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) return constant.getSExtValue();
  auto apply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!apply) return {};
  SmallVector<Attribute> args, results;
  for (Value operand : apply.getMapOperands()) {
    auto n = coordinate(operand, loop, iv, depth + 1);
    if (!n) return {};
    args.push_back(IntegerAttr::get(IndexType::get(value.getContext()), *n));
  }
  if (failed(apply.getAffineMap().constantFold(args, results))) return {};
  return cast<IntegerAttr>(results.front()).getInt();
}

// 沿循环结果向上追，判断 value 最终是否来自 GemmFragOp（MMA 产物）。
static bool mmaResult(Value value) {
  for (unsigned depth = 0; depth < 16; ++depth) {
    if (value.getDefiningOp<GemmFragOp>()) return true;
    auto result = dyn_cast<OpResult>(value);
    auto loop = result ? dyn_cast<Loop>(result.getOwner()) : Loop{};
    if (!loop) return false;
    value = loop.getBody()->getTerminator()->getOperand(result.getResultNumber());
  }
  return false;
}

// Fuse the pointwise add loop into the MN producer, keeping the completed K
// reduction intact: add(old[m,n], reduce_k(..., zero)), never seed MMA with old.
// consumer 是做逐 fragment 相加的循环，producer 是它前面产出 delta tile 的 MN 循环。
// 命名：output = consumer 的 InsertFrag，add = 相加，delta = 从 producer 结果取的片，
// old = 从旧 S 取的片，insertion = producer 里写 delta 的 InsertFrag，
// tile = 整块类型，payload = 单个 fragment 的寄存器形状。
static bool fuseTypedAccumulator(Loop consumer) {
  if (consumer.getNumResults() != 1 || consumer->hasAttr("frisk.pipelined")) return false;
  auto output = consumer.getBody()->getTerminator()->getOperand(0).getDefiningOp<InsertFragOp>();
  if (!output || output->getBlock() != consumer.getBody() ||
      output.getDestination() != consumer.getRegionIterArgs()[0]) return false;
  auto add = output.getValue().getDefiningOp<ElementwiseFragOp>();
  if (!add || add.getKind() != "add" || !add.getResult().hasOneUse()) return false;
  // add 的两个操作数都可能是 delta，所以两种顺序都试。
  for (unsigned side = 0; side < 2; ++side) {
    auto delta = add.getInputs()[side].getDefiningOp<ExtractFragOp>();
    auto old = add.getInputs()[1-side].getDefiningOp<ExtractFragOp>();
    if (!delta || !old || !delta.getResult().hasOneUse()) continue;
    auto producer = delta.getSource().getDefiningOp<Loop>();
    if (!producer || producer.getNumResults() != 1 || !producer.getResult(0).hasOneUse() ||
        producer->getBlock() != consumer->getBlock() || !producer->isBeforeInBlock(consumer) ||
        producer->hasAttr("frisk.pipelined") || !producer.hasConstantBounds() ||
        !consumer.hasConstantBounds() || producer.getLowerBoundMap() != consumer.getLowerBoundMap() ||
        producer.getUpperBoundMap() != consumer.getUpperBoundMap() || producer.getStep() != consumer.getStep()) continue;
    auto insertion = producer.getBody()->getTerminator()->getOperand(0).getDefiningOp<InsertFragOp>();
    if (!insertion || insertion->getBlock() != producer.getBody() ||
        insertion.getDestination() != producer.getRegionIterArgs()[0] ||
        !insertion.getResult().hasOneUse() ||
        (!mmaResult(insertion.getValue()) && !insertion->hasAttr("frisk.mma_fragment"))) continue;
    auto fragment = cast<FragmentType>(delta.getResult().getType());
    auto payload = fragment.getRegisterType();
    auto tile = cast<VectorType>(producer.getResult(0).getType());
    if (payload != cast<FragmentType>(insertion.getValue().getType()).getRegisterType() ||
        tile != output.getResult().getType() || old.getSource().getType() != tile ||
        tile.getNumElements() > 65536) continue;
    // All carrier uses must be the single insertion, and the consumer is just
    // address calculation, two extracts and the add/insert. No side effects.
    if (!producer.getRegionIterArgs()[0].hasOneUse() || !consumer.getRegionIterArgs()[0].hasOneUse()) continue;
    bool valid = true;
    for (Operation &op : consumer.getBody()->without_terminator())
      if (&op != output && &op != add && &op != delta && &op != old &&
          !isa<affine::AffineApplyOp, arith::ConstantOp>(op)) valid = false;
    // setup：producer 和 consumer 之间的纯计算。融合时要移到 producer 之前，所以要求
    // 无副作用、可投机、无 region、不碰 memref、不依赖 producer。
    SmallVector<Operation *> setup;
    for (Operation *op = producer->getNextNode(); op != consumer; op = op->getNextNode()) {
      if (op->getNumRegions() || !isMemoryEffectFree(op) || !isSpeculatable(op) ||
          llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<MemRefType>(t); })) { valid = false; break; }
      for (Value v : op->getOperands()) if (v.getDefiningOp() == producer) valid = false;
      setup.push_back(op);
    }
    if (!valid) continue;
    // 逐次迭代常量求值，证明：
    //   1. producer 和 consumer 在同一个 iv 下访问同一个 fragment 坐标（p == q == r == s）。
    //   2. grid 中坐标互不重复，且总元素数等于 tile 元素数，即完整不重叠覆盖。
    // grid：已访问的 fragment 坐标集合。count 上限 256，是证明成本的预算。
    // Complete, disjoint grid coverage and equal producer/consumer positions.
    std::set<SmallVector<int64_t>> grid;
    int64_t lb = producer.getConstantLowerBound(), ub = producer.getConstantUpperBound();
    unsigned count = 0;
    for (int64_t iv = lb; iv < ub;) {
      if (++count > 256) { valid = false; break; }
      SmallVector<int64_t> point;
      for (unsigned d = 0; d < payload.getRank(); ++d) {
        auto p = coordinate(insertion.getIndices()[d], producer, iv);
        auto q = coordinate(delta.getIndices()[d], consumer, iv);
        auto r = coordinate(old.getIndices()[d], consumer, iv);
        auto s = coordinate(output.getIndices()[d], consumer, iv);
        if (!p || !q || !r || !s || *p != *q || *p != *r || *p != *s ||
            *p < 0 || *p >= tile.getDimSize(d) / payload.getDimSize(d)) { valid = false; break; }
        point.push_back(*p);
      }
      if (!valid || !grid.insert(point).second) { valid = false; break; }
      if (__builtin_add_overflow(iv, producer.getStepAsInt(), &iv)) { valid = false; break; }
    }
    if (!valid || grid.size() * payload.getNumElements() != uint64_t(tile.getNumElements())) continue;
    // The old tile must dominate the producer after moving independent setup.
    DominanceInfo dominance(producer->getParentOfType<func::FuncOp>());
    if (!dominance.dominates(old.getSource(), producer) &&
        !llvm::is_contained(setup, old.getSource().getDefiningOp())) continue;
    for (Operation *op : setup) op->moveBefore(producer);
    // 开始改写：把 consumer 循环体克隆到 producer 里 insertion 的位置，delta 直接替换成 MMA 结果。
    OpBuilder b(insertion);
    Value value = insertion.getValue();
    // fragment 类型上带有布局信息。两边寄存器切片相同但类型标记可能不同，用 unpack + pack 换标记，
    // 不改变数据。
    if (value.getType() != fragment) {
      // Both fragments refer to the same register slice of the same tile.
      // Re-attach the consumer's type without changing lane/register ownership.
      value = b.create<UnpackFragOp>(add.getLoc(), payload, value);
      value = b.create<PackFragOp>(add.getLoc(), fragment, value);
    }
    // consumer 的 iv / iter_arg 映射到 producer 的 iv / iter_arg，delta 映射到 MMA 结果。
    IRMapping mapping;
    mapping.map(consumer.getInductionVar(), producer.getInductionVar());
    mapping.map(consumer.getRegionIterArgs()[0], producer.getRegionIterArgs()[0]);
    mapping.map(delta.getResult(), value);
    for (Operation &op : consumer.getBody()->without_terminator()) {
      if (&op == delta) continue;
      auto *clone = b.clone(op, mapping);
      if (&op == add) clone->setAttr("frisk.fragment_accumulate", b.getUnitAttr());
    }
    producer.getBody()->getTerminator()->setOperand(0, mapping.lookup(output.getResult()));
    consumer.replaceAllUsesWith(producer.getResults());
    consumer.erase();
    insertion.erase();
    producer->setAttr("frisk.accumulator_fused", b.getUnitAttr());
    return true;
  }
  return false;
}

// A fully covered, single-use insertion chain represents independent MMA
// fragments in the *same packed thread coordinates* as the elementwise add.
// Move the add to each fragment's insertion point, leaving its K reduction
// (including zero initialization and operation order) untouched.
// 旧路径。tileOperand 表示 add 的哪个操作数是 insert 链（另一个是 accumulator），
// 浮点操作数顺序要保留，所以调用方两种都试。
static bool fuseAccumulator(arith::AddFOp add, unsigned tileOperand,
                            DominanceInfo &dominance) {
  auto tileTy = dyn_cast<VectorType>(add.getType());
  if (!tileTy || tileTy.isScalable() || tileTy.getRank() != 2 ||
      tileTy.getNumElements() > 65536)
    return false;

  Value accumulator = add->getOperand(1 - tileOperand);
  Value tile = add->getOperand(tileOperand);
  // inserts：从 add 往上追到的 insert 链，顺序是从后往前。
  // covered：tile 每个元素是否已被写过，用于证明完整不重叠覆盖。
  SmallVector<vector::InsertStridedSliceOp> inserts;
  llvm::SmallBitVector covered(tileTy.getNumElements());
  while (auto insert = tile.getDefiningOp<vector::InsertStridedSliceOp>()) {
    auto fragmentTy = insert.getValueToStore().getType();
    if (!insert->hasAttr("frisk.mma_fragment") || !tile.hasOneUse() ||
        insert->getBlock() != add->getBlock() ||
        fragmentTy.getRank() != 2 || fragmentTy.isScalable() ||
        !dominance.dominates(accumulator, insert.getOperation()))
      return false;
    auto offsets = insert.getOffsets();
    auto strides = insert.getStrides();
    if (offsets.size() != 2 || strides.size() != 2 ||
        cast<IntegerAttr>(strides[0]).getInt() != 1 ||
        cast<IntegerAttr>(strides[1]).getInt() != 1)
      return false;
    int64_t row = cast<IntegerAttr>(offsets[0]).getInt();
    int64_t col = cast<IntegerAttr>(offsets[1]).getInt();
    if (row < 0 || col < 0 ||
        row + fragmentTy.getDimSize(0) > tileTy.getDimSize(0) ||
        col + fragmentTy.getDimSize(1) > tileTy.getDimSize(1))
      return false;
    for (int64_t r = 0; r < fragmentTy.getDimSize(0); ++r)
      for (int64_t c = 0; c < fragmentTy.getDimSize(1); ++c) {
        auto index = (row + r) * tileTy.getDimSize(1) + col + c;
        if (covered.test(index))
          return false;
        covered.set(index);
      }
    inserts.push_back(insert);
    tile = insert.getDest();
  }
  if (inserts.empty() || !covered.all())
    return false;

  // 按原始写入顺序（reverse 后）把 add 下推到每个 insert：insert 改成写 old + delta，
  // 目的改成上一步的 updated。最后原 add 由链尾替换。
  IRRewriter rewriter(add.getContext());
  Value updated = accumulator;
  for (auto insert : llvm::reverse(inserts)) {
    rewriter.setInsertionPoint(insert);
    SmallVector<int64_t> offsets;
    for (Attribute offset : insert.getOffsets())
      offsets.push_back(cast<IntegerAttr>(offset).getInt());
    auto fragmentTy = insert.getValueToStore().getType();
    auto old = rewriter.create<vector::ExtractStridedSliceOp>(
        insert.getLoc(), updated, offsets, fragmentTy.getShape(),
        SmallVector<int64_t>{1, 1});
    Value delta = insert.getValueToStore();
    auto sum = rewriter.create<arith::AddFOp>(
        add.getLoc(), tileOperand == 0 ? delta : old.getResult(),
        tileOperand == 0 ? old.getResult() : delta);
    sum.setFastmathAttr(add.getFastmathAttr());
    sum->setAttr("frisk.fragment_accumulate", rewriter.getUnitAttr());
    rewriter.modifyOpInPlace(insert, [&] {
      insert.getValueToStoreMutable().assign(sum.getResult());
      insert.getDestMutable().assign(updated);
    });
    updated = insert.getResult();
  }
  rewriter.replaceOp(add, updated);
  return true;
}

class FuseFragmentAccumulatorPass
    : public impl::FuseFragmentAccumulatorBase<FuseFragmentAccumulatorPass> {
  void runOnOperation() override {
    if (failed(normalizeTypedFragmentIR(getOperation()))) return signalPassFailure();
    // 先走 fragment 路径（内层优先），再走旧的 vector 路径。
    SmallVector<Loop> loops;
    getOperation().walk([&](Loop loop) { loops.push_back(loop); });
    // Inner loops first; this matcher deletes only its consumer loop.
    for (auto loop : llvm::reverse(loops)) fuseTypedAccumulator(loop);
    DominanceInfo dominance(getOperation());
    SmallVector<arith::AddFOp> adds;
    getOperation().walk([&](arith::AddFOp add) { adds.push_back(add); });
    for (auto add : adds)
      if (!fuseAccumulator(add, 0, dominance))
        fuseAccumulator(add, 1, dominance);
  }
};
} // namespace

std::unique_ptr<Pass> createFuseFragmentAccumulatorPass() {
  return std::make_unique<FuseFragmentAccumulatorPass>();
}
} // namespace mlir::frisk
