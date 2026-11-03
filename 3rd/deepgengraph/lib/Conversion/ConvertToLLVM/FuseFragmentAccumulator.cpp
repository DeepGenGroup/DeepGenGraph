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
static bool fuseTypedAccumulator(Loop consumer) {
  if (consumer.getNumResults() != 1 || consumer->hasAttr("frisk.pipelined")) return false;
  auto output = consumer.getBody()->getTerminator()->getOperand(0).getDefiningOp<InsertFragOp>();
  if (!output || output->getBlock() != consumer.getBody() ||
      output.getDestination() != consumer.getRegionIterArgs()[0]) return false;
  auto add = output.getValue().getDefiningOp<ElementwiseFragOp>();
  if (!add || add.getKind() != "add" || !add.getResult().hasOneUse()) return false;
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
    SmallVector<Operation *> setup;
    for (Operation *op = producer->getNextNode(); op != consumer; op = op->getNextNode()) {
      if (op->getNumRegions() || !isMemoryEffectFree(op) || !isSpeculatable(op) ||
          llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<MemRefType>(t); })) { valid = false; break; }
      for (Value v : op->getOperands()) if (v.getDefiningOp() == producer) valid = false;
      setup.push_back(op);
    }
    if (!valid) continue;
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
    OpBuilder b(insertion);
    Value value = insertion.getValue();
    if (value.getType() != fragment) {
      // Both fragments refer to the same register slice of the same tile.
      // Re-attach the consumer's type without changing lane/register ownership.
      value = b.create<UnpackFragOp>(add.getLoc(), payload, value);
      value = b.create<PackFragOp>(add.getLoc(), fragment, value);
    }
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
static bool fuseAccumulator(arith::AddFOp add, unsigned tileOperand,
                            DominanceInfo &dominance) {
  auto tileTy = dyn_cast<VectorType>(add.getType());
  if (!tileTy || tileTy.isScalable() || tileTy.getRank() != 2 ||
      tileTy.getNumElements() > 65536)
    return false;

  Value accumulator = add->getOperand(1 - tileOperand);
  Value tile = add->getOperand(tileOperand);
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
