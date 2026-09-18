#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_FUSEFRAGMENTACCUMULATOR
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

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
