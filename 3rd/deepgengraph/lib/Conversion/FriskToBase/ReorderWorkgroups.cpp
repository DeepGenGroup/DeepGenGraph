#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <limits>

namespace mlir::frisk {
#define GEN_PASS_DEF_REORDERWORKGROUPS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
namespace {
class ReorderWorkgroups : public impl::ReorderWorkgroupsBase<ReorderWorkgroups> {
  void runOnOperation() override {
    auto f = getOperation();
    if (groupSize == 0 || f.isDeclaration()) return;
    auto axis = [](StringRef s) -> int {
      if (s == "x") return 0;
      if (s == "y") return 1;
      if (s == "z") return 2;
      return -1;
    };
    int q = axis(queryAxis), h = axis(headAxis);
    auto reject = [&](StringRef message) { f.emitError(message); signalPassFailure(); };
    if (q < 0 || h < 0 || q == h || groupSize < 0)
      return reject("workgroup reorder requires distinct x/y/z axes and positive group-size");
    OpBuilder b(f.getContext());
    auto config = b.getDenseI64ArrayAttr({groupSize, q, h, reverse ? 1 : 0});
    if (auto old = f->getAttr("frisk.workgroup_order")) {
      if (old != config) reject("workgroup order already set with different options");
      return;
    }
    SmallVector<gpu::BlockIdOp> ids;
    int64_t sizes[3] = {0, 0, 0};
    bool invalid = false;
    f.walk([&](gpu::BlockIdOp id) {
      int d = int(id.getDimension());
      if (d != q && d != h) return;
      ids.push_back(id);
      auto range = id->getAttrOfType<IntegerAttr>("range");
      if (!range || range.getInt() <= 0 ||
          (sizes[d] && sizes[d] != range.getInt())) invalid = true;
      else sizes[d] = range.getInt();
    });
    if (ids.empty()) return; // Host/helper functions have no launch coordinates.
    if (invalid || !sizes[q] || !sizes[h])
      return reject("workgroup reorder needs consistent static positive block_id range attributes on both axes");
    if (sizes[q] > std::numeric_limits<int64_t>::max() / sizes[h])
      return reject("workgroup grid product overflows index arithmetic");
    // Newly created physical IDs are excluded from replacement. Entry-block
    // definitions dominate every original use, including nested loop bounds.
    b.setInsertionPointToStart(&f.front());
    auto loc = f.getLoc();
    auto id = [&](int d) -> Value {
      auto v = b.create<gpu::BlockIdOp>(loc, gpu::Dimension(d));
      v->setAttr("range", b.getIndexAttr(sizes[d]));
      return v;
    };
    Value pq = id(q), ph = id(h);
    auto c = [&](int64_t n) -> Value { return b.create<arith::ConstantIndexOp>(loc, n); };
    auto add = [&](Value a, Value v) -> Value { return b.create<arith::AddIOp>(loc, a, v); };
    auto mul = [&](Value a, Value v) -> Value { return b.create<arith::MulIOp>(loc, a, v); };
    auto div = [&](Value a, Value v) -> Value { return b.create<arith::DivUIOp>(loc, a, v); };
    auto rem = [&](Value a, Value v) -> Value { return b.create<arith::RemUIOp>(loc, a, v); };
    Value linear = q < h ? add(pq, mul(ph, c(sizes[q])))
                         : add(ph, mul(pq, c(sizes[h])));
    int64_t group = std::min<int64_t>(groupSize, sizes[q]);
    Value groupSpan = c(group * sizes[h]);
    Value start = mul(div(linear, groupSpan), c(group));
    Value width = c(group);
    if (sizes[q] % group) {
      Value remaining = b.create<arith::SubIOp>(loc, c(sizes[q]), start);
      width = b.create<arith::MinUIOp>(loc, remaining, width);
    }
    Value within = rem(linear, groupSpan);
    Value logicalQ = add(start, rem(within, width));
    Value logicalH = div(within, width);
    if (reverse)
      logicalQ = b.create<arith::SubIOp>(loc, c(sizes[q] - 1), logicalQ);
    for (auto old : ids) {
      old.replaceAllUsesWith(int(old.getDimension()) == q ? logicalQ : logicalH);
      old.erase();
    }
    f->setAttr("frisk.workgroup_order", config);
  }
};
}
std::unique_ptr<Pass> createReorderWorkgroupsPass() {
  return std::make_unique<ReorderWorkgroups>();
}
}
