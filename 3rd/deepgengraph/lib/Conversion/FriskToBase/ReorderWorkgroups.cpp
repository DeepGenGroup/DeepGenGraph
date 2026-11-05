// ===========================================================================
// Pass: ReorderWorkgroups（frisk-reorder-workgroups）   优化汇总.py 第 2 项，block 阶段
//
// 作用：
//   只改“硬件 block 编号 -> 逻辑 (query tile, head)”的映射，不改计算内容。让同一个 head
//   的若干 query tile 在派发顺序上相邻，读同一份 K/V 时更容易命中 L2。
//
// 映射公式（G = group，Q/H = query/head 轴大小，linear = 硬件线性编号）：
//   groupSpan = G * H                一组 G 个 query 覆盖的 block 数
//   start     = linear / groupSpan * G
//   width     = min(Q - start, G)    只有 Q 不能被 G 整除时才需要这个尾组处理
//   within    = linear % groupSpan
//   逻辑 q    = start + within % width
//   逻辑 head = within / width
//   这是双射：每个硬件 block 恰好对应一个逻辑 block。
//   reverse 时再做 q = Q - 1 - q。
//
// 选项：
//   groupSize  每组连续的 query tile 数，0 表示关闭。超过 Q 时按 Q 处理。
//   reverse    翻转 query 顺序。
//   queryAxis / headAxis  query 和 head 分别对应哪个 block 维度（x/y/z）。
//
// 产物：函数属性 frisk.workgroup_order = {group, q, h, reverse}，用于防止用不同配置重复执行。
// ===========================================================================
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
    // q / h：query 轴和 head 轴的维度编号（0/1/2）。
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
    // ids：需要被替换的原始 block_id。
    // sizes[d]：第 d 维的 block 数，取自 range 属性；同一维的多个 block_id 必须一致。
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
    // 按 x + y*X 的方式还原硬件线性编号：维度编号小的是快变维。
    // 这里假设硬件大致按这个顺序派发（优化汇总.py 中标为未确认的前提）。
    Value linear = q < h ? add(pq, mul(ph, c(sizes[q])))
                         : add(ph, mul(pq, c(sizes[h])));
    // 下面按文件头的公式生成 IR，变量名与公式一一对应。
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
    // 把所有旧 block_id 的使用替换成逻辑坐标。新创建的物理 block_id 不在 ids 中，不会被替换。
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
