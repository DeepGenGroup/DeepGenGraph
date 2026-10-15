#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/IR/Block.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::frisk;

static void check(bool condition, const char *message) {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

static void initialize(LowerInfo &info, Value buffer, coordXY_t creg,
                       coordXY_t order) {
  info.buffer = buffer;
  info.base_layout = {};
  info.base_layout.thread_creg = creg;
  info.base_layout.thread_creg_order = order;
  info.base_layout.warp_layout = {4, 2};
  info.base_layout.warp_layout_order = {0, 1};
  info.base_layout.warp_repeat = {1, 2};
  info.base_layout.warp_repeat_order = {1, 0};
  info.base_layout.wg_layout = {3, 2};
  info.base_layout.wg_layout_order = {0, 1};
  info.block_layout = {2, 3};
  info.block_layout_order = {1, 0};
  info.block_repeat = {2, 3};
  info.warpInstUnroll = {3, 1};
  info.thread_own_data_size = creg * coordXY_t{6, 6};
  info.reuseable = {BufferAxisReuse::Hold, BufferAxisReuse::CanReuse};
  info.warpIdDivisor = 2;
}

// Verify the observable mapping, including all repeat/unroll coordinates and
// broadcast waves, instead of only checking that fields have been swapped.
static void testTranspose(MLIRContext &context, Value buffer, coordXY_t creg,
                          coordXY_t order, bool withTransform, int ignoreDim) {
  LowerInfo info(8);
  initialize(info, buffer, creg, order);
  info.ignoreDim = ignoreDim;
  MMAInstInfo instruction{};
  info.mmaInst = &instruction;
  LowerInfo source(8);
  info.convertFrom = &source;
  if (withTransform) {
    auto row = getAffineDimExpr(0, &context);
    auto col = getAffineDimExpr(1, &context);
    info.logicalTransform = AffineMap::get(
        2, 0, {row, col.floorDiv(4) * 4 + (col % 2) * 2 +
                        col.floorDiv(2) % 2}, &context);
  }
  auto oldMap = info.getAffineMap();
  auto oldBlockWidths = info.get_block_widths();
  auto oldTotalWidths = info.get_thread_total_widths();
  auto oldOwned = info.get_thread_own_data_size();
  auto oldWgWidths = info.base_layout.get_wg_widths();
  check(info.convert_to_row_major(), "column-major layout must transpose");
  check(info.getOperandLabels().empty(), "invalidate cached operands");
  check(info.base_layout.thread_creg == coordXY_t{creg[1], creg[0]} &&
            info.base_layout.thread_creg_order == coordXY_t{1, 0},
        "row-major creg");
  check(info.get_block_widths() ==
            coordXY_t{oldBlockWidths[1], oldBlockWidths[0]} &&
            info.get_thread_total_widths() ==
            coordXY_t{oldTotalWidths[1], oldTotalWidths[0]} &&
            info.get_thread_own_data_size() ==
            coordXY_t{oldOwned[1], oldOwned[0]} &&
            info.base_layout.get_wg_widths() ==
            coordXY_t{oldWgWidths[1], oldWgWidths[0]},
        "transpose coverage and per-thread storage");
  check(info.reuseable[0] == BufferAxisReuse::CanReuse &&
            info.reuseable[1] == BufferAxisReuse::Hold &&
            info.ignoreDim == (ignoreDim < 0 ? -1 : 1 - ignoreDim),
        "transpose axis semantics");
  check(info.buffer == buffer && info.mmaInst == &instruction &&
            info.convertFrom == &source && info.warp_threads == 8 &&
            info.warpIdDivisor == 2,
        "preserve associations and thread numbering");

  auto newMap = info.getAffineMap();
  for (int64_t tid = 0; tid < 96; ++tid)
    for (int64_t br0 = 0; br0 < 2; ++br0)
      for (int64_t br1 = 0; br1 < 3; ++br1)
        for (int64_t iu0 = 0; iu0 < 3; ++iu0)
          for (int64_t wr = 0; wr < 2; ++wr)
            for (int64_t reg = 0; reg < flat_size(creg); ++reg) {
              auto oldXY = oldMap.compose({tid, br0, br1, iu0, 0, wr, reg});
              auto newXY = newMap.compose({tid, br1, br0, 0, iu0, wr, reg});
              check(newXY[0] == oldXY[1] && newXY[1] == oldXY[0],
                    "every mapped coordinate must transpose");
            }
  check(!info.convert_to_row_major() && info.getAffineMap() == newMap,
        "conversion must be idempotent");
}

static void testNoTranspose(Value buffer, coordXY_t creg, coordXY_t order) {
  LowerInfo info(8);
  initialize(info, buffer, creg, order);
  auto oldMap = info.getAffineMap();
  auto widths = info.get_block_widths();
  check(!info.convert_to_row_major(), "row-major/scalar layout stays put");
  auto newMap = info.getAffineMap();
  check(widths == info.get_block_widths() &&
            info.base_layout.thread_creg == creg &&
            info.base_layout.thread_creg_order == coordXY_t{1, 0},
        "normalize degenerate order without transposing axes");
  for (int64_t reg = 0; reg < flat_size(creg); ++reg)
    check(oldMap.compose({7, 1, 2, 2, 0, 1, reg}) ==
              newMap.compose({7, 1, 2, 2, 0, 1, reg}),
          "order normalization preserves valid register indices");
  check(!info.convert_to_row_major() && info.getAffineMap() == newMap,
        "no-op conversion stays idempotent");
}

int main() {
  MLIRContext context;
  Block block;
  auto buffer = block.addArgument(MemRefType::get({192, 144},
                                                 Float32Type::get(&context)),
                                  UnknownLoc::get(&context));
  for (bool transform : {false, true}) {
    testTranspose(context, buffer, {4, 1}, {0, 1}, transform, -1);
    testTranspose(context, buffer, {4, 1}, {1, 0}, transform, 0);
    testTranspose(context, buffer, {2, 3}, {0, 1}, transform, 1);
  }
  for (coordXY_t order : {coordXY_t{0, 1}, coordXY_t{1, 0}}) {
    testNoTranspose(buffer, {1, 1}, order);
    testNoTranspose(buffer, {1, 4}, order);
  }
  testNoTranspose(buffer, {2, 3}, {1, 0});
  llvm::outs() << "LowerInfo row-major layout tests passed\n";
}
