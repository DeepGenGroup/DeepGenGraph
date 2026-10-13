#ifndef DEEPGENGRAPH_ANALYSIS_WMMALAYOUT_H
#define DEEPGENGRAPH_ANALYSIS_WMMALAYOUT_H

#include "deepgengraph/Analysis/LowerInfo.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::frisk {
// A square transpose exchanges register bits with the column lane bits.
// Keep this predicate shared by inference and conversion planning. No kernel
// names, matrix sizes, or instruction spelling participate in the decision.
inline bool isWmmaCTransposeLayout(const LowerInfo &info) {
  const auto &d = info.base_layout;
  int64_t rows = d.warp_layout[0], group = d.warp_layout[1];
  return !info.logicalTransform && info.warp_threads == 64 && rows > 0 && group > 1 &&
         llvm::isPowerOf2_64(rows) && llvm::isPowerOf2_64(group) &&
         rows * group == info.warp_threads &&
         d.warp_layout_order == coordXY_t{0, 1} &&
         d.thread_creg == coordXY_t{1, 1} &&
         d.warp_repeat == coordXY_t{1, group} && info.ignoreDim == -1;
}

inline bool makeWmmaCContiguous(LowerInfo &info) {
  if (!isWmmaCTransposeLayout(info))
    return false;
  info.base_layout.thread_creg = info.base_layout.warp_repeat;
  info.base_layout.warp_repeat = {1, 1};
  return true;
}

// Group g adjacent N instructions (g column lanes and g C registers).
// Physical column n = g^2*q + g*r + lane becomes
// logical column g^2*lane + g*q + r, within each g^3-column group.
// For g=4 the first B instruction reads 0,16,32,48,1,17,...;
// each C instruction then produces four consecutive logical columns per lane.
// The same bijection is attached to B and C; A and the K reduction are intact.
inline bool makeWmmaBPermutedC(LowerInfo &info) {
  if (!isWmmaCTransposeLayout(info) || !info.mmaInst)
    return false;
  int64_t g = info.get_warp_layout()[1];
  int64_t n = g * g, span = n * g;
  auto type = dyn_cast<ShapedType>(info.buffer.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      type.getDimSize(1) % span != 0 || info.mmaInst->n != n ||
      info.mmaInst->m != info.get_warp_layout()[0])
    return false;
  auto *ctx = info.buffer.getContext();
  auto row = getAffineDimExpr(0, ctx), col = getAffineDimExpr(1, ctx);
  auto logicalCol = col.floorDiv(span) * span + (col % g) * n +
                    ((col.floorDiv(n)) % g) * g + (col.floorDiv(g)) % g;
  info.logicalTransform = AffineMap::get(2, 0, {row, logicalCol}, ctx);
  return true;
}
} // namespace mlir::frisk
#endif
