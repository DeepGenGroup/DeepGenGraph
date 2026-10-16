#ifndef DEEPGENGRAPH_ANALYSIS_MATRIX_LOAD_H
#define DEEPGENGRAPH_ANALYSIS_MATRIX_LOAD_H

#include "deepgengraph/Analysis/LowerInfo.h"

namespace mlir::frisk {

// ISA register order is creg first, then warp_repeat. Keep it distinct from
// the row-major vector coordinates used by thread tiling.
inline coordXY_t matrixLoadCoordinate(const LinearLayout2DDesc &layout,
                                      int64_t lane, int64_t reg) {
  auto split = [](int64_t index, coordXY_t shape, coordXY_t order) {
    coordXY_t result{};
    result[order[0]] = index % shape[order[0]];
    result[order[1]] = index / shape[order[0]];
    return result;
  };
  auto l = split(lane, layout.warp_layout, layout.warp_layout_order);
  auto r = split(reg % flat_size(layout.thread_creg), layout.thread_creg,
                 layout.thread_creg_order);
  auto repeat = split(reg / flat_size(layout.thread_creg), layout.warp_repeat,
                      layout.warp_repeat_order);
  return l * layout.thread_creg + r + repeat * layout.get_warp_widths();
}

inline bool matchesMatrixLoad(const WarpMatrtixCopyShmToRegInfo &load,
                              MemRefType source, const LowerInfo &info) {
  if (!source || source.getRank() != 2 || !source.hasStaticShape() ||
      source.getMemorySpaceAsInt() != 3 || !info.mmaInst ||
      info.logicalTransform || info.warp_threads != 64 ||
      info.warpInstUnroll != coordXY_t{1, 1})
    return false;
  auto element = source.getElementType();
  if (!((element.isF16() && load.elementType == FriskDType::f16) ||
        (element.isF32() && load.elementType == FriskDType::f32)))
    return false;
  const auto &mma = info.base_layout;
  auto instructionShape = mma.get_warp_widths_total();
  if (load.shape[0] != instructionShape[0] || load.shape[1] <= 0 ||
      instructionShape[1] <= 0 || load.shape[1] % instructionShape[1] ||
      load.inShmLayout.get_warp_widths_total() != load.shape ||
      load.outRegLayout.get_warp_widths_total() != load.shape)
    return false;
  for (unsigned dim = 0; dim < 2; ++dim)
    if (source.getDimSize(dim) <= 0 || source.getDimSize(dim) % load.shape[dim] ||
        info.get_block_widths()[dim] <= 0 ||
        source.getDimSize(dim) % info.get_block_widths()[dim])
      return false;
  // Compare lane ownership AND register order after repeating WMMA B to the
  // matrix-read shape. This rejects ALT layouts with interleaved N registers.
  auto repeated = mma;
  repeated.warp_repeat[1] *= load.shape[1] / instructionShape[1];
  int64_t registers = flat_size(repeated.thread_creg * repeated.warp_repeat);
  if (registers != flat_size(load.outRegLayout.thread_creg *
                            load.outRegLayout.warp_repeat))
    return false;
  for (int64_t lane = 0; lane < 64; ++lane)
    for (int64_t reg = 0; reg < registers; ++reg)
      if (matrixLoadCoordinate(repeated, lane, reg) !=
          matrixLoadCoordinate(load.outRegLayout, lane, reg))
        return false;
  // These entries read a contiguous row packet per lane. The physical packer
  // preserves that packet and may relocate its base independently per lane.
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(source.getStridesAndOffset(strides, offset)) &&
         strides[1] == 1 && load.inShmLayout.thread_creg[0] == 1 &&
         load.inShmLayout.warp_repeat == coordXY_t{1, 1};
}

inline const WarpMatrtixCopyShmToRegInfo *getMatrixLoadInstruction(
    ReadMatrixFromShmOp op, HWSpecification &hw) {
  for (const auto &inst : hw.matrixLoadInsts)
    if (op.getM() == inst.m && op.getN() == inst.n &&
        op.getAlt() == inst.alt && op.getTranspose() == inst.transpose &&
        ((op.getElementBitWidth() == 16 && inst.elementType == FriskDType::f16) ||
         (op.getElementBitWidth() == 32 && inst.elementType == FriskDType::f32)))
      return &inst;
  return nullptr;
}
} // namespace mlir::frisk
#endif
