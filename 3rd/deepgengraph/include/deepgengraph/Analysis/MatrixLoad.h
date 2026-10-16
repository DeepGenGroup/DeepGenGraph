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

// Matrix reads accept an independent address for each lane's row packet.
// A logical permutation can therefore relocate packets (including across MMA
// instructions), but must preserve every element's order inside the packet.
// Prove this over the finite tile, without depending on tensor roles or names.
inline bool preservesMatrixLoadPackets(MemRefType source, AffineMap transform,
                                       int64_t packet) {
  if (!transform) return true;
  if (transform.getNumDims() != 2 || transform.getNumSymbols() ||
      transform.getNumResults() != 2 || packet <= 0 ||
      source.getDimSize(1) % packet || source.getNumElements() > (1 << 20))
    return false;
  auto index = IndexType::get(source.getContext());
  int64_t rows = source.getDimSize(0), cols = source.getDimSize(1);
  SmallVector<char> seen(source.getNumElements(), 0);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < cols; col += packet) {
      coordXY_t base{};
      for (int64_t element = 0; element < packet; ++element) {
        SmallVector<Attribute> folded;
        if (failed(transform.constantFold(
                {IntegerAttr::get(index, row), IntegerAttr::get(index, col + element)},
                folded))) return false;
        coordXY_t at{cast<IntegerAttr>(folded[0]).getInt(),
                     cast<IntegerAttr>(folded[1]).getInt()};
        if (at[0] < 0 || at[0] >= rows || at[1] < 0 || at[1] >= cols)
          return false;
        if (!element) base = at;
        if (base[1] % packet || at != coordXY_t{base[0], base[1] + element} ||
            seen[at[0] * cols + at[1]]++)
          return false;
      }
    }
  return true;
}

inline bool matchesMatrixLoad(const WarpMatrtixCopyShmToRegInfo &load,
                              MemRefType source, const LowerInfo &info) {
  if (!source || source.getRank() != 2 || !source.hasStaticShape() ||
      source.getMemorySpaceAsInt() != 3 || !info.mmaInst ||
      info.warp_threads != 64 ||
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
         load.inShmLayout.warp_repeat == coordXY_t{1, 1} &&
         preservesMatrixLoadPackets(source, info.logicalTransform,
                                    load.inShmLayout.thread_creg[1]);
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

// Choose a bijective placement of intact lane packets. Score both documented
// 128-bit lane partitions as a packet-address model; this is not a claim about
// the target's collective-read bank counter. Check every instruction tile and
// its logical permutation, rather than specializing for Q/K/V or one origin.
inline int64_t selectMatrixPacketRows(const WarpMatrtixCopyShmToRegInfo &inst,
                                     MemRefType type, const LowerInfo &info,
                                     const HWSpecification &hw) {
  if (type.getNumElements() > (1 << 20)) return 0;
  int64_t packet = inst.inShmLayout.thread_creg[1];
  SmallVector<coordXY_t> logical;
  for (int64_t row = 0; row < type.getDimSize(0); row += inst.shape[0])
    for (int64_t col = 0; col < type.getDimSize(1); col += inst.shape[1])
      for (int64_t lane = 0; lane < 64; ++lane) {
        auto at = matrixLoadCoordinate(inst.inShmLayout, lane, 0) + coordXY_t{row, col};
        if (info.logicalTransform) {
          auto mapped = info.logicalTransform.compose({at[0], at[1]});
          at = {mapped[0], mapped[1]};
        }
        logical.push_back(at);
      }
  auto score = [&](int64_t rows) -> unsigned {
    auto io = ShmDataIODesc::getPacked({type.getDimSize(0), type.getDimSize(1)},
        type.getElementTypeBitWidth() / 8, 1, packet, 0, 0, 0, rows);
    if (!io) return 0;
    auto map = io->getLogicalToPhysicalMap(type.getContext());
    unsigned worst = 0;
    for (size_t start = 0; start < logical.size(); start += 64) {
      SmallVector<int64_t> addresses;
      for (auto at : ArrayRef(logical).slice(start, 64))
        addresses.push_back(map.compose({at[0], at[1]})[0] * io->elementBytes);
      for (auto kind : {ShmAccessKind::Read, ShmAccessKind::Write}) {
        auto ways = io->getBankConflictWays(hw, addresses, kind);
        if (!ways) return 0;
        worst = std::max(worst, *ways);
      }
    }
    return worst;
  };
  unsigned best = score(0);
  int64_t selected = 0;
  for (int64_t rows = 1; best > 1 && rows < type.getDimSize(0); rows *= 2) {
    unsigned ways = score(rows);
    if (ways && ways < best) { best = ways; selected = rows; }
  }
  return selected;
}
} // namespace mlir::frisk
#endif
