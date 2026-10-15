#include "deepgengraph/Analysis/LowerInfo.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace mlir::frisk {

std::optional<ShmDataIODesc> ShmDataIODesc::getPacked(
    coordXY_t shape, int64_t elementBytes, int64_t axis, int64_t pack,
    int64_t pairStride, int64_t storeStride, int64_t storePair) {
  if ((axis != 0 && axis != 1) || shape[0] <= 0 || shape[1] <= 0 ||
      shape[0] > std::numeric_limits<int64_t>::max() / shape[1] ||
      shape[0] > std::numeric_limits<int64_t>::max() / 32 ||
      shape[1] > std::numeric_limits<int64_t>::max() / 32 ||
      elementBytes <= 0 || elementBytes > 16 || pack < 1 ||
      !llvm::isPowerOf2_64(pack) || pack > 16 / elementBytes ||
      shape[axis] % pack || pairStride < 0 || storeStride < 0 || storePair < 0)
    return std::nullopt;
  if (pairStride && (axis != 0 || pairStride < pack ||
      pairStride > shape[0] / 2 || pairStride % pack ||
      shape[0] % (2 * pairStride) || pack * elementBytes > 8))
    return std::nullopt;
  if ((storeStride || storePair) && (axis != 1 || pairStride ||
      storeStride < 2 || !llvm::isPowerOf2_64(storeStride) ||
      storePair < storeStride || storePair > shape[1] / 2 ||
      storePair % storeStride || shape[1] % (2 * storePair)))
    return std::nullopt;
  ShmDataIODesc desc;
  desc.shape = shape;
  desc.elementBytes = elementBytes;
  desc.packAxis = axis;
  desc.packElements = pack;
  desc.pairStride = pairStride;
  desc.storeStride = storeStride;
  desc.storePair = storePair;
  desc.plannedVectorAccessBytes = pack * elementBytes;
  return desc;
}

AffineMap ShmDataIODesc::getLogicalToPhysicalMap(MLIRContext *ctx) const {
  return mapLogicalToPhysical(AffineMap::getMultiDimIdentityMap(2, ctx));
}

AffineMap ShmDataIODesc::mapLogicalToPhysical(AffineMap logical) const {
  assert(logical && logical.getNumResults() == 2);
  auto simplify = [&](AffineExpr expr) {
    return simplifyAffineExpr(expr, logical.getNumDims(), logical.getNumSymbols());
  };
  auto row = logical.getResult(0), col = logical.getResult(1);
  int64_t pack = packElements;
  AffineExpr physical;
  if (pairStride) {
    // Simplification before pairing preserves proofs that adjacent register
    // elements share the same fragment, enabling vector coalescing later.
    auto fragment = simplify(row.floorDiv(pack));
    auto element = simplify(row % pack);
    int64_t groups = pairStride / pack;
    auto group = fragment.floorDiv(2 * groups) * groups + fragment % groups;
    if (2 * pack * elementBytes == 16 && shape[1] % 8 == 0) {
      auto columnGroup = col.floorDiv(8);
      col = columnGroup * 8 +
            ((col.floorDiv(4) + columnGroup.floorDiv(4) +
              group.floorDiv(groups == 1 ? 1 : 2)) % 2) * 4 +
            (col + columnGroup) % 4;
    }
    physical = (group * shape[1] + col) * (2 * pack) +
               (fragment.floorDiv(groups) % 2) * pack + element;
  } else if (storeStride) {
    auto reg = col.floorDiv(storeStride);
    int64_t group = storePair / storeStride;
    auto paired = reg.floorDiv(2 * group) * (2 * group) +
                  (reg % group) * 2 + (reg.floorDiv(group) % 2);
    physical = (col % storeStride * shape[0] + row) *
               (shape[1] / storeStride) + paired;
  } else if (packAxis == 1) {
    physical = (col.floorDiv(pack) * shape[0] + row) * pack + col % pack;
  } else {
    physical = (row.floorDiv(pack) * shape[1] + col) * pack + row % pack;
  }
  return simplifyAffineMap(AffineMap::get(logical.getNumDims(),
      logical.getNumSymbols(), physical));
}

AffineMap ShmDataIODesc::getPhysicalToLogicalMap(MLIRContext *ctx) const {
  auto p = getAffineDimExpr(0, ctx);
  int64_t pack = packElements;
  auto row = p.floorDiv(pack * shape[1]) * pack + p % pack;
  auto col = p.floorDiv(pack) % shape[1];
  if (packAxis == 1) {
    row = p.floorDiv(pack) % shape[0];
    col = p.floorDiv(pack * shape[0]) * pack + p % pack;
  }
  if (storeStride) {
    int64_t rowSize = shape[1] / storeStride;
    int64_t groups = storePair / storeStride;
    auto paired = p % rowSize;
    auto reg = paired.floorDiv(2 * groups) * (2 * groups) +
               (paired % 2) * groups + paired.floorDiv(2) % groups;
    row = p.floorDiv(rowSize) % shape[0];
    col = p.floorDiv(rowSize * shape[0]) + storeStride * reg;
  }
  if (pairStride) {
    int64_t groups = pairStride / pack;
    auto group = p.floorDiv(2 * pack * shape[1]);
    auto column = p.floorDiv(2 * pack) % shape[1];
    row = (group.floorDiv(groups) * (2 * groups) +
           (p.floorDiv(pack) % 2) * groups + group % groups) * pack + p % pack;
    col = column;
    if (2 * pack * elementBytes == 16 && shape[1] % 8 == 0) {
      auto columnGroup = column.floorDiv(8);
      col = columnGroup * 8 +
            ((column.floorDiv(4) - columnGroup.floorDiv(4) -
              group.floorDiv(groups == 1 ? 1 : 2)) % 2) * 4 +
            (column - columnGroup) % 4;
    }
  }
  return AffineMap::get(1, 0, {row, col}, ctx);
}

AffineMap ShmDataIODesc::getGlobalSourceMap(MLIRContext *ctx,
    coordXY_t sourceStrides, int64_t sourceOffset) const {
  auto inverse = getPhysicalToLogicalMap(ctx);
  return simplifyAffineMap(AffineMap::get(1, 0,
      inverse.getResult(0) * sourceStrides[0] +
      inverse.getResult(1) * sourceStrides[1] + sourceOffset));
}

unsigned ShmDataIODesc::selectDMATransferBytes(const HWSpecification &hw,
    MLIRContext *ctx, coordXY_t sourceStrides,
    unsigned sourceAlignmentBytes) const {
  if (!hw.dataCopyInfo.supportAsyncCopy || elementBytes <= 0 ||
      shape[0] <= 0 || shape[1] <= 0 || sourceStrides[0] <= 0 ||
      sourceStrides[1] <= 0 || sourceAlignmentBytes == 0 ||
      shape[0] > (1 << 20) / shape[1])
    return 0;
  int64_t total = shape[0] * shape[1];
  // Bound address arithmetic for the static proof.
  int64_t limit = std::numeric_limits<int64_t>::max() / elementBytes;
  if (sourceStrides[0] > limit / shape[0] ||
      sourceStrides[1] > limit / shape[1] ||
      sourceStrides[0] * shape[0] > limit - sourceStrides[1] * shape[1])
    return 0;
  auto source = getGlobalSourceMap(ctx, sourceStrides);
  unsigned best = 0;
  for (const auto &dma : hw.dataCopyInfo.dmaTransfers) {
    if (dma.bytesPerLane <= best || dma.alignmentBytes == 0 ||
        dma.ldsLaneStrideBytes != dma.bytesPerLane ||
        dma.bytesPerLane % elementBytes ||
        sourceAlignmentBytes % dma.alignmentBytes)
      continue;
    int64_t elements = dma.bytesPerLane / elementBytes;
    if (total % elements) continue;
    bool legal = true;
    int64_t base = 0;
    for (int64_t p = 0; p < total; ++p) {
      int64_t address = source.compose({p})[0];
      if (p % elements == 0) {
        base = address;
        legal = (address * elementBytes) % dma.alignmentBytes == 0;
      } else {
        legal = address == base + p % elements;
      }
      if (!legal) break;
    }
    if (legal) best = dma.bytesPerLane;
  }
  return best;
}

std::optional<unsigned> ShmDataIODesc::getBankConflictWays(
    const HWSpecification &hw, ArrayRef<int64_t> addresses,
    ShmAccessKind kind) const {
  auto phases = hw.getBankConflictPhases(plannedVectorAccessBytes, kind);
  int64_t bankBytes = hw.dataCopyInfo.singleBankBytes;
  int64_t banks = hw.dataCopyInfo.bankcount;
  if (!phases || bankBytes <= 0 || banks <= 0 ||
      addresses.size() != unsigned(hw.getWarpsize()))
    return std::nullopt;
  for (int64_t address : addresses)
    if (address < -1 || (address >= 0 &&
        (address % bankBytes || address > std::numeric_limits<int64_t>::max() -
                                              plannedVectorAccessBytes)))
      return std::nullopt;
  unsigned worst = 0;
  for (const auto &phase : *phases) {
    std::map<int64_t, std::set<int64_t>> readWords;
    std::map<int64_t, unsigned> writes;
    for (unsigned lane : phase) {
      int64_t address = addresses[lane];
      if (address == -1) continue;
      for (int64_t word = address / bankBytes;
           word < (address + plannedVectorAccessBytes) / bankBytes; ++word) {
        unsigned ways;
        if (kind == ShmAccessKind::Read) {
          auto &words = readWords[word % banks];
          words.insert(word);
          ways = words.size();
        } else {
          ways = ++writes[word % banks];
        }
        worst = std::max(worst, ways);
      }
    }
  }
  return worst;
}

} // namespace mlir::frisk
