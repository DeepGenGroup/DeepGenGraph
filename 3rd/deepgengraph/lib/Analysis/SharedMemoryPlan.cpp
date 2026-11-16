#include "deepgengraph/Analysis/SharedMemoryPlan.h"
#include "deepgengraph/Analysis/SharedBufferAliases.h"
#include "deepgengraph/Analysis/MatrixLoad.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>

namespace mlir::frisk {
namespace {
constexpr StringLiteral layoutAttrs[] = {
    "frisk.shared_pack_axis", "frisk.shared_pack", "frisk.shared_pair_stride",
    "frisk.shared_store_stride", "frisk.shared_store_pair", "frisk.shared_packet_rows",
    "frisk.shared_swizzle_src", "frisk.shared_swizzle_dst", "frisk.shared_swizzles"};

std::optional<coordXY_t> globalStrides(Value source) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || type.getRank() != 2) return std::nullopt;
  std::array<unsigned, 2> axes{0, 1};
  while (true) {
    if (auto view = source.getDefiningOp<BufferViewOp>()) {
      unsigned leading = view.getSourceType().getRank() - view.getViewType().getRank();
      for (auto &axis : axes) axis += leading;
      source = view.getSource();
    } else if (auto transpose = source.getDefiningOp<memref::TransposeOp>()) {
      for (auto &axis : axes)
        axis = cast<AffineDimExpr>(transpose.getPermutation().getResult(axis)).getPosition();
      source = transpose.getIn();
    } else break;
  }
  type = dyn_cast<MemRefType>(source.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || type.getMemorySpaceAsInt() != 1 ||
      failed(type.getStridesAndOffset(strides, offset))) return std::nullopt;
  coordXY_t result{strides[axes[0]], strides[axes[1]]};
  if (llvm::any_of(result, [](int64_t s) { return s <= 0 || ShapedType::isDynamic(s); }))
    return std::nullopt;
  return result;
}

bool isWholeCopy(CopyOp copy, MemRefType type) {
  auto src = dyn_cast<MemRefType>(copy.getSrc().getType());
  auto map = copy.getOffsetMap();
  return src && src.getShape() == type.getShape() && map.getNumInputs() == 0 &&
      map.getNumResults() == 1 && isa<AffineConstantExpr>(map.getResult(0)) &&
      cast<AffineConstantExpr>(map.getResult(0)).getValue() == type.getRank();
}

// Explicit scalar accesses need no MMA/layout anchor. Normalize affine loop
// domains and thread IDs into the same finite access description as LowerInfo.
std::optional<LogicalAccess> scalarAccess(Operation *op, MemRefType type, unsigned threads) {
  AffineMap map;
  SmallVector<Value> operands;
  if (auto load = dyn_cast<memref::LoadOp>(op)) operands.assign(load.getIndices().begin(), load.getIndices().end());
  else if (auto store = dyn_cast<memref::StoreOp>(op)) operands.assign(store.getIndices().begin(), store.getIndices().end());
  else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    map = load.getAffineMap(); operands.assign(load.getMapOperands().begin(), load.getMapOperands().end());
  } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    map = store.getAffineMap(); operands.assign(store.getMapOperands().begin(), store.getMapOperands().end());
  } else return std::nullopt;
  if (type.getRank() != 2 || !type.hasStaticShape()) return std::nullopt;
  if (!map) map = AffineMap::getMultiDimIdentityMap(operands.size(), op->getContext());
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  LogicalAccess result;
  result.shape = {type.getDimSize(0), type.getDimSize(1)};
  result.elementBytes = type.getElementTypeBitWidth() / 8;
  result.boundsMasked = false;
  result.upperBounds.fill(1); result.upperBounds[0] = threads;
  SmallVector<AffineExpr> replacements;
  unsigned dimension = 1;
  for (Value value : operands) {
    if (auto tid = value.getDefiningOp<gpu::ThreadIdOp>(); tid && tid.getDimension() == gpu::Dimension::x) {
      replacements.push_back(getAffineDimExpr(0, op->getContext())); continue;
    }
    auto arg = dyn_cast<BlockArgument>(value);
    auto loop = arg ? dyn_cast_or_null<affine::AffineForOp>(arg.getOwner()->getParentOp()) : affine::AffineForOp{};
    if (!loop || loop.getInductionVar() != value || !loop.hasConstantLowerBound() ||
        !loop.hasConstantUpperBound() || dimension >= 6) return std::nullopt;
    int64_t low = loop.getConstantLowerBound(), high = loop.getConstantUpperBound(), step = loop.getStepAsInt();
    if (high <= low || step <= 0) return std::nullopt;
    result.upperBounds[dimension] = llvm::divideCeil(uint64_t(high - low), uint64_t(step));
    replacements.push_back(getAffineDimExpr(dimension++, op->getContext()) * step + low);
  }
  result.coordinates = map.replaceDimsAndSymbols(
      ArrayRef(replacements).take_front(map.getNumDims()),
      ArrayRef(replacements).drop_front(map.getNumDims()), 7, 0);
  return result;
}

SharedAccessDesc describe(OpOperand &use, unsigned threads,
                          LowerInfoMap &layouts, HWSpecification &hw) {
  SharedAccessDesc result;
  result.op = use.getOwner(); result.operand = use.getOperandNumber();
  result.buffer = use.get();
  result.attributes = result.op->getAttrDictionary();
  auto fail = [&](StringRef reason) {
    result.kind = SharedAccessKind::Unknown; result.reason = reason.str(); return result;
  };
  auto type = cast<MemRefType>(use.get().getType());
  auto *info = layouts.getOperandLowerInfo(use);
  if (info)
    if (auto logical = info->getLogicalAccess(threads)) result.computationCoordinates = logical->coordinates;
  if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(result.op)) {
    if (use.getOperandNumber() != 1 || copy->hasAttr("frisk.source_map") ||
        cast<MemRefType>(copy.getSrc().getType()).getShape() != type.getShape())
      return fail("copy must write the complete logical allocation before physical packing");
    result.kind = SharedAccessKind::AsyncCopy;
    result.source = copy.getSrc();
    result.sourceStrides = globalStrides(copy.getSrc());
    result.sourceAlignment = copy.getAlignment();
    result.copyThreads = threads;
    if (auto attr = copy->getAttrOfType<IntegerAttr>("frisk.copy_threads")) {
      if (attr.getInt() <= 0 || attr.getInt() > threads || attr.getInt() % hw.getWarpsize())
        return fail("copy_threads must select complete waves");
      result.copyThreads = attr.getInt();
    }
    if (auto attr = copy->getAttrOfType<IntegerAttr>("frisk.dma_bytes")) {
      if (attr.getInt() <= 0 || !hw.dataCopyInfo.getDMATransfer(attr.getInt()))
        return fail("requested DMA width is not a target capability");
      result.requestedDMABytes = attr.getInt();
    }
    if (!result.sourceStrides) return fail("unknown source-view strides");
    if (copy->hasAttr("frisk.force_sync")) {
      if (result.requestedDMABytes) return fail("force_sync conflicts with explicit DMA width");
      result.kind = SharedAccessKind::SyncCopy;
    }
    return result;
  }
  if (auto copy = dyn_cast<CopyOp>(result.op)) {
    if (!isWholeCopy(copy, type)) return fail("partial or offset synchronous copy");
    if (use.getOperandNumber() == 1) {
      auto src = cast<MemRefType>(copy.getSrc().getType());
      if (src.getMemorySpaceAsInt() == 1) {
        result.kind = SharedAccessKind::SyncCopy;
        result.source = copy.getSrc(); result.sourceStrides = globalStrides(copy.getSrc());
        result.copyThreads = threads;
        if (!result.sourceStrides) return fail("unknown synchronous source-view strides");
        return result;
      }
      info = layouts.getOperandLowerInfo(copy->getOpOperand(0));
      result.computationCoordinates = {};
      result.kind = SharedAccessKind::Write;
    } else result.kind = SharedAccessKind::Read;
  } else if (auto read = dyn_cast<ReadMatrixFromShmOp>(result.op)) {
    auto *inst = getMatrixLoadInstruction(read, hw);
    if (!read.isBlockForm() || !inst || !info || !matchesMatrixLoad(*inst, type, *info))
      return fail("unknown matrix instruction footprint");
    auto physical = *info;
    physical.base_layout = inst->inShmLayout;
    physical.block_repeat = {type.getDimSize(0) / physical.get_block_widths()[0],
                             type.getDimSize(1) / physical.get_block_widths()[1]};
    result.logical = physical.getLogicalAccess(threads);
    result.kind = SharedAccessKind::MatrixRead;
    result.requiredPacketBytes = flat_size(inst->inShmLayout.thread_creg) * type.getElementTypeBitWidth() / 8;
    return result;
  } else if (isa<GemmOp, AddOp, SubOp, MulOp, DivOp, MaxOp, Exp2Op, CastOp,
                 ConvertLayoutOp>(result.op)) {
    result.kind = SharedAccessKind::Read;
  } else if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp, affine::AffineStoreOp>(result.op)) {
    result.logical = scalarAccess(result.op, type, threads);
    result.kind = isa<memref::StoreOp, affine::AffineStoreOp>(result.op) ? SharedAccessKind::Write : SharedAccessKind::Read;
    if (!result.logical) return fail("unknown scalar index or loop domain");
    return result;
  } else return fail("unsupported access or escaping descriptor");
  if (!info) return fail("missing operand-specific computation layout");
  auto snapshot = *info;
  snapshot.buffer = result.buffer;
  result.logical = snapshot.getLogicalAccess(threads);
  if (!result.logical) return fail("unknown logical access domain");
  return result;
}

bool sameLayout(const ShmDataIODesc &a, const ShmDataIODesc &b) {
  return a.packAxis == b.packAxis && a.packElements == b.packElements &&
      a.pairStride == b.pairStride && a.storeStride == b.storeStride &&
      a.storePair == b.storePair && a.packetRows == b.packetRows &&
      a.swizzleSourceBit == b.swizzleSourceBit && a.swizzleTargetBit == b.swizzleTargetBit &&
      a.xorSwizzles == b.xorSwizzles;
}

// Cache logical coordinates once per access. Lane is the inner dimension so
// each packet is evaluated across every active wave at every repeat.
struct Samples {
  unsigned registers = 0;
  SmallVector<coordXY_t> points;
};
std::optional<Samples> sample(const LogicalAccess &access, uint64_t limit) {
  uint64_t count = 1;
  for (auto bound : access.upperBounds) {
    if (bound <= 0 || uint64_t(bound) > limit / count) return std::nullopt;
    count *= bound;
  }
  Samples result;
  result.registers = access.upperBounds[6];
  result.points.reserve(count);
  std::array<int64_t, 7> at{};
  uint64_t iterations = count / access.upperBounds[0] / result.registers;
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    uint64_t rem = iteration;
    for (unsigned dim = 1; dim < 6; ++dim) {
      at[dim] = rem % access.upperBounds[dim]; rem /= access.upperBounds[dim];
    }
    for (at[0] = 0; at[0] < access.upperBounds[0]; ++at[0])
      for (at[6] = 0; at[6] < result.registers; ++at[6]) {
        auto xy = access.coordinates.compose(at);
        bool active = xy[0] >= 0 && xy[1] >= 0 && xy[0] < access.shape[0] && xy[1] < access.shape[1];
        if (!active && !access.boundsMasked) return std::nullopt;
        result.points.push_back(active ? coordXY_t{xy[0], xy[1]} : coordXY_t{-1, -1});
      }
  }
  return result;
}

std::optional<SharedAccessPlan> evaluateAccess(const SharedAccessDesc &access,
    const Samples &samples, const ShmDataIODesc &layout, HWSpecification &hw,
    unsigned alignment) {
  auto map = layout.getLogicalToPhysicalMap(access.op->getContext());
  SmallVector<int64_t> offsets;
  offsets.reserve(samples.points.size());
  for (auto xy : samples.points)
    offsets.push_back(xy[0] < 0 ? -1 : map.compose({xy[0], xy[1]})[0] * layout.elementBytes);
  unsigned wave = hw.getWarpsize(), registers = samples.registers;
  if (offsets.size() % (wave * registers)) return std::nullopt;
  unsigned chosen = 0;
  for (unsigned bytes : {16u, 8u, 4u, 2u, 1u}) {
    if (alignment % bytes) continue;
    if (access.requiredPacketBytes && bytes != access.requiredPacketBytes) continue;
    if (bytes % layout.elementBytes) continue;
    unsigned elements = bytes / layout.elementBytes;
    if (!elements || registers % elements) continue;
    bool legal = true;
    for (size_t i = 0; i < offsets.size() && legal; i += elements) {
      auto first = offsets[i];
      if (first >= 0 && first % bytes) { legal = false; break; }
      for (unsigned e = 0; e < elements; ++e)
        if ((first < 0 && offsets[i + e] >= 0) ||
            (first >= 0 && offsets[i + e] != first + e * layout.elementBytes)) {
          legal = false; break;
        }
    }
    if (legal) { chosen = bytes; break; }
  }
  if (!chosen) return std::nullopt;
  SharedAccessPlan plan;
  plan.vectorBytes = chosen;
  // Matrix packet placement is checked above. Ordinary ds_read phases are
  // not a measured model of collective matrix instructions: leave unknown.
  bool known = access.kind != SharedAccessKind::MatrixRead;
  bool packetKnown = access.kind == SharedAccessKind::MatrixRead;
  unsigned worst = 0;
  uint64_t cost = 0, packetCost = 0;
  unsigned elements = chosen / layout.elementBytes;
  auto kind = access.kind == SharedAccessKind::Write ? ShmAccessKind::Write : ShmAccessKind::Read;
  auto phases = hw.getBankConflictPhases(chosen, kind);
  for (size_t start = 0; start < offsets.size(); start += wave * registers)
    for (unsigned reg = 0; reg < registers; reg += elements) {
      SmallVector<int64_t> addresses;
      for (unsigned lane = 0; lane < wave; ++lane)
        addresses.push_back(offsets[start + lane * registers + reg]);
      if (llvm::all_of(addresses, [](int64_t a) { return a < 0; })) continue;
      ++plan.waveInstructions;
      if (access.kind == SharedAccessKind::MatrixRead) {
        // Retain the packet-placement heuristic used by matrix-load planning.
        // Evaluating only ordinary reads (or ignoring this access as unknown)
        // favors column-packet layouts with heavily conflicting row packets.
        unsigned packetWays = 0;
        for (auto partition : {ShmAccessKind::Read, ShmAccessKind::Write}) {
          auto ways = layout.getBankConflictWays(hw, addresses, chosen, partition);
          if (!ways) packetKnown = false;
          else packetWays = std::max(packetWays, *ways);
        }
        packetCost += packetWays;
        continue;
      }
      auto ways = layout.getBankConflictWays(hw, addresses, chosen, kind);
      if (!ways || !phases) known = false;
      else { worst = std::max(worst, *ways); cost += *ways * phases->size(); }
    }
  if (known) { plan.worstBankWays = worst; plan.bankCost = cost; }
  if (packetKnown) plan.packetCost = packetCost;
  return plan;
}

SharedLayoutCandidate evaluate(const SharedBufferGroup &group, ShmDataIODesc layout,
    ArrayRef<std::optional<Samples>> samples, HWSpecification &hw,
    const SharedPlanningOptions &options, bool requireExportable) {
  SharedLayoutCandidate result;
  result.layout = layout;
  auto reject = [&](StringRef reason) { result.reason = reason.str(); return result; };
  auto *ctx = group.allocations.front()->getContext();
  uint64_t total = group.shape[0] * group.shape[1];
  // Also check the inverse as a proof, not merely matching total element counts.
  auto forward = layout.getLogicalToPhysicalMap(ctx), inverse = layout.getPhysicalToLogicalMap(ctx);
  for (int64_t row = 0; row < group.shape[0]; ++row)
    for (int64_t col = 0; col < group.shape[1]; ++col) {
      int64_t p = forward.compose({row, col})[0];
      if (p < 0 || uint64_t(p) >= total || inverse.compose({p}) != SmallVector<int64_t>({row, col}))
        return reject("layout is not a bounded bijection");
    }
  result.exportable = true;
  result.bankCost = 0;
  for (auto [i, access] : llvm::enumerate(group.accesses)) {
    if (access.kind == SharedAccessKind::AsyncCopy || access.kind == SharedAccessKind::SyncCopy) {
      SharedCopyPlan copy;
      copy.access = i; copy.copyThreads = access.copyThreads;
      copy.totalElements = total;
      copy.sourceCoordinates = inverse;
      copy.sourceOffsets = layout.getGlobalSourceMap(ctx, *access.sourceStrides);
      if (access.kind == SharedAccessKind::SyncCopy) {
        if (options.requireMaximumDMA && isa<AsyncCopyGmToShmOp>(access.op))
          return reject("synchronous copy cannot satisfy requested maximum-width DMA");
        copy.exportable = true;
        // The synchronous implementation chooses its own register transfers.
        // Unknown copy cost must not appear as zero in automatic ranking.
        result.bankCost.reset();
        result.copies.push_back(copy); continue;
      }
      unsigned best = 0;
      for (const auto &dma : hw.dataCopyInfo.dmaTransfers) {
        if (!dma.alignmentBytes || group.alignment % dma.alignmentBytes) continue;
        if (options.requireMaximumDMA && !access.requestedDMABytes && hw.dataCopyInfo.getMaxDMATransfer() &&
            dma.bytesPerLane != hw.dataCopyInfo.getMaxDMATransfer()->bytesPerLane) continue;
        if (access.requestedDMABytes && dma.bytesPerLane != access.requestedDMABytes) continue;
        if (requireExportable && !llvm::is_contained(options.exportableDMABytes, dma.bytesPerLane)) continue;
        if (dma.bytesPerLane <= best || total * group.elementBytes % (hw.getWarpsize() * dma.bytesPerLane)) continue;
        if (ShmDataIODesc::selectDMATransferBytes(hw, copy.sourceOffsets, total,
              group.elementBytes, access.sourceAlignment, dma.bytesPerLane) != dma.bytesPerLane) continue;
        best = dma.bytesPerLane;
      }
      if (!best) {
        if (!options.allowSynchronousFallback || access.requestedDMABytes || options.requireMaximumDMA)
          return reject("no legal complete-wave DMA for a producer's source strides/alignment/width");
        copy.exportable = true;
        result.bankCost.reset();
        result.copies.push_back(copy);
        continue;
      }
      copy.bytesPerLane = best;
      if (options.reduceCopyWaves) {
        unsigned consumerBytes = best;
        for (const auto &a : group.accesses)
          if (a.logical) consumerBytes = std::min<unsigned>(consumerBytes,
              a.logical->upperBounds[6] * group.elementBytes);
        if (!consumerBytes || best % consumerBytes ||
            copy.copyThreads % ((best / consumerBytes) * hw.getWarpsize()))
          return reject("copy-wave reduction cannot preserve complete waves");
        copy.copyThreads /= best / consumerBytes;
      }
      copy.exportable = llvm::is_contained(options.exportableDMABytes, best);
      copy.waveInstructions = total * group.elementBytes / (hw.getWarpsize() * best);
      copy.rounds = llvm::divideCeil(total * group.elementBytes, uint64_t(copy.copyThreads) * best);
      auto tid = getAffineDimExpr(0, ctx), round = getAffineDimExpr(1, ctx);
      copy.destination = AffineMap::get(2, 0, (round * copy.copyThreads + tid) * (best / group.elementBytes));
      result.dmaInstructions += copy.waveInstructions;
      result.exportable &= copy.exportable;
      result.copies.push_back(copy);
    } else {
      if (access.kind == SharedAccessKind::MatrixRead &&
          (layout.packAxis != 1 || layout.packElements * layout.elementBytes != access.requiredPacketBytes ||
           layout.pairStride || layout.storeStride))
        return reject("matrix packet is not supported by the physical packer");
      if (!samples[i]) return reject("unknown or over-budget logical access domain");
      auto accessPlan = evaluateAccess(access, *samples[i], layout, hw, group.alignment);
      if (!accessPlan) return reject("layout breaks a consumer or producer's implicit packet");
      accessPlan->access = i;
      result.ldsInstructions += accessPlan->waveInstructions;
      if (!accessPlan->bankCost) result.bankCost.reset();
      else if (result.bankCost) *result.bankCost += *accessPlan->bankCost;
      result.accesses.push_back(*accessPlan);
    }
  }
  result.addressCost = layout.xorSwizzles.size() + (layout.swizzleSourceBit >= 0) +
      (layout.pairStride != 0) + (layout.storeStride != 0) + (layout.packetRows != 0);
  result.legal = true;
  if (!result.exportable) result.reason = "hardware-legal; selected DMA width has no enabled exporter";
  return result;
}

bool dominates(const SharedLayoutCandidate &a, const SharedLayoutCandidate &b) {
  if (!a.legal || !b.legal || a.exportable != b.exportable || !a.bankCost || !b.bankCost) return false;
  auto ac = std::make_tuple(*a.bankCost, a.dmaInstructions, a.ldsInstructions, a.addressCost);
  auto bc = std::make_tuple(*b.bankCost, b.dmaInstructions, b.ldsInstructions, b.addressCost);
  return ac != bc && *a.bankCost <= *b.bankCost && a.dmaInstructions <= b.dmaInstructions &&
      a.ldsInstructions <= b.ldsInstructions && a.addressCost <= b.addressCost;
}
} // namespace

std::optional<ShmDataIODesc> readSharedLayout(Operation *storage, MemRefType type) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getElementTypeBitWidth() % 8)
    return std::nullopt;
  for (auto key : layoutAttrs)
    if (key != "frisk.shared_swizzles" && storage->hasAttr(key) &&
        !storage->getAttrOfType<IntegerAttr>(key)) return std::nullopt;
  auto get = [&](StringRef key, int64_t fallback = 0) {
    auto attr = storage->getAttrOfType<IntegerAttr>(key);
    return attr ? attr.getInt() : fallback;
  };
  auto io = ShmDataIODesc::getPacked({type.getDimSize(0), type.getDimSize(1)},
      type.getElementTypeBitWidth() / 8, get("frisk.shared_pack_axis"), get("frisk.shared_pack", 1),
      get("frisk.shared_pair_stride"), get("frisk.shared_store_stride"), get("frisk.shared_store_pair"),
      get("frisk.shared_packet_rows"));
  if (!io) return std::nullopt;
  if (storage->hasAttr("frisk.shared_swizzle_src") || storage->hasAttr("frisk.shared_swizzle_dst")) {
    auto src = get("frisk.shared_swizzle_src", -1), dst = get("frisk.shared_swizzle_dst", -1);
    if (src < 4 || src > 30 || dst < 4 || dst > 30 || !io->setXorSwizzle(src, dst))
      return std::nullopt;
  }
  if (auto attr = storage->getAttr("frisk.shared_swizzles")) {
    auto bits = dyn_cast<DenseI64ArrayAttr>(attr);
    if (!bits || bits.size() % 2) return std::nullopt;
    for (unsigned i = 0; i < bits.size(); i += 2)
      if (bits[i] < 0 || bits[i] > 30 || bits[i+1] < 0 || bits[i+1] > 30 ||
          !io->appendXorSwizzle(bits[i], bits[i+1])) return std::nullopt;
  }
  return io;
}

void writeSharedLayout(Operation *storage, const ShmDataIODesc &io) {
  OpBuilder b(storage->getContext());
  for (auto key : layoutAttrs) storage->removeAttr(key);
  storage->setAttr("frisk.shared_pack_axis", b.getI64IntegerAttr(io.packAxis));
  storage->setAttr("frisk.shared_pack", b.getI64IntegerAttr(io.packElements));
  auto set = [&](StringRef key, int64_t value) {
    if (value) storage->setAttr(key, b.getI64IntegerAttr(value));
  };
  set("frisk.shared_pair_stride", io.pairStride);
  set("frisk.shared_store_stride", io.storeStride);
  set("frisk.shared_store_pair", io.storePair);
  set("frisk.shared_packet_rows", io.packetRows);
  if (io.swizzleSourceBit >= 0) {
    storage->setAttr("frisk.shared_swizzle_src", b.getI64IntegerAttr(io.swizzleSourceBit));
    storage->setAttr("frisk.shared_swizzle_dst", b.getI64IntegerAttr(io.swizzleTargetBit));
  }
  SmallVector<int64_t> bits;
  for (auto [src, dst] : io.xorSwizzles) bits.append({src, dst});
  if (!bits.empty()) storage->setAttr("frisk.shared_swizzles", b.getDenseI64ArrayAttr(bits));
}

SmallVector<SharedBufferGroup, 0> collectSharedBufferAccesses(
    func::FuncOp kernel, LowerInfoMap &layouts, HWSpecification &hw) {
  SmallVector<SharedBufferGroup, 0> result;
  llvm::SmallPtrSet<Operation *, 16> visited;
  DenseMap<Operation *, unsigned> operationOrder;
  kernel.walk([&](Operation *op) { operationOrder.try_emplace(op, operationOrder.size()); });
  auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
  kernel.walk([&](Operation *op) {
    if (!isa<AllocBufferOp, memref::AllocOp>(op) || visited.contains(op) || op->hasAttr("frisk.shared_plan")) return;
    auto type = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (!type || type.getMemorySpaceAsInt() != 3) return;
    SharedBufferGroup group;
    group.id = result.size();
    auto reject = [&](StringRef reason) {
      if (group.complete) group.reason = reason.str();
      group.complete = false;
    };
    SharedBufferAliases aliases;
    if (!aliases.collect(op->getResult(0))) reject("unknown alias, view, origin or escaping descriptor");
    group.allocations.assign(aliases.allocations.begin(), aliases.allocations.end());
    if (group.allocations.empty()) group.allocations.push_back(op);
    group.aliases.assign(aliases.values.begin(), aliases.values.end());
    for (auto *alloc : group.allocations) {
      visited.insert(alloc);
      group.allocationAttributes.push_back(alloc->getAttrDictionary());
    }
    if (!type.hasStaticShape() || type.getRank() != 2 || !type.getLayout().isIdentity() ||
        !type.getElementType().isIntOrFloat() || type.getElementTypeBitWidth() % 8)
      reject("requires a static two-dimensional logical allocation");
    else {
      group.shape = {type.getDimSize(0), type.getDimSize(1)};
      group.elementBytes = type.getElementTypeBitWidth() / 8;
    }
    if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 || threads.getInt() % hw.getWarpsize())
      reject("unknown or incomplete-wave workgroup");
    else group.threads = threads.getInt();
    group.alignment = 16;
    for (auto *alloc : group.allocations) {
      unsigned guaranteed = std::max<int64_t>(1, group.elementBytes);
      if (auto alignment = alloc->getAttrOfType<IntegerAttr>("alignment"))
        if (alignment.getInt() > 0) guaranteed = alignment.getInt();
      group.alignment = std::gcd(group.alignment, guaranteed);
      for (auto key : layoutAttrs)
        if (alloc->getAttr(key) != op->getAttr(key)) reject("conflicting layout constraints across aliases");
    }
    llvm::sort(aliases.accesses, [&](OpOperand *a, OpOperand *b) {
      return std::make_pair(operationOrder.lookup(a->getOwner()), a->getOperandNumber()) <
             std::make_pair(operationOrder.lookup(b->getOwner()), b->getOperandNumber());
    });
    for (auto *use : aliases.accesses) {
      if (isa<memref::DeallocOp>(use->getOwner())) continue;
      auto access = describe(*use, group.threads, layouts, hw);
      if (access.kind == SharedAccessKind::Unknown) reject(access.reason);
      group.accesses.push_back(std::move(access));
    }
    result.push_back(std::move(group));
  });
  return result;
}

SharedMemoryPlan planSharedMemory(func::FuncOp kernel, LowerInfoMap &layouts,
    HWSpecification &hw, const SharedPlanningOptions &options) {
  SharedMemoryPlan plan;
  for (auto &group : collectSharedBufferAccesses(kernel, layouts, hw)) {
    SharedGroupPlan gp;
    gp.group = std::move(group);
    auto &g = gp.group;
    for (auto &a : g.accesses) {
      auto request = options.requestedDMABytes.find(a.op);
      if (request == options.requestedDMABytes.end()) continue;
      gp.required = true;
      if ((a.requestedDMABytes && a.requestedDMABytes != request->second) ||
          a.kind != SharedAccessKind::AsyncCopy) {
        g.complete = false; g.reason = "conflicting DMA width constraints";
      }
      a.requestedDMABytes = request->second;
    }
    for (auto *alloc : g.allocations)
      gp.required |= options.requestedSwizzles.count(alloc) || alloc->hasAttr("frisk.shared_swizzles") ||
                     alloc->hasAttr("frisk.shared_swizzle_src");
    for (auto &access : g.accesses) gp.required |= access.op->hasAttr("frisk.dma_bytes");
    if (options.requireMaximumDMA)
      for (auto &access : g.accesses) gp.required |= isa<AsyncCopyGmToShmOp>(access.op);
    if (!g.complete || g.shape[0] <= 0 || g.shape[1] <= 0 ||
        uint64_t(g.shape[0]) > options.maxProofElements / g.shape[1]) {
      if (g.complete) { g.complete = false; g.reason = "allocation exceeds finite proof budget"; }
      plan.groups.push_back(std::move(gp)); continue;
    }
    SmallVector<std::optional<Samples>> samples;
    for (auto &access : g.accesses)
      samples.push_back(access.logical ? sample(*access.logical, options.maxProofElements) : std::nullopt);
    auto *storage = g.allocations.front();
    auto type = cast<MemRefType>(storage->getResult(0).getType());
    auto original = readSharedLayout(storage, type);
    std::optional<SmallVector<std::pair<int, int>, 2>> swizzleRequest;
    for (auto *alloc : g.allocations) {
      auto request = options.requestedSwizzles.find(alloc);
      gp.required |= alloc->hasAttr("frisk.shared_swizzles") || alloc->hasAttr("frisk.shared_swizzle_src") ||
                     request != options.requestedSwizzles.end();
      if (request != options.requestedSwizzles.end() && original) {
        if (swizzleRequest && *swizzleRequest != request->second) original.reset();
        else swizzleRequest = request->second;
      }
    }
    if (swizzleRequest && original) {
      SmallVector<std::pair<int, int>, 2> existing;
      if (original->swizzleSourceBit >= 0) existing.emplace_back(original->swizzleSourceBit, original->swizzleTargetBit);
      llvm::append_range(existing, original->xorSwizzles);
      if (!existing.empty() && (existing.size() > swizzleRequest->size() ||
          !std::equal(existing.begin(), existing.end(), swizzleRequest->begin()))) original.reset();
      else {
        original->clearSwizzles();
        for (auto [src, dst] : *swizzleRequest)
          if (!original->appendXorSwizzle(src, dst)) { original.reset(); break; }
      }
    }
    for (const auto &a : g.accesses) gp.required |= a.requestedDMABytes != 0;
    if (!original) {
      g.complete = false; g.reason = "malformed physical-layout constraint";
      plan.groups.push_back(std::move(gp)); continue;
    }
    SmallVector<ShmDataIODesc, 0> bases;
    auto addBase = [&](const ShmDataIODesc &io) {
      if (llvm::none_of(bases, [&](const auto &b) { return sameLayout(b, io); })) bases.push_back(io);
    };
    addBase(*original);
    bool fixed = llvm::any_of(ArrayRef(layoutAttrs).take_front(6),
        [&](StringRef key) { return storage->hasAttr(key); });
    bool explicitSwizzle = original->hasSwizzle() || swizzleRequest.has_value() ||
                           storage->hasAttr("frisk.shared_swizzles");
    if (!options.searchLayouts && !fixed && !gp.required) {
      g.reason = "automatic physical layout search disabled";
      plan.groups.push_back(std::move(gp)); continue;
    }
    if (!fixed && options.searchLayouts) {
      auto addWithSwizzles = [&](ShmDataIODesc io) {
        io.swizzleSourceBit = original->swizzleSourceBit;
        io.swizzleTargetBit = original->swizzleTargetBit;
        io.xorSwizzles = original->xorSwizzles;
        addBase(io);
      };
      for (int axis : {0, 1})
        for (int pack = 1; pack * g.elementBytes <= 16; pack *= 2) {
          auto io = ShmDataIODesc::getPacked(g.shape, g.elementBytes, axis, pack);
          if (!io) continue;
          addWithSwizzles(*io);
          if (axis == 0 && pack * g.elementBytes <= 8)
            for (int64_t stride = pack; stride <= g.shape[0] / 2; stride *= 2) {
              auto paired = ShmDataIODesc::getPacked(g.shape, g.elementBytes, axis, pack, stride);
              if (paired) addWithSwizzles(*paired);
            }
          if (axis == 1)
            // rows=1 is the row-major packet layout, distinct from rows=0.
            // A fixed XOR (including explicit "off") constrains only XOR,
            // not the underlying physical packet placement.
            for (int64_t rows = 1; rows <= std::min<int64_t>(g.shape[0], hw.getWarpsize()); rows *= 2) {
              auto packets = ShmDataIODesc::getPacked(g.shape, g.elementBytes, axis, pack, 0, 0, 0, rows);
              if (packets) addWithSwizzles(*packets);
            }
        }
    }
    auto addCandidate = [&](ShmDataIODesc io) {
      auto candidate = evaluate(g, io, samples, hw, options, false);
      bool retry = candidate.legal && !candidate.exportable;
      gp.candidates.push_back(std::move(candidate));
      if (retry) {
        auto enabled = evaluate(g, io, samples, hw, options, true);
        if (enabled.legal) gp.candidates.push_back(std::move(enabled));
      }
    };
    for (const auto &base : bases) addCandidate(base);
    // Search generic bank-address bits, bounded by the allocation's complete
    // XOR periods. No kernel name, ABI argument or tensor role participates.
    if (options.searchSwizzles && !explicitSwizzle && hw.dataCopyInfo.bankcount > 0 &&
        hw.dataCopyInfo.singleBankBytes > 0) {
      unsigned bankBits = llvm::Log2_64(hw.dataCopyInfo.bankcount * hw.dataCopyInfo.singleBankBytes);
      unsigned granule = 16;
      for (const auto &dma : hw.dataCopyInfo.dmaTransfers)
        granule = std::min(granule, dma.bytesPerLane);
      unsigned lowBit = llvm::Log2_64(std::max<uint64_t>(granule, g.elementBytes));
      // Expand only the best few physical bases to bound compile time. This is
      // a deterministic heuristic; legality is still checked for every access.
      SmallVector<unsigned> order;
      for (unsigned i = 0; i < gp.candidates.size(); ++i)
        if (gp.candidates[i].legal && gp.candidates[i].bankCost) order.push_back(i);
      llvm::stable_sort(order, [&](unsigned a, unsigned b) {
        auto &x = gp.candidates[a]; auto &y = gp.candidates[b];
        return std::make_tuple(*x.bankCost, x.dmaInstructions, x.addressCost) <
               std::make_tuple(*y.bankCost, y.dmaInstructions, y.addressCost);
      });
      SmallVector<ShmDataIODesc, 0> seeds;
      for (unsigned i : ArrayRef(order).take_front(3)) seeds.push_back(gp.candidates[i].layout);
      for (const auto &seed : seeds)
        for (unsigned src = bankBits; src < bankBits + 3; ++src)
          for (unsigned dst = lowBit; dst < bankBits; ++dst) {
            auto io = seed;
            if (io.appendXorSwizzle(src, dst)) addCandidate(io);
          }
      // A second XOR can improve a different bank bit. Apply in order, and
      // let the inverse and every producer prove the composed permutation.
      std::optional<unsigned> best;
      for (unsigned i = 0; i < gp.candidates.size(); ++i) {
        auto &c = gp.candidates[i];
        if (c.legal && c.bankCost && (!best || *c.bankCost < *gp.candidates[*best].bankCost)) best = i;
      }
      if (best && gp.candidates[*best].layout.hasSwizzle()) {
        auto seed = gp.candidates[*best].layout;
        for (unsigned src = bankBits; src < bankBits + 3; ++src)
          for (unsigned dst = lowBit; dst < bankBits; ++dst) {
            auto io = seed;
            if (io.appendXorSwizzle(src, dst)) addCandidate(io);
          }
      }
    }
    for (auto &c : gp.candidates)
      for (auto &other : gp.candidates)
        if (dominates(other, c)) { c.dominated = true; break; }
    for (unsigned i = 0; i < gp.candidates.size(); ++i) {
      auto &c = gp.candidates[i];
      if (!c.legal || !c.exportable || c.dominated) continue;
      if (!gp.selected) { gp.selected = i; continue; }
      auto &prev = gp.candidates[*gp.selected];
      auto fallbackCount = [&](const SharedLayoutCandidate &candidate) {
        return llvm::count_if(candidate.copies, [&](const SharedCopyPlan &copy) {
          return !copy.bytesPerLane && g.accesses[copy.access].kind == SharedAccessKind::AsyncCopy;
        });
      };
      if (fallbackCount(c) != fallbackCount(prev)) {
        if (fallbackCount(c) < fallbackCount(prev)) gp.selected = i;
        continue;
      }
      // An unmodelled sub-dword access must not pin selection merely because
      // it appears before the fully assessed candidates in enumeration order.
      if (c.bankCost.has_value() != prev.bankCost.has_value()) {
        if (c.bankCost) gp.selected = i;
        continue;
      }
      // A synchronous copy can leave total cost unknown while every compute
      // reader/writer has a measured footprint. Compare those known portions
      // without letting unmodelled scalar accesses win with an apparent zero.
      auto partialScore = [](const SharedLayoutCandidate &candidate) {
        unsigned unknown = 0;
        uint64_t known = 0, packets = 0;
        for (const auto &access : candidate.accesses) {
          if (access.bankCost) known += *access.bankCost;
          else if (access.packetCost) packets += *access.packetCost;
          else ++unknown;
        }
        return std::make_tuple(unknown, known, packets, candidate.ldsInstructions,
                               candidate.dmaInstructions, candidate.addressCost);
      };
      if (!c.bankCost && !prev.bankCost && partialScore(c) < partialScore(prev)) gp.selected = i;
      // Among fully known plans prefer LDS service, then transfer count.
      if (c.bankCost && prev.bankCost &&
          std::make_tuple(*c.bankCost, c.dmaInstructions, c.ldsInstructions, c.addressCost) <
          std::make_tuple(*prev.bankCost, prev.dmaInstructions, prev.ldsInstructions, prev.addressCost))
        gp.selected = i;
    }
    if (!gp.selected) g.reason = "no exportable candidate satisfies all accesses and constraints";
    plan.groups.push_back(std::move(gp));
  }
  return plan;
}

LogicalResult SharedMemoryPlan::publish(LowerInfoMap &layouts) const {
  // No partially published kernel on an unsupported explicit request.
  for (const auto &gp : groups) {
    bool required = gp.required;
    for (auto *alloc : gp.group.allocations)
      required |= alloc->hasAttr("frisk.shared_swizzles") || alloc->hasAttr("frisk.shared_swizzle_src");
    for (auto &a : gp.group.accesses) required |= a.requestedDMABytes != 0;
    if (required && !gp.selected)
      return gp.group.allocations.front()->emitError("shared-memory planning failed: ") << gp.group.reason;
    if (gp.selected) {
      SharedBufferAliases aliases;
      if (!aliases.collect(gp.group.allocations.front()->getResult(0)) ||
          aliases.values.size() != gp.group.aliases.size() ||
          aliases.allocations.size() != gp.group.allocations.size() ||
          llvm::any_of(gp.group.aliases, [&](Value v) { return !aliases.values.contains(v); }) ||
          llvm::any_of(gp.group.allocations, [&](Operation *op) { return !aliases.allocations.contains(op); }))
        return gp.group.allocations.front()->emitError("shared-memory plan invalidated by aliases");
      auto kernel = gp.group.allocations.front()->getParentOfType<func::FuncOp>();
      auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
      if (!threads || threads.getInt() != gp.group.threads)
        return kernel.emitError("shared-memory plan invalidated by workgroup size");
      unsigned accesses = 0;
      for (auto *use : aliases.accesses) if (!isa<memref::DeallocOp>(use->getOwner())) ++accesses;
      if (accesses != gp.group.accesses.size())
        return gp.group.allocations.front()->emitError("shared-memory plan invalidated by new accesses");
      for (auto [i, alloc] : llvm::enumerate(gp.group.allocations))
        if (alloc->getAttrDictionary() != gp.group.allocationAttributes[i])
          return alloc->emitError("shared-memory plan invalidated by allocation constraints");
    }
    for (auto &a : gp.group.accesses) {
      if (a.op->getOperand(a.operand) != a.buffer || a.op->getAttrDictionary() != a.attributes)
        return a.op->emitError("shared-memory plan invalidated by an operand change");
      if (a.source && a.op->getOperand(0) != a.source)
        return a.op->emitError("shared-memory plan invalidated by copy source");
      if (a.computationCoordinates) {
        auto *info = layouts.getOperandLowerInfo(a.op->getOpOperand(a.operand));
        auto current = info ? info->getLogicalAccess(gp.group.threads) : std::nullopt;
        if (!current || current->coordinates != a.computationCoordinates ||
            (a.logical && a.kind != SharedAccessKind::MatrixRead &&
             (current->upperBounds != a.logical->upperBounds || current->shape != a.logical->shape)))
          return a.op->emitError("shared-memory plan invalidated by a computation layout change");
      }
    }
  }
  for (const auto &gp : groups) {
    if (!gp.selected) continue;
    const auto &chosen = gp.candidates[*gp.selected];
    OpBuilder b(gp.group.allocations.front()->getContext());
    for (auto *alloc : gp.group.allocations) {
      writeSharedLayout(alloc, chosen.layout);
      alloc->setAttr("frisk.shared_plan", b.getUnitAttr());
    }
    for (auto &copy : chosen.copies) {
      auto *op = gp.group.accesses[copy.access].op;
      if (copy.bytesPerLane) {
        op->setAttr("frisk.dma_bytes", b.getI64IntegerAttr(copy.bytesPerLane));
        op->setAttr("frisk.copy_threads", b.getI64IntegerAttr(copy.copyThreads));
        op->setAttr("frisk.dma_destination_map", AffineMapAttr::get(copy.destination));
        op->setAttr("frisk.dma_source_map", AffineMapAttr::get(copy.sourceCoordinates));
        op->setAttr("frisk.dma_total_elements", b.getI64IntegerAttr(copy.totalElements));
      } else if (isa<AsyncCopyGmToShmOp>(op)) {
        op->setAttr("frisk.force_sync", b.getUnitAttr());
      }
      // source_map is installed when the destination is physically flattened;
      // applying it to the still-logical destination would violate its verifier.
    }
    for (const auto &a : gp.group.accesses) {
      auto &use = a.op->getOpOperand(a.operand);
      if (auto *info = layouts.getOperandLowerInfo(use)) {
        info->shmIO = chosen.layout;
        info->shmIO->baseLayout = info->base_layout; // legacy snapshot adapter
        info->sharedGroup = gp.group.id;
      }
    }
  }
  return success();
}

void SharedMemoryPlan::dump(llvm::raw_ostream &os) const {
  if (execution) dumpSharedMemoryExecution(*execution, os);
  for (const auto &gp : groups) {
    const auto &g = gp.group;
    os << "Shared group " << g.id << " shape=" << g.shape[0] << 'x' << g.shape[1]
       << " aliases=" << g.aliases.size() << " accesses=" << g.accesses.size()
       << " complete=" << g.complete;
    if (!g.reason.empty()) os << " reason=" << g.reason;
    os << '\n';
    for (auto &a : g.accesses) {
      os << "  access " << a.op->getName() << " operand=" << a.operand << " kind=" << unsigned(a.kind);
      if (!a.reason.empty()) os << " reason=" << a.reason;
      os << '\n';
      if (a.logical) {
        os << "    logical=" << a.logical->coordinates << " bounds=";
        for (auto bound : a.logical->upperBounds) os << bound << ',';
        os << " bounds-masked=" << a.logical->boundsMasked << '\n';
      }
    }
    for (auto [i, c] : llvm::enumerate(gp.candidates)) {
      os << "  candidate " << i << " axis=" << c.layout.packAxis << " pack=" << c.layout.packElements
         << " pair=" << c.layout.pairStride << " rows=" << c.layout.packetRows << " xor=";
      if (c.layout.swizzleSourceBit >= 0) os << c.layout.swizzleSourceBit << ':' << c.layout.swizzleTargetBit << ',';
      for (auto [src, dst] : c.layout.xorSwizzles) os << src << ':' << dst << ',';
      os << " legal=" << c.legal << " exportable=" << c.exportable << " dominated=" << c.dominated << " banks=";
      if (c.bankCost) os << *c.bankCost; else os << "unknown";
      os << " dma=" << c.dmaInstructions << " lds=" << c.ldsInstructions;
      if (gp.selected && *gp.selected == i) os << " selected";
      if (!c.reason.empty()) os << " reason=" << c.reason;
      os << '\n';
      for (auto &copy : c.copies)
        os << "    copy access=" << copy.access << " bytes=" << copy.bytesPerLane
           << " threads=" << copy.copyThreads << " rounds=" << copy.rounds << " exportable=" << copy.exportable
           << " source=" << copy.sourceCoordinates << '\n';
      for (auto &access : c.accesses) {
        os << "    access=" << access.access << " bytes=" << access.vectorBytes
           << " instructions=" << access.waveInstructions << " worst-bank-ways=";
        if (access.worstBankWays) os << *access.worstBankWays; else os << "unknown";
        if (access.packetCost) os << " packet-cost=" << *access.packetCost;
        os << '\n';
      }
    }
  }
}
} // namespace mlir::frisk
