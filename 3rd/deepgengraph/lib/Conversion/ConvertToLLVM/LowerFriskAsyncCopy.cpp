#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
Value indexConstant(OpBuilder &b, Location loc, int64_t n) {
  return b.create<arith::ConstantIndexOp>(loc, n);
}

SmallVector<Value> coordinates(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<int64_t> shape) {
  SmallVector<Value> result(shape.size());
  for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
    Value size = indexConstant(b, loc, shape[dim]);
    result[dim] = b.create<arith::RemUIOp>(loc, linear, size);
    linear = b.create<arith::DivUIOp>(loc, linear, size);
  }
  return result;
}

// gfx936's direct-to-LDS operation adds lane_id * packetBytes to a
// wave-uniform destination base. Never pass the per-thread destination.
// Restrict DMA to layouts for which that mapping is exactly the logical copy.
unsigned selectDMA(frisk::AsyncCopyGmToShmOp op, unsigned requestedBytes = 0,
                   AffineMap sourceMap = {}) {
  auto src = cast<MemRefType>(op.getSrc().getType());
  auto dst = cast<MemRefType>(op.getDst().getType());
  unsigned bits = src.getElementTypeBitWidth();
  auto *hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000,
                                       op.getContext());
  if (bits != 8 && bits != 16 && bits != 32)
    return 0;
  SmallVector<int64_t> srcStrides, dstStrides;
  int64_t offset;
  if (failed(src.getStridesAndOffset(srcStrides, offset)) ||
      failed(dst.getStridesAndOffset(dstStrides, offset)))
    return 0;
  int64_t stride = 1;
  for (int64_t dim = dst.getRank() - 1; dim >= 0; --dim) {
    if (dst.getDimSize(dim) > 1 && dstStrides[dim] != stride)
      return 0;
    stride *= dst.getDimSize(dim);
  }
  // Check actual source addresses for each physical destination packet. The
  // physically contiguous source axis need not be the last logical axis.
  auto mapAttr = op->getAttrOfType<AffineMapAttr>("frisk.source_map");
  if (!sourceMap && mapAttr) sourceMap = mapAttr.getValue();
  if (dst.getNumElements() > (1 << 20) ||
      llvm::any_of(srcStrides, [](int64_t s) {
        return ShapedType::isDynamic(s) || s <= 0;
      })) return 0;
  if (!sourceMap) {
    SmallVector<AffineExpr> coords(src.getRank());
    auto flat = getAffineDimExpr(0, op.getContext());
    for (int64_t dim = src.getRank() - 1; dim >= 0; --dim) {
      coords[dim] = flat % src.getDimSize(dim);
      flat = flat.floorDiv(src.getDimSize(dim));
    }
    sourceMap = AffineMap::get(1, 0, coords, op.getContext());
  }
  if (sourceMap.getNumDims() != 1 || sourceMap.getNumSymbols() ||
      sourceMap.getNumResults() != unsigned(src.getRank())) return 0;
  auto address = getAffineConstantExpr(0, op.getContext());
  for (unsigned dim = 0; dim < unsigned(src.getRank()); ++dim)
    address = address + sourceMap.getResult(dim) * srcStrides[dim];
  auto offsets = AffineMap::get(1, 0, address);
  unsigned bytes = frisk::ShmDataIODesc::selectDMATransferBytes(
      *hw, offsets, dst.getNumElements(), bits / 8, op.getAlignment(), requestedBytes);
  // Automatic wide DMA uses complete waves, including the final round. This
  // also keeps the wide export's wave-uniform LDS-base contract explicit.
  if (!requestedBytes && bytes > 4 &&
      dst.getNumElements() % (hw->getWarpsize() * bytes * 8 / bits))
    bytes = frisk::ShmDataIODesc::selectDMATransferBytes(
        *hw, offsets, dst.getNumElements(), bits / 8, op.getAlignment(), 4);
  return bytes;
}

// Walk source storage in stride order; the tile may have padded rows or a
// logical transpose. This defines staging order without changing its ABI.
AffineMap sourceStorageOrder(frisk::AsyncCopyGmToShmOp op) {
  auto type = cast<MemRefType>(op.getSrc().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, [](int64_t s) { return ShapedType::isDynamic(s) || s <= 0; })) return {};
  SmallVector<unsigned> order;
  for (unsigned i = 0; i < type.getRank(); ++i) order.push_back(i);
  llvm::stable_sort(order, [&](unsigned a, unsigned b) { return strides[a] < strides[b]; });
  SmallVector<AffineExpr> indices(type.getRank());
  auto p = getAffineDimExpr(0, op.getContext());
  int64_t scale = 1;
  for (unsigned dim : order) {
    indices[dim] = p.floorDiv(scale) % type.getDimSize(dim);
    scale *= type.getDimSize(dim);
  }
  return AffineMap::get(1, 0, indices, op.getContext());
}

SmallVector<Value> mappedCoordinates(OpBuilder &b, Location loc, Value flat,
                                     AffineMap map) {
  SmallVector<Value> result;
  for (unsigned dim = 0; dim < map.getNumResults(); ++dim)
    result.push_back(b.create<affine::AffineApplyOp>(loc, map.getSubMap({dim}), ValueRange{flat}));
  return result;
}

// Use upstream's memref-level gather representation, but lower the pointer
// operation ourselves: upstream's chip database does not recognize gfx936.
struct GatherToGfx936 : ConvertOpToLLVMPattern<amdgpu::GatherToLDSOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(amdgpu::GatherToLDSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const final {
    if (!op->hasAttr("frisk.gfx936_dma"))
      return failure();
    auto loc = op.getLoc();
    Value src = getStridedElementPtr(r, loc, cast<MemRefType>(op.getSrc().getType()),
                                     adaptor.getSrc(), adaptor.getSrcIndices());
    Value dst = getStridedElementPtr(r, loc, cast<MemRefType>(op.getDst().getType()),
                                     adaptor.getDst(), adaptor.getDstIndices());
    auto i32 = r.getI32Type();
    Value address = r.create<LLVM::PtrToIntOp>(loc, i32, dst);
    address = r.create<ROCDL::ReadfirstlaneOp>(loc, i32, address);
    dst = r.create<LLVM::IntToPtrOp>(loc, dst.getType(), address);
    auto bytes = op->getAttrOfType<IntegerAttr>("frisk.dma_bytes");
    r.replaceOpWithNewOp<ROCDL::GlobalLoadAsyncLDSOp>(op, src, dst,
        r.getI32IntegerAttr(bytes.getInt()), r.getI32IntegerAttr(0), r.getI32IntegerAttr(0),
        ArrayAttr{}, ArrayAttr{}, ArrayAttr{});
    return success();
  }
};

struct LowerFriskAsyncCopyPass
    : PassWrapper<LowerFriskAsyncCopyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFriskAsyncCopyPass)
  LowerFriskAsyncCopyPass() = default;
  explicit LowerFriskAsyncCopyPass(bool enabled) { useAsyncCopy = enabled; }
  LowerFriskAsyncCopyPass(const LowerFriskAsyncCopyPass &other) : PassWrapper(other) {}
  Option<bool> useAsyncCopy{*this, "use-async-copy",
      llvm::cl::desc("Use gfx936 wave64 asyncmark DMA; false uses ordinary copies"),
      llvm::cl::init(true)};
  StringRef getArgument() const final { return "frisk-lower-async-copy"; }
  StringRef getDescription() const final {
    return "Lower tile copies and buffer readiness for gfx936 or synchronous mode";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<frisk::FriskDialect, amdgpu::AMDGPUDialect, affine::AffineDialect, vector::VectorDialect,
        ROCDL::ROCDLDialect, LLVM::LLVMDialect, arith::ArithDialect,
        memref::MemRefDialect, scf::SCFDialect, gpu::GPUDialect>();
  }
  void runOnOperation() final {
    SmallVector<frisk::AsyncCopyGmToShmOp> copies;
    SmallVector<frisk::WaitBufferReadyOp> waits;
    getOperation().walk([&](frisk::AsyncCopyGmToShmOp op) { copies.push_back(op); });
    getOperation().walk([&](frisk::WaitBufferReadyOp op) { waits.push_back(op); });
    // Validate all launches before changing IR. These ops are block collective
    // and must execute in block-uniform control flow (the caller's contract).
    struct StagingPool { Value buffer; frisk::WaitBufferReadyOp lastWait; };
    struct Repack { Value staging, destination; AffineMap map; int64_t threads; };
    SmallVector<StagingPool> pools;
    DenseMap<Operation *, Repack> repacks;
    unsigned directCount = 0, stagedCount = 0, fallbackCount = 0;
    int64_t stagingBytes = 0;
    for (auto op : copies) {
      auto function = op->getParentOfType<func::FuncOp>();
      auto threads = function ? function->getAttrOfType<IntegerAttr>("thread_num")
                              : IntegerAttr{};
      if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 ||
          (useAsyncCopy && threads.getInt() % 64)) {
        op.emitOpError("requires thread_num in [1,1024], a multiple of 64 in gfx936 async mode");
        return signalPassFailure();
      }
      if (op.getSrc().getDefiningOp<frisk::BufferViewOp>() ||
          op.getDst().getDefiningOp<frisk::BufferViewOp>()) {
        op.emitOpError("run finalize-thread-tiling to materialize tile views first");
        return signalPassFailure();
      }
    }
    for (auto op : copies) {
      OpBuilder b(op);
      auto loc = op.getLoc();
      auto srcType = cast<MemRefType>(op.getSrc().getType());
      auto dstType = cast<MemRefType>(op.getDst().getType());
      int64_t threads = op->getParentOfType<func::FuncOp>()
                            ->getAttrOfType<IntegerAttr>("thread_num").getInt();
      auto requestedBytes = op->getAttrOfType<IntegerAttr>("frisk.dma_bytes");
      bool enabled = useAsyncCopy && !op->hasAttr("frisk.force_sync");
      unsigned dmaBytes = enabled ? selectDMA(op, requestedBytes ? requestedBytes.getInt() : 0) : 0;
      bool dma = dmaBytes != 0;
      if (auto copying = op->getAttrOfType<IntegerAttr>("frisk.copy_threads")) {
        if (copying.getInt() <= 0 || copying.getInt() > threads || copying.getInt() % 64) {
          op.emitError("copy_threads must select complete waves within the block");
          return signalPassFailure();
        }
        threads = copying.getInt();
      }
      auto sourceMapAttr = op->getAttrOfType<AffineMapAttr>("frisk.source_map");
      AffineMap sourceMap = sourceMapAttr ? sourceMapAttr.getValue() : AffineMap{};
      Value destination = op.getDst();
      frisk::WaitBufferReadyOp ready;
      if (!dma && useAsyncCopy && !op->hasAttr("frisk.force_sync") && sourceMap) {
        for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
          if (auto wait = dyn_cast<frisk::WaitBufferReadyOp>(next);
              wait && wait.getBuffer() == op.getDst()) { ready = wait; break; }
          if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(next);
              copy && copy.getDst() == op.getDst()) break;
        }
        auto storageMap = sourceStorageOrder(op);
        unsigned stagedBytes = ready && storageMap
            ? selectDMA(op, requestedBytes ? requestedBytes.getInt() : 0, storageMap) : 0;
        if (stagedBytes) {
          auto stageType = MemRefType::get({srcType.getNumElements()}, srcType.getElementType(),
                                         AffineMap{}, dstType.getMemorySpace());
          Value staging;
          for (auto &pool : pools)
            if (pool.buffer.getType() == stageType && pool.lastWait->getBlock() == op->getBlock() &&
                pool.lastWait->isBeforeInBlock(op)) {
              staging = pool.buffer;
              pool.lastWait = ready;
              break;
            }
          if (!staging) {
            OpBuilder::InsertionGuard guard(b);
            // This pass runs after thread legalization, so a new memref.alloc
            // would become device malloc. Materialize static LDS explicitly.
            unsigned suffix = pools.size();
            std::string name;
            do { name = "async_lds_staging_" + std::to_string(suffix++); }
            while (getOperation().lookupSymbol(name));
            b.setInsertionPointToStart(getOperation().getBody());
            b.create<memref::GlobalOp>(loc, name, b.getStringAttr("private"), stageType,
                                       b.getUnitAttr(), false, b.getI64IntegerAttr(16));
            b.setInsertionPointToStart(&op->getParentOfType<func::FuncOp>().front());
            staging = b.create<memref::GetGlobalOp>(loc, stageType, name);
            pools.push_back({staging, ready});
            stagingBytes += srcType.getNumElements() * srcType.getElementTypeBitWidth() / 8;
          }
          // Compose logical source coordinates with their staging linear index.
          SmallVector<int64_t> strides;
          int64_t offset;
          (void)srcType.getStridesAndOffset(strides, offset);
          SmallVector<unsigned> order;
          for (unsigned dim = 0; dim < srcType.getRank(); ++dim) order.push_back(dim);
          llvm::stable_sort(order, [&](unsigned a, unsigned c) { return strides[a] < strides[c]; });
          AffineExpr flat = b.getAffineConstantExpr(0);
          int64_t scale = 1;
          for (unsigned dim : order) {
            flat = flat + sourceMap.getResult(dim) * scale;
            scale *= srcType.getDimSize(dim);
          }
          repacks[ready] = {staging, op.getDst(), AffineMap::get(1, 0, flat), threads};
          destination = staging;
          sourceMap = storageMap;
          dstType = stageType;
          dma = true;
          dmaBytes = stagedBytes;
          ++stagedCount;
        }
      }
      // A matrix-read packet may run across the global storage order. Keep
      // the requested width when staging can satisfy it, and reject only
      // after both direct and staged DMA have been considered.
      if (requestedBytes && !dma) {
        op.emitError("requested full-width DMA is not legal for this copy");
        return signalPassFailure();
      }
      if (dma && destination == op.getDst()) ++directCount;
      if (!dma) ++fallbackCount;
      int64_t width = dma ? dmaBytes * 8 / srcType.getElementTypeBitWidth() : 1;
      int64_t packets = srcType.getNumElements() / width;
      Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      // Disable complete waves, preserving hardware lane IDs within every DMA
      // packet. Completion/publication below remains block collective.
      scf::IfOp participants;
      if (op->hasAttr("frisk.copy_threads")) {
        Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
            tid, indexConstant(b, loc, threads));
        participants = b.create<scf::IfOp>(loc, active, false);
        b.setInsertionPointToStart(participants.thenBlock());
      }
      auto emitPacket = [&](Value linear) {
        Value flat = b.create<arith::MulIOp>(loc, linear,
                                            indexConstant(b, loc, width));
        SmallVector<Value> srcIndices;
        if (sourceMap) srcIndices = mappedCoordinates(b, loc, flat, sourceMap);
        else srcIndices = coordinates(b, loc, flat, srcType.getShape());
        if (dma) {
          Value wave = b.create<arith::DivUIOp>(loc, linear,
                                               indexConstant(b, loc, 64));
          Value base = b.create<arith::MulIOp>(loc, wave,
                                               indexConstant(b, loc, 64 * width));
          auto dstIndices = coordinates(b, loc, base, dstType.getShape());
          auto gather = b.create<amdgpu::GatherToLDSOp>(loc, op.getSrc(), srcIndices,
              destination, dstIndices, b.getI32Type(), true);
          gather->setAttr("frisk.gfx936_dma", b.getUnitAttr());
          gather->setAttr("frisk.dma_bytes", b.getI64IntegerAttr(dmaBytes));
        } else {
          Value value = b.create<memref::LoadOp>(loc, op.getSrc(), srcIndices);
          b.create<memref::StoreOp>(loc, value, op.getDst(),
                                    coordinates(b, loc, flat, dstType.getShape()));
        }
      };
      // A full, small tile has the same packet count in every lane. Emit
      // constant rounds so LLVM need not keep a divergent 64-bit induction
      // variable, loop bound and execution mask alive alongside MMA values.
      // Keep the bounded loop for large tiles and partial waves.
      if (dma && packets % threads == 0 && packets / threads <= 16) {
        for (int64_t round = 0; round < packets; round += threads) {
          Value linear = b.create<arith::AddIOp>(loc, tid, indexConstant(b, loc, round));
          emitPacket(linear);
        }
      } else {
        Value end = indexConstant(b, loc, packets);
        Value step = indexConstant(b, loc, threads);
        auto loop = b.create<scf::ForOp>(loc, tid, end, step);
        loop->setAttr("frisk.copy_implementation", b.getStringAttr(dma ? "gfx936-dma" : "synchronous"));
        b.setInsertionPointToStart(loop.getBody());
        emitPacket(loop.getInductionVar());
        b.setInsertionPointAfter(loop);
      }
      if (participants) b.setInsertionPointAfter(participants);
      if (dma) b.create<ROCDL::AsyncmarkOp>(loc);
      op.erase();
    }
    for (auto op : waits) {
      OpBuilder b(op);
      // Keep every named completion and its publication barrier while target
      // numerics remain unresolved. The CPU DMA model alone does not establish
      // target synchronization safety, nor explain a target numerical failure.
      // Count is groups, not bytes or per-thread packets. Zero is deliberately
      // conservative across loops, branches, fallback copies and buffer slots.
      if (useAsyncCopy) b.create<ROCDL::WaitAsyncmarkOp>(op.getLoc(), 0);
      b.create<gpu::BarrierOp>(op.getLoc());
      if (auto found = repacks.find(op); found != repacks.end()) {
        auto &repack = found->second;
        auto type = cast<MemRefType>(repack.destination.getType());
        auto loc = op.getLoc();
        int64_t width = 32 / type.getElementTypeBitWidth();
        auto packet = VectorType::get({width}, type.getElementType());
        Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value end = indexConstant(b, loc, type.getNumElements() / width);
        Value step = indexConstant(b, loc, repack.threads);
        auto loop = b.create<scf::ForOp>(loc, tid, end, step);
        loop->setAttr("frisk.copy_implementation", b.getStringAttr("lds-repack"));
        b.setInsertionPointToStart(loop.getBody());
        Value flat = b.create<arith::MulIOp>(loc, loop.getInductionVar(), indexConstant(b, loc, width));
        Value values = b.create<arith::ConstantOp>(loc, packet, b.getZeroAttr(packet));
        for (int64_t i = 0; i < width; ++i) {
          Value at = b.create<arith::AddIOp>(loc, flat, indexConstant(b, loc, i));
          auto indices = mappedCoordinates(b, loc, at, repack.map);
          Value value = b.create<memref::LoadOp>(loc, repack.staging, indices);
          values = b.create<vector::InsertOp>(loc, value, values,
                                              ArrayRef<OpFoldResult>{b.getIndexAttr(i)});
        }
        b.create<vector::StoreOp>(loc, values, repack.destination, ValueRange{flat});
        b.setInsertionPointAfter(loop);
        // Publish the packed destination and release all staging readers before
        // another issue may reuse the same staging allocation.
        b.create<gpu::BarrierOp>(loc);
      }
      op.erase();
    }
    if (!copies.empty())
      llvm::errs() << "LowerFriskAsyncCopy: " << directCount << " direct, " << stagedCount
                   << " staged, " << fallbackCount << " fallback; staging bytes " << stagingBytes << "\n";
    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(64);
    LLVMTypeConverter converter(&getContext(), options);
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherToGfx936>(converter);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isa<amdgpu::GatherToLDSOp>(op) || !op->hasAttr("frisk.gfx936_dma");
    });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createLowerFriskAsyncCopyPass(bool enabled) {
  return std::make_unique<LowerFriskAsyncCopyPass>(enabled);
}
