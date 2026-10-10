#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

namespace mlir::frisk {
#define GEN_PASS_DECL_IRDEEPOPTIMIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"
namespace {
#define GEN_PASS_DEF_IRDEEPOPTIMIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

// Only unroll indices entirely determined by constants and small static loops.
// Follow index dependencies, not the vector or its lane-dependent load address.
static bool collectIndexLoops(Value value,
                              llvm::SmallPtrSetImpl<Operation *> &loops,
                              llvm::SmallDenseSet<Value, 16> &visiting) {
  if (matchPattern(value, m_Constant()))
    return true;

  // An iter_arg recurrence may refer back to itself. Its initial value and
  // every other input to the recurrence must still be statically determined.
  if (!visiting.insert(value).second)
    return true;
  auto guard = llvm::make_scope_exit([&] { visiting.erase(value); });
  auto collectLoop = [&](affine::AffineForOp loop) {
    auto count = affine::getConstantTripCount(loop);
    if (!count || *count == 0 || *count > 64)
      return false;
    // A constant trip count alone does not imply constant IVs: the lower
    // bound could be a runtime offset. Outer static IVs are allowed here.
    for (Value bound : loop.getLowerBoundOperands())
      if (!collectIndexLoops(bound, loops, visiting))
        return false;
    for (Value bound : loop.getUpperBoundOperands())
      if (!collectIndexLoops(bound, loops, visiting))
        return false;
    loops.insert(loop);
    return true;
  };
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<affine::AffineForOp>(
        arg.getOwner()->getParentOp());
    if (!loop || !collectLoop(loop))
      return false;
    if (arg == loop.getInductionVar())
      return true;
    unsigned position = arg.getArgNumber() - 1;
    auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
    return collectIndexLoops(loop.getInits()[position], loops, visiting) &&
           collectIndexLoops(yield.getOperand(position), loops, visiting);
  }
  auto *op = value.getDefiningOp();
  if (!op || !isa<affine::AffineApplyOp, arith::AddIOp, arith::SubIOp,
                  arith::MulIOp, arith::DivSIOp, arith::DivUIOp,
                  arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
                  arith::IndexCastUIOp>(op))
    return false;
  for (Value operand : op->getOperands())
    if (!collectIndexLoops(operand, loops, visiting))
      return false;
  return true;
}

static void reorderUnrolledOperations(func::FuncOp function);
static void coalesceUnrolledSharedReads(func::FuncOp function);
static void coalesceUnrolledSharedScalars(func::FuncOp function);

class VectorOpLoopUnrollPass
    : public impl::IRDeepOptimizeBase<VectorOpLoopUnrollPass> {
  void runOnOperation() override {
    auto kernel = getOperation();
    RewritePatternSet patterns(&getContext());
    for (auto op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    // Bound cumulative code growth, not just individual loop trip counts.
    constexpr uint64_t maxExtraOperations = 65536;
    uint64_t extraOperations = 0;
    while (true) {
      // Full unrolling substitutes IVs and threads iter_args through clones;
      // folding then turns affine.apply/arithmetic and vector positions static.
      if (failed(applyPatternsGreedily(kernel, frozen))) {
        signalPassFailure();
        return;
      }
      llvm::SmallPtrSet<Operation *, 16> candidates;
      auto collect = [&](auto op) {
        for (OpFoldResult position : op.getMixedPosition()) {
          auto index = dyn_cast<Value>(position);
          if (!index)
            continue;
          llvm::SmallPtrSet<Operation *, 8> dependencies;
          llvm::SmallDenseSet<Value, 16> visiting;
          if (collectIndexLoops(index, dependencies, visiting))
            candidates.insert(dependencies.begin(), dependencies.end());
        }
      };
      kernel.walk([&](vector::InsertOp op) { collect(op); });
      kernel.walk([&](vector::ExtractOp op) { collect(op); });
      affine::AffineForOp selected;
      uint64_t selectedCost = 0;
      kernel.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) {
        if (selected || !candidates.contains(loop))
          return;
        uint64_t bodySize = 0;
        loop.getBody()->walk([&](Operation *) { ++bodySize; });
        auto count = affine::getConstantTripCount(loop);
        uint64_t cost = bodySize * (*count - 1);
        if (cost > maxExtraOperations - extraOperations)
          return;
        selected = loop;
        selectedCost = cost;
      });
      if (!selected)
        break;
      if (failed(affine::loopUnrollFull(selected))) {
        selected.emitError("failed to staticize register-vector indices");
        signalPassFailure();
        return;
      }
      extraOperations += selectedCost;
      // Recollect after every rewrite: unrolling invalidates nested op handles.
    }
    coalesceUnrolledSharedScalars(kernel);
    // Scalar packets can grow through several native widths, up to 128 bits.
    for (unsigned i = 0; i < 3; ++i)
      coalesceUnrolledSharedReads(kernel);
    if (reorderAfterUnroll)
      reorderUnrolledOperations(kernel);
  }
};

// Describe the entire memref, not an individual lane's access. Disjoint SSA
// indices in one thread need not be disjoint between different GPU threads.
struct BarrierMemoryRange {
  Value base;
  std::optional<int64_t> offset;
  std::optional<int64_t> bytes;
};

static std::optional<int64_t> getBarrierBufferBytes(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat())
    return std::nullopt;
  unsigned bits = type.getElementType().getIntOrFloatBitWidth();
  // For unusual widths (e.g. i24 or f80), LLVM's allocation stride may
  // include padding. Only use the native scalar widths of this GPU lowering.
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  int64_t bytes = bits / 8;
  for (int64_t size : type.getShape()) {
    if (size && bytes > std::numeric_limits<int64_t>::max() / size)
      return std::nullopt;
    bytes *= size;
  }
  return bytes;
}

static BarrierMemoryRange getBarrierMemoryRange(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return getBarrierMemoryRange(cast.getSource());
  if (auto view = value.getDefiningOp<memref::ViewOp>()) {
    auto source = getBarrierMemoryRange(view.getSource());
    auto shift = getConstantIntValue(view.getByteShift());
    auto bytes = getBarrierBufferBytes(view.getType());
    if (source.offset && shift && *shift >= 0 && bytes &&
        *source.offset <= std::numeric_limits<int64_t>::max() - *shift &&
        *source.offset + *shift <=
            std::numeric_limits<int64_t>::max() - *bytes)
      return {source.base, *source.offset + *shift, bytes};
    return {source.base, std::nullopt, std::nullopt};
  }
  if (auto *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
      auto source = getBarrierMemoryRange(view.getViewSource());
      // Subviews/reinterpret_casts may have dynamic offsets or strides. Keep
      // the underlying object identity, but do not guess their byte range.
      return {source.base, std::nullopt, std::nullopt};
    }
  }
  auto type = dyn_cast<MemRefType>(value.getType());
  // An arbitrary SSA memref (e.g. a thread-dependent select of subviews) can
  // have a different base pointer in each lane. Relative byte intervals alone
  // cannot prove those accesses disjoint across threads.
  bool fixedBase = value.getDefiningOp<memref::AllocOp>() ||
                   value.getDefiningOp<memref::AllocaOp>() ||
                   value.getDefiningOp<memref::GetGlobalOp>();
  return {value, fixedBase && type && type.getLayout().isIdentity()
                     ? std::optional<int64_t>(0) : std::nullopt,
          fixedBase && type ? getBarrierBufferBytes(type) : std::nullopt};
}

static bool barrierRangesAreDisjoint(Value lhs, Value rhs) {
  auto aType = cast<BaseMemRefType>(lhs.getType());
  auto bType = cast<BaseMemRefType>(rhs.getType());
  auto aSpace = dyn_cast_or_null<IntegerAttr>(aType.getMemorySpace());
  auto bSpace = dyn_cast_or_null<IntegerAttr>(bType.getMemorySpace());
  // Generic/default memory may alias either space. Only distinguish the
  // explicit global and workgroup address spaces used by this lowering.
  if (aSpace && bSpace &&
      ((aSpace.getInt() == 1 && bSpace.getInt() == 3) ||
       (aSpace.getInt() == 3 && bSpace.getInt() == 1)))
    return true;
  auto a = getBarrierMemoryRange(lhs), b = getBarrierMemoryRange(rhs);
  auto aGlobal = a.base.getDefiningOp<memref::GetGlobalOp>();
  auto bGlobal = b.base.getDefiningOp<memref::GetGlobalOp>();
  bool same = a.base == b.base ||
              (aGlobal && bGlobal && aGlobal.getNameAttr() == bGlobal.getNameAttr());
  if (same)
    return a.offset && a.bytes && b.offset && b.bytes &&
           (*a.offset + *a.bytes <= *b.offset ||
            *b.offset + *b.bytes <= *a.offset);
  auto isAllocation = [](Value base) {
    return base.getDefiningOp<memref::AllocOp>() ||
           base.getDefiningOp<memref::AllocaOp>() ||
           base.getDefiningOp<memref::GetGlobalOp>();
  };
  // Different function arguments, selects and region arguments can alias.
  return isAllocation(a.base) && isAllocation(b.base);
}

// Schedule only within straight-line windows. This is an IR issue-order
// heuristic, not a GPU cycle model: instruction selection and the backend
// scheduler still decide the machine schedule. Never rewrite arithmetic or MMA
// operands, speculate across control flow, or remove hardware hazard padding.
enum class IssueKind { Boundary, Support, Memory, Compute };

struct IssueNode {
  Operation *op;
  IssueKind kind;
  Value buffer;
  bool write = false;
  bool mma = false;
  SmallVector<unsigned, 4> successors, predecessors;
  unsigned pending = 0;
  unsigned readyAt = 0;
  unsigned feeds = 0;
};

// Match the COMPLETE register-only assembly emitted by ThreadLevelIRLegalize.
// A mnemonic substring or a user marker is insufficient: an asm may also have
// memory operations, clobbers, waits or other effects. Keep all MMAs in their
// original relative order, including independent accumulator chains.
static bool isSchedulableRegisterMMA(Operation *op) {
  auto assembly = dyn_cast<LLVM::InlineAsmOp>(op);
  if (!assembly || assembly.getConstraints() != "=v,v,v,0" ||
      !assembly.getHasSideEffects() || assembly.getIsAlignStack() ||
      assembly.getOperandAttrs() || op->getNumOperands() != 3 ||
      op->getNumResults() != 1)
    return false;
  auto half4 = VectorType::get({4}, Float16Type::get(op->getContext()));
  auto float4 = VectorType::get({4}, Float32Type::get(op->getContext()));
  if (op->getOperand(0).getType() != half4 ||
      op->getOperand(1).getType() != half4 ||
      op->getOperand(2).getType() != float4 ||
      op->getResult(0).getType() != float4)
    return false;
  return isPaddedRegisterMMA(assembly.getAsmString(), assembly.getConstraints());
}

static IssueNode classifyIssue(Operation *op) {
  IssueNode node{op, IssueKind::Boundary, {}};
  if (op->getNumRegions() || op->getNumSuccessors() ||
      op->hasTrait<OpTrait::IsTerminator>())
    return node;
  node.mma = isSchedulableRegisterMMA(op);
  if (node.mma) {
    node.kind = IssueKind::Compute;
    return node;
  }
  // Only ordinary synchronous accesses: no masked transfers, async copies,
  // atomics, volatile LLVM loads, calls or opaque assembly.
  if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
          affine::AffineStoreOp, affine::AffineVectorLoadOp,
          affine::AffineVectorStoreOp, vector::LoadOp, vector::StoreOp>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    cast<MemoryEffectOpInterface>(op).getEffects(effects);
    if (effects.size() == 1 && effects[0].getValue() &&
        isa<BaseMemRefType>(effects[0].getValue().getType()) &&
        isa<MemoryEffects::Read, MemoryEffects::Write>(effects[0].getEffect())) {
      node.kind = IssueKind::Memory;
      node.buffer = effects[0].getValue();
      node.write = isa<MemoryEffects::Write>(effects[0].getEffect());
    }
    return node;
  }
  // Collective operations (e.g. shuffle) and non-speculatable operations
  // remain boundaries even if their memory-effect interface says "no effects".
  if (!isMemoryEffectFree(op) || !isSpeculatable(op) ||
      op->getName().getDialectNamespace() == "gpu" ||
      isa<LLVM::InlineAsmOp>(op))
    return node;
  node.kind = IssueKind::Support;
  StringRef dialect = op->getName().getDialectNamespace();
  bool dataArithmetic = dialect == "arith" &&
      !op->hasTrait<OpTrait::ConstantLike>() &&
      llvm::any_of(op->getResultTypes(), [](Type type) {
        return isa<FloatType, VectorType>(type);
      });
  if (dataArithmetic || dialect == "math" ||
      isa<vector::FMAOp, vector::ReductionOp, vector::ContractionOp>(op))
    node.kind = IssueKind::Compute;
  return node;
}

// After fragment loops have been unrolled, paired K packets are visible in
// one block. Merge only two proven adjacent reads, never overfetch. Moving the
// second read to the first is legal only within a window without writes,
// barriers, control flow or unknown effects. Register-only MMAs are allowed,
// but their order and hazard padding are untouched.
static void coalesceUnrolledSharedReads(func::FuncOp function) {
  function.walk([&](Block *block) {
    SmallVector<vector::LoadOp> pending;
    SmallVector<Operation *> operations;
    for (Operation &op : *block)
      operations.push_back(&op);
    for (Operation *op : operations) {
      auto node = classifyIssue(op);
      if (node.kind == IssueKind::Boundary || node.write) {
        pending.clear();
        continue;
      }
      auto load = dyn_cast<vector::LoadOp>(op);
      if (!load)
        continue;
      auto memref = load.getMemRefType();
      auto type = load.getVectorType();
      if (memref.getMemorySpaceAsInt() != 3 || memref.getRank() != 1 ||
          !memref.isLastDimUnitStride() || type.getRank() != 1 ||
          type.isScalable() || !type.getElementType().isIntOrFloat() ||
          !llvm::isPowerOf2_64(type.getNumElements() * type.getElementTypeBitWidth()) ||
          type.getNumElements() * type.getElementTypeBitWidth() > 64)
        continue;
      bool merged = false;
      for (auto [i, first] : llvm::enumerate(pending)) {
        if (first.getBase() != load.getBase() || first.getVectorType() != type)
          continue;
        OpBuilder b(first);
        SmallVector<Value> indices{load.getIndices()[0], first.getIndices()[0]};
        auto difference = AffineMap::get(2, 0, b.getAffineDimExpr(0) - b.getAffineDimExpr(1));
        affine::fullyComposeAffineMapAndOperands(&difference, &indices);
        affine::canonicalizeMapAndOperands(&difference, &indices);
        difference = simplifyAffineMap(difference);
        if (auto constant = dyn_cast<AffineConstantExpr>(difference.getResult(0))) {
          if (constant.getValue() != type.getNumElements())
            continue;
        } else {
          // Modulo lane coordinates have intrinsic bounds, e.g.
          // ((lane % 64) / 16 + 4) / 8 == 0. Algebraic simplification alone
          // misses these. Prove the delta for ALL integer operands, without
          // assuming a particular thread id or adding alignment assertions.
          FlatLinearConstraints bounds(difference.getNumDims(), difference.getNumSymbols());
          if (failed(bounds.composeMatchingMap(difference)))
            continue;
          auto below = bounds;
          below.addBound(presburger::BoundType::UB, 0, type.getNumElements() - 1);
          if (!below.isIntegerEmpty())
            continue;
          bounds.addBound(presburger::BoundType::LB, 0, type.getNumElements() + 1);
          if (!bounds.isIntegerEmpty())
            continue;
        }
        int64_t width = type.getNumElements();
        auto wideType = VectorType::get({2 * width}, type.getElementType());
        auto wide = b.create<vector::LoadOp>(first.getLoc(), wideType,
                                            first.getBase(), first.getIndices());
        auto low = b.create<vector::ExtractStridedSliceOp>(first.getLoc(), wide,
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{width}, ArrayRef<int64_t>{1});
        auto high = b.create<vector::ExtractStridedSliceOp>(load.getLoc(), wide,
            ArrayRef<int64_t>{width}, ArrayRef<int64_t>{width}, ArrayRef<int64_t>{1});
        first.replaceAllUsesWith(low.getResult());
        load.replaceAllUsesWith(high.getResult());
        first.erase();
        load.erase();
        pending.erase(pending.begin() + i);
        merged = true;
        break;
      }
      if (!merged) {
        if (pending.size() == 64)
          pending.erase(pending.begin());
        pending.push_back(load);
      }
    }
  });
}

// Prove a constant displacement for every value of the affine operands. A
// constant-folded sample only proposes a displacement; the two emptiness
// checks prove it, including the intrinsic bounds of mod/floordiv expressions.
static std::optional<int64_t> sharedDisplacement(Value from, Value to) {
  auto *ctx = from.getContext();
  SmallVector<Value> operands{to, from};
  auto map = AffineMap::get(2, 0,
      getAffineDimExpr(0, ctx) - getAffineDimExpr(1, ctx));
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  if (auto constant = dyn_cast<AffineConstantExpr>(map.getResult(0)))
    return constant.getValue();
  SmallVector<Attribute> zero(operands.size(), IntegerAttr::get(IndexType::get(ctx), 0));
  SmallVector<Attribute> folded;
  if (failed(map.constantFold(zero, folded)))
    return std::nullopt;
  int64_t delta = cast<IntegerAttr>(folded.front()).getInt();
  if (delta == std::numeric_limits<int64_t>::min() ||
      delta == std::numeric_limits<int64_t>::max())
    return std::nullopt;
  FlatLinearConstraints bounds(map.getNumDims(), map.getNumSymbols());
  if (failed(bounds.composeMatchingMap(map)))
    return std::nullopt;
  auto lower = bounds;
  lower.addBound(presburger::BoundType::UB, 0, delta - 1);
  bounds.addBound(presburger::BoundType::LB, 0, delta + 1);
  if (!lower.isIntegerEmpty() || !bounds.isIntegerEmpty())
    return std::nullopt;
  return delta;
}

static bool isScalarSharedBuffer(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && type.isLastDimUnitStride() &&
      type.getMemorySpaceAsInt() == 3 && type.getElementType().isIntOrFloat() &&
      llvm::isPowerOf2_64(type.getElementTypeBitWidth()) &&
      type.getElementTypeBitWidth() <= 64;
}

static void coalesceUnrolledSharedScalars(func::FuncOp function) {
  function.walk([&](Block *block) {
    SmallVector<Operation *> operations;
    for (Operation &op : *block) operations.push_back(&op);
    SmallVector<memref::LoadOp> reads;
    SmallVector<memref::StoreOp> writes;
    auto flushWrites = [&]() {
      if (writes.size() < 2) { writes.clear(); return; }
      // Sort only disjoint stores to the same buffer in an effect-free window.
      // Place the packed write at the final original store, after all values
      // are available. Duplicate/unknown addresses retain their original order.
      SmallVector<std::pair<int64_t, memref::StoreOp>> sorted;
      for (auto store : writes) {
        auto delta = sharedDisplacement(writes.front().getIndices()[0], store.getIndices()[0]);
        if (!delta) { writes.clear(); return; }
        sorted.push_back({*delta, store});
      }
      llvm::sort(sorted, [](auto a, auto b) { return a.first < b.first; });
      for (unsigned i = 1; i < sorted.size(); ++i)
        if (sorted[i-1].first == sorted[i].first) { writes.clear(); return; }
      OpBuilder b(writes.back());
      unsigned limit = 128 / cast<MemRefType>(writes.front().getMemRef().getType()).getElementTypeBitWidth();
      SmallVector<Operation *> erase;
      for (unsigned i = 0; i < sorted.size();) {
        unsigned width = 1;
        while (width * 2 <= limit && i + width * 2 <= sorted.size() &&
               uint64_t(sorted[i + width * 2 - 1].first) - uint64_t(sorted[i].first) == uint64_t(width * 2 - 1))
          width *= 2;
        if (width == 1) { ++i; continue; }
        auto first = sorted[i].second;
        auto type = VectorType::get({int64_t(width)}, first.getValue().getType());
        Value packet = b.create<arith::ConstantOp>(first.getLoc(), type, b.getZeroAttr(type));
        for (unsigned j = 0; j < width; ++j) {
          packet = b.create<vector::InsertOp>(first.getLoc(), sorted[i+j].second.getValue(), packet,
              ArrayRef<OpFoldResult>{b.getIndexAttr(j)});
          erase.push_back(sorted[i+j].second);
        }
        b.create<vector::StoreOp>(first.getLoc(), packet, first.getMemRef(), first.getIndices());
        i += width;
      }
      for (Operation *op : erase) op->erase();
      writes.clear();
    };
    for (Operation *op : operations) {
      auto node = classifyIssue(op);
      auto load = dyn_cast<memref::LoadOp>(op);
      auto store = dyn_cast<memref::StoreOp>(op);
      if (store && isScalarSharedBuffer(store.getMemRef())) {
        reads.clear();
        if (!writes.empty() && (writes.front().getMemRef() != store.getMemRef() || writes.size() == 64))
          flushWrites();
        writes.push_back(store);
        continue;
      }
      if (node.kind == IssueKind::Boundary || node.kind == IssueKind::Memory)
        flushWrites();
      if (node.kind == IssueKind::Boundary || node.write) {
        reads.clear();
        continue;
      }
      if (!load || !isScalarSharedBuffer(load.getMemRef())) continue;
      bool merged = false;
      for (auto [i, first] : llvm::enumerate(reads)) {
        if (first.getMemRef() != load.getMemRef()) continue;
        auto delta = sharedDisplacement(first.getIndices()[0], load.getIndices()[0]);
        if (!delta || (*delta != 1 && *delta != -1)) continue;
        OpBuilder b(first);
        Value index = first.getIndices()[0];
        if (*delta == -1)
          index = b.create<affine::AffineApplyOp>(first.getLoc(),
              AffineMap::get(1, 0, b.getAffineDimExpr(0) - 1), index);
        auto type = VectorType::get({2}, first.getType());
        Value packet = b.create<vector::LoadOp>(first.getLoc(), type, first.getMemRef(), ValueRange{index});
        Value a = b.create<vector::ExtractOp>(first.getLoc(), packet,
            ArrayRef<OpFoldResult>{b.getIndexAttr(*delta == 1 ? 0 : 1)});
        Value c = b.create<vector::ExtractOp>(load.getLoc(), packet,
            ArrayRef<OpFoldResult>{b.getIndexAttr(*delta == 1 ? 1 : 0)});
        first.replaceAllUsesWith(a);
        load.replaceAllUsesWith(c);
        first.erase(); load.erase();
        reads.erase(reads.begin() + i);
        merged = true;
        break;
      }
      if (!merged) {
        if (reads.size() == 64) reads.erase(reads.begin());
        reads.push_back(load);
      }
    }
    flushWrites();
  });
}

static unsigned issueBit(IssueKind kind) {
  return kind == IssueKind::Memory ? 1 : kind == IssueKind::Compute ? 2 : 0;
}

// A rough live-data budget, not physical VGPR accounting. Ignore index/address
// scalars and track newly defined register data through their actual SSA uses.
static uint64_t registerDataBits(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.isScalable())
      return 4096;
    return std::min<uint64_t>(4096, vector.getNumElements() *
                                  registerDataBits(vector.getElementType()));
  }
  if (auto number = dyn_cast<FloatType>(type))
    return number.getWidth();
  if (auto number = dyn_cast<IntegerType>(type))
    return number.getWidth();
  return 0;
}

static void scheduleIssueWindow(SmallVectorImpl<IssueNode> &nodes) {
  bool hasMemory = false, hasCompute = false;
  for (const auto &node : nodes) {
    hasMemory |= node.kind == IssueKind::Memory;
    hasCompute |= node.kind == IssueKind::Compute;
  }
  if (!hasMemory || !hasCompute)
    return;
  llvm::DenseMap<Operation *, unsigned> indices;
  for (auto [index, node] : llvm::enumerate(nodes))
    indices[node.op] = index;
  auto edge = [&](unsigned from, unsigned to) {
    nodes[from].successors.push_back(to);
    nodes[to].predecessors.push_back(from);
    ++nodes[to].pending;
  };
  SmallVector<unsigned> accesses;
  std::optional<unsigned> previousMMA;
  uint64_t pairs = 0;
  for (auto [index, node] : llvm::enumerate(nodes)) {
    llvm::SmallDenseSet<unsigned, 8> definitions;
    for (Value operand : node.op->getOperands()) {
      auto found = indices.find(operand.getDefiningOp());
      if (found != indices.end())
        definitions.insert(found->second);
    }
    for (unsigned from : definitions)
      edge(from, index);
    if (node.mma) {
      if (previousMMA)
        edge(*previousMMA, index);
      previousMMA = index;
    }
    if (!node.buffer)
      continue;
    for (unsigned previous : accesses) {
      if (++pairs > 1048576)
        return; // No IR has moved; keep the original order beyond the budget.
      auto &other = nodes[previous];
      // Preserve RAW, WAR and WAW unless whole storage ranges are disjoint.
      // Different indices in shared memory are NOT a cross-thread alias proof.
      if ((node.write || other.write) &&
          !barrierRangesAreDisjoint(node.buffer, other.buffer))
        edge(previous, index);
    }
    accesses.push_back(index);
  }
  // Address arithmetic and vector packing should follow the issue class they
  // unlock; do not count them as useful compute to fake load/compute alternation.
  for (unsigned i = nodes.size(); i-- > 0;) {
    nodes[i].feeds = issueBit(nodes[i].kind);
    if (nodes[i].kind == IssueKind::Support)
      for (unsigned next : nodes[i].successors)
        nodes[i].feeds |= nodes[next].feeds;
  }
  // Keep small read packets together. Only support operations may originally
  // separate members; never merge across a compute, store or different buffer.
  // Vector loads are already packets. Four scalar elements match the MMA
  // register fragments and preserve the SLP opportunities lost by scalar M/C
  // alternation. No load is widened and no extra address is accessed here.
  SmallVector<SmallVector<unsigned, 4>> batches;
  SmallVector<unsigned> batchOf(nodes.size(), nodes.size());
  bool extend = false;
  Value buffer;
  Type element;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    auto &node = nodes[i];
    if (node.kind == IssueKind::Support)
      continue;
    if (!node.buffer || node.write) {
      extend = false;
      continue;
    }
    Type type = node.op->getResult(0).getType();
    bool scalar = !isa<VectorType>(type);
    if (!extend || node.buffer != buffer || type != element ||
        batches.back().size() == 4 || !scalar)
      batches.emplace_back();
    batchOf[i] = batches.size() - 1;
    batches.back().push_back(i);
    extend = scalar;
    buffer = node.buffer;
    element = type;
  }
  // Dependencies needed to complete an active packet. Such support/address
  // operations may move with it; no unrelated compute can split its loads.
  SmallVector<bool> needed(nodes.size(), false);
  std::optional<unsigned> activeBatch;
  auto activate = [&](unsigned batch) {
    activeBatch = batch;
    std::fill(needed.begin(), needed.end(), false);
    SmallVector<unsigned> todo(batches[batch].begin(), batches[batch].end());
    while (!todo.empty()) {
      unsigned index = todo.pop_back_val();
      if (needed[index])
        continue;
      needed[index] = true;
      todo.append(nodes[index].predecessors);
    }
  };
  llvm::DenseMap<Value, unsigned> remainingUses;
  for (const auto &node : nodes)
    for (Value result : node.op->getResults())
      remainingUses[result] = std::distance(result.use_begin(), result.use_end());
  SmallVector<unsigned> order;
  SmallVector<bool> issued(nodes.size(), false);
  uint64_t liveBits = 0;
  constexpr uint64_t lookaheadBits = 512;
  unsigned clock = 0, preferred = 1;
  while (order.size() != nodes.size()) {
    unsigned best = nodes.size(), bestRank = 5;
    unsigned nextReady = std::numeric_limits<unsigned>::max();
    unsigned desired = liveBits >= lookaheadBits ? 2 : preferred;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      auto &node = nodes[i];
      if (issued[i] || node.pending || (activeBatch && !needed[i]))
        continue;
      nextReady = std::min(nextReady, node.readyAt);
      if (node.readyAt > clock)
        continue;
      unsigned bit = issueBit(node.kind);
      unsigned rank = bit == desired ? 0 :
          node.kind == IssueKind::Support && (node.feeds & desired) ? 1 :
          bit ? 2 : 3;
      if (activeBatch)
        rank = batchOf[i] == *activeBatch ? 0 :
               node.kind == IssueKind::Support ? 1 : 2;
      if (rank < bestRank) {
        best = i;
        bestRank = rank;
      }
    }
    if (best == nodes.size()) {
      if (nextReady == std::numeric_limits<unsigned>::max())
        return; // Defensive: do not apply a partial/cyclic schedule.
      clock = std::max(clock, nextReady);
      continue;
    }
    auto &node = nodes[best];
    order.push_back(best);
    issued[best] = true;
    for (Value operand : node.op->getOperands()) {
      auto found = remainingUses.find(operand);
      if (found != remainingUses.end() && --found->second == 0)
        liveBits -= registerDataBits(operand.getType());
    }
    for (Value result : node.op->getResults())
      if (remainingUses[result])
        liveBits += registerDataBits(result.getType());
    if (!activeBatch && batchOf[best] != nodes.size())
      activate(batchOf[best]);
    if (activeBatch) {
      if (llvm::all_of(batches[*activeBatch], [&](unsigned i) { return issued[i]; })) {
        activeBatch.reset();
        preferred = 2;
      }
    } else if (unsigned bit = issueBit(node.kind)) {
      preferred = bit == 1 ? 2 : 1;
    }
    // Abstract readiness distances insert no waits. Within a packet we allow
    // address dependencies but never fill an artificial wait with unrelated ops.
    unsigned latency = node.buffer && !node.write ? 8 : node.mma ? 8 :
                       node.kind == IssueKind::Compute ? 2 : 1;
    for (unsigned next : node.successors) {
      --nodes[next].pending;
      nodes[next].readyAt = std::max(nodes[next].readyAt, clock + latency);
    }
    ++clock;
  }
  Operation *end = nodes.back().op->getNextNode();
  assert(end && "a scheduling window ends before a boundary or terminator");
  for (unsigned index : order)
    nodes[index].op->moveBefore(end);
}

static void reorderUnrolledOperations(func::FuncOp function) {
  // Hard window limits bound both graph construction and list scheduling.
  // Partitioning a long basic block preserves all inter-window dependencies.
  constexpr unsigned maxWindow = 4096;
  function.walk([&](Block *block) {
    SmallVector<IssueNode> window;
    auto flush = [&] {
      scheduleIssueWindow(window);
      window.clear();
    };
    // Snapshot before moving operations, so iterator order cannot change.
    SmallVector<Operation *> operations;
    for (Operation &op : *block)
      operations.push_back(&op);
    for (Operation *op : operations) {
      IssueNode node = classifyIssue(op);
      if (node.kind == IssueKind::Boundary) {
        flush();
        continue;
      }
      window.push_back(std::move(node));
      if (window.size() == maxWindow)
        flush();
    }
    flush();
  });
}

struct BarrierAccess {
  Value buffer;
  bool write;
};

struct BarrierNode {
  Operation *barrier = nullptr;
  bool unknown = false;
  SmallVector<BarrierAccess, 2> accesses;
  SmallVector<unsigned, 2> predecessors, successors;
};

// A small control-flow graph keeps both loop backedges and zero-trip paths.
// Only uniform for-loops expose their barriers as collective phase boundaries.
// Conditional regions are summarized without using their barriers as cutoffs.
class BarrierAnalysis {
public:
  explicit BarrierAnalysis(func::FuncOp function) : function(function) {
    kernel = function->hasAttr("thread_num") || function->hasAttr("gpu.kernel");
  }

  void run() {
    if (!llvm::hasSingleElement(function.getBody()))
      return; // Run before SCF/affine control flow is lowered to CFG blocks.
    unsigned entry = addNode();
    unsigned last = appendBlock(function.front(), entry);
    unsigned exit = addNode();
    connect(last, exit);
    // A callable helper may synchronize accesses in its caller. Kernel entry
    // and return, in contrast, cannot have concurrent accesses by this group.
    nodes[entry].unknown = nodes[exit].unknown = !kernel;
    for (unsigned index : barriers) {
      SmallVector<BarrierAccess> before, after;
      if (!collect(index, false, before) || !collect(index, true, after))
        continue;
      bool conflict = llvm::any_of(before, [&](const BarrierAccess &a) {
        return llvm::any_of(after, [&](const BarrierAccess &b) {
          return (a.write || b.write) &&
                 !barrierRangesAreDisjoint(a.buffer, b.buffer);
        });
      });
      if (!conflict) {
        nodes[index].barrier->erase();
        // Update immediately: simultaneously removing two individually
        // redundant barriers can remove the only synchronization between uses.
        nodes[index].barrier = nullptr;
      }
    }
  }

private:
  unsigned addNode() {
    nodes.emplace_back();
    return nodes.size() - 1;
  }
  void connect(unsigned from, unsigned to) {
    nodes[from].successors.push_back(to);
    nodes[to].predecessors.push_back(from);
  }

  bool isUniform(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &function.front())
        return kernel && value.getType().isIntOrIndexOrFloat();
      return arg.getArgNumber() == 0 &&
             collectiveLoops.contains(arg.getOwner()->getParentOp());
    }
    Operation *op = value.getDefiningOp();
    if (!op)
      return false;
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op))
      return true;
    return !op->getNumRegions() && isMemoryEffectFree(op) &&
           (isa<affine::AffineApplyOp>(op) ||
            op->getName().getDialectNamespace() == "arith") &&
           llvm::all_of(op->getOperands(), [&](Value v) { return isUniform(v); });
  }

  Block *getCollectiveBody(Operation *op) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (isUniform(loop.getLowerBound()) && isUniform(loop.getUpperBound()) &&
          isUniform(loop.getStep()))
        return loop.getBody();
    }
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      if (llvm::all_of(loop.getLowerBoundOperands(),
                       [&](Value v) { return isUniform(v); }) &&
          llvm::all_of(loop.getUpperBoundOperands(),
                       [&](Value v) { return isUniform(v); }))
        return loop.getBody();
    }
    return nullptr;
  }

  unsigned appendBlock(Block &block, unsigned previous) {
    for (Operation &op : block) {
      unsigned index = addNode();
      connect(previous, index);
      if (Block *body = getCollectiveBody(&op)) {
        collectiveLoops.insert(&op);
        unsigned exit = addNode();
        connect(index, exit); // May execute zero iterations.
        unsigned last = appendBlock(*body, index);
        connect(last, index); // Includes the final-iteration exit path.
        previous = exit;
        continue;
      }
      if (isa<gpu::BarrierOp>(&op)) {
        nodes[index].barrier = &op;
        barriers.push_back(index);
      } else {
        summarize(&op, nodes[index]);
      }
      previous = index;
    }
    return previous;
  }

  void summarize(Operation *op, BarrierNode &node) {
    if (isa<gpu::BarrierOp>(op)) {
      node.unknown = true; // A conditional barrier cannot protect every path.
      return;
    }
    if (isa<scf::ForOp, scf::IfOp, affine::AffineForOp, affine::AffineIfOp>(op)) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            summarize(&nested, node);
      return;
    }
    if (op->getNumRegions() || op->getNumSuccessors()) {
      node.unknown = true;
      return;
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(op) || isMemoryEffectFree(op))
      return;
    // Do not infer completion or synchronization semantics from memory effects
    // alone: async copies, atomics, fences, calls and opaque asm stay unknown.
    if (!isa<memref::LoadOp, memref::StoreOp, memref::CopyOp,
             affine::AffineLoadOp, affine::AffineStoreOp,
             affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
             vector::LoadOp, vector::StoreOp, vector::TransferReadOp,
             vector::TransferWriteOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp>(op)) {
      node.unknown = true;
      return;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    cast<MemoryEffectOpInterface>(op).getEffects(effects);
    for (const auto &effect : effects) {
      Value buffer = effect.getValue();
      if (!buffer || !isa<BaseMemRefType>(buffer.getType()) ||
          !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        node.unknown = true;
        continue;
      }
      auto type = cast<BaseMemRefType>(buffer.getType());
      auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
      if (space && space.getInt() == 5)
        continue; // Private memory cannot communicate between workitems.
      node.accesses.push_back({buffer, isa<MemoryEffects::Write>(effect.getEffect())});
    }
  }

  bool collect(unsigned barrier, bool forward,
               SmallVectorImpl<BarrierAccess> &accesses) {
    SmallVector<unsigned> worklist;
    auto enqueue = [&](unsigned index) {
      auto &edges = forward ? nodes[index].successors : nodes[index].predecessors;
      worklist.append(edges.begin(), edges.end());
    };
    enqueue(barrier);
    llvm::SmallDenseSet<unsigned, 32> visited;
    while (!worklist.empty()) {
      unsigned index = worklist.pop_back_val();
      if (!visited.insert(index).second || nodes[index].barrier)
        continue;
      if (nodes[index].unknown)
        return false;
      accesses.append(nodes[index].accesses.begin(), nodes[index].accesses.end());
      enqueue(index);
    }
    return true;
  }

  func::FuncOp function;
  bool kernel;
  SmallVector<BarrierNode> nodes;
  SmallVector<unsigned> barriers;
  llvm::SmallPtrSet<Operation *, 8> collectiveLoops;
};

class BarrierOptimizePass : public impl::IRDeepOptimizeBase<BarrierOptimizePass> {
public:
  StringRef getArgument() const override { return "barrier-optimize"; }
  StringRef getName() const override { return "BarrierOptimizePass"; }
  StringRef getDescription() const override {
    return "Remove workgroup barriers without cross-thread memory dependencies";
  }
  void runOnOperation() override { BarrierAnalysis(getOperation()).run(); }
};
} // namespace

std::unique_ptr<Pass> createIRDeepOptimizePass() {
  return std::make_unique<VectorOpLoopUnrollPass>();
}

std::unique_ptr<Pass> createBarrierOptimizePass() {
  return std::make_unique<BarrierOptimizePass>();
}


} // namespace mlir::frisk
