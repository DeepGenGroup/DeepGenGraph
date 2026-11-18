#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::frisk {
namespace {
using Loop = affine::AffineForOp;

static ShapedType matrix(Value value) {
  auto type = dyn_cast<ShapedType>(value.getType());
  return type && type.hasRank() && type.getRank() == 2 && type.hasStaticShape()
             ? type : ShapedType();
}

static Type resized(Type type, unsigned axis, int64_t extent) {
  auto shaped = cast<ShapedType>(type);
  SmallVector<int64_t> shape(shaped.getShape());
  shape[axis] = extent;
  if (auto memref = dyn_cast<MemRefType>(type))
    return MemRefType::get(shape, memref.getElementType(), MemRefLayoutAttrInterface(), memref.getMemorySpace());
  return VectorType::get(shape, shaped.getElementType());
}

static bool elementwise(Operation *op) {
  return isa<CastOp, AddOp, SubOp, MulOp, DivOp, MaxOp, Exp2Op>(op);
}

static bool onlyTileReads(Value value) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    if (user->hasTrait<ComputedTile>()) return true;
    auto copy = dyn_cast<CopyOp>(user);
    return copy && copy.getSrc() == value && copy.getDst() != value;
  });
}

static bool readOnlyGlobalViews(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto view = dyn_cast<BufferViewOp>(user)) {
      if (!readOnlyGlobalViews(view.getView())) return false;
    } else if (!onlyTileReads(value)) return false;
  }
  return true;
}

static FailureOr<CopyOp> stagingWriter(Value buffer) {
  auto alloc = buffer.getDefiningOp<AllocBufferOp>();
  if (!alloc || alloc.getMemorySpace() != 3) return failure();
  CopyOp writer;
  for (Operation *user : buffer.getUsers()) {
    if (auto copy = dyn_cast<CopyOp>(user); copy && copy.getDst() == buffer) {
      if (writer || copy.getNumResults() || copy.getSrc() == buffer) return failure();
      writer = copy;
    } else if (!user->hasTrait<ComputedTile>()) {
      // Reject mutation through aliases too. An arbitrary buffer_view is not
      // proven read-only; the accepted generator feeds computed tiles directly.
      return failure();
    }
  }
  if (!writer || !matrix(writer.getSrc()) ||
      matrix(writer.getSrc()).getShape() != matrix(buffer).getShape()) return failure();
  auto map = writer.getOffsetMap();
  bool whole = map.getNumResults() == 1 &&
      map.getResult(0) == getAffineConstantExpr(2, buffer.getContext());
  whole |= map.getNumResults() == 2 && llvm::all_of(map.getResults(), [](AffineExpr e) {
    auto c = dyn_cast<AffineConstantExpr>(e); return c && c.getValue() == 0;
  });
  if (!whole) return failure();
  return writer;
}

// A slice is pushed through the producer graph, including the *writer* of an
// owned allocation. Slicing only the GEMM reader would leave full-D LDS live.
// This helper accepts synchronous, single-writer block IR before scheduling.
class SliceBuilder {
  OpBuilder &b;
  IRMapping &mapping;
  Value origin;
  int64_t tile;
  DenseMap<std::pair<Value, unsigned>, Value> cache;
public:
  SliceBuilder(OpBuilder &b, IRMapping &mapping, Value origin, int64_t tile)
      : b(b), mapping(mapping), origin(origin), tile(tile) {}

  FailureOr<Value> slice(Value value, unsigned axis) {
    auto type = matrix(value);
    if (!type || type.getDimSize(axis) == 1)
      return mapping.lookupOrDefault(value);
    auto key = std::make_pair(value, axis);
    if (auto found = cache.find(key); found != cache.end()) return found->second;
    Operation *op = value.getDefiningOp();
    if (!op) return failure();
    if (!isa<AllocBufferOp>(op) && !onlyTileReads(value)) return failure();
    Type resultType = resized(value.getType(), axis, tile);
    Value result;
    if (auto view = dyn_cast<BufferViewOp>(op)) {
      // This pass runs before K ABI normalization. Accept direct global
      // arguments only: a computed alias could hide a later write to its
      // underlying allocation, invalidating rematerialized staging snapshots.
      auto argument = dyn_cast<BlockArgument>(view.getSource());
      if (!argument || !isa<func::FuncOp>(argument.getOwner()->getParentOp())) return failure();
      auto function = cast<func::FuncOp>(argument.getOwner()->getParentOp());
      if (argument.getOwner() != &function.getBody().front()) return failure();
      if (!readOnlyGlobalViews(view.getSource())) return failure();
      SmallVector<AffineExpr> offsets(view.getIndexMap().getResults());
      SmallVector<Value> indices;
      for (Value index : view.getIndices()) indices.push_back(mapping.lookupOrDefault(index));
      auto map = view.getIndexMap();
      // Add a dimension to the original origin. A reduction IV is an affine
      // dimension, not a symbol. Keep the original global strides,
      // including K's original D/S orientation before ABI normalization.
      offsets[offsets.size() - 2 + axis] = offsets[offsets.size() - 2 + axis] +
          b.getAffineDimExpr(map.getNumDims());
      indices.insert(indices.begin() + map.getNumDims(), origin);
      SmallVector<int64_t> shape(type.getShape()); shape[axis] = tile;
      result = b.create<BufferViewOp>(op->getLoc(), mapping.lookupOrDefault(view.getSource()),
          indices, AffineMap::get(map.getNumDims() + 1, map.getNumSymbols(), offsets, b.getContext()), shape);
    } else if (auto alloc = dyn_cast<AllocBufferOp>(op)) {
      auto writer = stagingWriter(value);
      if (failed(writer)) return failure();
      auto source = slice(writer->getSrc(), axis);
      if (failed(source)) return failure();
      SmallVector<int64_t> shape(type.getShape()); shape[axis] = tile;
      result = b.create<AllocBufferOp>(alloc.getLoc(), shape, type.getElementType(),
                                      alloc.getAlignment(), alloc.getMemorySpace());
      for (auto attr : alloc->getDiscardableAttrs())
        result.getDefiningOp()->setAttr(attr.getName(), attr.getValue());
      b.create<CopyOp>(writer->getLoc(), *source, result);
    } else if (elementwise(op) || isa<ZeroOp, arith::ConstantOp>(op)) {
      IRMapping operands;
      for (Value operand : op->getOperands()) {
        auto operandType = matrix(operand);
        if (operandType && operandType.getDimSize(axis) == type.getDimSize(axis)) {
          auto part = slice(operand, axis);
          if (failed(part)) return failure();
          operands.map(operand, *part);
        } else operands.map(operand, mapping.lookupOrDefault(operand));
      }
      auto *copy = b.clone(*op, operands);
      copy->getResult(0).setType(resultType);
      if (auto constant = dyn_cast<arith::ConstantOp>(copy)) {
        auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        if (!dense || !dense.isSplat()) return failure();
        constant.setValueAttr(DenseElementsAttr::get(cast<ShapedType>(resultType), dense.getSplatValue<Attribute>()));
      }
      result = copy->getResult(0);
    } else return failure();
    cache[key] = result;
    return result;
  }
};

// Track output-column identities through SSA, not equal numeric shapes. BM,
// BN and D can be identical; the score/probability must never follow O's axis.
class OutputColumns {
  DenseMap<Value, SmallVector<Value>> edges;
public:
  DenseSet<Value> values;
  explicit OutputColumns(func::FuncOp kernel, GemmOp pv) {
    auto connect = [&](Value a, Value c) {
      auto x = matrix(a), y = matrix(c);
      if (!x || !y || x.getDimSize(1) != y.getDimSize(1)) return;
      edges[a].push_back(c); edges[c].push_back(a);
    };
    kernel.walk([&](Operation *op) {
      if (auto gemm = dyn_cast<GemmOp>(op)) {
        connect(gemm.getB(), gemm.getC()); connect(gemm.getAcc(), gemm.getC());
      } else if (auto loop = dyn_cast<Loop>(op)) {
        auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
        for (unsigned i = 0; i < loop.getNumResults(); ++i) {
          connect(loop.getInits()[i], loop.getRegionIterArgs()[i]);
          connect(loop.getInits()[i], loop.getResult(i));
          connect(loop.getInits()[i], yield.getOperand(i));
        }
      } else if (auto copy = dyn_cast<CopyOp>(op)) {
        connect(copy.getSrc(), copy.getDst());
      } else if (elementwise(op)) {
        for (Value result : op->getResults())
          for (Value operand : op->getOperands()) connect(result, operand);
      }
    });
    SmallVector<Value> work{pv.getC()};
    while (!work.empty()) {
      Value value = work.pop_back_val();
      if (!values.insert(value).second) continue;
      for (Value next : edges[value]) work.push_back(next);
    }
  }
  bool affected(Operation *op) {
    return llvm::any_of(op->getOperands(), [&](Value v) { return values.contains(v); }) ||
           llvm::any_of(op->getResults(), [&](Value v) { return values.contains(v); });
  }
};

// Remove only dead virtual computations and unused owned staging writes.
// Global stores and arbitrary writes are never treated as dead SSA values.
static void cleanDeadStaging(func::FuncOp kernel) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    kernel.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op == kernel || op->hasTrait<OpTrait::IsTerminator>()) return;
      if (auto alloc = dyn_cast<AllocBufferOp>(op)) {
        bool onlyWrites = llvm::all_of(alloc->getUsers(), [&](Operation *user) {
          auto copy = dyn_cast<CopyOp>(user);
          return copy && !copy.getNumResults() && copy.getDst() == alloc.getResult();
        });
        if (onlyWrites) {
          for (Operation *writer : alloc->getUsers()) dead.push_back(writer);
          dead.push_back(op);
        }
      } else if (isOpTriviallyDead(op) ||
                 (op->hasTrait<ComputedTile>() && op->use_empty())) dead.push_back(op);
    });
    for (Operation *op : dead) { op->erase(); changed = true; }
  }
}

static LogicalResult tileQK(GemmOp qk, int64_t tile, Attribute pipelineHint,
                            bool simplify) {
  int64_t extent = matrix(qk.getA()).getDimSize(1);
  if (tile == extent) return success();
  OpBuilder b(qk);
  auto type = matrix(qk.getC());
  auto carrierType = VectorType::get(type.getShape(), type.getElementType());
  Value init = b.create<CastOp>(qk.getLoc(), carrierType, qk.getAcc());
  auto loop = b.create<Loop>(qk.getLoc(), 0, extent, tile, ValueRange{init});
  loop->setAttr("local_yield_transformed", b.getBoolAttr(true));
  // Expose the bounded reduction before retention and stream planning unless
  // the tuning input explicitly requests a rolled QK loop.
  if (simplify) {
    loop->setAttr("frisk.simplify_loop", b.getUnitAttr());
    loop->setAttr("frisk.loopUnrollFull", b.getBoolAttr(true));
  }
  if (pipelineHint) loop->setAttr("frisk.pipeline_hint", pipelineHint);
  b.setInsertionPointToStart(loop.getBody());
  IRMapping mapping;
  SliceBuilder slices(b, mapping, loop.getInductionVar(), tile);
  auto a = slices.slice(qk.getA(), 1), rhs = slices.slice(qk.getB(), 0);
  if (failed(a) || failed(rhs)) return qk.emitError("QK slice requires an owned single-writer unscheduled producer graph");
  Value acc = b.create<CastOp>(qk.getLoc(), type, loop.getRegionIterArgs()[0]);
  auto gemm = b.create<GemmOp>(qk.getLoc(), type, *a, *rhs, acc, false, false);
  for (auto attr : qk->getDiscardableAttrs()) gemm->setAttr(attr.getName(), attr.getValue());
  b.create<CopyOp>(qk.getLoc(), gemm.getC(), loop.getRegionIterArgs()[0]);
  b.create<affine::AffineYieldOp>(qk.getLoc(), loop.getRegionIterArgs());
  b.setInsertionPointAfter(loop);
  Value result = b.create<CastOp>(qk.getLoc(), type, loop.getResult(0));
  qk.getC().replaceAllUsesWith(result);
  qk.erase();
  return success();
}

static LogicalResult tileOutput(func::FuncOp kernel, GemmOp pv, int64_t tile) {
  int64_t extent = matrix(pv.getC()).getDimSize(1), count = extent / tile;
  if (count == 1) return success();
  OutputColumns columns(kernel, pv);
  // V staging is materialized lazily beside each PV, so four VO slices do not
  // require four simultaneously live LDS tiles. Collect the old producer path.
  DenseSet<Operation *> loads;
  std::function<LogicalResult(Value)> collect = [&](Value v) -> LogicalResult {
    Operation *op = v.getDefiningOp();
    if (!op) return failure();
    if (!loads.insert(op).second) return success();
    if (isa<BufferViewOp>(op)) return success();
    if (auto alloc = dyn_cast<AllocBufferOp>(op)) {
      auto writer = stagingWriter(v);
      if (failed(writer)) return failure();
      loads.insert(writer->getOperation());
      return collect(writer->getSrc());
    }
    if (elementwise(op)) {
      for (Value operand : op->getOperands())
        if (auto t = matrix(operand); t && t.getDimSize(1) == extent)
          if (failed(collect(operand))) return failure();
      return success();
    }
    return failure();
  };
  if (failed(collect(pv.getB()))) return pv.emitError("V slice requires an owned single-writer unscheduled producer graph");
  // Only these original producer operations are elided. Check that their
  // values do not escape to another consumer before replacing the body.
  for (Operation *op : loads)
    for (Operation *user : op->getUsers())
      if (!loads.contains(user) && user != pv) return op->emitError("V staging escapes the tiled PV producer graph");
  SmallVector<IRMapping> mappings(count);
  Block *oldBody = &kernel.getBody().front();
  Block *newBody = new Block(); kernel.getBody().push_back(newBody);
  for (BlockArgument arg : oldBody->getArguments()) {
    Value next = newBody->addArgument(arg.getType(), arg.getLoc());
    for (auto &mapping : mappings) mapping.map(arg, next);
  }
  OpBuilder b(kernel.getContext()); b.setInsertionPointToStart(newBody);
  std::function<LogicalResult(Operation *)> clone = [&](Operation *op) -> LogicalResult {
    if (loads.contains(op)) return success();
    if (auto loop = dyn_cast<Loop>(op); loop && columns.affected(loop)) {
      SmallVector<Value> inits; SmallVector<unsigned> starts;
      for (Value init : loop.getInits()) {
        starts.push_back(inits.size());
        unsigned n = columns.values.contains(init) ? count : 1;
        for (unsigned i = 0; i < n; ++i) inits.push_back(mappings[i].lookupOrDefault(init));
      }
      SmallVector<Value> lower, upper;
      for (Value v : loop.getLowerBoundOperands()) lower.push_back(mappings[0].lookupOrDefault(v));
      for (Value v : loop.getUpperBoundOperands()) upper.push_back(mappings[0].lookupOrDefault(v));
      auto next = b.create<Loop>(loop.getLoc(), lower,
          loop.getLowerBoundMap(), upper,
          loop.getUpperBoundMap(), loop.getStepAsInt(), inits);
      for (auto attr : loop->getDiscardableAttrs()) next->setAttr(attr.getName(), attr.getValue());
      for (unsigned i = 0; i < count; ++i) {
        mappings[i].map(loop.getInductionVar(), next.getInductionVar());
        for (unsigned j = 0; j < loop.getNumResults(); ++j) {
          unsigned index = starts[j] + (columns.values.contains(loop.getResult(j)) ? i : 0);
          mappings[i].map(loop.getRegionIterArgs()[j], next.getRegionIterArgs()[index]);
          mappings[i].map(loop.getResult(j), next.getResult(index));
        }
      }
      b.setInsertionPointToStart(next.getBody());
      for (Operation &nested : loop.getBody()->without_terminator())
        if (failed(clone(&nested))) return failure();
      SmallVector<Value> yields;
      auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
      for (Value value : yield.getOperands())
        for (unsigned i = 0, n = columns.values.contains(value) ? count : 1; i < n; ++i)
          yields.push_back(mappings[i].lookupOrDefault(value));
      b.create<affine::AffineYieldOp>(loop.getLoc(), yields);
      b.setInsertionPointAfter(next);
      return success();
    }
    if (op->getNumRegions() && columns.affected(op))
      return op->emitError("output columns escape the supported affine recurrence");
    if (columns.affected(op) && op != pv && !elementwise(op) &&
        !isa<ZeroOp, arith::ConstantOp, CopyOp>(op))
      return op->emitError("output columns escape supported tile operations");
    if (auto copy = dyn_cast<CopyOp>(op); copy && columns.affected(op)) {
      if (copy.getNumResults()) return copy.emitError("output columns require result-free block copies");
      auto map = copy.getOffsetMap();
      if (columns.values.contains(copy.getDst())) {
        if (map.getNumResults() != 1 ||
            map.getResult(0) != getAffineConstantExpr(2, b.getContext()))
          return copy.emitError("output columns require whole carrier copies");
      } else {
        auto destination = dyn_cast<MemRefType>(copy.getDst().getType());
        if (!destination || destination.getMemorySpaceAsInt() != 1 || destination.getRank() != 4 ||
            map.getNumResults() != 4)
          return copy.emitError("output columns require rank-four global stores");
      }
    }
    unsigned n = columns.affected(op) ? count : 1;
    for (unsigned i = 0; i < n; ++i) {
      if (op == pv) {
        Value origin = b.create<arith::ConstantIndexOp>(op->getLoc(), i * tile);
        SliceBuilder slices(b, mappings[i], origin, tile);
        auto v = slices.slice(pv.getB(), 1);
        if (failed(v)) return pv.emitError("cannot materialize V output slice");
        mappings[i].map(pv.getB(), *v);
      }
      Operation *next = b.clone(*op, mappings[i]);
      for (auto [oldResult, result] : llvm::zip(op->getResults(), next->getResults()))
        if (columns.values.contains(oldResult)) result.setType(resized(oldResult.getType(), 1, tile));
      if (auto constant = dyn_cast<arith::ConstantOp>(next); constant && n > 1) {
        auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        if (!dense || !dense.isSplat()) return constant.emitError("output carrier initializer must be a splat");
        constant.setValueAttr(DenseElementsAttr::get(cast<ShapedType>(constant.getType()), dense.getSplatValue<Attribute>()));
      }
      if (auto copy = dyn_cast<CopyOp>(next); copy && n > 1 &&
          !columns.values.contains(cast<CopyOp>(op).getDst())) {
        auto map = copy.getOffsetMap(); SmallVector<AffineExpr> offsets(map.getResults());
        offsets.back() = offsets.back() + i * tile;
        copy.setOffsetMap(AffineMap::get(map.getNumDims(), map.getNumSymbols(), offsets, b.getContext()));
      }
      if (n == 1)
        for (unsigned j = 1; j < count; ++j)
          for (auto [oldResult, result] : llvm::zip(op->getResults(), next->getResults())) mappings[j].map(oldResult, result);
    }
    return success();
  };
  for (Operation &op : *oldBody) if (failed(clone(&op))) return failure();
  oldBody->dropAllDefinedValueUses(); oldBody->erase();
  return success();
}

class TileAttention : public PassWrapper<TileAttention, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileAttention)
  TileAttention() = default;
  TileAttention(const TileAttention &other) : PassWrapper(other) {}
  Option<int64_t> qkTile{*this, "qk-tile", llvm::cl::init(0)};
  Option<int64_t> voTile{*this, "vo-tile", llvm::cl::init(0)};
  Option<bool> allowReassociation{*this, "allow-reassociation", llvm::cl::init(false)};
  StringRef getArgument() const final { return "frisk-tile-attention"; }
  StringRef getDescription() const final { return "Tile shared attention QK reduction and V/O columns before fragment planning"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<FriskDialect, affine::AffineDialect, arith::ArithDialect>();
  }
  void runOnOperation() final {
    auto kernel = getOperation();
    if (!kernel->hasAttr("thread_num")) return;
    if (kernel->hasAttr("frisk.attention_tiled")) return;
    auto hints = kernel->getAttrOfType<DenseI64ArrayAttr>("frisk.attention_tiles");
    if (!qkTile && !voTile && !hints) return;
    SmallVector<GemmOp> dots; kernel.walk([&](GemmOp op) { dots.push_back(op); });
    if (dots.size() != 2 || dots[0].getTransA() || dots[0].getTransB() ||
        dots[1].getTransA() || dots[1].getTransB() ||
        !isa<Loop>(dots[0]->getParentOp()) || dots[0]->getParentOp() != dots[1]->getParentOp()) {
      kernel.emitError("attention tiling requires shared full-D QK and PV in one affine key loop");
      return signalPassFailure();
    }
    auto a = matrix(dots[0].getA()), v = matrix(dots[1].getB());
    if (!a || !v || !a.getElementType().isF16() || !v.getElementType().isF16() ||
        (hints && hints.size() != 2)) {
      kernel.emitError("attention tiles require static rank-two FP16 operands and two hints");
      return signalPassFailure();
    }
    int64_t dq = a.getDimSize(1), dv = v.getDimSize(1);
    int64_t q = qkTile ? int64_t(qkTile) : hints ? hints[0] : dq;
    int64_t o = voTile ? int64_t(voTile) : hints ? hints[1] : dv;
    if (q <= 0 || q % 16 || dq % q || o <= 0 || o % 16 || dv % o) {
      kernel.emitError("QK/VO tiles must be positive multiples of 16 dividing their D extents");
      return signalPassFailure();
    }
    if (q != dq && (!allowReassociation || kernel->hasAttr("llvm.strictfp"))) {
      kernel.emitError("QK tiles changing the reduction grouping require allow-reassociation");
      return signalPassFailure();
    }
    auto score = matrix(dots[0].getC());
    if (q != dq && (!score || !score.getElementType().isF32())) {
      kernel.emitError("QK block reduction requires a static rank-two FP32 product");
      return signalPassFailure();
    }
    bool scheduled = false;
    kernel.walk([&](Operation *op) {
      scheduled |= op->hasAttr("pipeline.stage") || op->hasAttr("pipeline.order") ||
                   op->hasAttr("pipeline.scheduled");
    });
    if (scheduled) {
      kernel.emitError("attention tiles must be chosen before an authored pipeline schedule");
      return signalPassFailure();
    }
    // Work transactionally: unsupported producer graphs leave the caller's
    // function untouched and produce a diagnostic, never a half-tiled kernel.
    OwningOpRef<func::FuncOp> trial = cast<func::FuncOp>(kernel->clone());
    dots.clear(); trial->walk([&](GemmOp op) { dots.push_back(op); });
    GemmOp pv = dots[1];
    auto simplify = kernel->getAttrOfType<BoolAttr>("frisk.qk_simplify_loop");
    if (failed(tileQK(dots[0], q, kernel->getAttr("frisk.qk_pipeline_hint"),
                      !simplify || simplify.getValue())) ||
        failed(tileOutput(*trial, pv, o))) return signalPassFailure();
    cleanDeadStaging(*trial);
    trial->getOperation()->setAttr("frisk.attention_tiled", DenseI64ArrayAttr::get(&getContext(), {q, o}));
    trial->getOperation()->removeAttr("frisk.attention_tiles");
    trial->getOperation()->removeAttr("frisk.qk_pipeline_hint");
    trial->getOperation()->removeAttr("frisk.qk_simplify_loop");
    if (failed(verify(*trial))) return signalPassFailure();
    kernel.getBody().takeBody(trial->getBody());
    kernel->setAttrs(trial->getOperation()->getAttrs());
  }
};
} // namespace
std::unique_ptr<Pass> createTileAttentionPass() { return std::make_unique<TileAttention>(); }
} // namespace mlir::frisk
