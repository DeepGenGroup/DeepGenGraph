// Rewrite online-softmax before fragment scheduling. A bounded disposable
// expansion proves register slices; the committed graph retains GEMM/load loops.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Conversion/MaskAddBypass.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>
#include <set>

namespace mlir::frisk {
namespace {
constexpr StringLiteral Marker = "frisk.lazy_softmax_base";
constexpr StringLiteral Origin = "frisk.lazy_analysis_origin";
VectorType registers(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) return fragment.getRegisterType();
  return dyn_cast<VectorType>(type);
}
int64_t count(Type type) {
  if (auto v = registers(type)) return v.isScalable() ? 0 : v.getNumElements();
  return 1;
}
struct Slice {
  Value value;
  int64_t element = 0;
  bool operator==(const Slice &s) const { return value == s.value && element == s.element; }
  bool operator<(const Slice &s) const {
    return value != s.value ? std::less<const void *>{}(
        value.getAsOpaquePointer(), s.value.getAsOpaquePointer()) : element < s.element;
  }
};

// Exact folding, including affine.apply indices left by analysis-only unroll.
Attribute evaluate(Value v, std::optional<unsigned> tid = {}, unsigned depth = 0) {
  if (depth > 64 || !v.getType().isIntOrIndex()) return {};
  Attribute constant;
  if (matchPattern(v, m_Constant(&constant))) return constant;
  if (auto id = v.getDefiningOp<gpu::ThreadIdOp>(); id && tid)
    return IntegerAttr::get(v.getType(), id.getDimension() == gpu::Dimension::x ? *tid : 0);
  auto *op = v.getDefiningOp();
  if (!op || (op->getName().getDialectNamespace() != "arith" &&
              !isa<affine::AffineApplyOp>(op))) return {};
  SmallVector<Attribute> inputs;
  for (Value input : op->getOperands()) {
    auto attr = evaluate(input, tid, depth + 1);
    if (!attr) return {};
    inputs.push_back(attr);
  }
  Operation *copy = op->clone();
  SmallVector<OpFoldResult> results;
  Attribute result;
  if (succeeded(copy->fold(inputs, results)) && results.size() == 1)
    result = dyn_cast<Attribute>(results[0]);
  copy->destroy();
  return result;
}
std::optional<int64_t> integer(Value v) {
  if (auto attr = dyn_cast_or_null<IntegerAttr>(evaluate(v))) return attr.getInt();
  return {};
}
SmallVector<int64_t> coordinates(int64_t flat, VectorType type) {
  SmallVector<int64_t> pos(type.getRank());
  for (int64_t d = type.getRank() - 1; d >= 0; --d) {
    pos[d] = flat % type.getDimSize(d);
    flat /= type.getDimSize(d);
  }
  return pos;
}
// A fragment's indices count tiles, not scalar elements.
std::optional<SmallVector<int64_t>> origins(ValueRange indices, VectorType fragment) {
  if (indices.size() != unsigned(fragment.getRank())) return {};
  SmallVector<int64_t> result;
  for (auto [index, extent] : llvm::zip(indices, fragment.getShape())) {
    auto n = integer(index);
    if (!n || *n < 0 || *n > INT64_MAX / extent) return {};
    result.push_back(*n * extent);
  }
  return result;
}
std::optional<Slice> view(Slice s) {
  auto *op = s.value.getDefiningOp();
  if (!op) return {};
  if (isa<PackFragOp, UnpackFragOp, vector::ShapeCastOp>(op))
    return Slice{op->getOperand(0), s.element};
  if (auto cast = dyn_cast<ElementwiseFragOp>(op);
      cast && cast.getKind() == "cast" &&
      registers(cast.getInputs()[0].getType()) == registers(cast.getType()))
    return Slice{cast.getInputs()[0], s.element};
  if (auto cast = dyn_cast<ElementwiseFragOp>(op);
      cast && cast.getKind() == "cast" &&
      cast.getInputs()[0].getType() == cast.getType())
    return Slice{cast.getInputs()[0], s.element};
  if (auto tile = dyn_cast<ToThreadTileOp>(op)) {
    // A splat is independent of the ownership map. Never assume this for a
    // general block vector, whose selected register can differ across lanes.
    Attribute value;
    if (matchPattern(tile->getOperand(0), m_Constant(&value)))
      if (auto dense = dyn_cast<DenseElementsAttr>(value); dense && dense.isSplat())
        return Slice{tile->getOperand(0), 0};
  }
  if (auto e = dyn_cast<ExtractFragOp>(op)) {
    auto dst = e.getType().getRegisterType();
    auto src = cast<VectorType>(e.getSource().getType());
    auto start = origins(e.getIndices(), dst);
    if (!start) return {};
    auto pos = coordinates(s.element, dst);
    int64_t flat = 0;
    for (unsigned d = 0; d < pos.size(); ++d) {
      pos[d] += (*start)[d];
      if (pos[d] >= src.getDimSize(d)) return {};
      flat = flat * src.getDimSize(d) + pos[d];
    }
    return Slice{e.getSource(), flat};
  }
  if (auto i = dyn_cast<InsertFragOp>(op)) {
    auto src = i.getValue().getType().getRegisterType();
    auto dst = cast<VectorType>(i.getType());
    auto start = origins(i.getIndices(), src);
    if (!start) return {};
    auto pos = coordinates(s.element, dst);
    int64_t flat = 0;
    for (unsigned d = 0; d < pos.size(); ++d) {
      int64_t p = pos[d] - (*start)[d];
      if (p < 0 || p >= src.getDimSize(d)) return Slice{i.getDestination(), s.element};
      flat = flat * src.getDimSize(d) + p;
    }
    return Slice{i.getValue(), flat};
  }
  if (auto e = dyn_cast<vector::ExtractOp>(op)) {
    if (!e.getDynamicPosition().empty()) return {};
    auto shape = e.getSourceVectorType().getShape();
    auto pos = e.getStaticPosition();
    int64_t offset = 0;
    for (unsigned d = 0; d < shape.size(); ++d)
      offset = offset * shape[d] + (d < pos.size() ? pos[d] : 0);
    return Slice{e.getSource(), offset + s.element};
  }
  if (auto f = dyn_cast<vector::FromElementsOp>(op))
    return Slice{f->getOperand(s.element), 0};
  if (auto b = dyn_cast<BroadcastFragOp>(op)) {
    auto src = registers(b.getSource().getType());
    if (!src) return Slice{b.getSource(), 0};
    auto dst = b.getType().getRegisterType();
    auto pos = coordinates(s.element, dst);
    int64_t flat = 0;
    for (int64_t d = 0; d < src.getRank(); ++d)
      flat = flat * src.getDimSize(d) + (src.getDimSize(d) == 1 ? 0 : pos[d]);
    return Slice{b.getSource(), flat};
  }
  if (auto b = dyn_cast<vector::BroadcastOp>(op); b && !registers(b.getSource().getType()))
    return Slice{b.getSource(), 0};
  return {};
}
Slice resolve(Slice s) {
  for (unsigned i = 0; i < 512; ++i) {
    auto next = view(s);
    if (!next) break;
    s = *next;
  }
  return s;
}

std::optional<APFloat> floatConstant(Slice s, unsigned depth = 0) {
  if (depth > 128) return {};
  s = resolve(s);
  Attribute attr;
  if (matchPattern(s.value, m_Constant(&attr))) {
    if (auto dense = dyn_cast<DenseElementsAttr>(attr))
      attr = dense.getValues<Attribute>()[s.element];
    if (auto f = dyn_cast<FloatAttr>(attr)) return f.getValue();
  }
  if (auto tile = s.value.getDefiningOp<ToThreadTileOp>()) {
    if (matchPattern(tile.getBlockTile(), m_Constant(&attr)))
      if (auto dense = dyn_cast<DenseFPElementsAttr>(attr); dense && dense.isSplat())
        return dense.getSplatValue<APFloat>();
  }
  auto op = s.value.getDefiningOp<ElementwiseFragOp>();
  if (!op || op.getInputs().size() != 2) return {};
  auto lhs = floatConstant({op.getInputs()[0], s.element}, depth + 1);
  auto rhs = floatConstant({op.getInputs()[1], s.element}, depth + 1);
  if (!lhs || !rhs) return {};
  if (op.getKind() == "add") lhs->add(*rhs, APFloat::rmNearestTiesToEven);
  else if (op.getKind() == "sub") lhs->subtract(*rhs, APFloat::rmNearestTiesToEven);
  else if (op.getKind() == "mul") lhs->multiply(*rhs, APFloat::rmNearestTiesToEven);
  else return {};
  return lhs;
}
bool constant(Value value, bool inf = false) {
  for (int64_t i = 0; i < count(value.getType()); ++i) {
    auto f = floatConstant({value, i});
    if (!f || !(inf ? f->isNegInfinity() : f->isZero())) return false;
  }
  return true;
}
bool kind(Operation *op, StringRef name) {
  auto e = dyn_cast_or_null<ElementwiseFragOp>(op);
  return e && e.getKind() == name;
}
bool splat(Value value, Slice expected) {
  for (int64_t i = 0; i < count(value.getType()); ++i)
    if (!(resolve({value, i}) == expected)) return false;
  return true;
}
bool transparent(Operation *op) {
  if (auto cast = dyn_cast<ElementwiseFragOp>(op);
      cast && cast.getKind() == "cast" &&
      registers(cast.getInputs()[0].getType()) == registers(cast.getType())) return true;
  return isa<ExtractFragOp, InsertFragOp, PackFragOp, UnpackFragOp,
             BroadcastFragOp, vector::ExtractOp, vector::FromElementsOp,
             vector::ShapeCastOp, vector::BroadcastOp>(op);
}
struct Loop {
  Operation *op;
  Block *body;
  Value iv;
  ValueRange init;
  Block::BlockArgListType args;
  static std::optional<Loop> get(Operation *op) {
    if (auto f = dyn_cast<affine::AffineForOp>(op)) {
      if (!f.hasConstantLowerBound() || f.getConstantLowerBound() || f.getStepAsInt() <= 0)
        return {};
      return Loop{op, f.getBody(), f.getInductionVar(), f.getInits(), f.getRegionIterArgs()};
    }
    if (auto f = dyn_cast<scf::ForOp>(op)) {
      auto lower = integer(f.getLowerBound()), step = integer(f.getStep());
      if (!lower || *lower || !step || *step <= 0) return {};
      return Loop{op, f.getBody(), f.getInductionVar(), f.getInitArgs(), f.getRegionIterArgs()};
    }
    return {};
  }
  void setInit(unsigned i, Value value) {
    if (auto f = dyn_cast<affine::AffineForOp>(op)) f.getInitsMutable()[i].set(value);
    else cast<scf::ForOp>(op).getInitArgsMutable()[i].set(value);
  }
};

// Restrict candidate loop control to values uniform over the whole workgroup.
// No assumptions about a load, a lane id, or a caller-provided annotation.
bool uniform(Value value, unsigned depth = 0) {
  if (depth > 64) return false;
  if (matchPattern(value, m_Constant())) return true;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op)) return true;
  if (!isa<affine::AffineApplyOp>(op) && op->getName().getDialectNamespace() != "arith")
    return false;
  return llvm::all_of(op->getOperands(), [&](Value v) { return uniform(v, depth + 1); });
}
bool uniform(Loop loop) {
  if (auto f = dyn_cast<affine::AffineForOp>(loop.op))
    return llvm::all_of(f.getControlOperands(), [](Value v) { return uniform(v); });
  auto f = cast<scf::ForOp>(loop.op);
  return uniform(f.getUpperBound());
}

struct Row {
  Slice maximum;
  ElementwiseFragOp diff, alpha;
  SmallVector<ElementwiseFragOp> shifts, scales;
};
struct Output {
  Operation *root;
  Value oldState; // empty for scores
};
struct Plan {
  Loop loop;
  unsigned baseIndex;
  ReduceFragOp peak;
  SmallVector<Row> rows;
  SmallVector<Output> outputs;
  SmallVector<ExtractFragOp> seeds;
  Operation *split = nullptr;
};

// The only expanded IR is this proof copy. Observers retain the value of each
// original tile result through loop unrolling, without affecting the source IR.
class Analysis {
public:
  OwningOpRef<func::FuncOp> copy;
  SmallVector<Operation *> originals;
  DenseMap<Operation *, Operation *> cloned;
  DenseMap<Value, UnrealizedConversionCastOp> observed;
  llvm::SmallPtrSet<Operation *, 32> observers;
  explicit Analysis(func::FuncOp function) {
    IRMapping mapping;
    copy = cast<func::FuncOp>(function->clone(mapping));
    function.walk([&](Operation *op) { originals.push_back(op); });
    for (auto [i, original] : llvm::enumerate(originals)) {
      Operation *op = original == function ? copy->getOperation() : mapping.lookup(original);
      cloned[original] = op;
      op->setAttr(Origin, IntegerAttr::get(IntegerType::get(function.getContext(), 64), i));
      if (op->getNumResults() != 1 || !registers(op->getResult(0).getType())) continue;
      OpBuilder b(op);
      b.setInsertionPointAfter(op);
      auto observer = b.create<UnrealizedConversionCastOp>(op->getLoc(),
          op->getResultTypes(), op->getResults());
      observed[original->getResult(0)] = observer;
      observers.insert(observer);
    }
  }
  Operation *original(Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>(Origin);
    return id ? originals[id.getInt()] : nullptr;
  }
  Value value(Value original) { return observed.lookup(original)->getOperand(0); }
  bool expand() {
    SmallVector<affine::AffineForOp> loops;
    copy->walk<WalkOrder::PostOrder>([&](affine::AffineForOp f) {
      if (f->hasAttr("frisk.fragment_loop")) loops.push_back(f);
    });
    uint64_t budget = 131072;
    for (auto loop : loops) {
      if (!loop.hasConstantBounds()) return false;
      int64_t span = loop.getConstantUpperBound() - loop.getConstantLowerBound();
      if (span <= 0) return false;
      uint64_t trips = llvm::divideCeil(uint64_t(span), uint64_t(loop.getStepAsInt()));
      uint64_t size = 0;
      loop.walk([&](Operation *) { ++size; });
      if (trips > 256 || size > budget / trips) return false;
      budget -= size * trips;
      if (failed(affine::loopUnrollFull(loop))) return false;
    }
    // Observers inside unrolled loops have been cloned. Recompute their set.
    observers.clear();
    copy->walk([&](UnrealizedConversionCastOp op) { observers.insert(op); });
    return true;
  }
  bool usesOnly(Value value, const llvm::SmallPtrSetImpl<Operation *> &allowed, Block *body) {
    SmallVector<Slice> work;
    std::set<Slice> seen;
    for (int64_t i = 0; i < count(value.getType()); ++i) work.push_back({value, i});
    for (unsigned i = 0; i < work.size(); ++i) {
      if (work.size() > 65536) return false;
      Slice s = work[i];
      if (!seen.insert(s).second) continue;
      for (Operation *user : s.value.getUsers()) {
        if (allowed.contains(user) || observers.contains(user)) continue;
        if (user->getBlock() != body || !transparent(user) || user->getNumResults() != 1)
          return false;
        Value result = user->getResult(0);
        for (int64_t e = 0; e < count(result.getType()); ++e) {
          auto input = view({result, e});
          if (!input) return false;
          if (*input == s) work.push_back({result, e});
        }
      }
    }
    return true;
  }
};

Operation *rootIn(Operation *op, Block *body) {
  while (op && op->getBlock() != body) op = op->getParentOp();
  return op;
}

bool match(Plan &p, Analysis &a, std::string &why) {
  auto fail = [&](StringRef reason) { why = reason.str(); return false; };
  Loop proof = *Loop::get(a.cloned.lookup(p.loop.op));
  auto *yield = proof.body->getTerminator();
  Value base;
  for (auto [i, init] : llvm::enumerate(proof.init)) {
    auto type = registers(init.getType());
    if (!type || type.getRank() != 2 || type.isScalable() || type.getDimSize(1) != 1 ||
        !type.getElementType().isF32() || type.getDimSize(0) > 8 || !constant(init, true)) continue;
    if (base || !p.loop.op->getResult(i).use_empty()) return fail("ambiguous or escaped running max");
    p.baseIndex = i;
    base = proof.args[i];
    p.rows.resize(type.getDimSize(0));
  }
  if (!base) return fail("missing -inf running-max state");
  llvm::SmallPtrSet<Operation *, 16> baseUsers;
  llvm::SmallPtrSet<Operation *, 32> scales, shifts, maxima, gemms;
  llvm::MapVector<Value, SmallVector<bool>> states;
  ReduceFragOp reduction;
  for (auto [r, row] : llvm::enumerate(p.rows)) {
    row.maximum = resolve({yield->getOperand(p.baseIndex), int64_t(r)});
    Operation *max = row.maximum.value.getDefiningOp();
    if (!kind(max, "max") || count(row.maximum.value.getType()) != 1)
      return fail("running max is not a scalar row max");
    Slice old{base, int64_t(r)};
    Value peak = splat(max->getOperand(0), old) ? max->getOperand(1)
                 : splat(max->getOperand(1), old) ? max->getOperand(0) : Value();
    if (!peak) return fail("maximum does not consume old row");
    Slice s = resolve({peak});
    auto reduce = s.value.getDefiningOp<ReduceFragOp>();
    if (!reduce || reduce.getKind() != "max" || reduce.getDim() != 1 ||
        reduce.getWarpThreads() != 64 || reduce.getLaneExtent() != 4 || reduce.getLaneStride() != 16 ||
        registers(reduce.getType()) != registers(base.getType()) ||
        s.element != int64_t(r) || (reduction && reduction != reduce))
      return fail("missing common row reduction");
    reduction = reduce;
    maxima.insert(max);
    baseUsers.insert(max);
    for (Operation &op : *proof.body) {
      if (!kind(&op, "sub") || !splat(op.getOperand(1), row.maximum)) continue;
      if (splat(op.getOperand(0), old)) {
        if (row.diff) return fail("duplicate alpha exponent");
        row.diff = cast<ElementwiseFragOp>(op);
        baseUsers.insert(&op);
      } else row.shifts.push_back(cast<ElementwiseFragOp>(op));
    }
    if (!row.diff || row.shifts.empty()) return fail("missing shifts or alpha exponent");
    for (Operation &op : *proof.body)
      if (kind(&op, "exp2") && splat(op.getOperand(0), {row.diff.getResult(), 0})) {
        if (row.alpha) return fail("duplicate alpha");
        row.alpha = cast<ElementwiseFragOp>(op);
      }
    if (!row.alpha) return fail("missing alpha");
    for (Operation &op : *proof.body)
      if (kind(&op, "mul") && splat(op.getOperand(1), {row.alpha.getResult(), 0}))
        row.scales.push_back(cast<ElementwiseFragOp>(op));
    if (row.scales.empty()) return fail("missing state scales");
    llvm::SmallPtrSet<Operation *, 16> maxUsers{yield, row.diff};
    for (auto shift : row.shifts) { maxUsers.insert(shift); shifts.insert(shift); }
    if (!a.usesOnly(max->getResult(0), maxUsers, proof.body) ||
        !a.usesOnly(row.diff, llvm::SmallPtrSet<Operation *, 16>{row.alpha}, proof.body))
      return fail("escaped maximum or exponent");
    for (auto [i, value] : llvm::enumerate(yield->getOperands()))
      if (i != p.baseIndex)
        for (int64_t e = 0; e < count(value.getType()); ++e)
          if (resolve({value, e}) == row.maximum) return fail("maximum escapes through another state");
    llvm::SmallPtrSet<Operation *, 16> alphaUsers;
    for (auto scale : row.scales) {
      alphaUsers.insert(scale);
      scales.insert(scale);
      for (int64_t e = 0; e < count(scale.getType()); ++e) {
        Slice s = resolve({scale.getInputs()[0], e});
        auto arg = dyn_cast<BlockArgument>(s.value);
        if (!arg || arg.getOwner() != proof.body || arg == proof.iv)
          return fail("rescaled input is not a loop state");
        auto type = registers(arg.getType());
        if (!type || type.getRank() != 2 || type.getDimSize(0) != int64_t(p.rows.size()) ||
            !type.getElementType().isF32() || s.element / type.getDimSize(1) != int64_t(r) ||
            !constant(proof.init[arg.getArgNumber() - 1])) return fail("invalid state row or initializer");
        auto &covered = states[s.value];
        covered.resize(type.getNumElements());
        if (covered[s.element]) return fail("state scaled twice");
        covered[s.element] = true;
      }
    }
    if (!a.usesOnly(row.alpha, alphaUsers, proof.body)) return fail("escaped alpha");
  }
  if (!a.usesOnly(base, baseUsers, proof.body) ||
      !a.usesOnly(reduction, maxima, proof.body)) return fail("escaped base or peak");
  for (auto &entry : states)
    if (!llvm::all_of(entry.second, [](bool v) { return v; }) ||
        !a.usesOnly(entry.first, scales, proof.body)) return fail("partial or escaped state");

  auto scoreType = registers(reduction.getSource().getType());
  if (scoreType.getRank() != 2) return fail("expected a score matrix");
  SmallVector<bool> scoreCoverage(scoreType.getNumElements(), false);
  llvm::SmallPtrSet<Operation *, 16> seedOps;
  for (auto [r, row] : llvm::enumerate(p.rows)) for (auto shift : row.shifts) {
    for (int64_t e = 0; e < count(shift.getType()); ++e) {
      Slice score = resolve({shift.getInputs()[0], e});
      bool found = false;
      for (int64_t col = 0; col < scoreType.getDimSize(1); ++col) {
        int64_t index = r * scoreType.getDimSize(1) + col;
        if (resolve({reduction.getSource(), index}) == score && !scoreCoverage[index]) {
          scoreCoverage[index] = true; found = true; break;
        }
      }
      if (!found) return fail("score reduction and shifts disagree");
      auto mma = score.value.getDefiningOp<GemmFragOp>();
      if (!mma) return fail("score is not GEMM (mask/bias unsupported)");
      // The computation layout must keep the four MMAC accumulator registers
      // on one row. Other MMA layouts need their own coordinate proof.
      while (mma) {
        auto type = mma.getResult().getType();
        auto layout = type.getLayout();
        if (type.getRegisterType().getShape() != ArrayRef<int64_t>({1, 4}) ||
            mma.getInstName() != "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3" ||
            mma.getInstConstraints() != "=v,v,v,0" ||
            !layout.getAs<DenseI64ArrayAttr>("warp_layout") ||
            layout.getAs<DenseI64ArrayAttr>("warp_layout").asArrayRef() != ArrayRef<int64_t>({16, 4}))
          return fail("unsupported QK MMA layout");
        gemms.insert(mma);
        Value acc = mma.getAcc();
        Slice previous = resolve({acc, score.element});
        if (auto prior = previous.value.getDefiningOp<GemmFragOp>()) {
          if (previous.element != score.element) return fail("permuted MMA accumulator");
          mma = prior;
          continue;
        }
        if (!constant(acc)) return fail("nonzero QK seed");
        // Unrolled K carries resolve to the original tile extraction. Keep
        // that extraction and every GEMM loop in the committed graph.
        auto extract = acc.getDefiningOp<ExtractFragOp>();
        if (!extract) return fail("QK seed is not a tile extraction");
        auto original = dyn_cast_or_null<ExtractFragOp>(a.original(extract));
        auto srcType = original ? registers(original.getSource().getType()) : VectorType();
        auto start = origins(extract.getIndices(), registers(extract.getType()));
        if (!original || !srcType || srcType.getRank() != 2 ||
            srcType.getDimSize(0) != int64_t(p.rows.size()) || !start || (*start)[0] != int64_t(r) ||
            !a.observed.count(original.getSource()) ||
            !constant(a.value(original.getSource()))) return fail("incompatible seed row");
        if (seedOps.insert(original).second) p.seeds.push_back(original);
        break;
      }
    }
  }
  if (!llvm::all_of(scoreCoverage, [](bool v) { return v; })) return fail("partial score coverage");
  llvm::SmallPtrSet<Operation *, 32> scoreUsers(gemms.begin(), gemms.end());
  scoreUsers.insert(shifts.begin(), shifts.end());
  scoreUsers.insert(reduction);
  for (Operation *mma : gemms)
    if (!a.usesOnly(mma->getResult(0), scoreUsers, proof.body)) return fail("escaped QK accumulator");
  for (Operation *mma : gemms)
    if (constant(mma->getOperand(2)) &&
        !a.usesOnly(mma->getOperand(2), gemms, proof.body)) return fail("escaped QK seed");
  bool allSeedUses = true;
  a.copy->walk([&](ExtractFragOp extract) {
    if (seedOps.contains(a.original(extract)) &&
        !a.usesOnly(extract, gemms, proof.body)) allSeedUses = false;
  });
  if (!allSeedUses) return fail("seed extraction also feeds another computation");

  // Rebuild complete pure assembly results, not individual dynamic fragments.
  // This keeps the original GEMM/load loop nests available to scheduling.
  llvm::SmallPtrSet<Operation *, 16> roots;
  for (Operation *op : shifts) roots.insert(rootIn(a.original(op), p.loop.body));
  for (Operation *op : scales) roots.insert(rootIn(a.original(op), p.loop.body));
  unsigned scoreRoots = 0;
  for (Operation &root : *p.loop.body) {
    if (!roots.contains(&root)) continue;
    if (root.getNumResults() != 1) return fail("multiple assembly results");
    bool pure = true;
    root.walk([&](Operation *nested) {
      if (isa<affine::AffineForOp, affine::AffineYieldOp>(nested)) return;
      if (!isOrdinaryRegisterComputation(nested)) pure = false;
    });
    if (!pure) return fail("non-register rescale assembly");
    auto type = registers(root.getResult(0).getType());
    if (!type || type.getRank() != 2 || type.getDimSize(0) != int64_t(p.rows.size()))
      return fail("partial assembly result");
    Value proofValue = a.value(root.getResult(0)), oldState;
    bool scoreRoot = false;
    for (int64_t i = 0; i < type.getNumElements(); ++i) {
      Slice s = resolve({proofValue, i});
      Operation *op = s.value.getDefiningOp();
      if (shifts.contains(op)) {
        if (oldState || type != scoreType ||
            !(resolve({op->getOperand(0), s.element}) == resolve({reduction.getSource(), i})))
          return fail("incomplete score assembly");
        scoreRoot = true;
      } else if (scales.contains(op)) {
        Slice state = resolve({op->getOperand(0), s.element});
        if (scoreRoot || state.element != i || (oldState && oldState != state.value) ||
            registers(state.value.getType()) != type) return fail("incomplete state assembly");
        oldState = state.value;
      } else return fail("mixed assembly result");
    }
    if (scoreRoot) ++scoreRoots;
    if (oldState) oldState = p.loop.args[cast<BlockArgument>(oldState).getArgNumber() - 1];
    p.outputs.push_back({&root, oldState});
    if (!p.split) p.split = &root;
  }
  if (scoreRoots != 1 || p.outputs.size() != states.size() + 1)
    return fail("incomplete output tiles");
  p.peak = cast<ReduceFragOp>(a.original(reduction));
  DominanceInfo dom(p.loop.op->getParentOp());
  if (!dom.properlyDominates(p.peak.getResult(), p.split)) return fail("peak unavailable before rescale");
  for (auto seed : p.seeds)
    if (!dom.dominates(p.loop.args[p.baseIndex], seed)) return fail("base unavailable at seed");
  return true;
}

FragmentType localType(VectorType type) {
  return FragmentType::get(type.getContext(), type, DictionaryAttr::get(type.getContext()));
}
Value pack(OpBuilder &b, Location loc, Value value) {
  if (isa<FragmentType>(value.getType())) return value;
  return b.create<PackFragOp>(loc, localType(cast<VectorType>(value.getType())), value);
}
Value asType(OpBuilder &b, Location loc, Value value, Type type) {
  if (value.getType() == type) return value;
  if (auto f = dyn_cast<FragmentType>(value.getType()))
    value = b.create<UnpackFragOp>(loc, f.getRegisterType(), value);
  if (isa<FragmentType>(type)) value = b.create<PackFragOp>(loc, type, value);
  return value;
}
Value compute(OpBuilder &b, Location loc, StringRef kind, Value lhs, Value rhs = {}) {
  SmallVector<Value> inputs{lhs};
  if (rhs) inputs.push_back(rhs);
  return b.create<ElementwiseFragOp>(loc, lhs.getType(), kind, inputs);
}
Value broadcast(OpBuilder &b, Location loc, Value value, Type type) {
  return b.create<BroadcastFragOp>(loc, type, value);
}
void apply(Plan &p, float threshold) {
  auto loc = p.loop.op->getLoc();
  Value base = p.loop.args[p.baseIndex];
  OpBuilder b(p.loop.op);
  auto baseType = registers(base.getType());
  Value zero = b.create<arith::ConstantOp>(loc, baseType, b.getZeroAttr(baseType));
  p.loop.setInit(p.baseIndex, asType(b, loc, zero, base.getType()));
  // Seed each original first-QK tile extraction. Only its input changes;
  // subsequent K chunks still consume their predecessor accumulator.
  for (auto seed : p.seeds) {
    b.setInsertionPoint(rootIn(seed, p.loop.body));
    Value negativeBase = pack(b, loc, base);
    Value tile = broadcast(b, loc, negativeBase, localType(cast<VectorType>(seed.getSource().getType())));
    seed.getSourceMutable().assign(asType(b, loc, tile, seed.getSource().getType()));
  }
  b.setInsertionPoint(p.split);
  Value peakVector = asType(b, loc, p.peak, baseType);
  Value peak;
  for (int64_t row = 0; row < baseType.getDimSize(0); ++row) {
    Value scalar = b.create<vector::ExtractOp>(loc, peakVector, ArrayRef<int64_t>{row, 0});
    peak = peak ? b.create<arith::MaxNumFOp>(loc, peak, scalar) : scalar;
  }
  Value limit = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(threshold));
  Value over = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, peak, limit);
  Value any = b.create<WaveAnyOp>(loc, b.getI1Type(), over, 64);
  Value zeroIV = b.create<arith::ConstantOp>(loc, p.loop.iv.getType(), b.getZeroAttr(p.loop.iv.getType()));
  Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, p.loop.iv, zeroIV);
  Value take = b.create<arith::OrIOp>(loc, first, any);
  SmallVector<Value> fast;
  SmallVector<Type> types;
  for (auto &output : p.outputs) {
    Type type = output.root->getResult(0).getType();
    fast.push_back(asType(b, loc, output.oldState ? output.oldState : p.peak.getSource(), type));
    types.push_back(type);
  }
  fast.push_back(base);
  types.push_back(base.getType());
  auto branch = b.create<scf::IfOp>(loc, types, take, true);
  branch->setAttr(Marker, b.getUnitAttr());
  b.setInsertionPointToStart(&branch.getThenRegion().front());
  Value z = pack(b, loc, zero);
  Value positive = compute(b, loc, "max", p.peak, z);
  Value deltaVector = b.create<arith::SelectOp>(loc, first, peakVector, asType(b, loc, positive, baseType));
  Value delta = pack(b, loc, deltaVector);
  Value negative = compute(b, loc, "sub", z, delta);
  Value exponent = compute(b, loc, "exp2", negative);
  Value one = b.create<arith::ConstantOp>(loc, baseType,
      DenseElementsAttr::get(baseType, b.getF32FloatAttr(1)));
  // On the first tile old O/L are zero: avoid 0*inf for negative peaks.
  Value alphaVector = b.create<arith::SelectOp>(loc, first, one, asType(b, loc, exponent, baseType));
  Value alpha = pack(b, loc, alphaVector);
  SmallVector<Value> slow;
  for (auto &output : p.outputs) {
    Value input = pack(b, loc, output.oldState ? output.oldState : p.peak.getSource());
    Value factor = broadcast(b, loc, output.oldState ? alpha : delta, input.getType());
    Value updated = compute(b, loc, output.oldState ? "mul" : "sub", input, factor);
    slow.push_back(asType(b, loc, updated, output.root->getResult(0).getType()));
  }
  Value next = compute(b, loc, "sub", pack(b, loc, base), delta);
  slow.push_back(asType(b, loc, next, base.getType()));
  b.create<scf::YieldOp>(loc, slow);
  b.setInsertionPointToStart(&branch.getElseRegion().front());
  b.create<scf::YieldOp>(loc, fast);
  for (auto [i, output] : llvm::enumerate(p.outputs))
    output.root->getResult(0).replaceAllUsesWith(branch.getResult(i));
  p.loop.body->getTerminator()->setOperand(p.baseIndex, branch.getResults().back());
  // Erase dead pure assembly, including the former max/alpha loops. Never
  // erase or speculate a collective, memory operation, or synchronization.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    p.loop.op->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (!op->use_empty() || op->hasTrait<OpTrait::IsTerminator>()) return;
      bool pure = true;
      op->walk([&](Operation *nested) {
        if (isa<affine::AffineForOp, affine::AffineYieldOp, scf::YieldOp>(nested)) return;
        if (!isOrdinaryRegisterComputation(nested)) pure = false;
      });
      if (pure) dead.push_back(op);
    });
    // Parents are after children; erase only leaves on this sweep.
    for (Operation *op : dead) if (op->use_empty()) { op->erase(); changed = true; }
  }
}

class FragmentLazySoftmaxBasePass
    : public PassWrapper<FragmentLazySoftmaxBasePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FragmentLazySoftmaxBasePass)
  FragmentLazySoftmaxBasePass() = default;
  FragmentLazySoftmaxBasePass(const FragmentLazySoftmaxBasePass &other) : PassWrapper(other) {}
  FragmentLazySoftmaxBasePass(float t, unsigned wave) { threshold = t; waveSize = wave; }
  Option<float> threshold{*this, "threshold", llvm::cl::init(8.0f)};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(64)};
  StringRef getArgument() const final { return "frisk-fragment-lazy-softmax-base"; }
  StringRef getDescription() const final { return "Delay online-softmax rebasing before fragment scheduling"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<FriskDialect, arith::ArithDialect, affine::AffineDialect,
             scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto function = getOperation();
    if (!std::isfinite(threshold) || threshold <= 0 || threshold > 15 || waveSize != 64) {
      function.emitError("lazy softmax requires threshold in (0,15] and wave-size=64");
      return signalPassFailure();
    }
    auto threads = function->getAttrOfType<IntegerAttr>("thread_num");
    if (function.empty() || function->hasAttr(Marker) || mask_add_bypass::isProtected(function) ||
        !threads || threads.getInt() <= 0 || threads.getInt() > 1024 || threads.getInt() % 64) return;
    for (NamedAttribute attr : function->getAttrs()) {
      std::string text;
      llvm::raw_string_ostream os(text); attr.getValue().print(os);
      if (StringRef(text).contains("+wavefrontsize32") || StringRef(text).contains("-wavefrontsize64")) return;
    }
    // A failed proof must leave the input unchanged, including normalization.
    OwningOpRef<func::FuncOp> working(cast<func::FuncOp>(function->clone()));
    if (failed(normalizeTypedFragmentIR(*working))) return signalPassFailure();
    Analysis analysis(*working);
    if (!analysis.expand()) return;
    for (Operation &op : working->getBody().front()) {
      auto loop = Loop::get(&op);
      if (!loop || !uniform(*loop)) continue;
      if (!llvm::any_of(*loop->body, [](Operation &nested) {
            auto r = dyn_cast<ReduceFragOp>(nested);
            return r && r.getKind() == "max";
          })) continue;
      Plan plan{*loop};
      std::string why;
      if (!match(plan, analysis, why)) {
        llvm::errs() << "Fragment lazy softmax base: skipped, " << why << '\n';
        continue;
      }
      apply(plan, threshold);
      function.getBody().takeBody(working->getBody());
      function->setAttr(Marker, StringAttr::get(function.getContext(), "fragment-v1"));
      llvm::errs() << "Fragment lazy softmax base: rows=" << plan.rows.size()
                   << ", seed-extractions=" << plan.seeds.size()
                   << ", scaled-state=" << plan.outputs.size() - 1 << ", threshold=" << threshold << '\n';
      break;
    }
  }
};
} // namespace
std::unique_ptr<Pass> createFragmentLazySoftmaxBasePass(float threshold, unsigned waveSize) {
  return std::make_unique<FragmentLazySoftmaxBasePass>(threshold, waveSize);
}
} // namespace mlir::frisk
