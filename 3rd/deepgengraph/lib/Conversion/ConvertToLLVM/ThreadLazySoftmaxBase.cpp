// Delay online-softmax rebasing in structured ThreadIR. N=-B seeds QK MMA;
// only the first tile or a wave with peak(S-B)>T rescales scores and old O/L.
// Run after scheduling: moving exp2/PV or LDS operations across this region
// would change the established schedule. The region contains only register
// arithmetic; all collectives, memory accesses and synchronization stay outside.
#include "deepgengraph/Conversion/ConvertToLLVM/ThreadLazySoftmaxBase.h"
#include "ThreadScheduleUtils.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>
#include <functional>
#include <optional>

namespace mlir::frisk {
namespace {
using namespace thread_schedule;
constexpr StringLiteral Marker = "frisk.lazy_softmax_base";

int64_t elements(Type type) {
  if (auto v = dyn_cast<VectorType>(type))
    return v.isScalable() ? 0 : v.getNumElements();
  return type.isF32() ? 1 : 0;
}
struct Slice {
  Value value;
  int64_t index = 0;
  bool operator==(const Slice &s) const { return value == s.value && index == s.index; }
};
// Resolve register views, without relying on SSA names or layout attributes.
Slice resolve(Slice s) {
  for (unsigned n = 0; n < 128; ++n) {
    Operation *op = s.value.getDefiningOp();
    if (auto e = dyn_cast_or_null<vector::ExtractOp>(op)) {
      if (!e.getDynamicPosition().empty()) break;
      int64_t offset = 0;
      auto shape = e.getSourceVectorType().getShape();
      auto pos = e.getStaticPosition();
      for (unsigned i = 0; i < shape.size(); ++i)
        offset = offset * shape[i] + (i < pos.size() ? pos[i] : 0);
      s = {e.getSource(), offset + s.index};
    } else if (auto b = dyn_cast_or_null<vector::BroadcastOp>(op)) {
      if (elements(b.getSource().getType()) != 1) break;
      s = {b.getSource(), 0};
    } else if (auto f = dyn_cast_or_null<vector::FromElementsOp>(op)) {
      s = {f->getOperand(s.index), 0};
    } else if (auto c = dyn_cast_or_null<vector::ShapeCastOp>(op)) {
      s.value = c.getSource();
    } else if (auto i = dyn_cast_or_null<vector::InsertOp>(op)) {
      if (!i.getDynamicPosition().empty()) break;
      auto shape = i.getDestVectorType().getShape();
      auto pos = i.getStaticPosition();
      int64_t offset = 0;
      for (unsigned d = 0; d < shape.size(); ++d)
        offset = offset * shape[d] + (d < pos.size() ? pos[d] : 0);
      int64_t size = elements(i.getValueToStore().getType());
      s = s.index >= offset && s.index < offset + size
              ? Slice{i.getValueToStore(), s.index - offset} : Slice{i.getDest(), s.index};
    } else break;
  }
  return s;
}
bool splatOf(Value value, Slice expected) {
  int64_t count = elements(value.getType());
  if (!count) return false;
  for (int64_t i = 0; i < count; ++i)
    if (!(resolve({value, i}) == expected)) return false;
  return true;
}
bool constant(Value value, bool negativeInfinity = false) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr))) return false;
  auto matches = [&](Attribute a) {
    auto f = dyn_cast<FloatAttr>(a);
    return f && (negativeInfinity ? f.getValue().isNegInfinity() : f.getValue().isZero());
  };
  if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    return llvm::all_of(dense.getValues<Attribute>(), matches);
  return matches(attr);
}
bool view(Operation *op) {
  return isa<vector::ExtractOp, vector::BroadcastOp, vector::FromElementsOp,
             vector::ShapeCastOp, vector::InsertOp>(op) && pure(op);
}
struct Loop {
  Operation *op;
  Block *body;
  Value iv;
  ValueRange init;
  Block::BlockArgListType args;
  static std::optional<Loop> get(Operation *op) {
    if (auto f = dyn_cast<affine::AffineForOp>(op)) {
      if (!f.hasConstantLowerBound() || f.getConstantLowerBound() != 0 || f.getStepAsInt() != 1)
        return {};
      return Loop{op, f.getBody(), f.getInductionVar(), f.getInits(), f.getRegionIterArgs()};
    }
    if (auto f = dyn_cast<scf::ForOp>(op)) {
      APInt lower, step;
      if (!matchPattern(f.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
          !matchPattern(f.getStep(), m_ConstantInt(&step)) || !step.isOne()) return {};
      return Loop{op, f.getBody(), f.getInductionVar(), f.getInitArgs(), f.getRegionIterArgs()};
    }
    return {};
  }
  void setInit(unsigned index, Value value) {
    if (auto f = dyn_cast<affine::AffineForOp>(op)) f.getInitsMutable()[index].set(value);
    else cast<scf::ForOp>(op).getInitArgsMutable()[index].set(value);
  }
};
struct Row {
  Operation *maximum = nullptr;
  Value peak;
  arith::SubFOp diff;
  math::Exp2Op alpha;
  SmallVector<arith::SubFOp> shifts;
  SmallVector<arith::MulFOp> scales;
};
struct Plan {
  Loop loop;
  unsigned baseIndex = 0;
  SmallVector<Row> rows;
  llvm::MapVector<Operation *, unsigned> seeds;
  llvm::MapVector<Value, SmallVector<bool>> states;
  Operation *split = nullptr;
};

// All uses of a rewritten recurrence must reach its recognized consumers via
// register views. In particular, reject an escaped maximum or unscaled O/L use.
bool usesOnly(Value value, const llvm::SmallPtrSetImpl<Operation *> &allowed,
              Block *body) {
  SmallVector<Value> work{value};
  llvm::SmallPtrSet<Operation *, 32> seen;
  for (unsigned i = 0; i < work.size(); ++i)
    for (Operation *user : work[i].getUsers()) {
      if (allowed.contains(user)) continue;
      if (user->getBlock() != body || !view(user)) return false;
      if (seen.insert(user).second) llvm::append_range(work, user->getResults());
    }
  return true;
}

bool match(Plan &p, std::string &why) {
  auto fail = [&](StringRef reason) { why = reason.str(); return false; };
  auto &loop = p.loop;
  Operation *yield = loop.body->getTerminator();
  Value base;
  for (auto [i, init] : llvm::enumerate(loop.init)) {
    auto type = dyn_cast<VectorType>(init.getType());
    if (!type || type.isScalable() || type.getRank() != 2 || type.getDimSize(1) != 1 ||
        !type.getElementType().isF32() || type.getDimSize(0) > 8 || !constant(init, true)) continue;
    if (base) return fail("several running-max states");
    if (!loop.op->getResult(i).use_empty()) return fail("running max escapes the loop");
    p.baseIndex = i;
    base = loop.args[i];
    p.rows.resize(type.getDimSize(0));
  }
  if (!base) return fail("missing -inf running-max state");
  llvm::SmallPtrSet<Operation *, 16> baseUsers;
  for (auto [r, row] : llvm::enumerate(p.rows)) {
    Slice next = resolve({yield->getOperand(p.baseIndex), int64_t(r)});
    auto maximum = next.value.getDefiningOp<arith::MaxNumFOp>();
    if (!maximum || maximum->getBlock() != loop.body || elements(maximum.getType()) != 1)
      return fail("running max is not a row maxnum");
    Slice old{base, int64_t(r)};
    Value peak = splatOf(maximum.getRhs(), old) ? maximum.getLhs()
                 : splatOf(maximum.getLhs(), old) ? maximum.getRhs() : Value();
    if (!peak) return fail("maxnum does not consume the matching old row");
    Slice scalar = resolve({peak});
    if (!scalar.value.getType().isF32()) return fail("row peak is not a scalar");
    row.peak = scalar.value;
    row.maximum = maximum;
    if (!usesOnly(row.peak, llvm::SmallPtrSet<Operation *, 16>{maximum}, loop.body))
      return fail("row peak has an unknown use");
    baseUsers.insert(maximum);
    for (Operation &op : *loop.body) {
      auto sub = dyn_cast<arith::SubFOp>(op);
      if (!sub || !splatOf(sub.getRhs(), {maximum.getResult()})) continue;
      if (splatOf(sub.getLhs(), old)) {
        if (row.diff) return fail("several alpha exponents");
        row.diff = sub;
        baseUsers.insert(sub);
      } else row.shifts.push_back(sub);
    }
    if (!row.diff || row.shifts.empty()) return fail("missing score shifts or alpha exponent");
    for (Operation &op : *loop.body)
      if (auto exp = dyn_cast<math::Exp2Op>(op);
          exp && splatOf(exp.getOperand(), {row.diff.getResult()})) {
        if (row.alpha) return fail("several alpha exponentials");
        row.alpha = exp;
      }
    if (!row.alpha) return fail("missing alpha exp2");
    llvm::SmallPtrSet<Operation *, 16> maxUsers{yield, row.diff};
    for (auto shift : row.shifts) maxUsers.insert(shift);
    if (!usesOnly(maximum.getResult(), maxUsers, loop.body) ||
        !usesOnly(row.diff, llvm::SmallPtrSet<Operation *, 16>{row.alpha}, loop.body))
      return fail("maximum or exponent has an unknown use");
    for (auto [i, value] : llvm::enumerate(yield->getOperands()))
      if (i != p.baseIndex)
        for (int64_t e = 0; e < elements(value.getType()); ++e)
          if (resolve({value, e}) == Slice{maximum.getResult()})
            return fail("maximum escapes through another loop state");
    for (Operation &op : *loop.body)
      if (auto mul = dyn_cast<arith::MulFOp>(op);
          mul && splatOf(mul.getRhs(), {row.alpha.getResult()})) row.scales.push_back(mul);
    if (row.scales.empty()) return fail("alpha scales no state");
    llvm::SmallPtrSet<Operation *, 16> scaleUsers;
    for (auto scale : row.scales) {
      scaleUsers.insert(scale);
      for (int64_t e = 0; e < elements(scale.getType()); ++e) {
        Slice s = resolve({scale.getLhs(), e});
        auto arg = dyn_cast<BlockArgument>(s.value);
        if (!arg || arg.getOwner() != loop.body || arg == loop.iv)
          return fail("rescaled input is not a loop state");
        unsigned index = arg.getArgNumber() - 1;
        auto type = dyn_cast<VectorType>(arg.getType());
        if (!type || type.getRank() != 2 || type.getDimSize(0) != int64_t(p.rows.size()) ||
            !type.getElementType().isF32() || !constant(loop.init[index]) ||
            s.index / type.getDimSize(1) != int64_t(r))
          return fail("rescaled state is not zero-initialized or has a different row");
        auto &covered = p.states[s.value];
        covered.resize(type.getNumElements());
        if (covered[s.index]) return fail("state is scaled more than once");
        covered[s.index] = true;
      }
    }
    if (!usesOnly(row.alpha, scaleUsers, loop.body)) return fail("unknown alpha use");
  }
  if (!usesOnly(base, baseUsers, loop.body)) return fail("unknown running-max use");
  llvm::SmallPtrSet<Operation *, 32> allScales;
  for (auto &row : p.rows) for (auto scale : row.scales) allScales.insert(scale);
  for (auto &entry : p.states)
    if (!llvm::all_of(entry.second, [](bool b) { return b; }) ||
        !usesOnly(entry.first, allScales, loop.body)) return fail("partly scaled or escaped state");

  // Find the cut after every QK score and peak, before any rewritten consumer.
  for (auto &row : p.rows) {
    SmallVector<Operation *> consumers;
    for (auto op : row.shifts) consumers.push_back(op);
    for (auto op : row.scales) consumers.push_back(op);
    for (Operation *op : consumers)
      if (!p.split || op->isBeforeInBlock(p.split)) p.split = op;
  }
  DominanceInfo dom(loop.op->getParentOp());
  auto before = [&](Value value) { return dom.properlyDominates(value, p.split); };
  std::function<bool(Value)> canPrepare = [&](Value value) {
    if (before(value)) return true;
    Operation *op = value.getDefiningOp();
    return op && op->getBlock() == loop.body && view(op) &&
           llvm::all_of(op->getOperands(), canPrepare);
  };
  for (auto [r, row] : llvm::enumerate(p.rows)) {
    if (!before(row.peak)) return fail("peak unavailable at rescale");
    for (auto scale : row.scales)
      if (!canPrepare(scale.getLhs())) return fail("state view unavailable at rescale");
    for (auto shift : row.shifts) {
      if (!before(shift.getLhs())) return fail("score unavailable at rescale");
      Operation *mmaOp = shift.getLhs().getDefiningOp();
      if (!mma(mmaOp)) return fail("score is not a direct MMA result (mask/bias unsupported)");
      // Every interior accumulator has exactly one MMA consumer. No store,
      // mask, bias, or other externally visible use may observe its new seed.
      while (mma(mmaOp->getOperand(2).getDefiningOp())) {
        Operation *previous = mmaOp->getOperand(2).getDefiningOp();
        if (previous->getBlock() != loop.body || !previous->getResult(0).hasOneUse())
          return fail("MMA chain has extra users or escapes the loop");
        mmaOp = previous;
      }
      if (mmaOp->getBlock() != loop.body || !constant(mmaOp->getOperand(2)) ||
          !p.seeds.insert({mmaOp, r}).second) return fail("QK chain does not have a unique zero seed");
    }
  }
  // Changing score seeds is valid only when all remaining users compute row
  // peaks. The row reductions are already established by ThreadIR lowering.
  llvm::SmallPtrSet<Operation *, 32> shifts, peaks;
  for (auto &row : p.rows) {
    peaks.insert(row.peak.getDefiningOp());
    for (auto shift : row.shifts) shifts.insert(shift);
  }
  SmallVector<Value> work;
  llvm::SmallPtrSet<Operation *, 32> seen;
  for (auto &row : p.rows) for (auto shift : row.shifts) work.push_back(shift.getLhs());
  for (unsigned i = 0; i < work.size(); ++i)
    for (Operation *user : work[i].getUsers()) {
      if (shifts.contains(user) || peaks.contains(user)) continue;
      // ROCDL shuffle lowering uses bitcasts, address arithmetic and ds_bpermute.
      if (user->getBlock() != loop.body ||
          !(view(user) || isa<arith::MaxNumFOp, arith::BitcastOp, LLVM::BitcastOp,
                              ROCDL::DsBpermuteOp>(user)))
        return fail("score has a non-reduction user");
      if (seen.insert(user).second) llvm::append_range(work, user->getResults());
    }
  return true;
}

Value broadcast(OpBuilder &b, Location loc, Value scalar, Type type) {
  if (type == scalar.getType()) return scalar;
  return b.create<vector::BroadcastOp>(loc, type, scalar);
}
Value number(OpBuilder &b, Location loc, Type type, double value) {
  Attribute attr = b.getF32FloatAttr(value);
  if (auto v = dyn_cast<VectorType>(type)) attr = DenseElementsAttr::get(v, attr);
  return b.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
}

void apply(Plan &p, float threshold) {
  Loop &loop = p.loop;
  Value base = loop.args[p.baseIndex];
  auto loc = p.split->getLoc();
  OpBuilder b(p.split);
  // Materialize the original state views before the branch. Only transparent
  // register views move; the scheduled arithmetic/MMAs/memory ops never move.
  std::function<void(Value)> prepare = [&](Value value) {
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != loop.body || op->isBeforeInBlock(p.split)) return;
    assert(view(op));
    for (Value operand : op->getOperands()) prepare(operand);
    op->moveBefore(p.split);
  };
  SmallVector<Value> fast;
  for (auto &row : p.rows) for (auto shift : row.shifts) fast.push_back(shift.getLhs());
  for (auto &row : p.rows) for (auto scale : row.scales) {
    prepare(scale.getLhs());
    fast.push_back(scale.getLhs());
  }
  fast.push_back(base);

  // Seed each first QK MMA with -B for its row, before the chain starts.
  Operation *firstSeed = p.seeds.front().first;
  for (auto &entry : p.seeds)
    if (entry.first->isBeforeInBlock(firstSeed)) firstSeed = entry.first;
  b.setInsertionPoint(firstSeed);
  SmallVector<Value> seeds;
  for (unsigned r = 0; r < p.rows.size(); ++r) {
    Value scalar = b.create<vector::ExtractOp>(loc, base, ArrayRef<int64_t>{int64_t(r), 0});
    seeds.push_back(broadcast(b, loc, scalar, firstSeed->getOperand(2).getType()));
  }
  for (auto &entry : p.seeds) entry.first->setOperand(2, seeds[entry.second]);
  b.setInsertionPoint(loop.op);
  loop.setInit(p.baseIndex, number(b, loc, base.getType(), 0));

  b.setInsertionPoint(p.split);
  Value peak = p.rows.front().peak;
  for (unsigned r = 1; r < p.rows.size(); ++r)
    peak = b.create<arith::MaxNumFOp>(loc, peak, p.rows[r].peak);
  Value over = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, peak,
                                      number(b, loc, b.getF32Type(), threshold));
  Value ballot = b.create<ROCDL::BallotOp>(loc, b.getI64Type(), over);
  Value zero64 = b.create<arith::ConstantIntOp>(loc, 0, 64);
  Value any = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, ballot, zero64);
  Value zeroIV = b.create<arith::ConstantOp>(loc, loop.iv.getType(), b.getZeroAttr(loop.iv.getType()));
  Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, loop.iv, zeroIV);
  Value take = b.create<arith::OrIOp>(loc, first, any);
  SmallVector<Type> types;
  for (Value value : fast) types.push_back(value.getType());
  auto branch = b.create<scf::IfOp>(loc, types, take, true);
  branch->setAttr(Marker, b.getUnitAttr());
  b.setInsertionPointToStart(&branch.getThenRegion().front());
  SmallVector<Value> delta, alpha, nextBase;
  for (auto [r, row] : llvm::enumerate(p.rows)) {
    Value zero = number(b, loc, b.getF32Type(), 0);
    Value positive = b.create<arith::MaxNumFOp>(loc, row.peak, zero);
    Value d = b.create<arith::SelectOp>(loc, first, row.peak, positive);
    delta.push_back(d);
    Value negated = b.create<arith::SubFOp>(loc, zero, d);
    auto exp = b.create<math::Exp2Op>(loc, negated);
    exp->setAttrs(row.alpha->getAttrs());
    // The first old O/L are zero. Selecting 1 avoids 0*inf for negative peaks.
    alpha.push_back(b.create<arith::SelectOp>(loc, first, number(b, loc, b.getF32Type(), 1), exp));
    Value old = b.create<vector::ExtractOp>(loc, base, ArrayRef<int64_t>{int64_t(r), 0});
    nextBase.push_back(b.create<arith::SubFOp>(loc, old, d));
  }
  SmallVector<Value> slow;
  for (auto [r, row] : llvm::enumerate(p.rows)) for (auto shift : row.shifts) {
    auto sub = b.create<arith::SubFOp>(loc, shift.getLhs(), broadcast(b, loc, delta[r], shift.getType()));
    sub->setAttrs(shift->getAttrs());
    slow.push_back(sub);
  }
  for (auto [r, row] : llvm::enumerate(p.rows)) for (auto scale : row.scales) {
    auto mul = b.create<arith::MulFOp>(loc, scale.getLhs(), broadcast(b, loc, alpha[r], scale.getType()));
    mul->setAttrs(scale->getAttrs());
    slow.push_back(mul);
  }
  slow.push_back(b.create<vector::FromElementsOp>(loc, base.getType(), nextBase));
  b.create<scf::YieldOp>(loc, slow);
  b.setInsertionPointToStart(&branch.getElseRegion().front());
  b.create<scf::YieldOp>(loc, fast);

  unsigned index = 0;
  for (auto &row : p.rows) for (auto shift : row.shifts) {
    shift.replaceAllUsesWith(branch.getResult(index++));
    shift.erase();
  }
  for (auto &row : p.rows) for (auto scale : row.scales) {
    scale.replaceAllUsesWith(branch.getResult(index++));
    scale.erase();
  }
  loop.body->getTerminator()->setOperand(p.baseIndex, branch.getResult(index));
  // Delete only now-unused pure register operations, never collectives or
  // schedule barriers. In particular this removes the old max/alpha DAG.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*loop.body)))
      if (op.use_empty() && pure(&op)) { op.erase(); changed = true; }
  }
}

class ThreadLazySoftmaxBasePass
    : public PassWrapper<ThreadLazySoftmaxBasePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadLazySoftmaxBasePass)
  ThreadLazySoftmaxBasePass() = default;
  ThreadLazySoftmaxBasePass(const ThreadLazySoftmaxBasePass &other) : PassWrapper(other) {}
  ThreadLazySoftmaxBasePass(float t, unsigned wave) { threshold = t; waveSize = wave; }
  StringRef getArgument() const final { return "frisk-thread-lazy-softmax-base"; }
  StringRef getDescription() const final { return "Delay online-softmax rebasing in structured ThreadIR"; }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<arith::ArithDialect, math::MathDialect, vector::VectorDialect,
             scf::SCFDialect, ROCDL::ROCDLDialect>();
  }
  Option<float> threshold{*this, "threshold", llvm::cl::init(8.0f)};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(64)};
  void runOnOperation() final {
    auto f = getOperation();
    if (!std::isfinite(threshold) || threshold <= 0 || threshold > 15 || waveSize != 64) {
      f.emitError("lazy softmax requires threshold in (0,15] and wave-size=64");
      return signalPassFailure();
    }
    if (f.empty() || protectedFunction(f) || f->hasAttr(Marker)) return;
    auto threads = f->getAttrOfType<IntegerAttr>("thread_num");
    if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 || threads.getInt() % 64) return;
    // Reject explicit target ABI conflicts, including passthrough attributes.
    auto conflict = [](Attribute attr) {
      std::string text;
      llvm::raw_string_ostream os(text);
      attr.print(os);
      return StringRef(text).contains("+wavefrontsize32") || StringRef(text).contains("-wavefrontsize64");
    };
    for (NamedAttribute attr : f->getAttrs()) if (conflict(attr.getValue())) return;
    SmallVector<Operation *> candidates;
    for (Block &block : f.getBody()) for (Operation &op : block)
      if (Loop::get(&op)) candidates.push_back(&op);
    for (Operation *op : candidates) {
      Plan p{*Loop::get(op)};
      std::string why;
      if (!match(p, why)) {
        llvm::errs() << "Thread lazy softmax base: skipped, " << f.getSymName() << ": " << why << '\n';
        continue;
      }
      apply(p, threshold);
      markFunction(f, Marker, "thread-v1");
      llvm::errs() << "Thread lazy softmax base: rows=" << p.rows.size()
                   << ", seeded-accumulators=" << p.seeds.size()
                   << ", scaled-state=" << p.states.size() << ", threshold=" << threshold << '\n';
      break;
    }
  }
};
} // namespace

bool parseLazySoftmaxBaseOption(StringRef arg, bool &enabled, float &threshold, bool &consumed) {
  consumed = arg.starts_with("--lazy-softmax-base");
  if (!consumed) return true;
  if (!arg.consume_front("--lazy-softmax-base=")) return false;
  if (arg == "off" || arg == "0") { enabled = false; return true; }
  double value = 0;
  if (arg.getAsDouble(value) || !std::isfinite(value) || value <= 0 || value > 15) return false;
  enabled = true;
  threshold = float(value);
  return true;
}
std::unique_ptr<Pass> createThreadLazySoftmaxBasePass(float threshold, unsigned waveSize) {
  return std::make_unique<ThreadLazySoftmaxBasePass>(threshold, waveSize);
}
} // namespace mlir::frisk
