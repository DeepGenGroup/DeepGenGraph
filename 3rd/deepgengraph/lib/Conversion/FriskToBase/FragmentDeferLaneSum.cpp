// Move a linear lane-local recurrence ahead of its collective sum. Analysis
// specializes only a disposable copy: the scheduled graph retains its fragment
// loops, and the actual rewrite consists of a local reduction and an exit sum.
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
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <set>

namespace mlir::frisk {
namespace {
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
bool zero(Slice s) {
  s = resolve(s);
  Attribute attr;
  if (!matchPattern(s.value, m_Constant(&attr))) return false;
  if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    attr = dense.getValues<Attribute>()[s.element];
  auto f = dyn_cast<FloatAttr>(attr);
  return f && f.getValue().isZero();
}
struct Loop {
  Operation *op;
  Block *body;
  ValueRange init;
  Block::BlockArgListType args;
  ValueRange yielded;
  static std::optional<Loop> get(Operation *op) {
    if (auto f = dyn_cast<affine::AffineForOp>(op))
      return Loop{op, f.getBody(), f.getInits(), f.getRegionIterArgs(),
                  f.getBody()->getTerminator()->getOperands()};
    if (auto f = dyn_cast<scf::ForOp>(op))
      return Loop{op, f.getBody(), f.getInitArgs(), f.getRegionIterArgs(),
                  f.getBody()->getTerminator()->getOperands()};
    return {};
  }
};
unsigned groupMask(ReduceFragOp reduce) {
  return (reduce.getLaneExtent() - 1) * reduce.getLaneStride();
}
class Proof {
  func::FuncOp function;
  unsigned wave, threads;
  std::set<Slice> active;
public:
  Proof(func::FuncOp f, unsigned w, unsigned t) : function(f), wave(w), threads(t) {}
  bool uniform(Slice s, unsigned group, bool collectives = true, unsigned depth = 0) {
    if (depth > 256) return false;
    s = resolve(s);
    if (matchPattern(s.value, m_Constant())) return true;
    if (s.value.getType().isIntOrIndex()) {
      SmallVector<Attribute> values;
      for (unsigned tid = 0; tid < threads; ++tid) {
        auto attr = evaluate(s.value, tid);
        if (!attr) { values.clear(); break; }
        values.push_back(attr);
      }
      if (!values.empty()) {
        for (unsigned tid = 0; tid < threads; ++tid)
          for (unsigned bit = 1; bit < wave; bit *= 2)
            if ((group & bit) && values[tid] != values[tid ^ bit]) return false;
        return true;
      }
    }
    if (auto arg = dyn_cast<BlockArgument>(s.value)) {
      if (arg.getOwner() == &function.getBody().front()) return true;
      auto loop = Loop::get(arg.getOwner()->getParentOp());
      if (!loop) return false;
      if (arg.getArgNumber() == 0) return bounds(*loop, group, depth + 1);
      if (!active.insert(s).second) return true;
      unsigned i = arg.getArgNumber() - 1;
      bool ok = uniform({loop->init[i], s.element}, group, collectives, depth + 1) &&
                uniform({loop->yielded[i], s.element}, group, collectives, depth + 1);
      active.erase(s);
      return ok;
    }
    auto *op = s.value.getDefiningOp();
    if (!op) return false;
    if (isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op)) return true;
    if (auto reduce = dyn_cast<ReduceFragOp>(op))
      return collectives && reduce.getWarpThreads() == wave &&
             (groupMask(reduce) & group) == group;
    StringRef dialect = op->getName().getDialectNamespace();
    if ((!isa<ElementwiseFragOp>(op) && dialect != "arith" && dialect != "math" &&
         !isa<affine::AffineApplyOp>(op)) || !isMemoryEffectFree(op)) return false;
    for (Value v : op->getOperands())
      if (!uniform({v, registers(v.getType()) ? s.element : 0}, group, collectives, depth + 1))
        return false;
    return true;
  }
  bool bounds(Loop loop, unsigned group, unsigned depth = 0) {
    if (auto f = dyn_cast<affine::AffineForOp>(loop.op)) {
      for (Value v : f.getControlOperands())
        if (!uniform({v}, group, false, depth + 1)) return false;
    } else {
      auto scfLoop = cast<scf::ForOp>(loop.op);
      for (Value v : {scfLoop.getLowerBound(), scfLoop.getUpperBound(), scfLoop.getStep()})
        if (!uniform({v}, group, false, depth + 1)) return false;
    }
    return true;
  }
  bool control(Loop loop) {
    for (Operation *op = loop.op; op != function; op = op->getParentOp()) {
      if (auto f = Loop::get(op)) { if (!bounds(*f, wave - 1)) return false; }
      else if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (!uniform({branch.getCondition()}, wave - 1, false)) return false;
      } else if (auto branch = dyn_cast<affine::AffineIfOp>(op)) {
        for (Value v : branch.getOperands())
          if (!uniform({v}, wave - 1, false)) return false;
      } else return false;
    }
    return true;
  }
};
// Follow every affected component, including dead/overwritten carrier fields.
// Arithmetic and side effects are endpoints, never transparent aliases.
bool flowsTo(Slice s, Operation *end, unsigned operand, int64_t element,
             unsigned &reads, unsigned &budget, unsigned depth = 0) {
  if (depth > 512 || !budget) return false;
  --budget;
  for (OpOperand &use : s.value.getUses()) {
    Operation *user = use.getOwner();
    if (user == end && use.getOperandNumber() == operand && s.element == element) {
      ++reads; continue;
    }
    if (user->getNumResults() != 1) return false;
    Value out = user->getResult(0);
    int64_t n = count(out.getType());
    if (!n || n > 128) return false;
    for (int64_t i = 0; i < n; ++i) {
      auto input = view({out, i});
      if (!input) return false;
      if (*input == s && !flowsTo({out, i}, end, operand, element, reads, budget, depth + 1))
        return false;
    }
  }
  return true;
}
bool onlyFlows(Slice s, Operation *end, unsigned operand, int64_t element) {
  unsigned reads = 0, budget = 65536;
  return flowsTo(s, end, operand, element, reads, budget) && reads == 1;
}
bool kind(Operation *op, StringRef name) {
  if (!op) return false;
  if (auto elementwise = dyn_cast<ElementwiseFragOp>(op)) return elementwise.getKind() == name;
  return name == "add" ? isa<arith::AddFOp>(op) : isa<arith::MulFOp>(op);
}

// No unknown execution-mask changes. Ordinary reads may supply x, but do not
// establish uniformity of alpha. mask_frag's predicate region is scalar-only.
bool safeFunction(func::FuncOp function, unsigned wave) {
  if (mask_add_bypass::isProtected(function)) return false;
  auto conflict = [&](StringRef features) {
    SmallVector<StringRef> tokens; features.split(tokens, ',');
    return llvm::is_contained(tokens, "+wavefrontsize" + std::to_string(wave == 64 ? 32 : 64)) ||
           llvm::is_contained(tokens, "-wavefrontsize" + std::to_string(wave));
  };
  for (StringRef key : {"target-features", "llvm.target_features"})
    if (auto attr = function->getAttrOfType<StringAttr>(key); attr && conflict(attr.getValue()))
      return false;
  for (StringRef key : {"passthrough", "llvm.passthrough"})
    if (auto attrs = function->getAttrOfType<ArrayAttr>(key))
      for (Attribute attr : attrs)
        if (auto pair = dyn_cast<ArrayAttr>(attr); pair && pair.size() == 2)
          if (auto name = dyn_cast<StringAttr>(pair[0]); name && name.getValue() == "target-features")
            if (auto f = dyn_cast<StringAttr>(pair[1]); f && conflict(f.getValue())) return false;
  bool safe = true;
  function.walk([&](Operation *op) {
    if (isa<CallOpInterface>(op)) safe = false;
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "math" && dialect != "vector" &&
        dialect != "memref" && dialect != "affine" && dialect != "scf" && dialect != "func") {
      StringRef name = op->getName().getStringRef();
      if (!isa<LoadFragOp, StoreFragOp, ExtractFragOp, InsertFragOp, PackFragOp,
               UnpackFragOp, GemmFragOp, ElementwiseFragOp, BroadcastFragOp,
               ReduceFragOp, LaneReduceFragOp, ConvertLayoutFragOp, MaskFragOp,
               MaskYieldOp, ToThreadTileOp, FromThreadTileOp>(op) &&
          name != "gpu.thread_id" && name != "gpu.block_id" && name != "gpu.block_dim" &&
          name != "gpu.grid_dim" && name != "gpu.lane_id" && name != "gpu.barrier" &&
          name != "frisk.sync_threads_in_block" &&
          name != "frisk.async_copy_gm_to_shm" && name != "frisk.wait_buffer_ready" &&
          name != "frisk.async_task_completed" && name != "frisk.read_matrix_from_shm" &&
          name != "frisk.buffer_view") safe = false;
    }
    if (op->getNumRegions() && op != function &&
        !isa<affine::AffineForOp, scf::ForOp, affine::AffineIfOp, scf::IfOp, MaskFragOp>(op))
      safe = false;
  });
  return safe;
}

// Unroll pure fragment assembly in the analysis copy only. Do not duplicate
// loads, MMA, collectives, masks, unknown operations, or the recurrence itself.
void specialize(func::FuncOp function) {
  SmallVector<affine::AffineForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) { loops.push_back(loop); });
  uint64_t budget = 65536;
  for (auto loop : loops) {
    if (!loop->hasAttr("frisk.fragment_loop") || !loop.hasConstantBounds()) continue;
    int64_t span = loop.getConstantUpperBound() - loop.getConstantLowerBound();
    if (span <= 0) continue;
    uint64_t trips = llvm::divideCeil(uint64_t(span), uint64_t(loop.getStepAsInt()));
    if (trips > 64) continue;
    bool pure = true;
    uint64_t size = 0;
    for (Operation &op : *loop.getBody()) {
      ++size;
      StringRef dialect = op.getName().getDialectNamespace();
      if (!isa<ExtractFragOp, InsertFragOp, PackFragOp, UnpackFragOp,
               ElementwiseFragOp, BroadcastFragOp, affine::AffineApplyOp,
               affine::AffineYieldOp>(op) && dialect != "arith" && dialect != "math" &&
          dialect != "vector") pure = false;
      if (op.getNumRegions() || !isMemoryEffectFree(&op)) pure = false;
    }
    if (!pure || size > budget / trips) continue;
    budget -= size * trips;
    (void)affine::loopUnrollFull(loop);
  }
}

struct Candidate {
  ReduceFragOp reduce;
  SmallVector<Operation *> arithmetic;
};
std::optional<Candidate> match(Loop loop, unsigned index, Proof &proof, unsigned wave) {
  Value state = loop.args[index];
  auto type = registers(state.getType());
  if (!type || !type.getElementType().isF32() || type.isScalable() ||
      type.getNumElements() > 128) return {};
  Candidate candidate;
  for (int64_t j = 0; j < type.getNumElements(); ++j) {
    if (!zero({loop.init[index], j})) return {};
    Slice back = resolve({loop.yielded[index], j});
    Operation *update = back.value.getDefiningOp();
    if (!kind(update, "add") || update->getBlock() != loop.body) return {};
    bool matched = false;
    for (unsigned side = 0; side < 2; ++side) {
      Slice sum = resolve({update->getOperand(side), back.element});
      auto reduce = sum.value.getDefiningOp<ReduceFragOp>();
      if (!reduce || reduce.getKind() != "add" || reduce.getLaneExtent() <= 1 ||
          reduce.getWarpThreads() != wave || reduce->getBlock() != loop.body ||
          reduce.getType().getRegisterType() != type || sum.element != j ||
          (candidate.reduce && candidate.reduce != reduce)) continue;
      Slice old = resolve({update->getOperand(1 - side), back.element});
      Operation *scale = nullptr;
      unsigned stateOperand = 1 - side;
      int64_t stateElement = back.element;
      if (auto *mul = old.value.getDefiningOp(); kind(mul, "mul") && mul->getBlock() == loop.body) {
        for (unsigned operand = 0; operand < 2; ++operand)
          if (resolve({mul->getOperand(operand), old.element}) == Slice{state, j} &&
              proof.uniform({mul->getOperand(1 - operand), old.element}, groupMask(reduce))) {
            scale = mul; stateOperand = operand; stateElement = old.element; break;
          }
        if (!scale || !onlyFlows(old, update, 1 - side, back.element)) continue;
      } else if (!(old == Slice{state, j})) continue;
      if (!onlyFlows({state, j}, scale ? scale : update, stateOperand, stateElement) ||
          !onlyFlows(sum, update, side, back.element) ||
          !onlyFlows(back, loop.body->getTerminator(), index, j)) continue;
      candidate.reduce = reduce;
      candidate.arithmetic.push_back(update);
      if (scale) candidate.arithmetic.push_back(scale);
      matched = true;
      break;
    }
    if (!matched) return {};
  }
  return candidate;
}

class FragmentDeferLaneSumPass
    : public PassWrapper<FragmentDeferLaneSumPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FragmentDeferLaneSumPass)
  FragmentDeferLaneSumPass(bool approx = false, unsigned wave = 0) {
    approximate = approx; waveSize = wave;
  }
  FragmentDeferLaneSumPass(const FragmentDeferLaneSumPass &other) : PassWrapper(other) {}
  Option<bool> approximate{*this, "approximate", llvm::cl::init(false)};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(0)};
  StringRef getArgument() const final { return "frisk-fragment-defer-lane-sum"; }
  StringRef getDescription() const final { return "Defer lane sums in the fragment work graph before scheduling"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FriskDialect, arith::ArithDialect, affine::AffineDialect>();
  }
  void runOnOperation() override {
    auto function = getOperation();
    if (waveSize && waveSize != 32 && waveSize != 64) {
      function.emitError("wave-size must be 0, 32 or 64");
      return signalPassFailure();
    }
    auto threads = function->getAttrOfType<IntegerAttr>("thread_num");
    if (!approximate || !waveSize || !threads || threads.getInt() <= 0 ||
        threads.getInt() > 1024 || threads.getInt() % waveSize ||
        function.empty() || !safeFunction(function, waveSize)) return;

    OwningOpRef<func::FuncOp> shadow(cast<func::FuncOp>(function->clone()));
    SmallVector<Operation *> originals, copies;
    function.walk([&](Operation *op) { originals.push_back(op); });
    shadow->walk([&](Operation *op) { copies.push_back(op); });
    DenseMap<Operation *, Operation *> originalFor;
    SmallVector<Operation *> candidates;
    for (auto [id, pair] : llvm::enumerate(llvm::zip(originals, copies))) {
      auto [original, copy] = pair;
      // Unrolling clones this proof-only origin, allowing us to clear FP
      // assumptions on exactly the rewritten arithmetic in the original IR.
      if (isa<ElementwiseFragOp, arith::AddFOp, arith::MulFOp>(copy))
        copy->setAttr("frisk.defer_analysis_origin", IntegerAttr::get(
            IntegerType::get(function.getContext(), 64), id));
      if (isa<ReduceFragOp>(copy)) originalFor[copy] = original;
      if (auto loop = Loop::get(copy)) {
        bool hasReduction = llvm::any_of(*loop->body, [](Operation &op) { return isa<ReduceFragOp>(op); });
        if (hasReduction) { candidates.push_back(copy); originalFor[copy] = original; }
      }
    }
    specialize(*shadow);
    Proof proof(*shadow, waveSize, threads.getInt());
    unsigned states = 0, shuffles = 0;
    for (Operation *copy : candidates) {
      auto loop = *Loop::get(copy);
      if (!proof.control(loop)) continue;
      for (unsigned i = 0; i < loop.args.size(); ++i) {
        auto candidate = match(loop, i, proof, waveSize);
        if (!candidate) continue;
        auto reduce = candidate->reduce;
        auto original = cast<ReduceFragOp>(originalFor.lookup(reduce));
        // Also update the proof copy, so later candidates cannot rely on a
        // collective or a uniform state that this rewrite has made partial.
        auto originalLoop = *Loop::get(originalFor.lookup(copy));
        Value result = originalLoop.op->getResult(i);
        OpBuilder b(originalLoop.op);
        b.setInsertionPointAfter(originalLoop.op);
        auto loc = originalLoop.op->getLoc();
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : result.getUses()) uses.push_back(&use);
        auto fragment = dyn_cast<FragmentType>(result.getType());
        Value partial = result;
        if (!fragment) {
          fragment = FragmentType::get(function.getContext(), registers(result.getType()), b.getDictionaryAttr({}));
          partial = b.create<PackFragOp>(loc, fragment, result);
        }
        Value final = b.create<LaneReduceFragOp>(loc, fragment, partial,
            original.getLaneExtent(), original.getLaneStride(), original.getWarpThreads());
        if (!isa<FragmentType>(result.getType()))
          final = b.create<UnpackFragOp>(loc, result.getType(), final);
        for (OpOperand *use : uses) use->set(final);
        unsigned elements = count(result.getType());
        states += elements;
        for (int64_t lanes = original.getLaneExtent(); lanes > 1; lanes >>= 1)
          shuffles += elements;
        original.setLaneExtent(1);
        reduce.setLaneExtent(1);
        for (Operation *arithmetic : candidate->arithmetic) {
          auto origin = arithmetic->getAttrOfType<IntegerAttr>("frisk.defer_analysis_origin");
          Operation *op = originals[origin.getInt()];
          if (isa<arith::AddFOp, arith::MulFOp>(op))
            op->setAttr("fastmath", arith::FastMathFlagsAttr::get(
                op->getContext(), arith::FastMathFlags::none));
          else if (isa<ElementwiseFragOp>(op)) op->removeAttr("fastmath");
        }
      }
    }
    llvm::errs() << "DeferLaneSum: " << function.getSymName() << ": " << states
                 << " states, " << shuffles << " loop shuffles deferred (approx)\n";
  }
};
} // namespace
std::unique_ptr<Pass> createFragmentDeferLaneSumPass(bool approximate, unsigned waveSize) {
  return std::make_unique<FragmentDeferLaneSumPass>(approximate, waveSize);
}
} // namespace mlir::frisk
