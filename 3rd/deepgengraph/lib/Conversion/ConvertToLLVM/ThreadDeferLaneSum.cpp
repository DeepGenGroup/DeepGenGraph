// Keep lane-local recurrences in structured thread IR, before GPU shuffles
// become target instructions. No kernel names, tensor layouts or SSA names
// participate in matching.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>

namespace mlir::frisk {
namespace {
// A scalar or one statically indexed element of a fixed vector.
struct Slice {
  Value value;
  int64_t element = -1;
  bool operator==(const Slice &s) const {
    return value == s.value && element == s.element;
  }
  bool operator<(const Slice &s) const {
    return value != s.value ? std::less<const void *>{}(
        value.getAsOpaquePointer(), s.value.getAsOpaquePointer()) : element < s.element;
  }
};
int64_t count(Type type) {
  if (auto v = dyn_cast<VectorType>(type))
    return v.isScalable() ? 0 : v.getNumElements();
  return 1;
}
Slice part(Value v, int64_t index) {
  return {v, isa<VectorType>(v.getType()) ? index : -1};
}
// Return the input slice for a transparent vector view. Used in both def-use
// directions, so escaped fields cannot silently become lane-local values.
std::optional<Slice> view(Slice s) {
  auto *op = s.value.getDefiningOp();
  if (!op) return {};
  if (auto e = dyn_cast<vector::ExtractOp>(op)) {
    if (!e.getDynamicPosition().empty()) return {};
    auto shape = e.getSourceVectorType().getShape();
    int64_t offset = 0;
    auto pos = e.getStaticPosition();
    for (unsigned i = 0; i < shape.size(); ++i)
      offset = offset * shape[i] + (i < pos.size() ? pos[i] : 0);
    return Slice{e.getSource(), offset + std::max<int64_t>(s.element, 0)};
  }
  if (auto b = dyn_cast<vector::BroadcastOp>(op)) {
    if (!isa<VectorType>(b.getSource().getType())) return Slice{b.getSource()};
  }
  if (auto f = dyn_cast<vector::FromElementsOp>(op)) {
    if (s.element >= 0 && s.element < int64_t(f->getNumOperands()))
      return Slice{f->getOperand(s.element)};
  }
  if (auto c = dyn_cast<vector::ShapeCastOp>(op))
    return Slice{c.getSource(), s.element};
  return {};
}
Slice resolve(Slice s) {
  for (unsigned i = 0; i < 128; ++i) {
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
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (s.element < 0) return false;
    attr = dense.getValues<Attribute>()[s.element];
  }
  if (auto f = dyn_cast<FloatAttr>(attr)) return f.getValue().isZero();
  if (auto i = dyn_cast<IntegerAttr>(attr)) return i.getValue().isZero();
  return false;
}
std::optional<int64_t> integer(Value v) {
  APInt i;
  if (!matchPattern(v, m_ConstantInt(&i)) || i.getBitWidth() > 64) return {};
  return i.getSExtValue();
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
struct Reduction {
  Value leaf;
  unsigned mask = 0;
  SmallVector<unsigned> bits; // outermost first
  SmallVector<Operation *> nodes; // users before definitions
};
Reduction reduction(Value value, StringRef kind, unsigned wave) {
  Reduction r;
  Value current = value;
  while (auto *op = current.getDefiningOp()) {
    if (op->getName().getStringRef() != kind || op->getNumOperands() != 2)
      break;
    gpu::ShuffleOp shuffle;
    for (unsigned side = 0; side < 2; ++side) {
      auto s = op->getOperand(side).getDefiningOp<gpu::ShuffleOp>();
      if (s && s.getShuffleResult() == op->getOperand(side) &&
          s.getValue() == op->getOperand(1 - side)) { shuffle = s; break; }
    }
    if (!shuffle || shuffle.getMode() != gpu::ShuffleMode::XOR ||
        !shuffle.getValid().use_empty()) break;
    auto bit = integer(shuffle.getOffset()), width = integer(shuffle.getWidth());
    if (!bit || !width || *width != wave || *bit <= 0 || *bit >= wave ||
        !llvm::isPowerOf2_64(*bit) || (r.mask & *bit)) return {};
    r.mask |= *bit;
    r.bits.push_back(*bit);
    r.nodes.push_back(op);
    r.nodes.push_back(shuffle);
    current = shuffle.getValue();
  }
  if (r.mask) r.leaf = current;
  return r;
}

class Proof {
  func::FuncOp function;
  unsigned wave, threads;
  std::set<Slice> active;
  // Evaluate fixed integer address expressions for the one-dimensional launch
  // contract, rather than guessing uniformity from syntax (e.g. tid & ~mask).
  Attribute evaluate(Value v, unsigned tid, unsigned depth = 0) {
    if (depth > 64 || !v.getType().isIntOrIndex()) return {};
    Attribute constant;
    if (matchPattern(v, m_Constant(&constant))) return constant;
    if (auto id = v.getDefiningOp<gpu::ThreadIdOp>())
      return IntegerAttr::get(v.getType(), id.getDimension() == gpu::Dimension::x ? tid : 0);
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
  bool fixedUniform(Value v, unsigned group) {
    if (!v.getType().isIntOrIndex() || threads > 1024) return false;
    SmallVector<Attribute> values;
    for (unsigned tid = 0; tid < threads; ++tid) {
      auto attr = evaluate(v, tid);
      if (!attr) return false;
      values.push_back(attr);
    }
    for (unsigned tid = 0; tid < threads; ++tid)
      for (unsigned bit = 1; bit < wave; bit *= 2)
        if ((group & bit) && values[tid] != values[tid ^ bit]) return false;
    return true;
  }
public:
  Proof(func::FuncOp function, unsigned wave, unsigned threads)
      : function(function), wave(wave), threads(threads) {}
  bool uniform(Slice s, unsigned group, bool collectives = true,
               unsigned depth = 0) {
    if (depth > 128) return false;
    s = resolve(s);
    if (matchPattern(s.value, m_Constant())) return true;
    if (s.element == -1 && fixedUniform(s.value, group)) return true;
    if (auto arg = dyn_cast<BlockArgument>(s.value)) {
      if (arg.getOwner() == &function.getBody().front()) return true;
      auto loop = Loop::get(arg.getOwner()->getParentOp());
      if (!loop) return false;
      if (arg.getArgNumber() == 0) return bounds(*loop, group, depth + 1);
      // Induction proof: initial state and the next state must be uniform.
      if (!active.insert(s).second) return true;
      unsigned index = arg.getArgNumber() - 1;
      bool ok = uniform(part(loop->init[index], s.element), group, collectives, depth + 1) &&
                uniform(part(loop->yielded[index], s.element), group, collectives, depth + 1);
      active.erase(s);
      return ok;
    }
    auto *op = s.value.getDefiningOp();
    if (!op) return false;
    if (isa<gpu::BlockIdOp, gpu::GridDimOp, gpu::BlockDimOp>(op)) return true;
    if (collectives && s.element == -1 &&
        isa<arith::AddFOp, arith::AddIOp, arith::MaxNumFOp, arith::MinNumFOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp>(op)) {
      auto r = reduction(s.value, op->getName().getStringRef(), wave);
      if ((r.mask & group) == group && r.mask) return true;
    }
    StringRef dialect = op->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math" &&
         !isa<affine::AffineApplyOp>(op)) || !isMemoryEffectFree(op)) return false;
    for (Value operand : op->getOperands())
      if (!uniform(part(operand, s.element), group, collectives, depth + 1)) return false;
    return true;
  }
  bool bounds(Loop loop, unsigned group, unsigned depth = 0) {
    if (auto f = dyn_cast<affine::AffineForOp>(loop.op)) {
      for (Value v : f.getControlOperands())
        if (!uniform({v}, group, false, depth + 1)) return false;
      return true;
    }
    auto f = cast<scf::ForOp>(loop.op);
    for (Value v : {f.getLowerBound(), f.getUpperBound(), f.getStep()})
      if (!uniform({v}, group, false, depth + 1)) return false;
    return true;
  }
  bool control(Loop loop) {
    // All lanes enter/leave the candidate together. Structured ifs inside its
    // body may diverge and reconverge, but matching only accepts direct-body
    // updates and reductions. Unknown region control is never assumed safe.
    for (Operation *op = loop.op; op != function; op = op->getParentOp()) {
      if (auto parent = Loop::get(op)) {
        if (!bounds(*parent, wave - 1)) return false;
      } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (!uniform({branch.getCondition()}, wave - 1, false)) return false;
      } else if (auto branch = dyn_cast<affine::AffineIfOp>(op)) {
        for (Value v : branch.getOperands())
          if (!uniform({v}, wave - 1, false)) return false;
      } else return false;
    }
    return true;
  }
};

// Each affected element may only flow through static views to this exact
// endpoint operand. Other fields of a packed vector remain unrestricted.
bool flowsTo(Slice s, Operation *endpoint, unsigned operandNumber,
             unsigned &reads, unsigned depth = 0) {
  if (depth > 128) return false;
  for (OpOperand &use : s.value.getUses()) {
    auto *user = use.getOwner();
    if (user == endpoint && use.getOperandNumber() == operandNumber) {
      ++reads;
      continue;
    }
    if (user->getNumResults() != 1) return false;
    Value out = user->getResult(0);
    int64_t n = count(out.getType());
    if (!n || n > 128) return false;
    for (int64_t i = 0; i < n; ++i) {
      Slice next = part(out, i);
      auto input = view(next);
      if (!input) return false;
      if (*input == s && !flowsTo(next, endpoint, operandNumber, reads, depth + 1))
        return false;
    }
  }
  return true;
}
bool onlyFlows(Slice s, Operation *end, unsigned operand) {
  unsigned reads = 0;
  return flowsTo(s, end, operand, reads) && reads == 1;
}

bool safeFunction(func::FuncOp function) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none"})
    if (function->hasAttr(name)) return false;
  for (StringRef key : {"passthrough", "llvm.passthrough"})
    if (auto attrs = function->getAttrOfType<ArrayAttr>(key))
      for (Attribute attr : attrs) {
        if (auto pair = dyn_cast<ArrayAttr>(attr))
          if (!pair.empty()) attr = pair[0];
        if (auto name = dyn_cast<StringAttr>(attr))
          if (name.getValue() == "strictfp" || name.getValue() == "optnone") return false;
      }
  bool safe = true;
  function.walk([&](Operation *op) {
    if (isa<CallOpInterface>(op)) safe = false;
    if (auto assembly = dyn_cast<LLVM::InlineAsmOp>(op)) {
      if (!isRegisterMMA(assembly.getAsmString(), assembly.getConstraints()))
        safe = false;
      return;
    }
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "math" && dialect != "vector" &&
        dialect != "memref" && dialect != "affine" && dialect != "scf" &&
        dialect != "func") {
      StringRef name = op->getName().getStringRef();
      if (name != "gpu.thread_id" && name != "gpu.block_id" &&
          name != "gpu.block_dim" && name != "gpu.grid_dim" &&
          name != "gpu.lane_id" && name != "gpu.shuffle" && name != "gpu.barrier" &&
          name != "frisk.async_copy_gm_to_shm" && name != "frisk.wait_buffer_ready" &&
          name != "frisk.read_matrix_from_shm") safe = false;
    }
    if (op->getNumRegions() && op != function &&
        !isa<affine::AffineForOp, scf::ForOp, affine::AffineIfOp, scf::IfOp>(op))
      safe = false;
  });
  return safe;
}

bool rewrite(Loop loop, unsigned index, int64_t element, Proof &proof,
             unsigned wave, bool approximate, unsigned &shuffles) {
  Value state = loop.args[index];
  Type type = getElementTypeOrSelf(state.getType());
  if ((!type.isF32() && !type.isInteger(32)) ||
      (type.isF32() && !approximate) || !zero(part(loop.init[index], element))) return false;
  Slice back = resolve(part(loop.yielded[index], element));
  auto *update = back.value.getDefiningOp();
  if (!update || update->getBlock() != loop.body ||
      (type.isF32() ? !isa<arith::AddFOp>(update) : !isa<arith::AddIOp>(update))) return false;
  // Avoid changing multiple vector elements through one arithmetic operation.
  if (count(back.value.getType()) != 1) return false;
  for (unsigned side = 0; side != 2; ++side) {
    Slice sum = resolve(part(update->getOperand(side), back.element));
    if (sum.element != -1) continue;
    auto r = reduction(sum.value, type.isF32() ? "arith.addf" : "arith.addi", wave);
    if (!r.mask) continue;
    if (llvm::any_of(r.nodes, [&](Operation *op) { return op->getBlock() != loop.body; })) continue;
    Slice old = resolve(part(update->getOperand(1 - side), back.element));
    Operation *scale = nullptr;
    unsigned stateOperand = 1 - side;
    auto *mul = old.value.getDefiningOp();
    if (mul && (type.isF32() ? isa<arith::MulFOp>(mul) : isa<arith::MulIOp>(mul)) &&
        count(old.value.getType()) == 1 && mul->getBlock() == loop.body) {
      for (unsigned operand = 0; operand < 2; ++operand) {
        if (resolve(part(mul->getOperand(operand), old.element)) == part(state, element) &&
            proof.uniform(part(mul->getOperand(1 - operand), old.element), r.mask)) {
          scale = mul; stateOperand = operand; break;
        }
      }
      if (!scale || !onlyFlows(old, update, 1 - side)) continue;
    } else if (!(old == part(state, element))) continue;
    if (!onlyFlows(part(state, element), scale ? scale : update, stateOperand) ||
        !onlyFlows(sum, update, side) ||
        !onlyFlows(back, loop.body->getTerminator(), index)) continue;
    llvm::SmallPtrSet<Operation *, 16> nodes(r.nodes.begin(), r.nodes.end());
    bool exclusive = true;
    for (Operation *op : r.nodes)
      for (Value result : op->getResults())
        if (result != sum.value)
          for (Operation *user : result.getUsers())
            if (!nodes.contains(user)) exclusive = false;
    if (!exclusive) continue;

    sum.value.replaceAllUsesWith(r.leaf);
    // Partial recurrences may overflow even when reduced ones do not.
    for (Operation *op : {update, scale}) if (op) {
      if (type.isF32())
        op->setAttr("fastmath", arith::FastMathFlagsAttr::get(
            op->getContext(), arith::FastMathFlags::none));
      else
        op->setAttr("overflowFlags", arith::IntegerOverflowFlagsAttr::get(
            op->getContext(), arith::IntegerOverflowFlags::none));
    }
    for (Operation *op : r.nodes) op->erase();

    Value result = loop.op->getResult(index);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : result.getUses()) uses.push_back(&use);
    OpBuilder builder(loop.op);
    builder.setInsertionPointAfter(loop.op);
    Location loc = loop.op->getLoc();
    Value final = result;
    SmallVector<int64_t> position;
    if (auto v = dyn_cast<VectorType>(result.getType())) {
      int64_t flat = element;
      position.resize(v.getRank());
      for (int i = v.getRank() - 1; i >= 0; --i) {
        position[i] = flat % v.getDimSize(i); flat /= v.getDimSize(i);
      }
      final = builder.create<vector::ExtractOp>(loc, result, position);
    }
    auto width = builder.create<arith::ConstantIntOp>(loc, wave, 32);
    for (unsigned bit : llvm::reverse(r.bits)) {
      auto offset = builder.create<arith::ConstantIntOp>(loc, bit, 32);
      auto peer = builder.create<gpu::ShuffleOp>(loc, final, offset, width, gpu::ShuffleMode::XOR);
      final = type.isF32() ? Value(builder.create<arith::AddFOp>(loc, final, peer.getShuffleResult()))
                           : Value(builder.create<arith::AddIOp>(loc, final, peer.getShuffleResult()));
    }
    if (isa<VectorType>(result.getType()))
      final = builder.create<vector::InsertOp>(loc, final, result, position);
    for (OpOperand *use : uses) use->set(final);
    shuffles += r.bits.size();
    return true;
  }
  return false;
}

class ThreadDeferLaneSumPass
    : public PassWrapper<ThreadDeferLaneSumPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadDeferLaneSumPass)
  ThreadDeferLaneSumPass(DeferLaneSumOptions options = {}) {
    approximate = options.approximate; waveSize = options.waveSize;
  }
  ThreadDeferLaneSumPass(const ThreadDeferLaneSumPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "frisk-thread-defer-lane-sum"; }
  StringRef getDescription() const final {
    return "Defer fixed XOR lane sums across zero-initialized thread recurrences";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, vector::VectorDialect>();
  }
private:
  Option<bool> approximate{*this, "approximate", llvm::cl::init(false)};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(0)};
  void runOnOperation() override {
    if (waveSize != 0 && waveSize != 32 && waveSize != 64) {
      getOperation().emitError("wave-size must be 0, 32 or 64");
      return signalPassFailure();
    }
    auto function = getOperation();
    auto threads = function->getAttrOfType<IntegerAttr>("thread_num");
    // thread_num is the existing one-dimensional launch contract. An explicit
    // wave size is required; gpu.shuffle width alone is not a hardware ABI.
    if (!waveSize || !threads || threads.getInt() <= 0 ||
        threads.getInt() % waveSize || function.empty() || !safeFunction(function)) return;
    // Reject explicit target-feature conflicts with the caller's launch ABI.
    auto conflicts = [&](StringRef features) {
      SmallVector<StringRef> tokens;
      features.split(tokens, ',');
      return llvm::is_contained(tokens, "+wavefrontsize" + std::to_string(waveSize == 64 ? 32 : 64)) ||
             llvm::is_contained(tokens, "-wavefrontsize" + std::to_string(waveSize));
    };
    for (StringRef key : {"target-features", "llvm.target_features"})
      if (auto attr = function->getAttrOfType<StringAttr>(key))
        if (conflicts(attr.getValue())) return;
    for (StringRef key : {"passthrough", "llvm.passthrough"})
      if (auto attrs = function->getAttrOfType<ArrayAttr>(key))
        for (Attribute attr : attrs)
          if (auto pair = dyn_cast<ArrayAttr>(attr); pair && pair.size() == 2)
            if (auto name = dyn_cast<StringAttr>(pair[0]); name && name.getValue() == "target-features")
              if (auto features = dyn_cast<StringAttr>(pair[1]))
                if (conflicts(features.getValue())) return;
    Proof proof(function, waveSize, threads.getInt());
    SmallVector<Operation *> loops;
    function.walk([&](Operation *op) { if (Loop::get(op)) loops.push_back(op); });
    unsigned states = 0, shuffles = 0;
    for (Operation *op : loops) {
      auto loop = *Loop::get(op);
      bool nested = false;
      op->walk([&](Operation *child) { if (child != op && Loop::get(child)) nested = true; });
      if (nested || !proof.control(loop)) continue;
      for (unsigned i = 0; i < loop.args.size(); ++i) {
        int64_t n = count(loop.args[i].getType());
        if (n > 128) continue;
        for (int64_t j = 0; j < n; ++j)
          states += rewrite(loop, i, isa<VectorType>(loop.args[i].getType()) ? j : -1,
                            proof, waveSize, approximate, shuffles);
      }
    }
    llvm::errs() << "DeferLaneSum: " << function.getSymName() << ": " << states
                 << " states, " << shuffles << " loop shuffles deferred ("
                 << (approximate ? "approx" : "legal") << ")\n";
  }
};
} // namespace
std::unique_ptr<Pass> createThreadDeferLaneSumPass(DeferLaneSumOptions options) {
  return std::make_unique<ThreadDeferLaneSumPass>(options);
}
} // namespace mlir::frisk
