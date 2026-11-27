// Schedule softmax producer fragments against their real ThreadIR MMAC uses.
// Keep the existing FP32 sum DAG; no lifetime-asm pattern matching is required.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "ThreadScheduleUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace mlir::frisk {
namespace {
using namespace thread_schedule;
constexpr StringLiteral Marker = "frisk.softmax_fragments";
struct Packet { Operation *packed, *exp; Ops nodes; };

bool matchPacket(arith::TruncFOp cast, Packet &packet) {
  auto *ctx = cast.getContext();
  if (cast.getIn().getType() != VectorType::get({4}, Float32Type::get(ctx)) ||
      cast.getType() != VectorType::get({4}, Float16Type::get(ctx)) ||
      cast.getResult().use_empty()) return false;
  for (OpOperand &use : cast.getResult().getUses())
    if (!mma(use.getOwner()) || use.getOperandNumber() != 0 ||
        use.getOwner()->getBlock() != cast->getBlock()) return false;
  auto exp = cast.getIn().getDefiningOp<math::Exp2Op>();
  if (!exp || exp->getBlock() != cast->getBlock()) return false;
  // Stable softmax has score-max; the unshifted form has score+mask (or
  // a finite-mask select). The scheduling proof depends on SSA and effects,
  // not on the arithmetic used to form the exponent. Move only recognized
  // pure producers; an arbitrary exponent remains a fixed dependency.
  packet = {cast, exp, {exp, cast}};
  if (auto *score = exp.getOperand().getDefiningOp();
      score && score->getBlock() == cast->getBlock() && pure(score) &&
      isa<arith::SubFOp, arith::AddFOp, arith::SelectOp>(score))
    packet.nodes.insert(packet.nodes.begin(), score);
  return true;
}

bool reductionNode(Operation *op) {
  // The lane-local recurrence is still wrapped in vector<1xf32> here.
  // Treating its broadcast/add as fixed forces every exp2 feeding the row sum
  // to execute before PV, even though the half conversions themselves sink.
  if (isa<arith::AddFOp>(op)) {
    Type type = op->getResult(0).getType();
    if (auto vector = dyn_cast<VectorType>(type)) type = vector.getElementType();
    return type.isF32();
  }
  return isa<vector::ExtractOp, vector::InsertOp, vector::FromElementsOp,
             vector::BroadcastOp, vector::ShapeCastOp>(op);
}
bool knownFixed(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>()) return true;
  if (crossIterationIssue(op)) return true;
  if (op->getNumRegions()) return false;
  if (pure(op) || mma(op)) return true;
  if (isScheduleBarrier(op)) return true;
  if (isa<LDSLoadOp, ReadMatrixFromShmOp, gpu::BarrierOp,
          SyncThreadsInBlockOp, gpu::ShuffleOp>(op)) return true;
  StringRef name = op->getName().getStringRef();
  if (name == "frisk.async_copy_gm_to_shm" || name == "frisk.wait_buffer_ready") return true;
  return false;
}

unsigned schedule(Block &block, SoftmaxFragmentMode mode) {
  SmallVector<Packet, 4> packets;
  Ops original;
  DenseMap<Operation *, unsigned> index;
  for (Operation &op : block) {
    index[&op] = original.size(); original.push_back(&op);
    if (auto cast = dyn_cast<arith::TruncFOp>(op)) {
      Packet packet;
      if (matchPacket(cast, packet)) packets.push_back(std::move(packet));
    }
  }
  if (packets.size() < 2) return 0;
  llvm::SmallPtrSet<Operation *, 32> moving, reductions, anchors;
  Ops work;
  unsigned first = original.size(), last = 0;
  for (auto &packet : packets) {
    anchors.insert(packet.packed);
    for (Operation *op : packet.nodes) {
      // Two conversions may share an exponent/score. Schedule that producer
      // once, retaining all of its original uses and floating-point flags.
      moving.insert(op);
      first = std::min(first, index[op]);
    }
    work.push_back(packet.exp);
    for (Operation *user : packet.packed->getUsers()) last = std::max(last, index[user]);
  }
  // Follow only existing register sum edges and vector wrappers, never a PHI,
  // collective, conversion to half, or unrelated producer.
  for (unsigned n = 0; n < work.size(); ++n)
    for (Operation *user : work[n]->getUsers())
      if (user->getBlock() == &block && !moving.contains(user) &&
          reductionNode(user) && reductions.insert(user).second) work.push_back(user);
  for (Operation *op : reductions) moving.insert(op);
  // Validate the whole possible movement interval, including live-out sums.
  for (Operation *op : moving) last = std::max(last, index[op]);
  for (unsigned n = first; n <= last; ++n)
    if (!moving.contains(original[n]) && !knownFixed(original[n])) return 0;
  // Nested-region uses cannot be repaired by a flat SSA scheduler.
  for (Operation *op : moving) for (Operation *user : op->getUsers())
    if (user->getBlock() != &block) return 0;

  Ops order;
  llvm::SmallPtrSet<Operation *, 32> emitted, visiting;
  auto ready = [&](Operation *op) {
    return llvm::all_of(op->getOperands(), [&](Value value) {
      auto *def = value.getDefiningOp();
      return !def || def->getBlock() != &block || emitted.contains(def);
    });
  };
  auto drainSums = [&]() {
    for (Operation *op : original)
      if (reductions.contains(op) && !emitted.contains(op) && ready(op)) {
        order.push_back(op); emitted.insert(op);
      }
  };
  std::function<bool(Operation *)> emit = [&](Operation *op) {
    if (emitted.contains(op)) return true;
    if (!moving.contains(op) || !visiting.insert(op).second) return false;
    for (Value value : op->getOperands())
      if (auto *def = value.getDefiningOp(); def && def->getBlock() == &block)
        if (!emit(def)) return false;
    visiting.erase(op);
    drainSums();
    if (!emitted.contains(op)) { order.push_back(op); emitted.insert(op); }
    drainSums();
    return true;
  };
  for (Operation *op : original) {
    if (moving.contains(op)) {
      if (mode == SoftmaxFragmentMode::Fragment && anchors.contains(op) && !emit(op)) return 0;
      continue;
    }
    if (op->hasTrait<OpTrait::IsTerminator>())
      for (Operation *node : original) if (moving.contains(node) && !emit(node)) return 0;
    for (Value value : op->getOperands())
      if (auto *def = value.getDefiningOp(); def && def->getBlock() == &block)
        if (!emit(def)) return 0;
    order.push_back(op); emitted.insert(op);
  }
  if (order.size() != original.size() || order == original) return 0;
  if (!preservesScheduleBarriers(original, order)) return 0;
  DenseMap<Operation *, unsigned> proposed;
  for (auto [n, op] : llvm::enumerate(order)) proposed[op] = n;
  for (Operation *op : order) for (Value value : op->getOperands())
    if (auto *def = value.getDefiningOp(); def && def->getBlock() == &block)
      if (proposed[def] >= proposed[op]) return 0;
  Operation *end = block.getTerminator();
  for (Operation *op : order) if (op != end) op->moveBefore(end);
  for (auto &packet : packets) guard(packet.packed->getNextNode(), Marker);
  return packets.size();
}

class ThreadScheduleSoftmaxFragmentsPass
    : public PassWrapper<ThreadScheduleSoftmaxFragmentsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadScheduleSoftmaxFragmentsPass)
  ThreadScheduleSoftmaxFragmentsPass() = default;
  ThreadScheduleSoftmaxFragmentsPass(const ThreadScheduleSoftmaxFragmentsPass &other) : PassWrapper(other) {}
  explicit ThreadScheduleSoftmaxFragmentsPass(SoftmaxFragmentMode value) {
    mode = value == SoftmaxFragmentMode::PV ? "pv" : value == SoftmaxFragmentMode::Fragment ? "fragment" : "off";
  }
  Option<std::string> mode{*this, "mode", llvm::cl::init("off")};
  StringRef getArgument() const final { return "frisk-thread-schedule-softmax-fragments"; }
  StringRef getDescription() const final { return "Schedule softmax fragments and their original sums in ThreadIR"; }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<FriskDialect, LLVM::LLVMDialect, math::MathDialect>();
  }
  void runOnOperation() final {
    if (mode != "off" && mode != "fragment" && mode != "pv") {
      getOperation().emitError("softmax fragment mode must be off, fragment or pv");
      return signalPassFailure();
    }
    if (mode == "off") return;
    if (failed(checkGuard(getOperation()))) return signalPassFailure();
    bool conflict = false;
    getOperation().walk([&](func::FuncOp f) {
      if (auto old = f->getAttrOfType<StringAttr>(Marker))
        if (old.getValue() != mode) { f.emitError("conflicting softmax fragment mode"); conflict = true; }
    });
    if (conflict) return signalPassFailure();
    getOperation().walk([&](func::FuncOp f) {
      if (f.isExternal() || protectedFunction(f) || f->hasAttr(Marker)) return;
      foldVectorViews(f);
      unsigned count = 0;
      f.walk([&](Block *block) {
        count += schedule(*block, mode == "pv" ? SoftmaxFragmentMode::PV : SoftmaxFragmentMode::Fragment);
      });
      if (count) markFunction(f, Marker, mode);
      llvm::errs() << "Thread SoftmaxFragments: " << f.getName() << ": " << count
                   << " fragments, mode=" << mode << "; original FP32 sum DAG and memory/MMAC order retained\n";
    });
  }
};
} // namespace
std::unique_ptr<Pass> createThreadScheduleSoftmaxFragmentsPass(SoftmaxFragmentMode mode) {
  return std::make_unique<ThreadScheduleSoftmaxFragmentsPass>(mode);
}
} // namespace mlir::frisk
