// Hoist a bounded batch of ordinary global reads across exp2 in late ThreadIR.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "ThreadSchedulePlan.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir::frisk {
namespace {
using namespace thread_schedule;
constexpr StringLiteral Marker = "frisk.global_prefetch";

Value readBuffer(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op)) return load.getMemref();
  if (auto load = dyn_cast<affine::AffineLoadOp>(op)) return load.getMemRef();
  if (auto load = dyn_cast<affine::AffineVectorLoadOp>(op)) return load.getMemRef();
  if (auto load = dyn_cast<vector::LoadOp>(op)) return load.getBase();
  if (auto load = dyn_cast<LDSLoadOp>(op)) return load.getSource();
  return {};
}

Value writtenBuffer(Operation *op) {
  if (auto store = dyn_cast<memref::StoreOp>(op)) return store.getMemref();
  if (auto store = dyn_cast<affine::AffineStoreOp>(op)) return store.getMemRef();
  if (auto store = dyn_cast<affine::AffineVectorStoreOp>(op)) return store.getMemRef();
  if (auto store = dyn_cast<vector::StoreOp>(op)) return store.getBase();
  return {};
}

// Count payload words, not physical VGPRs or address temporaries. The legacy
// LLVM heuristic admits at most 128 bits per packet and eight words per batch.
unsigned packetWords(Type type) {
  uint64_t count = 1;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.isScalable()) return 0;
    count = vector.getNumElements();
    type = vector.getElementType();
  }
  if (!isa<IntegerType, FloatType>(type)) return 0;
  unsigned bits = type.getIntOrFloatBitWidth();
  if (!bits || count > 128 / bits) return 0;
  return (count * bits + 31) / 32;
}

bool collectAddress(Value value, Operation *point,
                    llvm::SmallPtrSetImpl<Operation *> &seen, Ops &ordered) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != point->getBlock() || op->isBeforeInBlock(point))
    return true;
  if (op == point || seen.size() >= 96) return false;
  if (!seen.insert(op).second) return true;
  if (!address(op)) return false;
  for (Value operand : op->getOperands())
    if (!collectAddress(operand, point, seen, ordered)) return false;
  ordered.push_back(op);
  return true;
}

unsigned prefetch(Block &block, unsigned maxWords) {
  Operation *point = nullptr;
  Ops loads, moving;
  llvm::SmallPtrSet<Operation *, 32> selected;
  unsigned words = 0;
  for (Operation &op : block) {
    if (!point) {
      if (auto exp = dyn_cast<math::Exp2Op>(op)) {
        Type type = exp.getType();
        if (auto vector = dyn_cast<VectorType>(type)) type = vector.getElementType();
        // Vector f32 exp2 becomes scalar f32 calls during target lowering.
        // Keep the legacy matcher restricted to f32 math.
        if (type.isF32()) point = &op;
      }
      continue;
    }
    // Never leave a mask/region, cross readiness, or disturb an existing plan.
    if (op.getNumRegions() || op.hasTrait<OpTrait::IsTerminator>() ||
        isScheduleBarrier(&op) || protectedLDSPair(&op)) break;
    if (Value buffer = readBuffer(&op)) {
      if (cast<MemRefType>(buffer.getType()).getMemorySpaceAsInt() != 1) continue;
      unsigned size = packetWords(op.getResult(0).getType());
      if (!size || size > maxWords - words) break;
      llvm::SmallPtrSet<Operation *, 32> seen;
      Ops dependencies;
      if (!llvm::all_of(op.getOperands(), [&](Value operand) {
            return collectAddress(operand, point, seen, dependencies);
          })) break;
      // Commit only after validating the complete batch below.
      for (Operation *dependency : dependencies)
        if (selected.insert(dependency).second) moving.push_back(dependency);
      selected.insert(&op);
      moving.push_back(&op);
      loads.push_back(&op);
      words += size;
    } else if (Value buffer = writtenBuffer(&op)) {
      // LDS writes cannot alias global memory; all other writes end the batch.
      if (cast<MemRefType>(buffer.getType()).getMemorySpaceAsInt() != 3) break;
    } else if (!isa<gpu::ShuffleOp>(op) && !pure(&op)) {
      break;
    }
  }
  if (loads.empty()) return 0;
  Ops original;
  for (Operation *op = point; op; op = op->getNextNode()) {
    original.push_back(op);
    if (op == loads.back()) break;
  }
  Ops order = moving;
  for (Operation *op : original)
    if (!selected.contains(op)) order.push_back(op);
  LocalSchedulePlan plan{original, order};
  if (!plan.validate()) return 0;
  plan.apply();
  return loads.size();
}

class ThreadPrefetchGlobalPacketsPass
    : public PassWrapper<ThreadPrefetchGlobalPacketsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadPrefetchGlobalPacketsPass)
  ThreadPrefetchGlobalPacketsPass(unsigned words = 8) { maxWords = words; }
  ThreadPrefetchGlobalPacketsPass(const ThreadPrefetchGlobalPacketsPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "frisk-thread-prefetch-global-packets"; }
  StringRef getDescription() const final {
    return "Hoist bounded ordinary global packets across exp2 before target lowering";
  }

private:
  Option<unsigned> maxWords{*this, "max-words", llvm::cl::init(8)};
  void runOnOperation() override {
    auto function = getOperation();
    if (function.isDeclaration() || protectedFunction(function) || function->hasAttr(Marker)) return;
    if (auto module = function->getParentOfType<ModuleOp>())
      if (auto triple = module->getAttrOfType<StringAttr>("llvm.target_triple"))
        if (!triple.getValue().empty() && !llvm::Triple(triple.getValue()).isAMDGPU()) return;
    unsigned packets = 0;
    function.walk([&](Block *block) { packets += prefetch(*block, maxWords); });
    // Replaying must not move another maxWords past the same exp2. Also keep
    // the legacy LLVM entry point from applying a second prefetch budget.
    markFunction(function, Marker, std::to_string(maxWords));
    llvm::errs() << "PrefetchGlobalPackets: " << packets << " packets, at most "
                 << maxWords << " register words per block phase (ThreadIR)\n";
  }
};
} // namespace

std::unique_ptr<Pass> createThreadPrefetchGlobalPacketsPass(unsigned maxWords) {
  return std::make_unique<ThreadPrefetchGlobalPacketsPass>(maxWords);
}
} // namespace mlir::frisk
