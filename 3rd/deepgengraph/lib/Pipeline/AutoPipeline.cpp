// Opt-in block-tile prefetching. Keep the consumer recurrence in its original
// loop; only invariant-address global reads advance by num_stages - 1 tiles.
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

namespace mlir::pipeline {
#define GEN_PASS_DEF_AUTOPIPELINE
#include "deepgengraph/Pipeline/Passes.h.inc"
} // namespace mlir::pipeline

using namespace mlir;

namespace {
constexpr StringLiteral hintName = "frisk.pipeline_hint";

struct Transfer {
  Operation *op;
  Value source, buffer;
  frisk::AllocBufferOp alloc;
  SmallVector<Value> handles;
  SmallVector<Value> slots;
};

// These block-level computations read their operands and produce fresh logical
// tiles. Gemm's legacy unqualified MemWrite effect describes its result, not
// mutation of A/B. Do not extend this exception to arbitrary ComputedTile ops.
bool readsOperands(Operation *op) {
  return isa<frisk::GemmOp, frisk::AddOp, frisk::SubOp, frisk::MulOp,
             frisk::DivOp, frisk::Exp2Op, frisk::CastOp,
             frisk::ReadMatrixFromShmOp>(op);
}

class PrefetchLoop {
public:
  PrefetchLoop(affine::AffineForOp loop, int64_t stages)
      : loop(loop), stages(stages) {}

  // Analysis is transactional: a rejected hint never leaves half a schedule.
  bool analyze();
  void build();
  StringRef reason = "unsupported loop";

private:
  bool reject(StringRef why) { reason = why; return false; }
  bool address(Value value);
  bool safeEffects(Operation *op);
  bool localDestination(Value value);
  Value bound(OpBuilder &b, bool lower);
  Value offset(OpBuilder &b, Value iv, int64_t tiles);
  SmallVector<Value> selectSlots(OpBuilder &b, Value ordinal, int64_t lead);
  void issue(OpBuilder &b, Value iv, ValueRange slots);

  affine::AffineForOp loop;
  int64_t stages;
  SmallVector<Transfer> transfers;
  DenseSet<Operation *> producers, allocations, addresses, waits;
  DenseSet<Value> rotated;
};

bool PrefetchLoop::address(Value value) {
  if (value == loop.getInductionVar())
    return true;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getOwner() != loop.getBody();
  Operation *def = value.getDefiningOp();
  if (!loop->isAncestor(def))
    return true;
  if (def->getBlock() != loop.getBody() || def->getNumRegions() ||
      !isMemoryEffectFree(def) || !isSpeculatable(def))
    return false;
  if (addresses.contains(def))
    return true;
  for (Value operand : def->getOperands())
    if (!address(operand))
      return false;
  addresses.insert(def);
  return true;
}

bool PrefetchLoop::localDestination(Value value) {
  if (!isa<BaseMemRefType>(value.getType()))
    return isa<VectorType>(value.getType());
  if (rotated.contains(value))
    return false;
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getMemorySpaceAsInt() == 1)
    return false;
  // Private storage/results in this iteration cannot alias the global source.
  Operation *def = value.getDefiningOp();
  return def && loop->isAncestor(def) &&
         (isa<frisk::AllocBufferOp, memref::AllocOp, memref::AllocaOp>(def) ||
          readsOperands(def) || isa<frisk::ZeroOp>(def));
}

bool PrefetchLoop::safeEffects(Operation *op) {
  if (producers.contains(op) || waits.contains(op) ||
      isa<frisk::AllocBufferOp, frisk::ZeroOp>(op) || readsOperands(op))
    return true;
  if (auto copy = dyn_cast<frisk::CopyOp>(op))
    return localDestination(copy.getDst());
  if (auto reduce = dyn_cast<frisk::ReduceOp>(op))
    return localDestination(reduce->getOperand(1));
  // Reject opaque/async regions. Ordinary structured control flow is executed
  // at the original iteration and recursively checked for global mutations.
  if (isa<scf::IfOp, scf::ForOp, affine::AffineIfOp, affine::AffineForOp>(op)) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!safeEffects(&nested))
            return false;
    return true;
  }
  if (op->getNumRegions())
    return false;
  if (isMemoryEffectFree(op))
    return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  for (auto &effect : instances) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()))
      continue;
    if (!isa<MemoryEffects::Write>(effect.getEffect()) ||
        !effect.getValue() || !localDestination(effect.getValue()))
      return false;
  }
  return true;
}

bool PrefetchLoop::analyze() {
  for (Operation *parent = loop; parent && !isa<func::FuncOp>(parent);
       parent = parent->getParentOp())
    if (parent->hasAttr("pipeline.stage") || parent->hasAttr("pipeline.order") ||
        parent->hasAttr("pipeline.scheduled"))
      return reject("loop or ancestor already has an explicit pipeline schedule");
  // Require block-uniform bounds and enclosing control flow, since slot waits
  // are collective. Checking a body operation includes this loop's bounds.
  if (!frisk::isInBlockUniformControlFlow(loop.getBody()->getTerminator()))
    return reject("loop bounds/control flow are not proven block-uniform");
  if (loop.getStepAsInt() > std::numeric_limits<int64_t>::max() / stages)
    return reject("stage distance times loop step overflows");
  if (stages > 64)
    return reject("more than 64 stages exceeds the prefetch code-size budget");
  for (Type type : loop.getResultTypes())
    if (isa<BaseMemRefType>(type))
      return reject("memref loop-carried aliases are not supported");

  for (Operation &op : loop.getBody()->without_terminator()) {
    Value source, destination;
    if (auto async = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
      source = async.getSrc();
      destination = async.getDst();
      if (async->hasAttr("frisk.source_map"))
        return reject("run automatic pipelining before physical packing");
    } else if (auto copy = dyn_cast<frisk::CopyOp>(op)) {
      source = copy.getSrc();
      destination = copy.getDst();
      auto map = copy.getOffsetMap();
      auto dst = dyn_cast<MemRefType>(destination.getType());
      bool whole = dst && map.getNumInputs() == 0 &&
          ((map.getNumResults() == 1 &&
            map.getResult(0) == getAffineConstantExpr(dst.getRank(), loop.getContext())) ||
           (map.getNumResults() == unsigned(dst.getRank()) &&
            llvm::all_of(map.getResults(), [](AffineExpr e) {
              auto c = dyn_cast<AffineConstantExpr>(e);
              return c && c.getValue() == 0;
            })));
      if (!whole)
        continue;
    } else {
      continue;
    }
    auto src = dyn_cast<MemRefType>(source.getType());
    auto dst = dyn_cast<MemRefType>(destination.getType());
    if (!src || !dst || src.getMemorySpaceAsInt() != 1 ||
        dst.getMemorySpaceAsInt() != 3 || !src.hasStaticShape() ||
        src.getShape() != dst.getShape() ||
        src.getElementType() != dst.getElementType())
      continue;
    auto alloc = destination.getDefiningOp<frisk::AllocBufferOp>();
    if (!alloc || alloc->getBlock() != loop.getBody() ||
        !rotated.insert(destination).second)
      return reject("prefetch destination must have one writer and be allocated in the loop");
    if (!address(source))
      return reject("prefetch address depends on a loop-carried value or non-speculatable operation");
    Transfer transfer{&op, source, destination, alloc, {destination}, {}};
    if (auto copy = dyn_cast<frisk::CopyOp>(op); copy && copy.hasValueResult()) {
      transfer.handles.push_back(copy.getValueResult());
      rotated.insert(copy.getValueResult());
    }
    transfers.push_back(std::move(transfer));
    producers.insert(&op);
    allocations.insert(alloc);
  }
  if (transfers.empty())
    return reject("no full-tile global-to-shared transfer with owned storage");

  for (auto &transfer : transfers) {
    SmallVector<Operation *> consumers;
    SmallVector<Operation *> ready;
    for (Value handle : transfer.handles)
      for (OpOperand &use : handle.getUses()) {
        Operation *user = use.getOwner();
        if (user == transfer.op)
          continue;
        if (auto wait = dyn_cast<frisk::WaitBufferReadyOp>(user);
            wait && wait->getBlock() == loop.getBody()) {
          waits.insert(user);
          ready.push_back(user);
          continue;
        }
        // Descriptor escapes and aliases require a more general lifetime
        // proof. Direct reads cover block GEMMs, arithmetic and scalar tests.
        bool read = readsOperands(user) ||
                    isa<memref::LoadOp, affine::AffineLoadOp>(user) ||
                    (isa<frisk::CopyOp>(user) && use.getOperandNumber() == 0);
        if (!read || user->getBlock() != loop.getBody() ||
            !transfer.op->isBeforeInBlock(user))
          return reject("prefetch storage escapes, aliases, is mutated, or is read before its copy");
        consumers.push_back(user);
      }
    if (consumers.empty())
      return reject("prefetch storage has no direct consumer");
    if (isa<frisk::AsyncCopyGmToShmOp>(transfer.op))
      for (Operation *consumer : consumers)
        if (!llvm::any_of(ready, [&](Operation *wait) {
              return transfer.op->isBeforeInBlock(wait) &&
                     wait->isBeforeInBlock(consumer);
            }))
          return reject("async consumer has no matching wait after its issue");
  }
  for (Operation &op : loop.getBody()->without_terminator())
    if (!safeEffects(&op))
      return reject("unknown effects, external writes, or unsupported synchronization in loop");
  return true;
}

Value PrefetchLoop::bound(OpBuilder &b, bool lower) {
  auto map = lower ? loop.getLowerBoundMap() : loop.getUpperBoundMap();
  auto operands = lower ? loop.getLowerBoundOperands() : loop.getUpperBoundOperands();
  if (map.getNumResults() == 1)
    return b.createOrFold<affine::AffineApplyOp>(loop.getLoc(), map, operands);
  if (lower)
    return b.create<affine::AffineMaxOp>(loop.getLoc(), map, operands);
  return b.create<affine::AffineMinOp>(loop.getLoc(), map, operands);
}

Value PrefetchLoop::offset(OpBuilder &b, Value iv, int64_t tiles) {
  return b.createOrFold<affine::AffineApplyOp>(loop.getLoc(),
      AffineMap::get(1, 0, b.getAffineDimExpr(0) + tiles * loop.getStepAsInt()), iv);
}

SmallVector<Value> PrefetchLoop::selectSlots(OpBuilder &b, Value ordinal,
                                            int64_t lead) {
  auto loc = loop.getLoc();
  Value count = b.create<arith::ConstantIndexOp>(loc, stages);
  Value slot = b.create<arith::RemUIOp>(loc, ordinal, count);
  if (lead) {
    Value ahead = b.create<arith::ConstantIndexOp>(loc, lead);
    slot = b.create<arith::AddIOp>(loc, slot, ahead);
    slot = b.create<arith::RemUIOp>(loc, slot, count);
  }
  SmallVector<Value> selected;
  for (auto &transfer : transfers)
    selected.push_back(transfer.slots.back());
  // All buffers use the same phase. Select descriptors, never duplicate GEMMs.
  for (int64_t i = stages - 2; i >= 0; --i) {
    Value index = b.create<arith::ConstantIndexOp>(loc, i);
    Value isSlot = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, slot, index);
    SmallVector<Type> types;
    for (Value value : selected)
      types.push_back(value.getType());
    auto choose = b.create<scf::IfOp>(loc, types, isSlot, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choose.getThenRegion().front());
      SmallVector<Value> values;
      for (auto &transfer : transfers)
        values.push_back(transfer.slots[i]);
      b.create<scf::YieldOp>(loc, values);
      b.setInsertionPointToStart(&choose.getElseRegion().front());
      b.create<scf::YieldOp>(loc, selected);
    }
    selected.assign(choose.getResults().begin(), choose.getResults().end());
  }
  return selected;
}

void PrefetchLoop::issue(OpBuilder &b, Value iv, ValueRange slots) {
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), iv);
  for (Operation &op : loop.getBody()->without_terminator())
    if (addresses.contains(&op))
      b.clone(op, mapping);
  for (auto [i, transfer] : llvm::enumerate(transfers)) {
    Value source = mapping.lookupOrDefault(transfer.source);
    if (isa<frisk::AsyncCopyGmToShmOp>(transfer.op)) {
      mapping.map(transfer.buffer, slots[i]);
      b.clone(*transfer.op, mapping);
    } else {
      auto type = cast<MemRefType>(source.getType());
      int64_t alignment = std::max<int64_t>(1, type.getElementTypeBitWidth() / 8);
      if (alignment & (alignment - 1))
        alignment = 1;
      if (auto attr = transfer.op->getAttrOfType<IntegerAttr>("alignment"))
        alignment = attr.getInt();
      b.create<frisk::AsyncCopyGmToShmOp>(transfer.op->getLoc(), source, slots[i], alignment);
    }
  }
}

void PrefetchLoop::build() {
  OpBuilder b(loop);
  auto loc = loop.getLoc();
  Value lb = bound(b, true), ub = bound(b, false);
  for (auto &transfer : transfers)
    for (int64_t i = 0; i < stages; ++i) {
      auto *alloc = b.clone(*transfer.alloc);
      alloc->setAttr("pipeline_slot", b.getI64IntegerAttr(i));
      transfer.slots.push_back(alloc->getResult(0));
    }
  // Guard every prologue tile, including tile zero: zero-trip loops issue no
  // DMA. Short and non-multiple trip counts need no special epilogue.
  Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lb, ub);
  Value extent = b.create<arith::SubIOp>(loc, ub, lb);
  for (int64_t i = 0; i < stages - 1; ++i) {
    Value distance = b.create<arith::ConstantIndexOp>(loc, i * loop.getStepAsInt());
    Value enough = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, extent, distance);
    Value valid = b.create<arith::AndIOp>(loc, nonempty, enough);
    auto guard = b.create<scf::IfOp>(loc, valid, false);
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value iv = offset(b, lb, i);
    SmallVector<Value> slots;
    for (auto &transfer : transfers)
      slots.push_back(transfer.slots[i]);
    issue(b, iv, slots);
  }
  auto steady = b.create<affine::AffineForOp>(loc,
      loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), loop.getInits());
  for (NamedAttribute attr : loop->getDiscardableAttrs())
    if (attr.getName() != hintName)
      steady->setAttr(attr.getName(), attr.getValue());
  steady->setAttr("pipeline.scheduled", b.getUnitAttr());
  steady->setAttr("pipeline.num_stages", b.getI64IntegerAttr(stages));
  steady->setAttr("pipeline.slots", b.getI64IntegerAttr(stages));
  b.setInsertionPointToStart(steady.getBody());
  Value iv = steady.getInductionVar();
  // The unsigned difference is valid even for an interval spanning zero with
  // a distance larger than INT64_MAX. Do not overflow a speculative future IV
  // while testing whether the last few tiles exist.
  Value delta = b.create<arith::SubIOp>(loc, iv, lb);
  Value step = b.create<arith::ConstantIndexOp>(loc, loop.getStepAsInt());
  Value ordinal = b.create<arith::DivUIOp>(loc, delta, step);
  auto current = selectSlots(b, ordinal, 0);
  // Wait BEFORE issuing the future tile: backends may implement readiness as
  // wait-all. Moving these waits after the issue would serialize the prefetch.
  for (Value buffer : current)
    b.create<frisk::WaitBufferReadyOp>(loc, buffer);
  // Release all previous-iteration readers before reusing their slot. This is
  // explicit even on backends where readiness also includes a block barrier.
  b.create<frisk::SyncThreadsInBlockOp>(loc);
  Value remaining = b.create<arith::SubIOp>(loc, ub, iv);
  Value distance = b.create<arith::ConstantIndexOp>(loc, (stages - 1) * loop.getStepAsInt());
  Value valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remaining, distance);
  auto guard = b.create<scf::IfOp>(loc, valid, false);
  {
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value futureIv = offset(b, iv, stages - 1);
    auto future = selectSlots(b, ordinal, stages - 1);
    issue(b, futureIv, future);
  }
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), iv);
  mapping.map(loop.getRegionIterArgs(), steady.getRegionIterArgs());
  for (auto [i, transfer] : llvm::enumerate(transfers))
    for (Value handle : transfer.handles)
      mapping.map(handle, current[i]);
  for (Operation &op : loop.getBody()->without_terminator())
    if (!producers.contains(&op) && !allocations.contains(&op) && !waits.contains(&op))
      b.clone(op, mapping);
  SmallVector<Value> results;
  for (Value value : loop.getBody()->getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  // The no-iter-args builder supplies an empty terminator.
  if (!steady.getBody()->empty() &&
      steady.getBody()->back().hasTrait<OpTrait::IsTerminator>())
    steady.getBody()->back().erase();
  b.setInsertionPointToEnd(steady.getBody());
  b.create<affine::AffineYieldOp>(loc, results);
  loop.replaceAllUsesWith(steady.getResults());
  loop.erase();
}

struct AutoPipelinePass : pipeline::impl::AutoPipelineBase<AutoPipelinePass> {
  void runOnOperation() override {
    SmallVector<std::pair<Operation *, int64_t>> hints;
    bool invalid = false;
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      auto hint = op->getAttr(hintName);
      if (!hint)
        return;
      int64_t stages = 2;
      if (auto integer = dyn_cast<IntegerAttr>(hint);
          integer && integer.getValue().getBitWidth() <= 64 &&
          !integer.getValue().isNegative())
        stages = integer.getInt();
      else if (!isa<UnitAttr>(hint)) {
        op->emitError("frisk.pipeline_hint must be a positive integer (or a unit attribute for 2)");
        invalid = true;
        return;
      }
      if (stages < 1) {
        op->emitError("frisk.pipeline_hint num_stages must be at least 1");
        invalid = true;
        return;
      }
      hints.push_back({op, stages});
    });
    if (invalid)
      return signalPassFailure();
    for (auto [op, stages] : hints) {
      if (stages == 1)
        continue;
      auto loop = dyn_cast<affine::AffineForOp>(op);
      if (!loop) {
        op->emitRemark("automatic pipeline skipped: requires a Frisk Base affine.for");
        continue;
      }
      PrefetchLoop plan(loop, stages);
      if (!plan.analyze()) {
        loop.emitRemark() << "automatic pipeline skipped: " << plan.reason;
        continue;
      }
      plan.build();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pipeline::createAutoPipelinePass() {
  return std::make_unique<AutoPipelinePass>();
}
