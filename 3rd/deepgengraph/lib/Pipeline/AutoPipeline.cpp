// Opt-in block-tile prefetching. Keep the consumer recurrence in its original
// loop; only invariant-address global reads advance by num_stages - 1 tiles.
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "deepgengraph/Pipeline/SharedPipelinePlan.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/ScheduleAttrs.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
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
  return isa<frisk::GemmOp, frisk::AddOp, frisk::SubOp, frisk::MaxOp, frisk::MulOp,
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
  SmallVector<Value> issue(OpBuilder &b, Value iv, ValueRange slots);

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
  // A causal mask computes a fresh tile from its coordinate arguments. Keep
  // the region in the consumer iteration, and accept it only when every
  // nested operation is free of memory effects (including nested control
  // flow). An opaque or effectful mask body must still reject prefetching.
  if (isa<frisk::MaskOp>(op)) {
    for (Region &region : op->getRegions())
      if (region.walk([](Operation *nested) {
            return isMemoryEffectFree(nested) ? WalkResult::advance()
                                             : WalkResult::interrupt();
          }).wasInterrupted())
        return false;
    return true;
  }
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
    if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(transfer.op)) {
      for (Operation *user : copy.getToken().getUsers()) {
        if (!isa<frisk::WaitBufferReadyOp>(user) || user->getBlock() != loop.getBody())
          return reject("async task escapes the iteration or has an unsupported consumer");
        waits.insert(user);
        ready.push_back(user);
      }
    }
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

SmallVector<Value> PrefetchLoop::issue(OpBuilder &b, Value iv, ValueRange slots) {
  SmallVector<Value> tasks;
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), iv);
  for (Operation &op : loop.getBody()->without_terminator())
    if (addresses.contains(&op))
      b.clone(op, mapping);
  for (auto [i, transfer] : llvm::enumerate(transfers)) {
    Value source = mapping.lookupOrDefault(transfer.source);
    if (isa<frisk::AsyncCopyGmToShmOp>(transfer.op)) {
      mapping.map(transfer.buffer, slots[i]);
      tasks.push_back(b.clone(*transfer.op, mapping)->getResult(0));
    } else {
      auto type = cast<MemRefType>(source.getType());
      int64_t alignment = std::max<int64_t>(1, type.getElementTypeBitWidth() / 8);
      if (alignment & (alignment - 1))
        alignment = 1;
      if (auto attr = transfer.op->getAttrOfType<IntegerAttr>("alignment"))
        alignment = attr.getInt();
      tasks.push_back(b.create<frisk::AsyncCopyGmToShmOp>(
          transfer.op->getLoc(), source, slots[i], alignment).getToken());
    }
  }
  return tasks;
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
  auto taskType = frisk::friskAsyncTaskType::get(b.getContext());
  SmallVector<Type> taskTypes(transfers.size(), taskType);
  Value completed = b.create<frisk::AsyncTaskCompletedOp>(loc, taskType);
  SmallVector<Value> emptyTasks(transfers.size(), completed), initialTasks;
  for (int64_t i = 0; i < stages - 1; ++i) {
    Value distance = b.create<arith::ConstantIndexOp>(loc, i * loop.getStepAsInt());
    Value enough = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, extent, distance);
    Value valid = b.create<arith::AndIOp>(loc, nonempty, enough);
    auto guard = b.create<scf::IfOp>(loc, taskTypes, valid, true);
    {
      OpBuilder::InsertionGuard insertion(b);
      b.setInsertionPointToStart(&guard.getThenRegion().front());
      Value iv = offset(b, lb, i);
      SmallVector<Value> slots;
      for (auto &transfer : transfers) slots.push_back(transfer.slots[i]);
      b.create<scf::YieldOp>(loc, issue(b, iv, slots));
      b.setInsertionPointToStart(&guard.getElseRegion().front());
      b.create<scf::YieldOp>(loc, emptyTasks);
    }
    llvm::append_range(initialTasks, guard.getResults());
  }
  unsigned originalCarriers = loop.getNumIterOperands();
  SmallVector<Value> inits(loop.getInits());
  llvm::append_range(inits, initialTasks);
  auto steady = b.create<affine::AffineForOp>(loc,
      loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), inits);
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
  auto taskQueue = steady.getRegionIterArgs().drop_front(originalCarriers);
  for (Value task : taskQueue.take_front(transfers.size()))
    b.create<frisk::WaitBufferReadyOp>(loc, task);
  // Release all previous-iteration readers before reusing their slot. This is
  // explicit even on backends where readiness also includes a block barrier.
  b.create<frisk::SyncThreadsInBlockOp>(loc);
  Value remaining = b.create<arith::SubIOp>(loc, ub, iv);
  Value distance = b.create<arith::ConstantIndexOp>(loc, (stages - 1) * loop.getStepAsInt());
  Value valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remaining, distance);
  auto guard = b.create<scf::IfOp>(loc, taskTypes, valid, true);
  {
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value futureIv = offset(b, iv, stages - 1);
    auto future = selectSlots(b, ordinal, stages - 1);
    b.create<scf::YieldOp>(loc, issue(b, futureIv, future));
    b.setInsertionPointToStart(&guard.getElseRegion().front());
    b.create<scf::YieldOp>(loc, emptyTasks);
  }
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), iv);
  mapping.map(loop.getRegionIterArgs(), steady.getRegionIterArgs().take_front(originalCarriers));
  for (auto [i, transfer] : llvm::enumerate(transfers))
    for (Value handle : transfer.handles)
      mapping.map(handle, current[i]);
  for (Operation &op : loop.getBody()->without_terminator())
    if (!producers.contains(&op) && !allocations.contains(&op) && !waits.contains(&op))
      b.clone(op, mapping);
  SmallVector<Value> results;
  for (Value value : loop.getBody()->getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  llvm::append_range(results, taskQueue.drop_front(transfers.size()));
  llvm::append_range(results, guard.getResults());
  // The no-iter-args builder supplies an empty terminator.
  if (!steady.getBody()->empty() &&
      steady.getBody()->back().hasTrait<OpTrait::IsTerminator>())
    steady.getBody()->back().erase();
  b.setInsertionPointToEnd(steady.getBody());
  b.create<affine::AffineYieldOp>(loc, results);
  // Keep completion explicit for every FIFO slot, including a prologue task
  // carried through a zero/short loop. Bounds guards normally leave completed
  // tasks here, but storage liveness must not depend on recognizing that guard
  // arithmetic. Draining the exact exit tokens closes the structured task-flow
  // proof and permits storage reuse before/after the entire pipeline region.
  b.setInsertionPointAfter(steady);
  for (Value task : steady.getResults().drop_front(originalCarriers))
    b.create<frisk::WaitBufferReadyOp>(loc, task);
  loop.replaceAllUsesWith(steady.getResults().take_front(originalCarriers));
  loop.erase();
}

// Logical scheduling belongs here, before matrix/fragment lowering. Keep each
// slice's independent allocation and serial stream order as a fallback until
// SharedMemoryPlan has proved the physical packet, reader and storage contract.
// The first issue has no stream anchor and retains ordinary safe DMA hoisting
// across preceding compute; later issues are anchored to earlier readiness.
static Value sharedSource(Value value) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    Operation *def = value.getDefiningOp();
    if (auto view = dyn_cast_or_null<frisk::BufferViewOp>(def)) value = view.getSource();
    else if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def)) value = view.getViewSource();
    else if (auto cast = dyn_cast_or_null<memref::CastOp>(def)) value = cast.getSource();
    else return value;
  }
  return {};
}

static void planSharedCopyStream(func::FuncOp kernel, Block &block, Value stream) {
  using namespace frisk;
  using pipeline::sharedPipelinePlanAttr;
  if (!isInBlockUniformControlFlow(block.getTerminator())) return;
  for (Operation *p = block.getParentOp(); p; p = p->getParentOp())
    if (hasAnyScheduleAttr(p)) return;
  struct Stage {
    AsyncCopyGmToShmOp issue;
    WaitBufferReadyOp ready;
    SmallVector<Operation *> readers;
  };
  SmallVector<Stage> stages;
  Value resource;
  MemRefType shape;
  auto origin = [](Value value) {
    for (unsigned depth = 0; depth < 32; ++depth) {
      Operation *def = value.getDefiningOp();
      if (auto view = dyn_cast_or_null<BufferViewOp>(def)) value = view.getSource();
      else if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def)) value = view.getViewSource();
      else if (auto cast = dyn_cast_or_null<memref::CastOp>(def)) value = cast.getSource();
      else return value;
    }
    return Value{};
  };
  for (Operation &op : block) {
    auto copy = dyn_cast<AsyncCopyGmToShmOp>(op);
    if (!copy) continue;
    if (sharedSource(copy.getSrc()) != stream) continue;
    // A committed or existing provisional schedule must never be rediscovered.
    if (copy->hasAttr(sharedPipelinePlanAttr) || hasAnyScheduleAttr(copy)) return;
    auto alloc = copy.getDst().getDefiningOp<AllocBufferOp>();
    auto src = dyn_cast<MemRefType>(copy.getSrc().getType());
    auto dst = dyn_cast<MemRefType>(copy.getDst().getType());
    if (!alloc || alloc->getBlock() != &block || !src || !dst ||
        src.getMemorySpaceAsInt() != 1 || dst.getMemorySpaceAsInt() != 3 ||
        !dst.hasStaticShape() || src.getShape() != dst.getShape() ||
        src.getElementType() != dst.getElementType()) continue;
    if (!copy.getToken().hasOneUse()) return;
    auto ready = dyn_cast<WaitBufferReadyOp>(*copy.getToken().getUsers().begin());
    if (!ready || ready->getBlock() != &block || !copy->isBeforeInBlock(ready)) return;
    bool matrixB = llvm::any_of(copy.getDst().getUses(), [](OpOperand &use) {
      return isa<GemmOp>(use.getOwner()) && use.getOperandNumber() == 1;
    });
    if (!matrixB) continue;
    Stage stage{copy, ready, {}};
    for (OpOperand &use : copy.getDst().getUses()) {
      Operation *user = use.getOwner();
      if (user == copy) continue;
      // Only direct matrix B readers establish this stream. Aliases, mutation,
      // extra readers and nested consumers need a richer logical contract.
      auto gemm = dyn_cast<GemmOp>(user);
      if (!gemm || use.getOperandNumber() != 1 || user->getBlock() != &block ||
          !ready->isBeforeInBlock(user) || hasAnyScheduleAttr(user)) return;
      stage.readers.push_back(user);
    }
    if (stage.readers.empty()) continue;
    Value root = origin(copy.getSrc());
    if (!root) return;
    if (stages.empty()) { resource = root; shape = dst; }
    if (root != resource || dst != shape) return;
    if (!stages.empty())
      for (Operation *reader : stages.back().readers)
        if (!reader->isBeforeInBlock(copy)) return;
    stages.push_back(std::move(stage));
  }
  if (stages.size() < 2 || stages.size() > 32) return;
  DenseSet<Operation *> allowed;
  for (auto &stage : stages) {
    allowed.insert(stage.issue);
    allowed.insert(stage.ready);
    allowed.insert(stage.issue.getDst().getDefiningOp());
    for (Operation *reader : stage.readers) allowed.insert(reader);
  }
  Operation *last = stages.back().ready;
  for (Operation *reader : stages.back().readers)
    if (last->isBeforeInBlock(reader)) last = reader;
  for (Operation *op = stages.front().ready; ; op = op->getNextNode()) {
    if (hasAnyScheduleAttr(op) || op->getNumRegions() ||
        (!allowed.contains(op) && !isPure(op))) return;
    if (op == last) break;
  }
  DominanceInfo dominance(kernel);
  for (unsigned i = 1; i < stages.size(); ++i) {
    DenseSet<Operation *> visited;
    std::function<bool(Value)> available = [&](Value value) {
      if (dominance.dominates(value, stages[i - 1].ready)) return true;
      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != &block || def->getNumRegions() || !isPure(def)) return false;
      if (!visited.insert(def).second) return true;
      return llvm::all_of(def->getOperands(), available);
    };
    if (!available(stages[i].issue.getSrc())) return;
  }
  OpBuilder b(kernel.getContext());
  for (auto [i, stage] : llvm::enumerate(stages))
    stage.issue->setAttr(sharedPipelinePlanAttr, b.getDenseI64ArrayAttr(
        {int64_t(i), int64_t(stages.size()), int64_t(i % 2), int64_t(i) - 1}));
  llvm::errs() << "AutoPipeline shared-copy plan: stages=" << stages.size()
               << ", slots=2, issue=after-previous-ready, function="
               << kernel.getSymName() << '\n';
}

static void planSharedCopyStreams(func::FuncOp kernel, Block &block) {
  // Expanded QK and PV now share a block. Analyze source streams separately;
  // effects between stages still prevent an unsafe partial/interleaved stream.
  // The current physical two-slot contract owns one stream per block. Prefer
  // the tail stream and leave other streams to ordinary dependency scheduling.
  SmallVector<Value> sources;
  for (Operation &op : block) {
    if (op.hasAttr(pipeline::sharedPipelinePlanAttr)) return;
    if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
      Value source = sharedSource(copy.getSrc());
      if (source && !llvm::is_contained(sources, source)) sources.push_back(source);
    }
  }
  for (Value source : llvm::reverse(sources)) {
    planSharedCopyStream(kernel, block, source);
    if (llvm::any_of(block, [](Operation &op) {
          return op.hasAttr(pipeline::sharedPipelinePlanAttr);
        })) break;
  }
}

static LogicalResult exposeMarkedInnerLoops(func::FuncOp kernel) {
  SmallVector<affine::AffineForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) {
    auto full = loop->getAttrOfType<BoolAttr>("frisk.loopUnrollFull");
    if (!full || !full.getValue()) return;
    bool inner = false;
    for (Operation *p = loop; p && p != kernel; p = p->getParentOp()) {
      if (p->hasAttr(hintName) || frisk::hasAnyScheduleAttr(p) ||
          p->hasAttr(pipeline::crossIterationPlanAttr)) return;
      if (p != loop && isa<affine::AffineForOp, scf::ForOp>(p)) inner = true;
    }
    if (inner) loops.push_back(loop);
  });
  uint64_t remaining = 65536;
  bool expanded = false;
  for (auto loop : loops) {
    auto count = affine::getConstantTripCount(loop);
    if (!count || !*count || *count > 64) continue;
    uint64_t size = 0;
    loop.getBody()->walk([&](Operation *) { ++size; });
    uint64_t cost = size * (*count - 1);
    if (cost > remaining) continue;
    // Frisk Base's legacy vector copy mutates subsequent uses of an iter_arg.
    // MLIR's generic unroller only understands SSA yields. Restore that SSA
    // recurrence first, or all clones would write/read the initial carrier.
    SmallVector<frisk::CopyOp> updates;
    DenseSet<Value> destinations;
    bool valid = true;
    loop.walk([&](frisk::CopyOp copy) {
      auto dst = dyn_cast<VectorType>(copy.getDst().getType());
      if (!dst || copy.hasValueResult()) return;
      auto arg = dyn_cast<BlockArgument>(copy.getDst());
      auto src = dyn_cast<ShapedType>(copy.getSrc().getType());
      auto map = copy.getOffsetMap();
      auto rank = map.getNumResults() == 1
                      ? dyn_cast<AffineConstantExpr>(map.getResult(0)) : AffineConstantExpr{};
      if (copy->getBlock() != loop.getBody() || !arg || arg.getOwner() != loop.getBody() ||
          arg.getArgNumber() == 0 || !src || src.getShape() != dst.getShape() ||
          src.getElementType() != dst.getElementType() || map.getNumInputs() != 0 ||
          !rank || rank.getValue() != dst.getRank() || !destinations.insert(arg).second) {
        valid = false;
        return;
      }
      for (Operation *user : arg.getUsers())
        if (user->getBlock() != loop.getBody()) valid = false;
      updates.push_back(copy);
    });
    if (!valid) continue;
    for (auto copy : updates) {
      OpBuilder b(copy);
      Value next = b.create<frisk::CastOp>(copy.getLoc(), copy.getDst().getType(), copy.getSrc());
      for (OpOperand &use : llvm::make_early_inc_range(copy.getDst().getUses()))
        if (use.getOwner() != copy && copy->isBeforeInBlock(use.getOwner())) use.set(next);
      copy.erase();
    }
    if (failed(affine::loopUnrollFull(loop))) return failure();
    remaining -= cost;
    expanded = true;
  }
  if (!expanded) return success();
  // Unrolling makes the temporary vector carriers redundant. Keep the direct
  // computed-tile edges visible to layout inference: otherwise an artificial
  // memref/vector boundary can materialize the QK accumulator in shared memory.
  SmallVector<frisk::CastOp> casts;
  kernel.walk([&](frisk::CastOp cast) { casts.push_back(cast); });
  for (auto outer : casts) {
    auto inner = outer->getOperand(0).getDefiningOp<frisk::CastOp>();
    if (!inner || !isa<VectorType>(inner->getResult(0).getType())) continue;
    Value original = inner->getOperand(0);
    if (original.getType() != outer->getResult(0).getType() ||
        !isa_and_nonnull<frisk::GemmOp, frisk::ZeroOp>(original.getDefiningOp())) continue;
    outer->getResult(0).replaceAllUsesWith(original);
    outer.erase();
    // Do not erase inner until after the traversal: it may be in casts too.
  }
  SmallVector<frisk::CastOp> dead;
  kernel.walk([&](frisk::CastOp cast) {
    if (cast->use_empty()) dead.push_back(cast);
  });
  for (auto cast : llvm::reverse(dead)) cast.erase();
  return success();
}

static void planCrossIteration(func::FuncOp kernel, Operation *loop) {
  using namespace frisk;
  using namespace pipeline;
  Block *body;
  Value iv;
  if (auto affine = dyn_cast<affine::AffineForOp>(loop)) {
    body = affine.getBody(); iv = affine.getInductionVar();
  } else if (auto scf = dyn_cast<scf::ForOp>(loop)) {
    auto step = scf.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!step || step.value() <= 0 || !scf.getInductionVar().getType().isIndex()) return;
    body = scf.getBody(); iv = scf.getInductionVar();
  } else return;
  if (loop->hasAttr(crossIterationPlanAttr) || loop->hasAttr(hintName) ||
      hasAnyScheduleAttr(loop) || !isInBlockUniformControlFlow(body->getTerminator())) return;
  DominanceInfo dominance(kernel);
  auto address = [&](Value source, bool &varies) {
    DenseSet<Operation *> visited;
    std::function<bool(Value)> visit = [&](Value value) {
      if (value == iv) { varies = true; return true; }
      if (dominance.dominates(value, loop)) return true;
      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != body || def->getNumRegions() || !isPure(def)) return false;
      if (!visited.insert(def).second) return true;
      return llvm::all_of(def->getOperands(), visit);
    };
    return visit(source);
  };
  auto localDestination = [&](Value value) {
    if (isa<VectorType>(value.getType())) return true;
    auto type = dyn_cast<MemRefType>(value.getType());
    Operation *def = value.getDefiningOp();
    return type && type.getMemorySpaceAsInt() != 1 && def &&
           def->getBlock() == body &&
           (isa<AllocBufferOp, ZeroOp>(def) || readsOperands(def));
  };
  AsyncCopyGmToShmOp opening, tail;
  WaitBufferReadyOp openingReady, tailReady;
  for (Operation &op : *body) {
    if (hasAnyScheduleAttr(&op) || op.hasAttr(crossIterationIssueAttr) ||
        op.hasAttr(crossIterationAnchorAttr) || op.getNumRegions()) return;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) {
      auto alloc = copy.getDst().getDefiningOp<AllocBufferOp>();
      auto src = dyn_cast<MemRefType>(copy.getSrc().getType());
      auto dst = dyn_cast<MemRefType>(copy.getDst().getType());
      if (!alloc || alloc->getBlock() != body || !src || !dst ||
          src.getMemorySpaceAsInt() != 1 || dst.getMemorySpaceAsInt() != 3 ||
          !dst.hasStaticShape() || src.getShape() != dst.getShape() ||
          src.getElementType() != dst.getElementType() || !copy.getToken().hasOneUse()) return;
      auto ready = dyn_cast<WaitBufferReadyOp>(*copy.getToken().getUsers().begin());
      if (!ready || ready->getBlock() != body || !copy->isBeforeInBlock(ready)) return;
      bool varies = false;
      if (!address(copy.getSrc(), varies)) return;
      bool matrixB = llvm::any_of(copy.getDst().getUses(), [](OpOperand &use) {
        return isa<GemmOp>(use.getOwner()) && use.getOperandNumber() == 1;
      });
      if (!opening && varies) {
        if (!matrixB) return;
        opening = copy; openingReady = ready;
      }
      tail = copy; tailReady = ready;
      continue;
    }
    if (auto wait = dyn_cast<WaitBufferReadyOp>(op)) {
      auto copy = wait.getToken().getDefiningOp<AsyncCopyGmToShmOp>();
      if (!copy || copy->getBlock() != body || !copy.getToken().hasOneUse()) return;
      continue;
    }
    if (isa<AllocBufferOp, ZeroOp>(op) || readsOperands(&op) ||
        op.hasTrait<OpTrait::IsTerminator>()) continue;
    if (auto copy = dyn_cast<CopyOp>(op)) {
      if (!localDestination(copy.getDst())) return;
      continue;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op)) {
      if (!localDestination(reduce->getOperand(1))) return;
      continue;
    }
    if (!isPure(&op)) return;
  }
  if (!opening || opening == tail || !openingReady->isBeforeInBlock(tailReady)) return;
  // The opening buffer must be dead before the anchor; the tail must still
  // have matrix work afterwards. Physical readers/aliases are revalidated late.
  for (OpOperand &use : opening.getDst().getUses()) {
    Operation *user = use.getOwner();
    if (user == opening) continue;
    if (!isa<GemmOp>(user) || use.getOperandNumber() != 1 || user->getBlock() != body ||
        !openingReady->isBeforeInBlock(user) || !user->isBeforeInBlock(tailReady)) return;
  }
  bool finalRead = false;
  for (OpOperand &use : tail.getDst().getUses()) {
    Operation *user = use.getOwner();
    if (user == tail) continue;
    if (!isa<GemmOp>(user) || use.getOperandNumber() != 1 || user->getBlock() != body ||
        !tailReady->isBeforeInBlock(user)) return;
    finalRead = true;
  }
  if (!finalRead) return;
  OpBuilder b(kernel.getContext());
  loop->setAttr(crossIterationPlanAttr, b.getUnitAttr());
  opening->setAttr(crossIterationIssueAttr, b.getUnitAttr());
  tail->setAttr(crossIterationAnchorAttr, b.getUnitAttr());
  llvm::errs() << "AutoPipeline cross-iteration plan: opening DMA, issue-after=tail-ready, function="
               << kernel.getSymName() << '\n';
}

struct AutoPipelinePass : pipeline::impl::AutoPipelineBase<AutoPipelinePass> {
  void runOnOperation() override {
    SmallVector<std::pair<Operation *, int64_t>> hints;
    if ((doubleBuffer || crossIteration) && failed(exposeMarkedInnerLoops(getOperation()))) {
      signalPassFailure();
      return;
    }
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
    if ((doubleBuffer || crossIteration) && !getOperation().isExternal()) {
      // Match the execution planner's supported scope: the function body or
      // one outer compute loop. Hinted nested pipelines retain their owner.
      Block &entry = getOperation().getBody().front();
      if (doubleBuffer) planSharedCopyStreams(getOperation(), entry);
      for (Operation &op : entry) {
        if (!isa<affine::AffineForOp, scf::ForOp>(op)) continue;
        if (doubleBuffer) planSharedCopyStreams(getOperation(), op.getRegion(0).front());
        if (crossIteration) planCrossIteration(getOperation(), &op);
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pipeline::createAutoPipelinePass() {
  return std::make_unique<AutoPipelinePass>();
}
