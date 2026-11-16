// Joint shared-memory execution planning: DMA issue times, logical buffer
// lifetimes, ping-pong slots and physical pool placement, before IR mutation.
#include "deepgengraph/Analysis/SharedMemoryPlan.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/ScheduleAttrs.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>
#include <optional>

namespace mlir::frisk {
namespace {
constexpr StringLiteral marker = "frisk.shared_pipeline";
using Loop = affine::AffineForOp;

static bool barrier(Operation *op) {
  return op && isa<gpu::BarrierOp, SyncThreadsInBlockOp>(op);
}
static Operation *topIn(Operation *op, Block *block) {
  while (op && op->getBlock() != block) op = op->getParentOp();
  return op;
}
static bool shared(Value v) {
  auto t = dyn_cast<MemRefType>(v.getType());
  return t && t.getMemorySpaceAsInt() == 3;
}
static std::optional<int64_t> constant(Value v) {
  APInt n;
  if (matchPattern(v, m_ConstantInt(&n)) && n.getBitWidth() <= 64)
    return n.getSExtValue();
  return std::nullopt;
}
// Only descriptors made from block-uniform values may be collective DMA inputs.
static bool uniform(Value v, func::FuncOp f, unsigned depth = 0) {
  if (depth > 64) return false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getOwner() == &f.getBody().front()) return true;
    auto *parent = arg.getOwner()->getParentOp();
    return arg.getArgNumber() == 0 && isa<Loop, scf::ForOp>(parent) &&
           isInBlockUniformControlFlow(arg.getOwner()->getTerminator());
  }
  auto *op = v.getDefiningOp();
  if (!op) return false;
  if (op->hasTrait<OpTrait::ConstantLike>() ||
      isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp, memref::AllocOp>(op))
    return true;
  if (op->getNumRegions() || !isPure(op) ||
      !(isa<affine::AffineApplyOp, ViewLikeOpInterface, memref::CastOp>(op) ||
        op->getName().getDialectNamespace() == "arith")) return false;
  return llvm::all_of(op->getOperands(), [&](Value x) { return uniform(x, f, depth + 1); });
}
static Value origin(Value v) {
  for (unsigned i = 0; i < 32; ++i) {
    auto *op = v.getDefiningOp();
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(op)) v = view.getViewSource();
    else if (auto cast = dyn_cast_or_null<memref::CastOp>(op)) v = cast.getSource();
    else return v;
  }
  return {};
}

struct Span { int64_t begin, size; };
// Full descriptor extent, deliberately conservative for readers. Unknown
// selectors, strided aliases and dynamic offsets fail closed.
static std::optional<Span> extent(Value v, memref::AllocOp pool, unsigned depth = 0) {
  if (depth > 32) return std::nullopt;
  auto t = dyn_cast<MemRefType>(v.getType());
  if (!t || !t.hasStaticShape() || !t.getLayout().isIdentity() ||
      !t.getElementType().isIntOrFloat() || t.getElementTypeBitWidth() % 8)
    return std::nullopt;
  int64_t size = t.getElementTypeBitWidth() / 8;
  for (int64_t dim : t.getShape()) {
    if (dim <= 0 || size > (1 << 20) / dim) return std::nullopt;
    size *= dim;
  }
  if (v == pool.getResult()) return Span{0, size};
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    auto s = extent(cast.getSource(), pool, depth + 1);
    if (s && s->size == size) return s;
  }
  if (auto view = v.getDefiningOp<memref::ViewOp>()) {
    auto source = extent(view.getSource(), pool, depth + 1);
    auto offset = constant(view.getByteShift());
    if (source && offset && *offset >= 0 && *offset <= source->size &&
        size <= source->size - *offset) return Span{source->begin + *offset, size};
  }
  return std::nullopt;
}

// Evaluate the lane/fragment index DAG without specializing the IR. Bound all
// enumeration and reject anything involving unknown values, loads or overflow.
static std::optional<int64_t> evaluate(Value v, unsigned lane,
                                     const DenseMap<Value, int64_t> &ivs,
                                     unsigned depth = 0) {
  if (depth > 64) return std::nullopt;
  if (auto it = ivs.find(v); it != ivs.end()) return it->second;
  if (auto n = constant(v)) return n;
  if (auto tid = v.getDefiningOp<gpu::ThreadIdOp>())
    return tid.getDimension() == gpu::Dimension::x ? lane : 0;
  if (auto apply = v.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> inputs, outputs;
    for (Value x : apply.getMapOperands()) {
      auto n = evaluate(x, lane, ivs, depth + 1);
      if (!n) return std::nullopt;
      inputs.push_back(IntegerAttr::get(IndexType::get(v.getContext()), *n));
    }
    if (failed(apply.getAffineMap().constantFold(inputs, outputs))) return std::nullopt;
    return cast<IntegerAttr>(outputs.front()).getInt();
  }
  auto *op = v.getDefiningOp();
  if (!op || op->getNumOperands() != 2 || !v.getType().isIndex()) return std::nullopt;
  auto a = evaluate(op->getOperand(0), lane, ivs, depth + 1);
  auto b = evaluate(op->getOperand(1), lane, ivs, depth + 1);
  if (!a || !b) return std::nullopt;
  int64_t out;
  if (isa<arith::AddIOp>(op) && !__builtin_add_overflow(*a, *b, &out)) return out;
  if (isa<arith::SubIOp>(op) && !__builtin_sub_overflow(*a, *b, &out)) return out;
  if (isa<arith::MulIOp>(op) && !__builtin_mul_overflow(*a, *b, &out)) return out;
  if (*a >= 0 && *b > 0) {
    if (isa<arith::DivUIOp>(op)) return *a / *b;
    if (isa<arith::RemUIOp>(op)) return *a % *b;
  }
  return std::nullopt;
}

static bool containedRead(ReadMatrixFromShmOp read, Block *scope, Span slice,
                          memref::AllocOp pool, unsigned threads) {
  auto s = extent(read.getSrc(), pool);
  auto t = cast<MemRefType>(read.getSrc().getType());
  if (!s || s->begin < slice.begin || s->begin + s->size > slice.begin + slice.size ||
      t.getRank() != 1 || read.getIndices().size() != 1 ||
      read.getIntrinsicName() != "llvm.hcu.ds.read.m32x16.b16" ||
      read.getResult().getType() != VectorType::get({4}, IntegerType::get(read.getContext(), 32)))
    return false;
  SmallVector<Loop> loops;
  uint64_t count = threads;
  for (auto *p = read->getParentOp(); p != scope->getParentOp(); p = p->getParentOp()) {
    auto loop = dyn_cast<Loop>(p);
    if (!loop || !loop.hasConstantBounds()) return false;
    int64_t lo = loop.getConstantLowerBound(), hi = loop.getConstantUpperBound();
    int64_t step = loop.getStepAsInt(), distance;
    if (step <= 0 || __builtin_sub_overflow(hi, lo, &distance) || distance <= 0 ||
        distance > 256 || count > 262144 / uint64_t(distance)) return false;
    count *= distance;
    loops.push_back(loop);
  }
  DenseMap<Value, int64_t> ivs;
  std::function<bool(unsigned)> visit = [&](unsigned i) {
    if (i < loops.size()) {
      auto loop = loops[i];
      for (int64_t iv = loop.getConstantLowerBound(); iv < loop.getConstantUpperBound();) {
        ivs[loop.getInductionVar()] = iv;
        if (!visit(i + 1)) return false;
        if (__builtin_add_overflow(iv, loop.getStepAsInt(), &iv)) return false;
      }
      return true;
    }
    for (unsigned lane = 0; lane < threads; ++lane) {
      auto n = evaluate(read.getIndices()[0], lane, ivs);
      if (!n || *n < 0 || *n >= t.getNumElements()) return false;
      int64_t address = s->begin + *n * (t.getElementTypeBitWidth() / 8) + read.getOffset();
      if (address < slice.begin || address > slice.begin + slice.size - 16) return false;
    }
    return true;
  };
  return visit(0);
}

static bool direct(AsyncCopyGmToShmOp issue, memref::AllocOp pool,
                   unsigned threads) {
  auto s = extent(issue.getDst(), pool);
  auto width = issue->getAttrOfType<IntegerAttr>("frisk.dma_bytes");
  auto src = cast<MemRefType>(issue.getSrc().getType());
  // Other live transfers may already use a narrower packet selected by the
  // shared-memory planner. Audit their actual complete-wave footprint instead
  // of relying on the removed late 16B -> 8B LLVM rewrite.
  int64_t bytes = width ? width.getInt() : 0;
  if (auto copying = issue->getAttrOfType<IntegerAttr>("frisk.copy_threads")) {
    if (copying.getInt() <= 0 || copying.getInt() > threads || copying.getInt() % 64) return false;
    threads = copying.getInt();
  }
  return s && src.getMemorySpaceAsInt() == 1 &&
         !issue->hasAttr("frisk.force_sync") && bytes > 0 && bytes <= 16 &&
         s->begin % bytes == 0 && s->size % (threads * bytes) == 0 &&
         s->size / (threads * bytes) <= 16 && issue.getAlignment() % bytes == 0 &&
         uniform(issue.getSrc(), issue->getParentOfType<func::FuncOp>()) &&
         selectAsyncDMATransferBytes(issue, bytes) == bytes;
}

// A stream is discovered by task completion and matrix readers, never by an
// argument number or a Q/K/V name. Each stage still owns an independent logical
// allocation; the common allocator will assign the alternating physical slots.
static bool collectPipeline(func::FuncOp f, Block &block,
                            const ShmLivenessResult &live, unsigned threads,
                            SharedBufferingPlan &pipeline, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  if (!isInBlockUniformControlFlow(block.getTerminator()))
    return fail("non-uniform or unsupported control flow");
  Operation *owner = block.getParentOp();
  if (owner != f && (!isa<Loop, scf::ForOp>(owner) ||
                     owner->getBlock() != &f.getBody().front()))
    return fail("requires one outer compute loop");
  DenseMap<Value, unsigned> buffers;
  for (auto [i, info] : llvm::enumerate(live.buffers))
    if (info.reusable) buffers[info.allocation->getResult(0)] = i;
  WaitBufferReadyOp ready;
  for (Operation &op : block) {
    if (auto wait = dyn_cast<WaitBufferReadyOp>(op)) ready = wait;
    bool bad = false;
    op.walk([&](ReadMatrixFromShmOp read) {
      if (read->hasAttr(marker) || !ready) { bad = true; return; }
      if (pipeline.stages.empty() || pipeline.stages.back().ready != ready) {
        auto issue = ready.getToken().getDefiningOp<AsyncCopyGmToShmOp>();
        if (!issue || issue->getBlock() != &block || !issue->isBeforeInBlock(ready) ||
            !issue.getToken().hasOneUse() || !buffers.count(issue.getDst())) {
          bad = true; return;
        }
        SharedPipelineStage stage;
        stage.issue = issue;
        stage.ready = ready;
        stage.bufferIndex = buffers.lookup(issue.getDst());
        stage.slot = pipeline.stages.size() % 2;
        if (!pipeline.stages.empty()) stage.issueAfter = pipeline.stages.back().ready;
        pipeline.stages.push_back(std::move(stage));
      }
      pipeline.stages.back().reads.push_back(read);
    });
    if (bad) return fail("already planned, incomplete task or unsupported logical allocation");
  }
  if (pipeline.stages.size() < 2 || pipeline.stages.size() > 32)
    return fail("requires 2..32 matrix-copy stages");
  Value resource = origin(pipeline.stages.front().issue.getSrc());
  if (!resource) return fail("unknown DMA source");
  llvm::SmallPtrSet<Operation *, 32> allowed;
  llvm::SmallDenseSet<unsigned, 16> distinct;
  for (auto [index, stage] : llvm::enumerate(pipeline.stages)) {
    const auto &info = live.buffers[stage.bufferIndex];
    auto alloc = info.allocation;
    if (!distinct.insert(stage.bufferIndex).second)
      return fail("stages must own distinct logical buffers before pooling");
    if (!pipeline.slotBytes) pipeline.slotBytes = info.sizeBytes;
    if (info.sizeBytes != pipeline.slotBytes || info.asyncAcrossIterations ||
        origin(stage.issue.getSrc()) != resource || !direct(stage.issue, alloc, threads))
      return fail("mixed sources, sizes or unsupported direct DMA");
    allowed.insert(alloc);
    allowed.insert(stage.issue);
    allowed.insert(stage.ready);
    llvm::SmallPtrSet<Operation *, 16> accesses{stage.issue, stage.ready};
    for (auto read : stage.reads) {
      if (!containedRead(read, &block, {0, int64_t(info.sizeBytes)}, alloc, threads))
        return fail("unknown matrix address or read outside DMA slice");
      accesses.insert(read);
      allowed.insert(read);
    }
    // All aliases/users must be accounted for before extending the lifetime.
    for (Operation *access : info.accesses)
      if (!accesses.contains(access)) return fail("buffer has another reader, writer or escape");
    if (stage.issueAfter &&
        !topIn(pipeline.stages[index - 1].reads.back(), &block)
             ->isBeforeInBlock(stage.issue))
      return fail("unsupported stage order");
  }
  Operation *last = topIn(pipeline.stages.back().reads.back(), &block);
  for (Operation *op = pipeline.stages.front().ready; ; op = op->getNextNode()) {
    bool unsafe = op->walk([&](Operation *nested) {
      for (Operation *p = nested; p; p = p->getParentOp())
        if (hasAnyScheduleAttr(p)) return WalkResult::interrupt();
      if (allowed.contains(nested)) return WalkResult::advance();
      if (isa<Loop, scf::ForOp>(nested))
        return isInBlockUniformControlFlow(nested->getRegion(0).front().getTerminator())
                   ? WalkResult::advance() : WalkResult::interrupt();
      if (nested->getNumRegions() || !isMemoryEffectFree(nested) || barrier(nested) ||
          isa<ToThreadTileOp, FromThreadTileOp>(nested)) return WalkResult::interrupt();
      return WalkResult::advance();
    }).wasInterrupted();
    if (unsafe) return fail("explicit synchronization, scheduling or unknown effect in stream");
    if (op == last) break;
  }
  // The ordinary physical preparation establishes loop-backedge protection.
  // Do not infer cross-wave completion from the last lane's LDS load.
  if (owner != f) {
    bool finish = false;
    for (Operation *op = last->getNextNode(); op; op = op->getNextNode())
      finish |= barrier(op);
    if (!finish) return fail("missing final reader barrier");
  }
  DominanceInfo dominance(f);
  for (auto &stage : pipeline.stages) {
    if (!stage.issueAfter) continue;
    Operation *anchor = stage.issueAfter->getNextNode();
    llvm::SetVector<Operation *> dependencies;
    std::function<bool(Value)> visit = [&](Value value) {
      if (dominance.dominates(value, anchor)) return true;
      auto *def = value.getDefiningOp();
      if (!def || def->getBlock() != &block || def->getNumRegions() || !isPure(def) ||
          isa<ToThreadTileOp, FromThreadTileOp>(def)) return false;
      if (dependencies.contains(def)) return true;
      for (Value operand : def->getOperands()) if (!visit(operand)) return false;
      dependencies.insert(def);
      return true;
    };
    // Destination allocations are replaced by the already-planned entry views.
    if (!visit(stage.issue.getSrc())) return fail("DMA address depends on unavailable effect");
    stage.addressDependencies.assign(dependencies.begin(), dependencies.end());
  }
  return true;
}

static std::string snapshot(func::FuncOp kernel) {
  std::string text;
  llvm::raw_string_ostream os(text);
  kernel.print(os, OpPrintingFlags().useLocalScope());
  return text;
}
} // namespace

SharedMemoryPlan planSharedMemoryExecution(func::FuncOp kernel, bool doubleBuffer) {
  SharedMemoryPlan result;
  auto &execution = result.execution.emplace();
  execution.kernel = kernel;
  execution.inputSnapshot = snapshot(kernel);
  bool deferred = kernel.isExternal();
  kernel.walk([&](Operation *op) {
    deferred |= isa<ToThreadTileOp, FromThreadTileOp>(op) ||
        llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<FragmentType>(t); }) ||
        llvm::any_of(op->getResultTypes(), [](Type t) { return isa<FragmentType>(t); });
  });
  if (deferred) return result;
  execution.lifetimes = analyzeShmLiveness(kernel);
  auto baseline = planShmReuse(execution.lifetimes);
  execution.storage = baseline;
  auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
  if (doubleBuffer && threads && threads.getInt() >= 64 && threads.getInt() <= 1024 &&
      threads.getInt() % 64 == 0 && llvm::hasSingleElement(kernel.getBody())) {
    llvm::SetVector<Block *> blocks;
    kernel.walk([&](WaitBufferReadyOp wait) { blocks.insert(wait->getBlock()); });
    for (Block *block : blocks) {
      if (block == &kernel.getBody().front() &&
          llvm::any_of(blocks, [&](Block *b) { return b != block; })) continue;
      bool hasReads = false;
      block->walk([&](ReadMatrixFromShmOp) { hasReads = true; });
      if (!hasReads) continue;
      SharedBufferingPlan pipeline;
      std::string reason;
      if (!collectPipeline(kernel, *block, execution.lifetimes, threads.getInt(), pipeline, reason)) {
        execution.rejectedPipelines.push_back(std::move(reason));
        continue;
      }
      // Project the earlier issue into the allocation lifetime WITHOUT moving
      // IR. Use the operation after readiness as the conservative first-use
      // point, so that readiness's collective barrier protects the old reader.
      auto projected = execution.lifetimes;
      for (const auto &stage : pipeline.stages) {
        if (!stage.issueAfter) continue;
        auto &accesses = projected.buffers[stage.bufferIndex].accesses;
        for (Operation *&access : accesses)
          if (access == stage.issue) access = stage.issueAfter->getNextNode();
      }
      auto storage = planShmReuse(projected);
      DenseMap<unsigned, uint64_t> offsets;
      for (auto placement : storage.placements) offsets[placement.bufferIndex] = placement.offsetBytes;
      bool valid = true;
      for (unsigned i = 0; i < 2; ++i) {
        unsigned buffer = pipeline.stages[i].bufferIndex;
        valid &= offsets.count(buffer) != 0;
        pipeline.slotOffsets[i] = offsets.lookup(buffer);
      }
      uint64_t a = pipeline.slotOffsets[0], b = pipeline.slotOffsets[1];
      valid &= a + pipeline.slotBytes <= b || b + pipeline.slotBytes <= a;
      for (const auto &stage : pipeline.stages)
        valid &= offsets.count(stage.bufferIndex) &&
                 offsets.lookup(stage.bufferIndex) == pipeline.slotOffsets[stage.slot];
      for (const auto &previous : execution.pipelines)
        for (const auto &stage : previous.stages)
          valid &= offsets.count(stage.bufferIndex) &&
                   offsets.lookup(stage.bufferIndex) == previous.slotOffsets[stage.slot];
      if (!valid || storage.pooledBytes > baseline.pooledBytes) {
        execution.rejectedPipelines.push_back("two alternating slots do not fit the existing LDS budget");
        continue;
      }
      execution.lifetimes = std::move(projected);
      execution.storage = std::move(storage);
      execution.pipelines.push_back(std::move(pipeline));
    }
  }
  execution.materializePool = threads && execution.storage.placements.size() >= 2 &&
                              execution.storage.pooledBytes < execution.storage.originalBytes;
  if (execution.materializePool) planShmHandoffs(execution.lifetimes, execution.storage);
  else execution.pipelines.clear();
  return result;
}

void dumpSharedMemoryExecution(const SharedExecutionPlan &plan, llvm::raw_ostream &os) {
  auto kernel = plan.kernel;
  os << "Shared execution plan: LDS=" << plan.storage.pooledBytes
     << ", buffers=" << plan.storage.placements.size()
     << ", handoff-barriers=" << plan.storage.barrierPositions.size() << '\n';
  for (const auto &pipeline : plan.pipelines) {
    os << "Shared memory pipeline: stages=" << pipeline.stages.size()
       << ", slot-bytes=" << pipeline.slotBytes << ", offsets="
       << pipeline.slotOffsets[0] << ',' << pipeline.slotOffsets[1]
       << ", LDS=" << plan.storage.pooledBytes << '\n';
    // Retain the tuning-log schema. These handoff barriers are now avoided by
    // planning the overlapping lifetimes, rather than inserted then removed.
    os << "V double buffer: " << pipeline.stages.size() << " slices, slot-bytes="
       << pipeline.slotBytes << ", displacement="
       << int64_t(pipeline.slotOffsets[1]) - int64_t(pipeline.slotOffsets[0])
       << ", removed-barriers=" << pipeline.stages.size() - 1
       << ", LDS=" << plan.storage.pooledBytes << ", function="
       << kernel.getSymName() << '\n';
    for (auto [i, stage] : llvm::enumerate(pipeline.stages))
      os << "  stage=" << i << " buffer=" << stage.bufferIndex << " slot=" << stage.slot
         << " issue=" << (stage.issueAfter ? "after-previous-ready" : "original") << '\n';
  }
  for (const auto &reason : plan.rejectedPipelines) {
    os << "Shared memory pipeline: skipped: " << reason << '\n';
    os << "V double buffer: skipped, function=" << kernel.getSymName()
       << ": " << reason << '\n';
  }
}

LogicalResult SharedMemoryPlan::materializeExecution() const {
  if (!execution) return failure();
  const auto &plan = *execution;
  if (snapshot(plan.kernel) != plan.inputSnapshot)
    return plan.kernel->emitError("shared execution plan invalidated before materialization");
  if (!plan.materializePool) return success();
  materializeShmReuse(plan.kernel, plan.lifetimes, plan.storage, nullptr,
                      /*hoistViews=*/!plan.pipelines.empty());
  for (const auto &pipeline : plan.pipelines)
    for (auto stage : pipeline.stages) {
      if (stage.issueAfter) {
        OpBuilder builder(stage.issueAfter->getNextNode());
        IRMapping mapping;
        for (Operation *dependency : stage.addressDependencies) builder.clone(*dependency, mapping);
        stage.issue.getSrcMutable().assign(mapping.lookupOrDefault(stage.issue.getSrc()));
        stage.issue->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
      }
      auto slot = IntegerAttr::get(IntegerType::get(plan.kernel->getContext(), 64), stage.slot);
      stage.issue->setAttr(marker, slot);
      stage.ready->setAttr(marker, slot);
      for (auto read : stage.reads) read->setAttr(marker, slot);
    }
  return success();
}
} // namespace mlir::frisk
