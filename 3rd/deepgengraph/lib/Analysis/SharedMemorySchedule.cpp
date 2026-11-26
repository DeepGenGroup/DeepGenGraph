// Joint shared-memory execution planning: DMA issue times, logical buffer
// lifetimes, ping-pong slots and physical pool placement, before IR mutation.
#include "deepgengraph/Analysis/SharedMemoryPlan.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/ScheduleAttrs.h"
#include "deepgengraph/Pipeline/SharedPipelinePlan.h"
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
constexpr StringLiteral crossMarker = "frisk.cross_bn_k0";
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

// Consume AutoPipeline's logical schedule. This phase may reject the entire
// candidate, but must not discover a new stage order, slot or issue anchor.
// Each stage still owns an independent allocation until physical validation.
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
  using pipeline::sharedPipelinePlanAttr;
  for (Operation &op : block) {
    if (!op.hasAttr(sharedPipelinePlanAttr)) continue;
    auto logical = op.getAttrOfType<DenseI64ArrayAttr>(sharedPipelinePlanAttr);
    auto issue = dyn_cast<AsyncCopyGmToShmOp>(op);
    unsigned index = pipeline.stages.size();
    if (!logical || logical.size() != 4 || logical[0] != index ||
        logical[1] < 2 || logical[1] > 32 || logical[2] != int64_t(index % 2) ||
        logical[3] != int64_t(index) - 1)
      return fail("invalid AutoPipeline stage/slot/issue-after contract");
    if (!issue || issue->hasAttr(marker) || !issue.getToken().hasOneUse() ||
        !buffers.count(issue.getDst()))
      return fail("already planned, incomplete task or unsupported logical allocation");
    auto ready = dyn_cast<WaitBufferReadyOp>(*issue.getToken().getUsers().begin());
    if (!ready || ready->getBlock() != &block || !issue->isBeforeInBlock(ready))
      return fail("AutoPipeline task lost its local readiness");
    SharedPipelineStage stage;
    stage.issue = issue;
    stage.ready = ready;
    stage.bufferIndex = buffers.lookup(issue.getDst());
    stage.slot = logical[2];
    if (logical[3] >= 0) stage.issueAfter = pipeline.stages[logical[3]].ready;
    // Reader footprints are available only after physical lowering. Recover
    // them from this declared buffer's uses, never from the nearest wait.
    for (Operation *access : live.buffers[stage.bufferIndex].accesses)
      if (auto read = dyn_cast<ReadMatrixFromShmOp>(access)) {
        Operation *top = topIn(read, &block);
        if (read->hasAttr(marker) || !top || !ready->isBeforeInBlock(top))
          return fail("AutoPipeline reader precedes readiness or is already scheduled");
        stage.reads.push_back(read);
      }
    if (stage.reads.empty()) return fail("AutoPipeline stage has no physical matrix readers");
    llvm::sort(stage.reads, [&](ReadMatrixFromShmOp a, ReadMatrixFromShmOp b) {
      Operation *x = topIn(a, &block), *y = topIn(b, &block);
      return x != y && x->isBeforeInBlock(y);
    });
    pipeline.stages.push_back(std::move(stage));
  }
  if (pipeline.stages.empty()) return fail("no AutoPipeline shared-copy plan");
  for (auto &stage : pipeline.stages)
    if (stage.issue->getAttrOfType<DenseI64ArrayAttr>(sharedPipelinePlanAttr)[1] !=
        int64_t(pipeline.stages.size()))
      return fail("incomplete AutoPipeline shared-copy plan");
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

static Block *loopBody(Operation *op) {
  if (auto loop = dyn_cast<Loop>(op)) return loop.getBody();
  if (auto loop = dyn_cast<scf::ForOp>(op)) return loop.getBody();
  return nullptr;
}
static Value induction(Operation *op) { return loopBody(op)->getArgument(0); }

static bool collectCrossIteration(func::FuncOp kernel, Operation *loop,
    const ShmLivenessResult &live, const ShmLivenessResult &original, unsigned threads,
    SharedCrossIterationPlan &plan, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  Block *body = loopBody(loop);
  if (!body || loop->getBlock() != &kernel.getBody().front() ||
      !isInBlockUniformControlFlow(body->getTerminator()))
    return fail("requires a block-uniform outer loop");
  if (hasAnyScheduleAttr(loop))
    return fail("existing pipeline schedule owns task/slot lifetimes; no second cross-iteration schedule");
  if (!loop->getAttrOfType<UnitAttr>(pipeline::crossIterationPlanAttr))
    return fail("no AutoPipeline cross-iteration plan");
  if (auto scf = dyn_cast<scf::ForOp>(loop)) {
    auto step = constant(scf.getStep());
    if (!step || *step <= 0 || !scf.getInductionVar().getType().isIndex())
      return fail("requires a positive constant index step");
  }
  plan.loop = loop;
  AsyncCopyGmToShmOp anchor;
  for (Operation &op : *body) {
    bool first = op.hasAttr(pipeline::crossIterationIssueAttr);
    bool last = op.hasAttr(pipeline::crossIterationAnchorAttr);
    if (!first && !last) continue;
    auto copy = dyn_cast<AsyncCopyGmToShmOp>(op);
    if (!copy || !copy.getToken().hasOneUse() ||
        (first && (!op.getAttrOfType<UnitAttr>(pipeline::crossIterationIssueAttr) || plan.issue)) ||
        (last && (!op.getAttrOfType<UnitAttr>(pipeline::crossIterationAnchorAttr) || anchor)))
      return fail("invalid AutoPipeline cross-iteration DMA contract");
    auto ready = dyn_cast<WaitBufferReadyOp>(*copy.getToken().getUsers().begin());
    if (!ready || ready->getBlock() != body || !copy->isBeforeInBlock(ready))
      return fail("AutoPipeline cross-iteration task lost local readiness");
    if (first) { plan.issue = copy; plan.ready = ready; }
    if (last) { anchor = copy; plan.issueAfter = ready; }
  }
  if (!plan.ready || !plan.issueAfter || plan.ready == plan.issueAfter ||
      !plan.ready->isBeforeInBlock(plan.issueAfter))
    return fail("missing distinct initial and final readiness");
  if (!plan.issue || plan.issue->getBlock() != body ||
      !plan.issue->isBeforeInBlock(plan.ready) || !plan.issue.getToken().hasOneUse() ||
      plan.issue->hasAttr(marker))
    return fail("initial readiness must name one local, unscheduled DMA");
  auto found = llvm::find_if(live.buffers, [&](const ShmLiveInterval &info) {
    return info.allocation->getResult(0) == plan.issue.getDst();
  });
  if (found == live.buffers.end() || !found->reusable || found->asyncAcrossIterations ||
      !llvm::is_contained(found->iterationLocalLoops, loop) ||
      !direct(plan.issue, found->allocation, threads))
    return fail("initial DMA must fully overwrite a reusable iteration-local allocation");
  plan.bufferIndex = std::distance(live.buffers.begin(), found);
  for (Operation *access : found->accesses) {
    Operation *top = topIn(access, body);
    if (!top || !top->isBeforeInBlock(plan.issueAfter))
      return fail("initial buffer still has readers at the next-iteration issue point");
    if (access != plan.issue && access != plan.ready && !plan.ready->isBeforeInBlock(top))
      return fail("initial buffer is accessed before its readiness");
  }
  // All memory effects in the loop are accounted for by local task completion
  // and reusable LDS allocations. In particular, moving a global read must not
  // cross a possibly aliasing global write, an unknown call or a conditional DMA.
  llvm::SmallPtrSet<Operation *, 32> accesses;
  for (const auto &info : original.buffers) if (info.reusable) {
    accesses.insert(info.allocation);
    accesses.insert(info.accesses.begin(), info.accesses.end());
  }
  bool finalRead = false, unsafe = false;
  loop->walk([&](Operation *op) {
    if (op == loop || unsafe) return;
    if (auto nested = dyn_cast<Loop>(op)) {
      if (!nested.hasConstantBounds() || !isInBlockUniformControlFlow(nested.getBody()->getTerminator())) unsafe = true;
      return;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) return;
    if (op->getNumRegions()) { unsafe = true; return; }
    if (barrier(op)) return;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op)) {
      auto *allocation = origin(copy.getDst()).getDefiningOp();
      auto alloc = dyn_cast_or_null<memref::AllocOp>(allocation);
      if (copy->getBlock() != body || !alloc || !accesses.contains(copy) ||
          !copy.getToken().hasOneUse() ||
          !isa<WaitBufferReadyOp>(*copy.getToken().getUsers().begin()) ||
          !direct(copy, alloc, threads)) unsafe = true;
      return;
    }
    if (auto wait = dyn_cast<WaitBufferReadyOp>(op)) {
      if (wait->getBlock() != body || !accesses.contains(wait)) unsafe = true;
      return;
    }
    if (auto read = dyn_cast<ReadMatrixFromShmOp>(op)) {
      auto alloc = dyn_cast_or_null<memref::AllocOp>(origin(read.getSrc()).getDefiningOp());
      auto span = alloc ? extent(read.getSrc(), alloc) : std::nullopt;
      if (!span || !accesses.contains(op) || !containedRead(read, body, *span, alloc, threads)) unsafe = true;
      if (plan.issueAfter->isBeforeInBlock(topIn(op, body))) finalRead = true;
      return;
    }
    if (isa<memref::LoadOp, affine::AffineLoadOp, vector::LoadOp, affine::AffineVectorLoadOp>(op)) {
      if (!accesses.contains(op)) unsafe = true;
      return;
    }
    if (isa<memref::AllocOp>(op) && accesses.contains(op)) return;
    if (!isMemoryEffectFree(op)) unsafe = true;
  });
  if (unsafe) return fail("unknown memory effect, task completion or physical reader");
  if (!finalRead) return fail("no final matrix-read overlap window");
  // Nothing may still produce DMA after the selected final readiness.
  for (Operation *op = plan.issueAfter->getNextNode(); op; op = op->getNextNode())
    if (isa<AsyncCopyGmToShmOp>(op)) return fail("DMA after final readiness");
  // The first copy is the opening producer, not a later unrelated stream.
  for (Operation &op : *body) {
    if (&op == plan.issue) break;
    if (isa<AsyncCopyGmToShmOp, WaitBufferReadyOp>(op)) return fail("producer before initial DMA");
  }
  DominanceInfo dominance(kernel);
  llvm::SetVector<Operation *> dependencies;
  bool varies = false;
  std::function<bool(Value)> visit = [&](Value v) {
    if (v == induction(loop)) { varies = true; return true; }
    if (dominance.dominates(v, loop)) return true;
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != body || def->getNumRegions() || !isPure(def)) return false;
    if (dependencies.contains(def)) return true;
    for (Value operand : def->getOperands()) if (!visit(operand)) return false;
    dependencies.insert(def);
    return true;
  };
  if (!visit(plan.issue.getSrc()) || !varies)
    return fail("source must have cloneable induction-dependent address expressions");
  plan.addressDependencies.assign(dependencies.begin(), dependencies.end());
  return true;
}

static void materializeCrossIteration(SharedCrossIterationPlan plan) {
  Operation *old = plan.loop;
  Block *body = loopBody(old);
  OpBuilder b(old);
  Location loc = old->getLoc();
  Value lo, hi, step;
  SmallVector<Value> init;
  if (auto loop = dyn_cast<Loop>(old)) {
    lo = b.create<affine::AffineMaxOp>(loc, loop.getLowerBoundMap(), loop.getLowerBoundOperands());
    hi = b.create<affine::AffineMinOp>(loc, loop.getUpperBoundMap(), loop.getUpperBoundOperands());
    step = b.create<arith::ConstantIndexOp>(loc, loop.getStepAsInt());
    llvm::append_range(init, loop.getInits());
  } else {
    auto scf = cast<scf::ForOp>(old);
    lo = scf.getLowerBound(); hi = scf.getUpperBound(); step = scf.getStep();
    llvm::append_range(init, scf.getInitArgs());
  }
  // The initial publication must not overwrite prologue readers in another
  // wave. Reuse an existing final handoff; otherwise insert it before priming.
  bool handoff = false;
  for (Operation *op = old->getPrevNode(); op; op = op->getPrevNode()) {
    if (barrier(op) || isa<WaitBufferReadyOp>(op)) { handoff = true; break; }
    if (!isMemoryEffectFree(op)) break;
  }
  if (!handoff) b.create<gpu::BarrierOp>(loc);
  auto taskType = friskAsyncTaskType::get(b.getContext());
  Value done = b.create<AsyncTaskCompletedOp>(loc, taskType);
  auto issue = [&](OpBuilder &builder, Value iv) {
    IRMapping map;
    map.map(induction(old), iv);
    for (Operation *dep : plan.addressDependencies) builder.clone(*dep, map);
    auto copy = cast<AsyncCopyGmToShmOp>(builder.clone(*plan.issue, map));
    copy->setAttr(marker, builder.getI64IntegerAttr(0));
    copy->setAttr(crossMarker, builder.getUnitAttr());
    return copy.getToken();
  };
  auto guarded = [&](Value condition, auto emit) {
    auto branch = b.create<scf::IfOp>(loc, TypeRange{taskType}, condition, true);
    branch->setAttr(crossMarker, b.getUnitAttr());
    OpBuilder thenBuilder = OpBuilder::atBlockBegin(branch.thenBlock());
    Value task = emit(thenBuilder);
    thenBuilder.create<scf::YieldOp>(loc, task);
    OpBuilder elseBuilder = OpBuilder::atBlockBegin(branch.elseBlock());
    elseBuilder.create<scf::YieldOp>(loc, done);
    return branch.getResult(0);
  };
  Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lo, hi);
  init.push_back(guarded(nonempty, [&](OpBuilder &builder) { return issue(builder, lo); }));
  Operation *replacement;
  if (auto loop = dyn_cast<Loop>(old)) {
    replacement = b.create<Loop>(loc, loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
        loop.getUpperBoundOperands(), loop.getUpperBoundMap(), loop.getStepAsInt(), init);
  } else replacement = b.create<scf::ForOp>(loc, lo, hi, step, init);
  for (NamedAttribute attr : old->getDiscardableAttrs()) replacement->setAttr(attr.getName(), attr.getValue());
  replacement->setAttr(crossMarker, b.getUnitAttr());
  // Keep original operation handles (pipeline stages and projected anchors).
  // Only the loop container and the additional task carrier are new.
  Block *nextBody = loopBody(replacement);
  if (!nextBody->empty()) nextBody->getTerminator()->erase();
  for (auto [from, to] : llvm::zip(body->getArguments(), nextBody->getArguments())) from.replaceAllUsesWith(to);
  nextBody->getOperations().splice(nextBody->end(), body->getOperations());
  Operation *nextAnchor = plan.issueAfter->getNextNode();
  while (barrier(nextAnchor)) nextAnchor = nextAnchor->getNextNode();
  b.setInsertionPoint(nextAnchor);
  Value iv = nextBody->getArgument(0);
  // Unsigned distance avoids speculating iv+step outside the valid range.
  Value remaining = b.create<arith::SubIOp>(loc, hi, iv);
  Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remaining, step);
  Value next = guarded(hasNext, [&](OpBuilder &builder) {
    Value advance = builder.create<arith::AddIOp>(loc, iv, step);
    // The original IV's uses have been remapped above.
    IRMapping map; map.map(iv, advance);
    for (Operation *dep : plan.addressDependencies) builder.clone(*dep, map);
    auto copy = cast<AsyncCopyGmToShmOp>(builder.clone(*plan.issue, map));
    copy->setAttr(marker, builder.getI64IntegerAttr(0));
    copy->setAttr(crossMarker, builder.getUnitAttr());
    return copy.getToken();
  });
  plan.ready.getTokenMutable().assign(nextBody->getArguments().back());
  plan.ready->setAttr(marker, b.getI64IntegerAttr(0));
  nextBody->getTerminator()->insertOperands(nextBody->getTerminator()->getNumOperands(), ValueRange{next});
  plan.issue.erase();
  for (auto [from, to] : llvm::zip(old->getResults(), replacement->getResults())) from.replaceAllUsesWith(to);
  // Explicit drain keeps task completion provable even on the zero-trip path.
  b.setInsertionPointAfter(replacement);
  b.create<WaitBufferReadyOp>(loc, replacement->getResults().back());
  old->erase();
}
} // namespace

SharedMemoryPlan planSharedMemoryExecution(func::FuncOp kernel, bool doubleBuffer, bool crossIteration) {
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
  const auto originalLifetimes = execution.lifetimes;
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
  if (crossIteration && execution.materializePool && !kernel->hasAttr(crossMarker) &&
      threads.getInt() >= 64 && threads.getInt() <= 1024 && threads.getInt() % 64 == 0) {
    for (Operation &op : kernel.getBody().front()) {
      if (!loopBody(&op)) continue;
      bool hasCopy = false;
      op.walk([&](AsyncCopyGmToShmOp) { hasCopy = true; });
      if (!hasCopy) continue;
      SharedCrossIterationPlan cross;
      std::string reason;
      if (!collectCrossIteration(kernel, &op, execution.lifetimes, originalLifetimes, threads.getInt(), cross, reason)) {
        execution.rejectedCrossIterations.push_back(std::move(reason));
        continue;
      }
      auto projected = execution.lifetimes;
      auto &buffer = projected.buffers[cross.bufferIndex];
      buffer.asyncAcrossIterations = true;
      buffer.cyclicLoop = &op;
      buffer.cyclicIssueAfter = cross.issueAfter;
      auto storage = planShmReuse(projected);
      DenseMap<unsigned, uint64_t> offsets;
      for (auto placed : storage.placements) offsets[placed.bufferIndex] = placed.offsetBytes;
      bool valid = storage.pooledBytes <= execution.storage.pooledBytes;
      for (const auto &pipeline : execution.pipelines)
        for (const auto &stage : pipeline.stages)
          valid &= offsets.count(stage.bufferIndex) &&
                   offsets.lookup(stage.bufferIndex) == pipeline.slotOffsets[stage.slot];
      if (!valid) {
        execution.rejectedCrossIterations.push_back("circular lifetime exceeds LDS budget or changes planned slots");
        continue;
      }
      execution.lifetimes = std::move(projected);
      execution.storage = std::move(storage);
      execution.crossIterations.push_back(std::move(cross));
    }
  }
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
  for (const auto &cross : plan.crossIterations)
    os << "Cross BN K0 plan: buffer=" << cross.bufferIndex
       << ", slot-bytes=" << plan.lifetimes.buffers[cross.bufferIndex].sizeBytes
       << ", LDS=" << plan.storage.pooledBytes << ", function=" << kernel.getSymName() << '\n';
  for (const auto &reason : plan.rejectedCrossIterations)
    os << "Cross BN K0 plan: skipped, function=" << kernel.getSymName() << ": " << reason << '\n';
}

LogicalResult SharedMemoryPlan::materializeExecution() const {
  if (!execution) return failure();
  const auto &plan = *execution;
  if (snapshot(plan.kernel) != plan.inputSnapshot)
    return plan.kernel->emitError("shared execution plan invalidated before materialization");
  if (!plan.materializePool) return success();
  materializeShmReuse(plan.kernel, plan.lifetimes, plan.storage, nullptr,
                      /*hoistViews=*/!plan.pipelines.empty() || !plan.crossIterations.empty());
  for (const auto &pipeline : plan.pipelines)
    for (auto stage : pipeline.stages) {
      if (stage.issueAfter) {
        Operation *anchor = stage.issueAfter->getNextNode();
        while (barrier(anchor)) anchor = anchor->getNextNode();
        OpBuilder builder(anchor);
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
  for (auto cross : plan.crossIterations) materializeCrossIteration(cross);
  if (!plan.crossIterations.empty()) {
    OpBuilder b(plan.kernel->getContext());
    plan.kernel->setAttr(crossMarker, b.getStringAttr("shared-plan-v1"));
    SmallVector<Attribute> attrs;
    if (auto old = plan.kernel->getAttrOfType<ArrayAttr>("llvm.passthrough"))
      llvm::append_range(attrs, old);
    attrs.push_back(b.getArrayAttr({b.getStringAttr(crossMarker), b.getStringAttr("shared-plan-v1")}));
    plan.kernel->setAttr("llvm.passthrough", b.getArrayAttr(attrs));
  }
  return success();
}
} // namespace mlir::frisk
