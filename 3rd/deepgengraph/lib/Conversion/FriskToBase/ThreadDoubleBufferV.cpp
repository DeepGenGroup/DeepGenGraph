// V double buffering after physical Thread IR storage planning. No allocation
// is added or resized. All checks finish before any operation is changed.
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "ScheduleAttrs.h"
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
#include "llvm/ADT/SetVector.h"
#include <functional>
#include <optional>

namespace mlir::frisk {
namespace {
constexpr StringLiteral marker = "frisk.v_double_buffer";
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

struct Stage {
  AsyncCopyGmToShmOp issue;
  WaitBufferReadyOp ready;
  SmallVector<ReadMatrixFromShmOp> reads;
  Operation *handoff = nullptr;
  llvm::SetVector<Operation *> dependencies;
};
struct Plan {
  SmallVector<Stage> stages;
  Operation *finish = nullptr;
  Span slice;
  int64_t spare = 0;
};

static bool direct(AsyncCopyGmToShmOp issue, memref::AllocOp pool,
                   unsigned threads) {
  auto s = extent(issue.getDst(), pool);
  auto width = issue->getAttrOfType<IntegerAttr>("frisk.dma_bytes");
  auto src = cast<MemRefType>(issue.getSrc().getType());
  return s && src.getMemorySpaceAsInt() == 1 &&
         !issue->hasAttr("frisk.force_sync") && width && width.getInt() == 16 &&
         s->begin % 16 == 0 && s->size % (threads * 16) == 0 &&
         s->size / (threads * 16) <= 16 && issue.getAlignment() >= 16 &&
         uniform(issue.getSrc(), issue->getParentOfType<func::FuncOp>()) &&
         selectAsyncDMATransferBytes(issue, 16) == 16;
}

static bool plan(func::FuncOp f, Block &block, memref::AllocOp pool,
                 unsigned threads, Plan &p, std::string &reason) {
  auto fail = [&](StringRef why) { reason = why.str(); return false; };
  if (!isInBlockUniformControlFlow(block.getTerminator()))
    return fail("non-uniform or unsupported control flow");
  // One outer compute loop, with bounded inner reader/computation loops.
  auto *owner = block.getParentOp();
  if (owner != f && (!isa<Loop, scf::ForOp>(owner) || owner->getBlock() != &f.getBody().front()))
    return fail("requires one outer compute loop");
  WaitBufferReadyOp ready;
  for (Operation &op : block) {
    if (auto wait = dyn_cast<WaitBufferReadyOp>(op)) ready = wait;
    bool bad = false;
    op.walk([&](ReadMatrixFromShmOp read) {
      if (read->hasAttr(marker) || !ready) { bad = true; return; }
      if (p.stages.empty() || p.stages.back().ready != ready) {
        auto issue = ready.getToken().getDefiningOp<AsyncCopyGmToShmOp>();
        if (!issue || issue->getBlock() != &block || !issue->isBeforeInBlock(ready) ||
            !issue.getToken().hasOneUse()) { bad = true; return; }
        p.stages.push_back(Stage{issue, ready, {}, nullptr, {}});
      }
      p.stages.back().reads.push_back(read);
    });
    if (bad) return fail("already transformed or missing local task completion");
  }
  if (p.stages.size() < 2 || p.stages.size() > 32)
    return fail("requires 2..32 V slices");
  auto initial = extent(p.stages.front().issue.getDst(), pool);
  if (!initial) return fail("unknown DMA destination alias");
  p.slice = *initial;
  int64_t capacity = pool.getType().getNumElements();
  p.spare = p.slice.begin + p.slice.size;
  if (p.spare + p.slice.size > capacity) p.spare = p.slice.begin - p.slice.size;
  if (p.spare < 0) return fail("no spare LDS slice");
  Value resource = origin(p.stages.front().issue.getSrc());
  if (!resource) return fail("unknown DMA source");
  llvm::SmallPtrSet<Operation *, 32> allowed;
  for (auto &stage : p.stages) {
    auto s = extent(stage.issue.getDst(), pool);
    if (!s || s->begin != p.slice.begin || s->size != p.slice.size)
      return fail("DMA slices differ or are already buffered");
    if (origin(stage.issue.getSrc()) != resource || !direct(stage.issue, pool, threads))
      return fail("mixed sources or unsupported direct DMA");
    allowed.insert(stage.issue);
    allowed.insert(stage.ready);
    for (auto read : stage.reads) {
      if (!containedRead(read, &block, p.slice, pool, threads))
        return fail("unknown matrix address or read outside DMA slice");
      allowed.insert(read);
    }
  }
  for (unsigned i = 1; i < p.stages.size(); ++i) {
    auto &prev = p.stages[i - 1];
    for (Operation *op = prev.ready->getNextNode(); op != p.stages[i].ready; op = op->getNextNode()) {
      if (barrier(op)) {
        if (prev.handoff) return fail("ambiguous reader handoff");
        prev.handoff = op;
      }
    }
    if (!prev.handoff ||
        !topIn(prev.reads.back(), &block)->isBeforeInBlock(prev.handoff) ||
        !prev.handoff->isBeforeInBlock(p.stages[i].issue))
      return fail("missing reader handoff or unsupported order");
    allowed.insert(prev.handoff);
  }
  for (Operation *op = topIn(p.stages.back().reads.back(), &block)->getNextNode();
       op; op = op->getNextNode())
    if (barrier(op)) { p.finish = op; break; }
  if (!p.finish) return fail("missing final reader barrier");
  allowed.insert(p.finish);
  auto *first = p.stages.front().ready.getOperation();
  // No hidden memory effects, synchronization, escapes or other LDS users in
  // the transformed region. Inspect inner loops rather than their summaries.
  auto safeRegionOp = [&](Operation *op) {
    for (Operation *parent = op; parent; parent = parent->getParentOp())
      if (hasAnyScheduleAttr(parent)) return false;
    if (allowed.contains(op)) return true;
    if (isa<Loop, scf::ForOp>(op)) return isInBlockUniformControlFlow(op->getRegion(0).front().getTerminator());
    if (op->getNumRegions()) return false;
    return isMemoryEffectFree(op) && !barrier(op) &&
           !isa<ToThreadTileOp, FromThreadTileOp>(op);
  };
  for (Operation *op = first; op != p.finish; op = op->getNextNode()) {
    if (op->walk([&](Operation *nested) {
          return safeRegionOp(nested) ? WalkResult::advance() : WalkResult::interrupt();
        }).wasInterrupted()) return fail("unknown operation or other live LDS user in V region");
  }

  // Audit the whole function. A read before V in the same iteration must see
  // bytes overwritten AND completed this iteration; a prologue read may die
  // before the loop. No shared reads after V or in unrelated regions survive.
  llvm::BitVector written(capacity);
  bool valid = true;
  auto audit = [&](Operation *op) {
    if (!valid || allowed.contains(op)) return;
    // Do not let an address-space cast disguise an LDS reader/writer as a
    // global access. The closed alias model is deliberately space preserving.
    if (isa<memref::MemorySpaceCastOp>(op)) { valid = false; return; }
    Operation *top = topIn(op, &block);
    bool before = top && top->isBeforeInBlock(first);
    if (auto issue = dyn_cast<AsyncCopyGmToShmOp>(op)) {
      // All DMA in the function must have an understood physical footprint
      // and exact local completion. This also excludes pending loop carriers.
      if (!direct(issue, pool, threads) || !issue.getToken().hasOneUse()) { valid = false; return; }
      auto wait = dyn_cast<WaitBufferReadyOp>(*issue.getToken().getUsers().begin());
      if (!wait || wait->getBlock() != issue->getBlock() || !issue->isBeforeInBlock(wait)) {
        valid = false;
        return;
      }
      // A pre-existing transfer may not remain in flight while we borrow its
      // storage. A publication for V does not establish another token's
      // completion at this IR level (nor a staged copy's completion).
      if (before && (issue->getBlock() != &block || !wait->isBeforeInBlock(first))) valid = false;
      if (!top && owner != f) {
        auto *entryTop = topIn(issue, &f.getBody().front());
        if (entryTop && entryTop->isBeforeInBlock(owner) &&
            (issue->getBlock() != &f.getBody().front() || !wait->isBeforeInBlock(owner))) valid = false;
      }
      return;
    }
    if (auto wait = dyn_cast<WaitBufferReadyOp>(op)) {
      auto issue = wait.getToken().getDefiningOp<AsyncCopyGmToShmOp>();
      if (!issue || issue->getBlock() != wait->getBlock()) { valid = false; return; }
      if (before && op->getBlock() == &block) {
        auto s = extent(issue.getDst(), pool);
        if (!s) { valid = false; return; }
        written.set(s->begin, s->begin + s->size);
      }
      return;
    }
    Value source;
    if (auto load = dyn_cast<memref::LoadOp>(op)) source = load.getMemRef();
    else if (auto load = dyn_cast<vector::LoadOp>(op)) source = load.getBase();
    else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) source = load.getMemRef();
    else if (auto load = dyn_cast<ReadMatrixFromShmOp>(op)) source = load.getSrc();
    if (source) {
      if (!shared(source)) return;
      auto s = extent(source, pool);
      if (!s) { valid = false; return; }
      if (before) {
        for (int64_t b = std::max(s->begin, p.spare);
             b < std::min(s->begin + s->size, p.spare + p.slice.size); ++b)
          if (!written[b]) { valid = false; return; }
      } else {
        Operation *entryTop = topIn(op, &f.getBody().front());
        if (owner == f || !entryTop || entryTop == owner || !entryTop->isBeforeInBlock(owner)) valid = false;
      }
      return;
    }
    if (isa<memref::StoreOp, vector::StoreOp, affine::AffineStoreOp>(op)) {
      for (Value v : op->getOperands()) if (shared(v)) valid = false;
      return;
    }
    if (op == f || isa<Loop, scf::ForOp, affine::AffineIfOp, scf::IfOp>(op) ||
        barrier(op) || op == pool || isMemoryEffectFree(op)) return;
    valid = false;
  };
  f.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!valid) return;
    audit(op);
    if (!valid) reason = "unknown alias/effect, surviving LDS reader or alternate slot live across iteration at " + op->getName().getStringRef().str();
  });
  if (!valid) return false;

  DominanceInfo dominance(f);
  for (unsigned i = 1; i < p.stages.size(); ++i) {
    auto &stage = p.stages[i];
    Operation *anchor = p.stages[i - 1].ready->getNextNode();
    std::function<bool(Value)> visit = [&](Value v) {
      auto *def = v.getDefiningOp();
      if (dominance.dominates(v, anchor)) return true;
      if (!def || def->getBlock() != &block || def->getNumRegions() ||
          !isPure(def) || isa<ToThreadTileOp, FromThreadTileOp>(def)) return false;
      if (stage.dependencies.contains(def)) return true;
      for (Value input : def->getOperands()) if (!visit(input)) return false;
      stage.dependencies.insert(def);
      return true;
    };
    for (Value v : stage.issue->getOperands())
      if (!visit(v)) return fail("DMA address depends on unavailable effect");
  }
  return true;
}

static void apply(Plan &p, memref::AllocOp pool) {
  for (unsigned i = 0; i < p.stages.size(); ++i) {
    auto &stage = p.stages[i];
    if (i) {
      OpBuilder b(p.stages[i - 1].ready->getNextNode());
      IRMapping mapping;
      for (Operation *dependency : stage.dependencies) b.clone(*dependency, mapping);
      for (OpOperand &operand : stage.issue->getOpOperands())
        operand.set(mapping.lookupOrDefault(operand.get()));
      if (i % 2) {
        // Each descriptor keeps its original type and relative reader offset.
        auto makeView = [&](Value old) -> Value {
          auto s = *extent(old, pool);
          auto offset = b.create<arith::ConstantIndexOp>(stage.issue.getLoc(),
              s.begin + p.spare - p.slice.begin);
          return b.create<memref::ViewOp>(stage.issue.getLoc(), cast<MemRefType>(old.getType()),
                                        pool, offset, ValueRange{});
        };
        stage.issue.getDstMutable().assign(makeView(stage.issue.getDst()));
        for (auto read : stage.reads) read.getSrcMutable().assign(makeView(read.getSrc()));
      }
      stage.issue->moveBefore(b.getInsertionBlock(), b.getInsertionPoint());
    }
    for (auto read : stage.reads)
      read->setAttr(marker, IntegerAttr::get(IntegerType::get(read.getContext(), 64), p.slice.size));
  }
  for (auto &stage : p.stages) if (stage.handoff) stage.handoff->erase();
}

class ThreadDoubleBufferVPass
    : public PassWrapper<ThreadDoubleBufferVPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadDoubleBufferVPass)
  StringRef getArgument() const final { return "frisk-thread-double-buffer-v"; }
  StringRef getDescription() const final { return "Double buffer matrix-read DMA in existing Thread IR LDS storage"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }
  void runOnOperation() final {
    auto f = getOperation();
    llvm::SetVector<Block *> blocks;
    f.walk([&](WaitBufferReadyOp wait) { blocks.insert(wait->getBlock()); });
    bool hasReads = false;
    f.walk([&](ReadMatrixFromShmOp) { hasReads = true; });
    if (!hasReads) return;
    auto skip = [&](StringRef why) {
      llvm::errs() << "V double buffer: skipped, function=" << f.getName() << ": " << why << '\n';
    };
    auto threads = f->getAttrOfType<IntegerAttr>("thread_num");
    if (!threads || threads.getInt() < 64 || threads.getInt() > 1024 || threads.getInt() % 64 ||
        !llvm::hasSingleElement(f.getBody())) { skip("requires a fixed 1D wave64 workgroup"); return; }
    SmallVector<memref::AllocOp> pools;
    f.walk([&](memref::AllocOp alloc) { if (shared(alloc)) pools.push_back(alloc); });
    if (pools.size() != 1) { skip("requires one static LDS allocation"); return; }
    auto pool = pools.front();
    auto type = pool.getType();
    if (!type.hasStaticShape() || type.getRank() != 1 || !type.getElementType().isInteger(8) ||
        !type.getLayout().isIdentity() || type.getNumElements() <= 0 || type.getNumElements() > (1 << 20) ||
        pool->getBlock() != &f.getBody().front() || pool.getAlignment().value_or(0) < 16) {
      skip("requires a static aligned byte pool"); return;
    }
    for (Block *block : blocks) {
      bool candidate = false;
      for (Operation &op : *block) op.walk([&](ReadMatrixFromShmOp) { candidate = true; });
      if (!candidate) continue;
      // An entry block containing a compute loop is not itself that stream.
      if (block == &f.getBody().front() && llvm::any_of(blocks, [&](Block *b) { return b != block; })) continue;
      Plan p;
      std::string reason;
      if (!plan(f, *block, pool, threads.getInt(), p, reason)) { skip(reason); continue; }
      apply(p, pool);
      llvm::errs() << "V double buffer: " << p.stages.size() << " slices, slot-bytes=" << p.slice.size
                   << ", displacement=" << p.spare - p.slice.begin << ", removed-barriers="
                   << p.stages.size() - 1 << ", LDS=" << type.getNumElements() << ", function=" << f.getName() << '\n';
    }
  }
};
} // namespace

bool parseVDoubleBufferOption(llvm::StringRef argument, bool &enabled, bool &consumed) {
  consumed = argument.starts_with("--v-double-buffer");
  if (!consumed) return true;
  if (!argument.consume_front("--v-double-buffer=") || (argument != "0" && argument != "1")) return false;
  enabled = argument == "1";
  return true;
}
std::unique_ptr<Pass> createThreadDoubleBufferVPass() {
  return std::make_unique<ThreadDoubleBufferVPass>();
}
} // namespace mlir::frisk
