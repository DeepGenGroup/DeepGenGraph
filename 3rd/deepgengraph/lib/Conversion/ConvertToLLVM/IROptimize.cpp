#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_IRDEEPOPTIMIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"

// Only unroll indices entirely determined by constants and small static loops.
// Follow index dependencies, not the vector or its lane-dependent load address.
static bool collectIndexLoops(Value value,
                              llvm::SmallPtrSetImpl<Operation *> &loops,
                              llvm::SmallDenseSet<Value, 16> &visiting) {
  if (matchPattern(value, m_Constant()))
    return true;

  // An iter_arg recurrence may refer back to itself. Its initial value and
  // every other input to the recurrence must still be statically determined.
  if (!visiting.insert(value).second)
    return true;
  auto guard = llvm::make_scope_exit([&] { visiting.erase(value); });
  auto collectLoop = [&](affine::AffineForOp loop) {
    auto count = affine::getConstantTripCount(loop);
    if (!count || *count == 0 || *count > 64)
      return false;
    // A constant trip count alone does not imply constant IVs: the lower
    // bound could be a runtime offset. Outer static IVs are allowed here.
    for (Value bound : loop.getLowerBoundOperands())
      if (!collectIndexLoops(bound, loops, visiting))
        return false;
    for (Value bound : loop.getUpperBoundOperands())
      if (!collectIndexLoops(bound, loops, visiting))
        return false;
    loops.insert(loop);
    return true;
  };
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<affine::AffineForOp>(
        arg.getOwner()->getParentOp());
    if (!loop || !collectLoop(loop))
      return false;
    if (arg == loop.getInductionVar())
      return true;
    unsigned position = arg.getArgNumber() - 1;
    auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
    return collectIndexLoops(loop.getInits()[position], loops, visiting) &&
           collectIndexLoops(yield.getOperand(position), loops, visiting);
  }
  auto *op = value.getDefiningOp();
  if (!op || !isa<affine::AffineApplyOp, arith::AddIOp, arith::SubIOp,
                  arith::MulIOp, arith::DivSIOp, arith::DivUIOp,
                  arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
                  arith::IndexCastUIOp>(op))
    return false;
  for (Value operand : op->getOperands())
    if (!collectIndexLoops(operand, loops, visiting))
      return false;
  return true;
}

class VectorOpLoopUnrollPass
    : public impl::IRDeepOptimizeBase<VectorOpLoopUnrollPass> {
  void runOnOperation() override {
    auto kernel = getOperation();
    RewritePatternSet patterns(&getContext());
    for (auto op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    // Bound cumulative code growth, not just individual loop trip counts.
    constexpr uint64_t maxExtraOperations = 65536;
    uint64_t extraOperations = 0;
    while (true) {
      // Full unrolling substitutes IVs and threads iter_args through clones;
      // folding then turns affine.apply/arithmetic and vector positions static.
      if (failed(applyPatternsGreedily(kernel, frozen))) {
        signalPassFailure();
        return;
      }
      llvm::SmallPtrSet<Operation *, 16> candidates;
      auto collect = [&](auto op) {
        for (OpFoldResult position : op.getMixedPosition()) {
          auto index = dyn_cast<Value>(position);
          if (!index)
            continue;
          llvm::SmallPtrSet<Operation *, 8> dependencies;
          llvm::SmallDenseSet<Value, 16> visiting;
          if (collectIndexLoops(index, dependencies, visiting))
            candidates.insert(dependencies.begin(), dependencies.end());
        }
      };
      kernel.walk([&](vector::InsertOp op) { collect(op); });
      kernel.walk([&](vector::ExtractOp op) { collect(op); });
      affine::AffineForOp selected;
      uint64_t selectedCost = 0;
      kernel.walk<WalkOrder::PostOrder>([&](affine::AffineForOp loop) {
        if (selected || !candidates.contains(loop))
          return;
        uint64_t bodySize = 0;
        loop.getBody()->walk([&](Operation *) { ++bodySize; });
        auto count = affine::getConstantTripCount(loop);
        uint64_t cost = bodySize * (*count - 1);
        if (cost > maxExtraOperations - extraOperations)
          return;
        selected = loop;
        selectedCost = cost;
      });
      if (!selected)
        break;
      if (failed(affine::loopUnrollFull(selected))) {
        selected.emitError("failed to staticize register-vector indices");
        signalPassFailure();
        return;
      }
      extraOperations += selectedCost;
      // Recollect after every rewrite: unrolling invalidates nested op handles.
    }
  }
};

// Describe the entire memref, not an individual lane's access. Disjoint SSA
// indices in one thread need not be disjoint between different GPU threads.
struct BarrierMemoryRange {
  Value base;
  std::optional<int64_t> offset;
  std::optional<int64_t> bytes;
};

static std::optional<int64_t> getBarrierBufferBytes(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat())
    return std::nullopt;
  unsigned bits = type.getElementType().getIntOrFloatBitWidth();
  // For unusual widths (e.g. i24 or f80), LLVM's allocation stride may
  // include padding. Only use the native scalar widths of this GPU lowering.
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  int64_t bytes = bits / 8;
  for (int64_t size : type.getShape()) {
    if (size && bytes > std::numeric_limits<int64_t>::max() / size)
      return std::nullopt;
    bytes *= size;
  }
  return bytes;
}

static BarrierMemoryRange getBarrierMemoryRange(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return getBarrierMemoryRange(cast.getSource());
  if (auto view = value.getDefiningOp<memref::ViewOp>()) {
    auto source = getBarrierMemoryRange(view.getSource());
    auto shift = getConstantIntValue(view.getByteShift());
    auto bytes = getBarrierBufferBytes(view.getType());
    if (source.offset && shift && *shift >= 0 && bytes &&
        *source.offset <= std::numeric_limits<int64_t>::max() - *shift &&
        *source.offset + *shift <=
            std::numeric_limits<int64_t>::max() - *bytes)
      return {source.base, *source.offset + *shift, bytes};
    return {source.base, std::nullopt, std::nullopt};
  }
  if (auto *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
      auto source = getBarrierMemoryRange(view.getViewSource());
      // Subviews/reinterpret_casts may have dynamic offsets or strides. Keep
      // the underlying object identity, but do not guess their byte range.
      return {source.base, std::nullopt, std::nullopt};
    }
  }
  auto type = dyn_cast<MemRefType>(value.getType());
  // An arbitrary SSA memref (e.g. a thread-dependent select of subviews) can
  // have a different base pointer in each lane. Relative byte intervals alone
  // cannot prove those accesses disjoint across threads.
  bool fixedBase = value.getDefiningOp<memref::AllocOp>() ||
                   value.getDefiningOp<memref::AllocaOp>() ||
                   value.getDefiningOp<memref::GetGlobalOp>();
  return {value, fixedBase && type && type.getLayout().isIdentity()
                     ? std::optional<int64_t>(0) : std::nullopt,
          fixedBase && type ? getBarrierBufferBytes(type) : std::nullopt};
}

static bool barrierRangesAreDisjoint(Value lhs, Value rhs) {
  auto aType = cast<BaseMemRefType>(lhs.getType());
  auto bType = cast<BaseMemRefType>(rhs.getType());
  auto aSpace = dyn_cast_or_null<IntegerAttr>(aType.getMemorySpace());
  auto bSpace = dyn_cast_or_null<IntegerAttr>(bType.getMemorySpace());
  // Generic/default memory may alias either space. Only distinguish the
  // explicit global and workgroup address spaces used by this lowering.
  if (aSpace && bSpace &&
      ((aSpace.getInt() == 1 && bSpace.getInt() == 3) ||
       (aSpace.getInt() == 3 && bSpace.getInt() == 1)))
    return true;
  auto a = getBarrierMemoryRange(lhs), b = getBarrierMemoryRange(rhs);
  auto aGlobal = a.base.getDefiningOp<memref::GetGlobalOp>();
  auto bGlobal = b.base.getDefiningOp<memref::GetGlobalOp>();
  bool same = a.base == b.base ||
              (aGlobal && bGlobal && aGlobal.getNameAttr() == bGlobal.getNameAttr());
  if (same)
    return a.offset && a.bytes && b.offset && b.bytes &&
           (*a.offset + *a.bytes <= *b.offset ||
            *b.offset + *b.bytes <= *a.offset);
  auto isAllocation = [](Value base) {
    return base.getDefiningOp<memref::AllocOp>() ||
           base.getDefiningOp<memref::AllocaOp>() ||
           base.getDefiningOp<memref::GetGlobalOp>();
  };
  // Different function arguments, selects and region arguments can alias.
  return isAllocation(a.base) && isAllocation(b.base);
}

struct BarrierAccess {
  Value buffer;
  bool write;
};

struct BarrierNode {
  Operation *barrier = nullptr;
  bool unknown = false;
  SmallVector<BarrierAccess, 2> accesses;
  SmallVector<unsigned, 2> predecessors, successors;
};

// A small control-flow graph keeps both loop backedges and zero-trip paths.
// Only uniform for-loops expose their barriers as collective phase boundaries.
// Conditional regions are summarized without using their barriers as cutoffs.
class BarrierAnalysis {
public:
  explicit BarrierAnalysis(func::FuncOp function) : function(function) {
    kernel = function->hasAttr("thread_num") || function->hasAttr("gpu.kernel");
  }

  void run() {
    if (!llvm::hasSingleElement(function.getBody()))
      return; // Run before SCF/affine control flow is lowered to CFG blocks.
    unsigned entry = addNode();
    unsigned last = appendBlock(function.front(), entry);
    unsigned exit = addNode();
    connect(last, exit);
    // A callable helper may synchronize accesses in its caller. Kernel entry
    // and return, in contrast, cannot have concurrent accesses by this group.
    nodes[entry].unknown = nodes[exit].unknown = !kernel;
    for (unsigned index : barriers) {
      SmallVector<BarrierAccess> before, after;
      if (!collect(index, false, before) || !collect(index, true, after))
        continue;
      bool conflict = llvm::any_of(before, [&](const BarrierAccess &a) {
        return llvm::any_of(after, [&](const BarrierAccess &b) {
          return (a.write || b.write) &&
                 !barrierRangesAreDisjoint(a.buffer, b.buffer);
        });
      });
      if (!conflict) {
        nodes[index].barrier->erase();
        // Update immediately: simultaneously removing two individually
        // redundant barriers can remove the only synchronization between uses.
        nodes[index].barrier = nullptr;
      }
    }
  }

private:
  unsigned addNode() {
    nodes.emplace_back();
    return nodes.size() - 1;
  }
  void connect(unsigned from, unsigned to) {
    nodes[from].successors.push_back(to);
    nodes[to].predecessors.push_back(from);
  }

  bool isUniform(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &function.front())
        return kernel && value.getType().isIntOrIndexOrFloat();
      return arg.getArgNumber() == 0 &&
             collectiveLoops.contains(arg.getOwner()->getParentOp());
    }
    Operation *op = value.getDefiningOp();
    if (!op)
      return false;
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op))
      return true;
    return !op->getNumRegions() && isMemoryEffectFree(op) &&
           (isa<affine::AffineApplyOp>(op) ||
            op->getName().getDialectNamespace() == "arith") &&
           llvm::all_of(op->getOperands(), [&](Value v) { return isUniform(v); });
  }

  Block *getCollectiveBody(Operation *op) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (isUniform(loop.getLowerBound()) && isUniform(loop.getUpperBound()) &&
          isUniform(loop.getStep()))
        return loop.getBody();
    }
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      if (llvm::all_of(loop.getLowerBoundOperands(),
                       [&](Value v) { return isUniform(v); }) &&
          llvm::all_of(loop.getUpperBoundOperands(),
                       [&](Value v) { return isUniform(v); }))
        return loop.getBody();
    }
    return nullptr;
  }

  unsigned appendBlock(Block &block, unsigned previous) {
    for (Operation &op : block) {
      unsigned index = addNode();
      connect(previous, index);
      if (Block *body = getCollectiveBody(&op)) {
        collectiveLoops.insert(&op);
        unsigned exit = addNode();
        connect(index, exit); // May execute zero iterations.
        unsigned last = appendBlock(*body, index);
        connect(last, index); // Includes the final-iteration exit path.
        previous = exit;
        continue;
      }
      if (isa<gpu::BarrierOp>(&op)) {
        nodes[index].barrier = &op;
        barriers.push_back(index);
      } else {
        summarize(&op, nodes[index]);
      }
      previous = index;
    }
    return previous;
  }

  void summarize(Operation *op, BarrierNode &node) {
    if (isa<gpu::BarrierOp>(op)) {
      node.unknown = true; // A conditional barrier cannot protect every path.
      return;
    }
    if (isa<scf::ForOp, scf::IfOp, affine::AffineForOp, affine::AffineIfOp>(op)) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            summarize(&nested, node);
      return;
    }
    if (op->getNumRegions() || op->getNumSuccessors()) {
      node.unknown = true;
      return;
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(op) || isMemoryEffectFree(op))
      return;
    // Do not infer completion or synchronization semantics from memory effects
    // alone: async copies, atomics, fences, calls and opaque asm stay unknown.
    if (!isa<memref::LoadOp, memref::StoreOp, memref::CopyOp,
             affine::AffineLoadOp, affine::AffineStoreOp,
             affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
             vector::LoadOp, vector::StoreOp, vector::TransferReadOp,
             vector::TransferWriteOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp>(op)) {
      node.unknown = true;
      return;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    cast<MemoryEffectOpInterface>(op).getEffects(effects);
    for (const auto &effect : effects) {
      Value buffer = effect.getValue();
      if (!buffer || !isa<BaseMemRefType>(buffer.getType()) ||
          !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        node.unknown = true;
        continue;
      }
      auto type = cast<BaseMemRefType>(buffer.getType());
      auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
      if (space && space.getInt() == 5)
        continue; // Private memory cannot communicate between workitems.
      node.accesses.push_back({buffer, isa<MemoryEffects::Write>(effect.getEffect())});
    }
  }

  bool collect(unsigned barrier, bool forward,
               SmallVectorImpl<BarrierAccess> &accesses) {
    SmallVector<unsigned> worklist;
    auto enqueue = [&](unsigned index) {
      auto &edges = forward ? nodes[index].successors : nodes[index].predecessors;
      worklist.append(edges.begin(), edges.end());
    };
    enqueue(barrier);
    llvm::SmallDenseSet<unsigned, 32> visited;
    while (!worklist.empty()) {
      unsigned index = worklist.pop_back_val();
      if (!visited.insert(index).second || nodes[index].barrier)
        continue;
      if (nodes[index].unknown)
        return false;
      accesses.append(nodes[index].accesses.begin(), nodes[index].accesses.end());
      enqueue(index);
    }
    return true;
  }

  func::FuncOp function;
  bool kernel;
  SmallVector<BarrierNode> nodes;
  SmallVector<unsigned> barriers;
  llvm::SmallPtrSet<Operation *, 8> collectiveLoops;
};

class BarrierOptimizePass : public impl::IRDeepOptimizeBase<BarrierOptimizePass> {
public:
  StringRef getArgument() const override { return "barrier-optimize"; }
  StringRef getName() const override { return "BarrierOptimizePass"; }
  StringRef getDescription() const override {
    return "Remove workgroup barriers without cross-thread memory dependencies";
  }
  void runOnOperation() override { BarrierAnalysis(getOperation()).run(); }
};
} // namespace

std::unique_ptr<Pass> createIRDeepOptimizePass() {
  return std::make_unique<VectorOpLoopUnrollPass>();
}

std::unique_ptr<Pass> createBarrierOptimizePass() {
  return std::make_unique<BarrierOptimizePass>();
}


} // namespace mlir::frisk
