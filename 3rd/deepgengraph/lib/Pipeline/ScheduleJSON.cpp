// Block-level scheduling exchange. Cost estimates are never correctness proofs.
#include "deepgengraph/Pipeline/ScheduleJSON.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "deepgengraph/Analysis/RegisterPressure.h"
#include "mlir/Analysis/Liveness.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskTraits.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

using namespace mlir;
namespace {
using Loop = affine::AffineForOp;
using JObject = llvm::json::Object;
using JArray = llvm::json::Array;
constexpr int64_t schemaVersion = 1;

struct Access { Value value; bool write; };
struct Node {
  Operation *op;
  std::string kind = "alu";
  SmallVector<Access> accesses;
  bool opaque = false, anchor = true;
  int64_t readBytes = 0, writeBytes = 0, compute = 0;
};
struct Edge { unsigned src, dst; bool sameStage; std::string kind; };
struct Buffer {
  Value value;
  int64_t bytes = 0, space = 0;
  SmallVector<unsigned> readers, writers, users;
  bool rotatable = false;
  bool iterationLocal = false;
  std::string lifetimeReason = "no proof of full write, ownership and no escape";
};
struct Problem {
  bool fragmentIR = false;
  Loop loop;
  std::string id, fingerprint;
  SmallVector<Node> nodes;
  SmallVector<Edge> edges;
  SmallVector<Buffer> buffers;
  DenseMap<Value, unsigned> bufferIDs;
  bool pipelineEligible = true;
  bool ssaCarriers = true;
  std::string pipelineReason;
  JArray candidates;
};

Value root(Value value) {
  for (unsigned n = 0; n < 64; ++n) {
    auto *def = value.getDefiningOp();
    if (auto copy = dyn_cast_or_null<frisk::CopyOp>(def))
      value = copy.getDstMemRef();
    else if (isa_and_nonnull<frisk::BufferViewOp>(def))
      value = def->getOperand(0);
    else if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def))
      value = view.getViewSource();
    else if (auto castOp = dyn_cast_or_null<memref::CastOp>(def))
      value = castOp.getSource();
    else return value;
  }
  return value; // Cycles cannot occur in the supported descriptor slice.
}
int64_t elements(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) return -1;
  return shaped.getNumElements();
}
int64_t bytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() ||
      !shaped.getElementType().isIntOrFloat()) return -1;
  int64_t result;
  if (__builtin_mul_overflow(shaped.getNumElements(),
      int64_t((shaped.getElementTypeBitWidth() + 7) / 8), &result)) return -1;
  return result;
}
int64_t space(Value value) {
  if (auto type = dyn_cast<MemRefType>(value.getType()))
    return type.getMemorySpaceAsInt();
  return 0;
}
int64_t elementBytes(Value value) {
  auto type = dyn_cast<ShapedType>(value.getType());
  return type && type.getElementType().isIntOrFloat()
      ? (type.getElementTypeBitWidth() + 7) / 8 : -1;
}
bool shaped(Value value) { return isa<ShapedType>(value.getType()); }
bool fresh(Value value) {
  auto *def = value.getDefiningOp();
  return def && (isa<frisk::AllocBufferOp, memref::AllocOp>(def) ||
                def->hasTrait<frisk::ComputedTile>());
}
bool mayAlias(Value a, Value b) {
  a = root(a); b = root(b);
  if (a == b) return true;
  if (!isa<MemRefType>(a.getType()) || !isa<MemRefType>(b.getType()))
    return false;
  // Unknown descriptors (e.g. a select) may point to a local allocation.
  // Only distinct owned roots and function-entry arguments prove disjointness.
  auto external = [](Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
  };
  if ((fresh(a) && (fresh(b) || external(b))) ||
      (fresh(b) && external(a))) return false;
  return space(a) == space(b); // External descriptors may alias: no name-based proof.
}
void access(Node &node, Value value, bool write) {
  if (!shaped(value)) return;
  node.accesses.push_back({root(value), write});
  int64_t size = bytes(value.getType());
  auto &total = write ? node.writeBytes : node.readBytes;
  if (size < 0 || total < 0 || __builtin_add_overflow(total, size, &total)) total = -1;
}
// Opaque regions stay atomic and form scheduling fences. Their aggregate scale
// is diagnostic only; it is not used to justify reordering their internals.
void classify(Node &node) {
  Operation *op = node.op;
  if (auto load = dyn_cast<frisk::LoadFragOp>(op)) {
    node.kind = space(load.getSource()) == 3 ? "shm_reg" : "copy";
    access(node, load.getSource(), false);
    node.readBytes = bytes(cast<frisk::FragmentType>(load.getType()).getRegisterType());
  } else if (auto store = dyn_cast<frisk::StoreFragOp>(op)) {
    node.kind = "copy";
    access(node, store.getDestination(), true);
    node.writeBytes = bytes(cast<frisk::FragmentType>(store.getValue().getType()).getRegisterType());
  } else if (isa<frisk::GemmFragOp, frisk::ReduceFragOp>(op)) {
    node.kind = isa<frisk::GemmFragOp>(op) ? "wmma" : "reduce";
    node.compute = 1; // Register SSA dependencies are collected independently.
  } else if (auto layout = dyn_cast<frisk::ConvertLayoutFragOp>(op);
             layout && layout.getShufflePlan()) {
    node.kind = "alu"; node.compute = 1;
  } else if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
    node.kind = "dma";
    access(node, copy.getSrc(), false); access(node, copy.getDst(), true);
  } else if (auto copy = dyn_cast<frisk::CopyOp>(op)) {
    access(node, copy.getSrcMemRef(), false); access(node, copy.getDstMemRef(), true);
    node.kind = space(copy.getSrcMemRef()) == 1 && space(copy.getDstMemRef()) == 3
                    ? "copy_gm_shm" : space(copy.getSrcMemRef()) == 3 &&
                      space(copy.getDstMemRef()) == 0 ? "shm_reg" : "copy";
  } else if (isa<frisk::WaitBufferReadyOp>(op)) {
    node.kind = "wait";
  } else if (isa<frisk::AllocBufferOp, memref::AllocOp>(op)) {
    node.kind = "alloc"; node.anchor = false;
  } else if (isa<frisk::ReadMatrixFromShmOp, frisk::CopyToRegOp>(op)) {
    node.kind = space(op->getOperand(0)) == 3 ? "shm_reg" : "copy";
    access(node, op->getOperand(0), false);
    for (Value v : op->getResults()) access(node, v, true);
  } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
    node.kind = "shm_reg"; access(node, load.getMemref(), false);
    node.readBytes = elementBytes(load.getMemref());
  } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    node.kind = "shm_reg"; access(node, load.getMemref(), false);
    node.readBytes = elementBytes(load.getMemref());
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    node.kind = "copy"; access(node, store.getMemref(), true);
    node.writeBytes = elementBytes(store.getMemref());
  } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    node.kind = "copy"; access(node, store.getMemref(), true);
    node.writeBytes = elementBytes(store.getMemref());
  } else if (isa<frisk::BufferViewOp, ViewLikeOpInterface, memref::CastOp>(op)) {
    node.kind = "address"; node.anchor = false;
  } else if (isPure(op) && !op->getNumRegions() && !op->hasTrait<frisk::ComputedTile>()) {
    bool address = llvm::all_of(op->getResultTypes(), [](Type t) { return t.isIndex(); });
    node.kind = address ? "address" : "alu"; node.anchor = false;
    for (Value v : op->getOperands()) access(node, v, false);
    for (Value v : op->getResults()) access(node, v, true);
    if (!address) node.compute = op->getNumResults() && shaped(op->getResult(0))
        ? elements(op->getResult(0).getType()) : 1;
  } else if (op->hasTrait<frisk::ComputedTile>()) {
    for (Value v : op->getOperands()) access(node, v, false);
    for (Value v : op->getResults()) access(node, v, true);
    if (isa<frisk::GemmOp>(op)) {
      node.kind = "wmma";
      auto gemm = cast<frisk::GemmOp>(op);
      auto a = dyn_cast<ShapedType>(gemm.getA().getType());
      auto c = dyn_cast<ShapedType>(gemm.getC().getType());
      if (a && c && a.hasStaticShape() && c.hasStaticShape() && a.getRank() == 2) {
        int64_t k = a.getDimSize(gemm.getTransA() ? 0 : 1), product;
        node.compute = __builtin_mul_overflow(c.getNumElements(), k, &product) ||
                       __builtin_mul_overflow(product, int64_t(2), &product)
                           ? -1 : product;
      } else node.compute = -1;
    } else node.compute = op->getNumResults() ? elements(op->getResult(0).getType()) : 1;
  } else if (auto reduce = dyn_cast<frisk::ReduceOp>(op)) {
    node.kind = "reduce";
    access(node, reduce.getSrc(), false); access(node, reduce.getDst(), true);
    node.compute = elements(reduce.getSrc().getType());
  } else {
    node.kind = "opaque"; node.opaque = true;
    // Include nested accesses/scales for inspection, but remain a hard fence.
    if (op->getNumRegions()) op->walk([&](Operation *nested) {
      if (nested == op || nested->getNumRegions() || nested->hasTrait<OpTrait::IsTerminator>()) return;
      Node child{nested};
      classify(child);
      llvm::append_range(node.accesses, child.accesses);
      node.readBytes = node.readBytes < 0 || child.readBytes < 0 ? -1 : node.readBytes + child.readBytes;
      node.writeBytes = node.writeBytes < 0 || child.writeBytes < 0 ? -1 : node.writeBytes + child.writeBytes;
      node.compute = node.compute < 0 || child.compute < 0 ? -1 : node.compute + child.compute;
    });
    else for (Value v : op->getOperands()) {
      access(node, v, false); access(node, v, true);
    }
  }
  if (op->getNumRegions()) { node.opaque = true; node.anchor = true; }
}
std::string hashFunction(func::FuncOp function) {
  std::string text;
  llvm::raw_string_ostream os(text);
  function.print(os, OpPrintingFlags().useLocalScope().printGenericOpForm());
  auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}
LogicalResult selectLoop(ModuleOp module, StringRef selector, Problem &p) {
  struct Candidate { Loop loop; std::string id; bool marked; };
  SmallVector<Candidate> candidates;
  module.walk([&](func::FuncOp function) {
    unsigned ordinal = 0;
    function.walk<WalkOrder::PreOrder>([&](Loop loop) {
      std::string id = (function.getSymName() + ":" + Twine(ordinal++)).str();
      bool work = false;
      loop.walk([&](Operation *op) {
        work |= isa<frisk::GemmOp, frisk::GemmFragOp, frisk::AsyncCopyGmToShmOp, frisk::CopyOp>(op);
      });
      if (!work && selector != id && !loop->hasAttr("cpsat.mainloop")) return;
      candidates.push_back({loop, id, loop->hasAttr("cpsat.mainloop")});
      p.candidates.push_back(JObject{{"id", id}, {"marked", loop->hasAttr("cpsat.mainloop")},
          {"top_level_ops", int64_t(loop.getBody()->getOperations().size() - 1)}});
    });
  });
  SmallVector<Candidate *> choices;
  bool marked = llvm::any_of(candidates, [](auto &c) { return c.marked; });
  for (auto &c : candidates) {
    if (!selector.empty()) { if (c.id == selector) choices.push_back(&c); continue; }
    if (marked) { if (c.marked) choices.push_back(&c); continue; }
    if (!llvm::any_of(candidates, [&](auto &parent) {
          return parent.loop != c.loop && parent.loop->isProperAncestor(c.loop);
        })) choices.push_back(&c);
  }
  if (choices.size() != 1) {
    auto diag = module.emitError("CP-SAT needs one mainloop; use --cpsat-loop=function:ordinal or cpsat.mainloop; candidates:");
    for (auto &c : candidates) diag << " " << c.id;
    return failure();
  }
  p.loop = choices.front()->loop; p.id = choices.front()->id;
  p.fingerprint = hashFunction(p.loop->getParentOfType<func::FuncOp>());
  return success();
}

// Undo the legacy destination-style vector update only for a full, unique
// tail write. All reads must precede it. This exposes the actual recurrence to
// affine unrolling and the SSA-carrier pipeline materializer.
unsigned normalizeTailCarriers(Loop loop) {
  unsigned changed = 0;
  Operation *yield = loop.getBody()->getTerminator();
  for (auto [index, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    auto type = dyn_cast<VectorType>(arg.getType());
    if (!type || yield->getOperand(index) != arg) continue;
    frisk::CopyOp writer;
    bool safe = true;
    for (Operation *user : arg.getUsers()) {
      auto copy = dyn_cast<frisk::CopyOp>(user);
      if (!copy || copy.getDstMemRef() != arg) continue;
      if (writer || user->getBlock() != loop.getBody()) { safe = false; break; }
      writer = copy;
    }
    if (!safe || !writer || writer->getNumResults() ||
        !writer.getMapOperands().empty()) continue;
    auto srcType = dyn_cast<MemRefType>(writer.getSrcMemRef().getType());
    auto map = writer.getOffsetMap();
    if (!srcType || srcType.getShape() != type.getShape() ||
        srcType.getElementType() != type.getElementType() ||
        map.getNumInputs() || map.getNumResults() != 1 ||
        map.getResult(0) != getAffineConstantExpr(type.getRank(), loop.getContext()))
      continue;
    for (Operation *user : arg.getUsers()) {
      if (user == writer || user == yield) continue;
      Operation *top = loop.getBody()->findAncestorOpInBlock(*user);
      if (!top || !top->isBeforeInBlock(writer)) safe = false;
      // Only known readers of a vector value; opaque mutation/escape is not
      // covered by this conversion's full-write proof.
      auto copy = dyn_cast<frisk::CopyOp>(user);
      if (!isPure(user) && !user->hasTrait<frisk::ComputedTile>() &&
          !(copy && copy.getSrcMemRef() == arg && copy.getDstMemRef() != arg))
        safe = false;
    }
    if (!safe) continue;
    OpBuilder builder(writer);
    Value next = builder.create<frisk::CopyToRegOp>(writer.getLoc(), type,
                                                   writer.getSrcMemRef());
    for (OpOperand &operand : yield->getOpOperands())
      if (operand.get() == arg) operand.set(next);
    writer.erase();
    ++changed;
  }
  return changed;
}

// Expose content dependencies before assigning solver identities. An extract
// from a fully assembled tile must not depend on unrelated inserted slices.
// This is SSA normalization, not scheduling: no computational op is moved.
struct FoldSchedulingExtract : OpRewritePattern<frisk::ExtractFragOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::ExtractFragOp op,
                                PatternRewriter &rewriter) const override {
    auto coordinates = [](ValueRange values) -> std::optional<SmallVector<int64_t>> {
      SmallVector<int64_t> result;
      for (Value v : values) {
        APInt index;
        if (!matchPattern(v, m_ConstantInt(&index)) || index.isNegative()) return std::nullopt;
        result.push_back(index.getSExtValue());
      }
      return result;
    };
    auto at = coordinates(op.getIndices());
    if (!at) return failure();
    auto type = cast<frisk::FragmentType>(op.getType());
    auto shape = type.getRegisterType().getShape();
    Value source = op.getSource();
    while (auto insert = source.getDefiningOp<frisk::InsertFragOp>()) {
      auto where = coordinates(insert.getIndices());
      if (!where) break;
      auto inserted = cast<frisk::FragmentType>(insert.getValue().getType());
      auto other = inserted.getRegisterType().getShape();
      bool disjoint = false, contained = true;
      SmallVector<int64_t> offsets;
      for (unsigned d = 0; d < shape.size(); ++d) {
        int64_t a = (*at)[d] * shape[d], b = (*where)[d] * other[d];
        disjoint |= a + shape[d] <= b || b + other[d] <= a;
        contained &= b <= a && a + shape[d] <= b + other[d];
        offsets.push_back(a - b);
      }
      if (contained) {
        if (inserted == type && shape == other) {
          rewriter.replaceOp(op, insert.getValue());
          return success();
        }
        // Insert/extract have register-coordinate semantics. Reproduce those
        // semantics explicitly, including a change of fragment role/layout.
        Value payload = rewriter.create<frisk::UnpackFragOp>(
            op.getLoc(), inserted.getRegisterType(), insert.getValue());
        if (shape != other) {
          SmallVector<Value> scalars;
          for (int64_t element = 0; element < type.getRegisterType().getNumElements(); ++element) {
            int64_t index = element;
            SmallVector<int64_t> coordinates(offsets);
            for (int64_t d = shape.size() - 1; d >= 0; --d) {
              coordinates[d] += index % shape[d];
              index /= shape[d];
            }
            scalars.push_back(rewriter.create<vector::ExtractOp>(op.getLoc(), payload, coordinates));
          }
          auto flat = VectorType::get({int64_t(scalars.size())}, type.getRegisterType().getElementType());
          payload = rewriter.create<vector::FromElementsOp>(op.getLoc(), flat, scalars);
          payload = rewriter.create<vector::ShapeCastOp>(op.getLoc(), type.getRegisterType(), payload);
        }
        rewriter.replaceOpWithNewOp<frisk::PackFragOp>(op, type, payload);
        return success();
      }
      if (!disjoint) break; // Partial overlap/different layouts need a real slice.
      source = insert.getDestination();
    }
    if (source == op.getSource()) return failure();
    rewriter.modifyOpInPlace(op, [&] { op.getSourceMutable().assign(source); });
    return success();
  }
};

struct FoldSchedulingUnpack : OpRewritePattern<frisk::UnpackFragOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::UnpackFragOp op,
                                PatternRewriter &rewriter) const override {
    auto pack = op.getSource().getDefiningOp<frisk::PackFragOp>();
    if (!pack) return failure();
    rewriter.replaceOp(op, pack.getSource());
    return success();
  }
};

struct FoldSchedulingIdentityCast : OpRewritePattern<frisk::ElementwiseFragOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(frisk::ElementwiseFragOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "cast" || op.getInputs().size() != 1 ||
        op.getInputs()[0].getType() != op.getType()) return failure();
    rewriter.replaceOp(op, op.getInputs()[0]);
    return success();
  }
};

LogicalResult prepareSchedulingLoop(ModuleOp module,
                                   const pipeline::ScheduleJSONOptions &options) {
  Problem selection;
  if (failed(selectLoop(module, options.loopSelector, selection))) return failure();
  Loop main = selection.loop;
  // Never erase an authored schedule by expanding its region.
  auto scheduled = [](Operation *op) {
    return op->hasAttr("pipeline.stage") || op->hasAttr("pipeline.order") ||
           op->hasAttr("pipeline.scheduled");
  };
  bool authored = false;
  main.walk([&](Operation *op) { authored |= scheduled(op); });
  for (Operation *op = main->getParentOp(); op; op = op->getParentOp())
    authored |= scheduled(op);
  if (authored) return main.emitError("CP-SAT unrolling requires an unscheduled scope");
  SmallVector<Loop> loops;
  main.walk<WalkOrder::PostOrder>([&](Loop loop) {
    if (loop != main) loops.push_back(loop);
  });
  uint64_t remaining = options.unrollMaxExpandedOps;
  unsigned unrolled = 0, normalized = 0;
  for (Loop loop : loops) {
    auto trips = affine::getConstantTripCount(loop);
    if (!trips) {
      if (options.unrollAll) return loop.emitError("Fragment CP-SAT requires statically bounded inner loops");
      continue;
    }
    if (!options.unrollAll && *trips > options.unrollMaxTripCount) continue;
    uint64_t size = 0;
    loop.getBody()->walk([&](Operation *) { ++size; });
    uint64_t cost = 0;
    if (*trips > 1 && __builtin_mul_overflow(size, *trips - 1, &cost))
      return loop.emitError("inner-loop expansion size overflow");
    if (!options.unrollAll && cost > remaining) continue;
    normalized += normalizeTailCarriers(loop);
    // A hidden destination-style update cannot be treated as an SSA recurrence
    // by generic affine unrolling. Leave unsupported loops atomic.
    bool hiddenWrite = false;
    loop.walk([&](frisk::CopyOp copy) {
      hiddenWrite |= llvm::is_contained(loop.getRegionIterArgs(), copy.getDstMemRef());
    });
    if (hiddenWrite) {
      if (options.unrollAll) return loop.emitError("hidden destination-style loop recurrence");
      continue;
    }
    if (!*trips) {
      loop->replaceAllUsesWith(loop.getInits());
      loop.erase();
    } else if (failed(affine::loopUnrollFull(loop))) {
      return main.emitError("failed CP-SAT inner-loop unrolling");
    }
    if (!options.unrollAll) remaining -= cost;
    ++unrolled;
  }
  normalized += normalizeTailCarriers(main);
  if (options.unrollAll) {
    // Fold cloned constant coordinates and deduplicate SSA values before IDs.
    RewritePatternSet folding(module.getContext());
    if (options.fragmentIR)
      folding.add<FoldSchedulingExtract, FoldSchedulingUnpack,
                  FoldSchedulingIdentityCast>(module.getContext());
    if (failed(applyPatternsGreedily(module, std::move(folding)))) return failure();
    PassManager pm(module.getContext());
    pm.addPass(createCSEPass());
    if (failed(pm.run(module))) return failure();
    bool mainAlive = false;
    module.walk([&](Loop loop) { mainAlive |= loop == main; });
    if (!mainAlive) return module.emitError("selected mainloop was eliminated during Fragment preparation");
    bool nested = false;
    main.walk([&](Operation *op) {
      nested |= op != main && isa<Loop, scf::ForOp, scf::WhileOp>(op);
    });
    if (nested) return main.emitError("Fragment CP-SAT requires all inner loops to be fully unrolled");
    main->setAttr("cpsat.unroll_all", UnitAttr::get(module.getContext()));
  }
  OpBuilder builder(main);
  main->setAttr("cpsat.unroll_max_trip_count", builder.getI64IntegerAttr(options.unrollMaxTripCount));
  main->setAttr("cpsat.unrolled_loops", builder.getI64IntegerAttr(unrolled));
  main->setAttr("cpsat.normalized_carriers", builder.getI64IntegerAttr(normalized));
  llvm::errs() << "CP-SAT preparation: unrolled " << unrolled << " inner loops; normalized "
               << normalized << " vector carriers; added-op budget used "
               << options.unrollMaxExpandedOps - remaining << "\n";
  return verify(module);
}

LogicalResult analyze(ModuleOp module, StringRef selector, Problem &p, bool fragmentIR = false) {
  p.fragmentIR = fragmentIR;
  if (failed(selectLoop(module, selector, p))) return failure();
  auto rejectPipeline = [&](StringRef reason) {
    p.pipelineEligible = false;
    if (p.pipelineReason.empty()) p.pipelineReason = reason.str();
  };
  if (!p.loop.getNumIterOperands()) rejectPipeline("PipelineSchedule requires iter_args");
  // The existing epilogue is at ub-step and assumes tile zero exists.
  // Reject zero-trip/unaligned bounds instead of speculating a DMA out of bounds.
  if (p.loop.hasConstantBounds()) {
    int64_t span;
    if (__builtin_sub_overflow(p.loop.getConstantUpperBound(), p.loop.getConstantLowerBound(), &span) ||
        span <= 0 || span % p.loop.getStepAsInt())
      rejectPipeline("pipeline v1 requires nonempty, step-aligned constant bounds");
  }
  // Existing schedules cannot be rescheduled piecemeal, including ancestor scopes.
  for (Operation *op = p.loop; op; op = op->getParentOp())
    if (op->hasAttr("pipeline.scheduled"))
      return p.loop.emitError("CP-SAT cannot reschedule an already pipelined scope");
  DenseMap<Operation *, unsigned> ids;
  for (Operation &op : p.loop.getBody()->without_terminator()) {
    if (fragmentIR)
      if (auto layout = dyn_cast<frisk::ConvertLayoutFragOp>(op);
          layout && !layout.getShufflePlan())
        return op.emitError("Fragment CP-SAT needs explicit LDS scratch for non-shuffle layout conversion");
    ids[&op] = p.nodes.size();
    p.nodes.push_back(Node{&op}); classify(p.nodes.back());
    if (p.nodes.back().opaque) rejectPipeline("opaque/nested operations are supported only in intra mode");
    if (auto copy = dyn_cast<frisk::CopyOp>(op))
      if (llvm::is_contained(p.loop.getRegionIterArgs(), copy.getDstMemRef()))
        p.ssaCarriers = false;
  }
  if (p.nodes.empty()) return p.loop.emitError("empty scheduling region");
  if (!p.ssaCarriers && p.loop.getNumIterOperands() != 1)
    rejectPipeline("pipeline v1 supports only one destination-style loop carrier");
  for (auto &node : p.nodes) {
    if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(node.op)) {
      if (!copy.getToken().hasOneUse())
        return copy.emitError("CP-SAT requires exactly one local wait for each DMA token");
      Operation *user = *copy.getToken().getUsers().begin();
      if (!isa<frisk::WaitBufferReadyOp>(user) || !ids.count(user))
        return copy.emitError("CP-SAT requires a direct local wait for each DMA token");
    }
  }
  auto addEdge = [&](unsigned src, unsigned dst, bool same, StringRef kind) {
    if (src != dst) p.edges.push_back({src, dst, same, kind.str()});
  };
  auto bufferID = [&](Value value) {
    value = root(value);
    auto [it, inserted] = p.bufferIDs.try_emplace(value, p.buffers.size());
    if (inserted) p.buffers.push_back(Buffer{value, bytes(value.getType()), space(value)});
    return it->second;
  };
  for (auto [i, node] : llvm::enumerate(p.nodes)) {
    for (auto a : node.accesses) {
      auto &b = p.buffers[bufferID(a.value)];
      auto &list = a.write ? b.writers : b.readers;
      if (!llvm::is_contained(list, i)) list.push_back(i);
      if (!llvm::is_contained(b.users, i)) b.users.push_back(i);
    }
  }
  // Rotating storage is deliberately restricted to a full single-writer DMA,
  // an owned alloc_buffer, no views/selects/escapes, and one local token wait.
  for (auto &b : p.buffers) {
    if (b.writers.size() != 1) continue;
    unsigned writer = b.writers.front();
    auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(p.nodes[writer].op);
    Operation *alloc = b.value.getDefiningOp();
    if (!copy || !isa_and_nonnull<frisk::AllocBufferOp, memref::AllocOp>(alloc) ||
        alloc->getNumOperands() || b.bytes <= 0 || b.space != 3 || copy.getDst() != b.value ||
        elements(copy.getSrc().getType()) != elements(b.value.getType()) ||
        !copy.getToken().hasOneUse()) continue;
    auto wait = dyn_cast<frisk::WaitBufferReadyOp>(*copy.getToken().getUsers().begin());
    if (!wait || !ids.count(wait)) continue;
    bool owned = true;
    for (Operation *user : b.value.getUsers()) {
      if (!ids.count(user) || isa<frisk::BufferViewOp, frisk::CopyOp>(user)) owned = false;
      if (ids.count(user) && user != copy && user != wait &&
          !wait->isBeforeInBlock(user)) owned = false;
    }
    // Source data must be immutable for the whole scheduled loop (including aliases).
    for (auto &node : p.nodes)
      for (auto a : node.accesses)
        if (a.write && mayAlias(a.value, copy.getSrc())) owned = false;
    if (!owned) continue;
    b.rotatable = true; p.nodes[writer].anchor = false;
    if (ids.count(alloc)) addEdge(ids.lookup(alloc), writer, true, "allocation_writer");
    if (!llvm::is_contained(b.users, ids.lookup(wait))) b.users.push_back(ids.lookup(wait));
  }
  for (auto [i, node] : llvm::enumerate(p.nodes)) {
    // Include captures inside opaque regions as well as direct operands.
    node.op->walk([&](Operation *nested) {
      for (Value value : nested->getOperands()) {
        Operation *def = value.getDefiningOp();
        if (!def || !ids.count(def) || ids.lookup(def) == i) continue;
        bool same = true;
        if (isa<frisk::AsyncCopyGmToShmOp>(def)) same = false;
        if (isa<frisk::AllocBufferOp, memref::AllocOp>(def)) same = false;
        addEdge(ids.lookup(def), i, same, "ssa");
      }
    });
    if (auto wait = dyn_cast<frisk::WaitBufferReadyOp>(node.op)) {
      auto copy = wait.getToken().getDefiningOp<frisk::AsyncCopyGmToShmOp>();
      if (!copy || !ids.count(copy)) {
        node.opaque = true; // A nonlocal wait is a fence even in intra mode.
        rejectPipeline("wait token is not a local DMA");
      }
      else {
        unsigned issue = ids.lookup(copy);
        addEdge(issue, i, false, "completion");
        for (unsigned j = issue + 1; j < p.nodes.size(); ++j) {
          if (j == i) continue;
          for (auto a : p.nodes[j].accesses)
            if (mayAlias(a.value, copy.getDst()) ||
                (a.write && mayAlias(a.value, copy.getSrc()))) {
              if (j < i) return node.op->emitError("access before DMA completion in scheduling input");
              addEdge(i, j, true, "ready"); break;
            }
        }
        // Intra-iteration scheduling at this phase cannot prove direct hardware
        // DMA/barrier motion. Preserve relative order with other launches/waits
        // and earlier memory readers; software pipelining uses new slot barriers.
        for (unsigned j = 0; j < p.nodes.size(); ++j) {
          if (j == i || j == issue) continue;
          if (p.nodes[j].kind == "dma" || p.nodes[j].kind == "wait" ||
              (j < i && !p.nodes[j].accesses.empty()))
            addEdge(std::min<unsigned>(i, j), std::max<unsigned>(i, j), false, "intra_sync");
        }
      }
    }
  }
  for (unsigned i = 0; i < p.nodes.size(); ++i) {
    for (unsigned j = i + 1; j < p.nodes.size(); ++j) {
      if (p.nodes[i].opaque || p.nodes[j].opaque) {
        addEdge(i, j, true, "fence"); continue;
      }
      for (auto a : p.nodes[i].accesses) for (auto b : p.nodes[j].accesses) {
        if (!(a.write || b.write) || !mayAlias(a.value, b.value)) continue;
        auto found = p.bufferIDs.find(root(a.value));
        bool rotate = a.value == b.value && found != p.bufferIDs.end() &&
                      p.buffers[found->second].rotatable;
        addEdge(i, j, !rotate, a.write ? (b.write ? "waw" : "raw") : "war");
      }
    }
  }
  // Non-versioned mutable storage must not gain cross-iteration overlap.
  for (auto &b : p.buffers) {
    if (b.rotatable || b.writers.empty()) continue;
    for (unsigned i : b.users) p.nodes[i].anchor = true;
    // The materializer treats an allocation as a storage producer. Advancing
    // non-versioned scratch alone would silently rotate it despite slots=1 in
    // the JSON model. Keep its definition with the anchored reads/writes.
    if (auto *def = b.value.getDefiningOp(); def && ids.count(def))
      p.nodes[ids.lookup(def)].anchor = true;
  }
  // Scalar/vector loop-carried results cannot be advanced by this materializer.
  for (auto [i, node] : llvm::enumerate(p.nodes)) {
    for (Value operand : node.op->getOperands())
      if (llvm::is_contained(p.loop.getRegionIterArgs(), operand)) node.anchor = true;
    for (Value result : node.op->getResults())
      for (Operation *user : result.getUsers())
        if (user == p.loop.getBody()->getTerminator()) node.anchor = true;
  }
  // Export ownership facts, not fixed live intervals: Python recomputes the
  // intervals for each candidate order. Unknown aliases/carriers stay pinned.
  for (auto &b : p.buffers) {
    if (b.space != 3 || b.writers.size() != 1) continue;
    unsigned writer = b.writers.front();
    auto *def = b.value.getDefiningOp();
    bool fullWrite = b.rotatable ||
        (def == p.nodes[writer].op && def->hasTrait<frisk::ComputedTile>());
    if (!fullWrite) continue;
    bool local = true;
    for (unsigned user : b.users)
      if (user < writer || p.nodes[user].opaque) local = false;
    for (Operation *user : b.value.getUsers())
      if (!ids.count(user) || isa<frisk::BufferViewOp, ViewLikeOpInterface,
                                 memref::CastOp, arith::SelectOp>(user)) local = false;
    if (local) {
      b.iterationLocal = true;
      b.lifetimeReason = b.rotatable ? "owned full-tile DMA with local completion"
                                    : "owned computed tile with only local direct uses";
    }
  }
  // Block-wide barriers must execute collectively. Restrict pipeline v1 to
  // constant bounds in a kernel, in addition to the nonempty check above.
  if (!p.loop.hasConstantBounds() ||
      !p.loop->getParentOfType<func::FuncOp>()->hasAttr("thread_num"))
    rejectPipeline("pipeline v1 requires constant bounds and thread_num");
  for (Operation *parent = p.loop->getParentOp(); !isa<func::FuncOp>(parent);
       parent = parent->getParentOp())
    if (!isa<Loop>(parent) || !cast<Loop>(parent).hasConstantBounds())
      rejectPipeline("pipeline v1 requires constant affine enclosing loops");
  return success();
}

JArray integers(ArrayRef<unsigned> values) {
  JArray result; for (auto v : values) result.push_back(int64_t(v)); return result;
}
bool zeroIssue(Operation *op) {
  // Only representation changes with static selectors. Dynamic extraction
  // may lower to lane selection instructions and must retain its cost.
  auto constant = [](Value v) { return matchPattern(v, m_Constant()); };
  if (auto extract = dyn_cast<frisk::ExtractFragOp>(op))
    return llvm::all_of(extract.getIndices(), constant);
  if (auto insert = dyn_cast<frisk::InsertFragOp>(op))
    return llvm::all_of(insert.getIndices(), constant);
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return extract.getDynamicPosition().empty();
  return isa<frisk::PackFragOp, frisk::UnpackFragOp, vector::BitCastOp,
             vector::ShapeCastOp, vector::FromElementsOp, frisk::BufferViewOp,
             memref::AllocOp, memref::SubViewOp, memref::CastOp,
             memref::TransposeOp, arith::ConstantOp>(op);
}
// Export content identities, not vector SSA names: insert/extract/pack do not
// allocate a second copy of the whole tile. The same 32-bit origin word is
// counted once even when several fragment views reference it.
JObject registerLifetimes(Problem &p) {
  p.loop.getContext()->getOrLoadDialect<LLVM::LLVMDialect>();
  auto function = p.loop->getParentOfType<func::FuncOp>();
  Liveness liveness(function);
  auto pressure = frisk::analyzeRegisterPressure(function, liveness);
  DenseMap<Operation *, unsigned> ids;
  for (auto [i, node] : llvm::enumerate(p.nodes)) ids[node.op] = i;
  struct Atom { int64_t producer; std::set<std::pair<int64_t, int64_t>> users; };
  DenseMap<Value, unsigned> origins;
  std::map<std::pair<unsigned, uint64_t>, Atom> atoms;
  bool supported = true;
  std::function<void(Value, uint64_t, uint64_t, int64_t, int64_t)> use;
  use = [&](Value origin, uint64_t offset, uint64_t width, int64_t user, int64_t distance) {
    auto cost = pressure.costs.lookup(origin);
    if (origin.getType().isIndex() || (!cost.dataWords && !cost.unknownValues)) return;
    if (cost.unknownValues || !width) { supported = false; return; }
    auto arg = dyn_cast<BlockArgument>(origin);
    if (arg && arg.getOwner() == p.loop.getBody() && arg.getArgNumber() > 0) {
      // Incoming carrier content is the previous iteration's yielded content.
      // Unchanged/cyclic slices and dynamic inserts are pinned conservatively
      // as live-in storage. Do not recurse indefinitely through a carrier.
      Value yielded = p.loop.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
      auto mapping = pressure.storage.lookup(yielded);
      if (!distance && mapping.exact) {
        uint64_t remaining = width;
        for (auto slice : mapping.slices) {
          if (offset >= slice.bitWidth) { offset -= slice.bitWidth; continue; }
          uint64_t take = std::min(remaining, slice.bitWidth - offset);
          use(slice.origin, slice.offset + offset, take, user, 1);
          remaining -= take;
          if (!remaining) break;
          offset = 0;
        }
        if (remaining) supported = false;
        return;
      }
    }
    unsigned identity = origins.try_emplace(origin, origins.size()).first->second;
    auto *def = origin.getDefiningOp();
    int64_t producer = def && ids.count(def) ? int64_t(ids.lookup(def)) : -1;
    // Narrow ALU results are scalar registers until packing/legalization.
    // Do not assume two independently converted f16 values share one VGPR.
    // LDS/matrix payloads already arrive packed and keep 32-bit accounting.
    uint64_t unit = 32;
    if (def && isa<frisk::ElementwiseFragOp, arith::TruncFOp>(def)) {
      uint64_t bits = pressure.elementBits.lookup(origin);
      if (bits && bits < 32) unit = bits;
    }
    for (uint64_t word = offset / unit; word < (offset + width + unit - 1) / unit; ++word) {
      auto &atom = atoms.try_emplace({identity, word}, Atom{producer, {}}).first->second;
      atom.users.insert({user, distance});
    }
  };
  auto useValue = [&](Value value, int64_t user) {
    auto found = pressure.storage.find(value);
    if (found == pressure.storage.end()) { supported = false; return; }
    for (auto slice : found->second.slices)
      use(slice.origin, slice.offset, slice.bitWidth, user, 0);
  };
  for (auto [i, node] : llvm::enumerate(p.nodes)) {
    if (node.opaque) supported = false;
    for (Value value : node.op->getOperands()) useValue(value, i);
    for (Value value : node.op->getResults()) useValue(value, i);
  }
  for (Value value : p.loop.getBody()->getTerminator()->getOperands())
    useValue(value, p.nodes.size()); // n denotes the logical loop yield boundary.
  if (auto *point = pressure.lookup(p.loop))
    for (Value value : point->liveBefore)
      if (llvm::is_contained(point->liveAfter, value))
        useValue(value, p.nodes.size()); // Enclosing live-through register data.
  // Group equal lifetimes to keep the CP-SAT cumulative constraint compact.
  std::map<std::pair<int64_t, std::set<std::pair<int64_t, int64_t>>>, int64_t> groups;
  int64_t invariantWords = 0;
  for (auto &[key, atom] : atoms) {
    if (atom.producer < 0) ++invariantWords;
    else ++groups[{atom.producer, atom.users}];
  }
  JArray values;
  for (auto &[key, words] : groups) {
    JArray users;
    for (auto [op, distance] : key.second)
      users.push_back(JObject{{"op", op}, {"distance", distance}});
    values.push_back(JObject{{"producer", key.first}, {"words", words}, {"users", std::move(users)}});
  }
  return JObject{{"contract", "fragment-origin-words-v2"}, {"supported", supported},
      {"invariant_words", invariantWords}, {"values", std::move(values)},
      {"accounting", "32-bit origin words; narrow ALU results charged per scalar; alias slices deduplicated; cyclic carriers; address/backend reserve required"}};
}
JObject serialize(Problem &p) {
  JArray ops, edges, buffers;
  for (auto [i, n] : llvm::enumerate(p.nodes)) {
    JArray reads, writes;
    for (auto a : n.accesses)
      (a.write ? writes : reads).push_back(int64_t(p.bufferIDs.lookup(root(a.value))));
    int64_t issueUnits = 1;
    if (auto elementwise = dyn_cast<frisk::ElementwiseFragOp>(n.op)) {
      auto type = cast<frisk::FragmentType>(elementwise.getType()).getRegisterType();
      issueUnits = type.getNumElements();
      // Packed f32 add/mul use two lanes per instruction. Transcendentals
      // and narrowing stay scalar unless a target packing proof is present.
      if (type.getElementType().isF32() &&
          (elementwise.getKind() == "add" || elementwise.getKind() == "mul" ||
           elementwise.getKind() == "sub")) issueUnits = (issueUnits + 1) / 2;
    } else if (auto reduce = dyn_cast<frisk::ReduceFragOp>(n.op)) {
      auto src = cast<frisk::FragmentType>(reduce.getSource().getType()).getRegisterType();
      auto dst = cast<frisk::FragmentType>(reduce.getType()).getRegisterType();
      issueUnits = std::max<int64_t>(1, src.getNumElements() - dst.getNumElements());
    }
    ops.push_back(JObject{{"id", int64_t(i)}, {"name", n.op->getName().getStringRef()},
        {"kind", n.kind}, {"read_bytes", n.readBytes}, {"write_bytes", n.writeBytes},
        {"compute_ops", n.compute}, {"reads", std::move(reads)}, {"writes", std::move(writes)},
        {"pipeline_anchor", n.anchor}, {"opaque", n.opaque},
        {"zero_issue", p.fragmentIR && zeroIssue(n.op)}, {"issue_units", issueUnits}});
  }
  for (auto &e : p.edges) edges.push_back(JObject{{"src", int64_t(e.src)},
      {"dst", int64_t(e.dst)}, {"same_stage", e.sameStage}, {"kind", e.kind}});
  for (auto [i, b] : llvm::enumerate(p.buffers)) {
    std::string type; llvm::raw_string_ostream os(type); b.value.getType().print(os);
    buffers.push_back(JObject{{"id", int64_t(i)}, {"type", type}, {"bytes", b.bytes},
        {"alignment", b.value.getDefiningOp() && b.value.getDefiningOp()->getAttrOfType<IntegerAttr>("alignment")
                          ? std::max<int64_t>(16, b.value.getDefiningOp()->getAttrOfType<IntegerAttr>("alignment").getInt()) : 16},
        {"memory_space", b.space}, {"rotatable", b.rotatable},
        {"liveness", JObject{{"iteration_local", b.iterationLocal}, {"reason", b.lifetimeReason}}},
        {"readers", integers(b.readers)}, {"writers", integers(b.writers)}, {"users", integers(b.users)}});
  }
  JObject loop{{"id", p.id}, {"step", p.loop.getStepAsInt()},
               {"pipeline_eligible", p.pipelineEligible}, {"pipeline_reason", p.pipelineReason}};
  if (p.loop.hasConstantBounds()) {
    loop["lower"] = p.loop.getConstantLowerBound(); loop["upper"] = p.loop.getConstantUpperBound();
  }
  int64_t directDMA = 0, directWait = 0, nestedDMA = 0, nestedWait = 0, syncCopies = 0;
  p.loop.walk([&](Operation *op) {
    bool direct = op->getBlock() == p.loop.getBody();
    if (isa<frisk::AsyncCopyGmToShmOp>(op)) (direct ? directDMA : nestedDMA)++;
    if (isa<frisk::WaitBufferReadyOp>(op)) (direct ? directWait : nestedWait)++;
    if (auto copy = dyn_cast<frisk::CopyOp>(op))
      syncCopies += space(copy.getSrcMemRef()) == 1 && space(copy.getDstMemRef()) == 3;
  });
  bool physicalStorage = p.fragmentIR && llvm::all_of(p.buffers, [](const Buffer &b) {
    return b.space != 3 || bool(b.value.getDefiningOp<memref::AllocOp>());
  });
  JObject result{{"schema", "deepgengraph.cpsat.problem"}, {"version", schemaVersion},
      {"storage_contract", physicalStorage ? "periodic-offsets-v1" : "conservative-v1"},
      {"fingerprint", p.fingerprint}, {"phase", p.fragmentIR ? "frisk-fragment-after-layout" : "frisk-base-before-pipeline-schedule"},
      {"dma_summary", JObject{{"direct_issues", directDMA}, {"direct_waits", directWait},
          {"nested_issues", nestedDMA}, {"nested_waits", nestedWait}, {"synchronous_gm_shm", syncCopies}}},
      {"loop", std::move(loop)}, {"candidates", std::move(p.candidates)},
      {"operations", std::move(ops)}, {"dependencies", std::move(edges)},
      {"buffers", std::move(buffers)}, {"limits", JObject{{"max_slots", 2}, {"max_stages", 2}}},
      {"wait_semantics", "wait_all"}};
  if (p.fragmentIR) result["register_liveness"] = registerLifetimes(p);
  return result;
}
LogicalResult writeJSON(ModuleOp module, StringRef path, JObject object) {
  std::error_code error; llvm::raw_fd_ostream os(path, error);
  if (error) return module.emitError() << "cannot write " << path << ": " << error.message();
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(object))) << '\n';
  os.flush();
  if (os.has_error()) { os.clear_error(); return module.emitError("failed writing schedule JSON"); }
  return success();
}
JObject baseline(Problem &p) {
  JArray operations, buffers;
  for (auto [i, node] : llvm::enumerate(p.nodes))
    operations.push_back(JObject{{"id", int64_t(i)}, {"stage", 0}, {"order", int64_t(i)}});
  for (auto [i, buffer] : llvm::enumerate(p.buffers))
    buffers.push_back(JObject{{"id", int64_t(i)}, {"slots", 1}});
  return JObject{{"schema", "deepgengraph.cpsat.schedule"}, {"version", schemaVersion},
      {"fingerprint", p.fingerprint}, {"loop_id", p.id}, {"mode", "intra"},
      {"status", "BASELINE"}, {"operations", std::move(operations)}, {"buffers", std::move(buffers)}};
}

// Verify exact physical ranges against the emitted periodic order. Never trust
// the solver's reported peak or cycle estimates for memory safety.
LogicalResult applyStorage(Problem &p, const JObject &storage,
                           ArrayRef<int64_t> stages, ArrayRef<int64_t> orders) {
  auto fail = [&](StringRef why) { return p.loop.emitError() << "invalid CP-SAT storage: " << why; };
  auto pool = storage.getInteger("pool_bytes"), budget = storage.getInteger("budget_bytes");
  auto alignment = storage.getInteger("alignment");
  auto *entries = storage.getArray("buffers");
  if (!p.fragmentIR || storage.getString("kind") != "periodic-offsets-v1" ||
      storage.getInteger("period") != 2 || !pool || !budget || !alignment || !entries ||
      *pool <= 0 || *pool > *budget || *alignment < 16 ||
      (*alignment & (*alignment - 1))) return fail("unsupported contract, size or alignment");
  struct Instance { int64_t begin, end, lo, hi; };
  SmallVector<Instance> instances;
  DenseMap<unsigned, SmallVector<int64_t, 2>> offsets;
  int64_t n = p.nodes.size();
  int64_t maximum = *std::max_element(stages.begin(), stages.end());
  for (auto &entry : *entries) {
    auto *object = entry.getAsObject();
    if (!object) return fail("expected buffer object");
    auto id = object->getInteger("id");
    auto *values = object->getArray("offsets");
    if (!id || *id < 0 || *id >= int64_t(p.buffers.size()) || offsets.count(*id) ||
        !values || values->size() != 2) return fail("invalid/duplicate buffer or parity offsets");
    auto &b = p.buffers[*id];
    auto alloc = b.value.getDefiningOp<memref::AllocOp>();
    if (b.space != 3 || !b.iterationLocal || !b.rotatable || b.bytes <= 0 ||
        !alloc || alloc->getNumOperands() || !alloc.getType().getLayout().isIdentity())
      return fail("storage requires an owned static full-write shared allocation");
    int64_t requiredAlignment = alloc.getAlignment().value_or(16);
    if (requiredAlignment <= 0 || *alignment % requiredAlignment)
      return fail("allocation alignment is not preserved");
    SmallVector<int64_t, 2> pair;
    for (auto &value : *values) {
      auto offset = value.getAsInteger();
      if (!offset || *offset < 0 || *offset % *alignment || b.bytes > *pool ||
          *offset > *pool - b.bytes) return fail("offset outside pool or misaligned");
      pair.push_back(*offset);
    }
    if (!maximum && pair[0] != pair[1]) return fail("intra schedule cannot rotate physical offsets");
    offsets[*id] = pair;
    auto position = [&](unsigned id) { return stages[id] * n + orders[id]; };
    int64_t begin = position(b.writers.front()), end = begin + 1;
    for (unsigned user : b.users) {
      if (position(user) < begin) return fail("use precedes full write");
      end = std::max(end, position(user) + 1);
    }
    for (int64_t iteration = -2; iteration < 4; ++iteration) {
      int64_t lo = pair[(iteration % 2 + 2) % 2];
      Instance current{begin + iteration * n, end + iteration * n, lo, lo + b.bytes};
      for (const auto &other : instances)
        if (current.begin < other.end && other.begin < current.end &&
            current.lo < other.hi && other.lo < current.hi)
          return fail("physical ranges alias overlapping version lifetimes");
      instances.push_back(current);
    }
  }
  for (auto [id, b] : llvm::enumerate(p.buffers))
    if (b.space == 3 && !offsets.count(id)) return fail("missing shared buffer placement");
  OpBuilder builder(p.loop);
  for (auto &[id, pair] : offsets) {
    auto *alloc = p.buffers[id].value.getDefiningOp();
    alloc->setAttr("cpsat.physical_offsets", builder.getDenseI64ArrayAttr(pair));
    alloc->setAttr("cpsat.physical_bytes", builder.getI64IntegerAttr(p.buffers[id].bytes));
    if (!maximum) alloc->setAttr("cpsat.physical_offset", builder.getI64IntegerAttr(pair[0]));
  }
  auto function = p.loop->getParentOfType<func::FuncOp>();
  if (function->hasAttr("cpsat.pool_bytes")) return fail("multiple physical schedules in one function");
  function->setAttr("cpsat.pool_bytes", builder.getI64IntegerAttr(*pool));
  function->setAttr("cpsat.pool_alignment", builder.getI64IntegerAttr(*alignment));
  function->setAttr("cpsat.preserve_schedule", builder.getUnitAttr());
  // Every storage overwrite is collective: prior readers on all waves must
  // finish before a new DMA may reuse the bytes. These barriers are cloned
  // with the writer, including prologue/epilogue, without moving any operation.
  for (auto &[id, pair] : offsets)
    p.nodes[p.buffers[id].writers.front()].op->setAttr("cpsat.storage_handoff", builder.getUnitAttr());
  return success();
}

LogicalResult apply(ModuleOp module, Problem &p, const JObject &solution) {
  auto fail = [&](StringRef why) { return p.loop.emitError() << "invalid CP-SAT schedule: " << why; };
  if (solution.getString("schema") != "deepgengraph.cpsat.schedule" ||
      solution.getInteger("version") != schemaVersion ||
      solution.getString("fingerprint") != p.fingerprint || solution.getString("loop_id") != p.id)
    return fail("schema, loop identity or fingerprint mismatch");
  auto mode = solution.getString("mode");
  if (!mode || (*mode != "intra" && *mode != "pipeline")) return fail("unknown mode");
  auto *ops = solution.getArray("operations");
  if (!ops || ops->size() != p.nodes.size()) return fail("operation count mismatch");
  SmallVector<int64_t> stages(p.nodes.size(), -1), orders(p.nodes.size(), -1);
  std::set<int64_t> seenOrders;
  int64_t maxStage = 0;
  for (auto &value : *ops) {
    auto *op = value.getAsObject(); if (!op) return fail("operation must be an object");
    auto id = op->getInteger("id"), stage = op->getInteger("stage"), order = op->getInteger("order");
    if (!id || !stage || !order || *id < 0 || *id >= int64_t(p.nodes.size()) ||
        *stage < 0 || *stage >= 2 || *order < 0 || *order >= int64_t(p.nodes.size()) ||
        stages[*id] != -1 || !seenOrders.insert(*order).second) return fail("invalid/duplicate id, stage or order");
    stages[*id] = *stage; orders[*id] = *order; maxStage = std::max(maxStage, *stage);
  }
  if (*mode == "intra" && maxStage) return fail("intra mode requires stage zero");
  if (maxStage && !p.pipelineEligible) return fail(p.pipelineReason);
  if (*std::min_element(stages.begin(), stages.end()) != 0) return fail("stages must start at zero");
  for (auto [i, node] : llvm::enumerate(p.nodes))
    if (maxStage && node.anchor && stages[i] != maxStage) return fail("loop state/compute must remain in final stage");
  for (auto &e : p.edges) {
    if (e.kind == "intra_sync" && maxStage) continue;
    if (stages[e.src] > stages[e.dst] ||
        ((e.sameStage || stages[e.src] == stages[e.dst]) && orders[e.src] >= orders[e.dst]) ||
        (e.sameStage && stages[e.src] != stages[e.dst])) return fail("dependency/order violation");
  }
  auto *bufferPlans = solution.getArray("buffers");
  if (!bufferPlans || bufferPlans->size() != p.buffers.size()) return fail("buffer count mismatch");
  SmallVector<int64_t> slots(p.buffers.size(), -1);
  for (auto &value : *bufferPlans) {
    auto *entry = value.getAsObject(); if (!entry) return fail("buffer must be an object");
    auto id = entry->getInteger("id"), count = entry->getInteger("slots");
    if (!id || !count || *id < 0 || *id >= int64_t(slots.size()) || slots[*id] != -1 || *count < 1 || *count > 2)
      return fail("invalid/duplicate buffer id or slots");
    slots[*id] = *count;
  }
  for (auto [i, b] : llvm::enumerate(p.buffers)) {
    int64_t required = 1;
    if (b.rotatable) for (unsigned user : b.users)
      required = std::max(required, 1 + stages[user] - stages[b.writers.front()]);
    if (slots[i] != required) return fail("buffer version count does not match stage lifetimes");
  }
  if (solution.get("storage") && !solution.getObject("storage"))
    return fail("storage must be an object");
  if (auto *storage = solution.getObject("storage"))
    if (failed(applyStorage(p, *storage, stages, orders))) return failure();
  if (!maxStage) {
    bool identity = true;
    SmallVector<Operation *> sorted(p.nodes.size());
    for (auto [i, node] : llvm::enumerate(p.nodes)) { sorted[orders[i]] = node.op; identity &= orders[i] == int64_t(i); }
    // Baseline replay is a true no-op, preserving all downstream scheduling.
    if (identity) return success();
    for (Operation *op : sorted) op->moveBefore(p.loop.getBody()->getTerminator());
    p.loop->setAttr("pipeline.scheduled", UnitAttr::get(module.getContext()));
    p.loop->setAttr("cpsat.mode", StringAttr::get(module.getContext(), "intra"));
  } else {
    if (p.loop->hasAttr("pipeline.stage") || p.loop->hasAttr("pipeline.order"))
      return fail("remove existing manual stage/order before importing a different plan");
    OpBuilder builder(p.loop);
    p.loop->setAttr("pipeline.stage", builder.getDenseI64ArrayAttr(stages));
    p.loop->setAttr("pipeline.order", builder.getDenseI64ArrayAttr(orders));
    // applyPipelineSchedule selects the first annotated loop: avoid ambiguity.
    auto function = p.loop->getParentOfType<func::FuncOp>();
    unsigned annotated = 0;
    function.walk([&](Loop l) { annotated += l->hasAttr("pipeline.stage"); });
    if (annotated != 1) return fail("multiple annotated loops in one function");
    mlir::pipeline::PipelineScheduleOptions settings;
    settings.ssaCarriers = p.ssaCarriers;
    if (failed(mlir::pipeline::applyPipelineSchedule(function, settings))) return failure();
  }
  return verify(module);
}
} // namespace

LogicalResult mlir::pipeline::processScheduleJSON(ModuleOp module,
                                                 const ScheduleJSONOptions &options) {
  if (options.unrollMaxTripCount || options.unrollAll) {
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone()));
    if (failed(prepareSchedulingLoop(*clone, options))) return failure();
    ScheduleJSONOptions prepared = options;
    prepared.unrollMaxTripCount = 0;
    prepared.unrollAll = false;
    if (failed(processScheduleJSON(*clone, prepared))) return failure();
    module.getBodyRegion().takeBody(clone->getBodyRegion());
    return success();
  }
  Problem p;
  if (failed(analyze(module, options.loopSelector, p, options.fragmentIR))) return failure();
  if (!options.baselinePath.empty() && failed(writeJSON(module, options.baselinePath, baseline(p)))) return failure();
  if (!options.exportPath.empty() && failed(writeJSON(module, options.exportPath, serialize(p)))) return failure();
  if (!options.importPath.empty()) {
    auto input = llvm::MemoryBuffer::getFile(options.importPath);
    if (!input) return module.emitError("cannot read CP-SAT solution JSON");
    auto parsed = llvm::json::parse((*input)->getBuffer());
    if (!parsed) return module.emitError() << "invalid JSON: " << llvm::toString(parsed.takeError());
    auto *object = parsed->getAsObject();
    if (!object) return module.emitError("CP-SAT solution must be an object");
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone()));
    Problem cloneProblem;
    if (failed(analyze(*clone, p.id, cloneProblem, options.fragmentIR)) || failed(apply(*clone, cloneProblem, *object))) return failure();
    module.getBodyRegion().takeBody(clone->getBodyRegion());
  }
  if (!options.dumpIRPath.empty()) {
    std::error_code error; llvm::raw_fd_ostream os(options.dumpIRPath, error);
    if (error) return module.emitError("cannot write CP-SAT IR snapshot");
    module.print(os); os << '\n';
  }
  return success();
}
