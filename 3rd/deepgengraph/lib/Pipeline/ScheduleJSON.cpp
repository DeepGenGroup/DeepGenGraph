// Block-level scheduling exchange. Cost estimates are never correctness proofs.
#include "deepgengraph/Pipeline/ScheduleJSON.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskTraits.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
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
  if (auto copy = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
    node.kind = "dma";
    access(node, copy.getSrc(), false); access(node, copy.getDst(), true);
  } else if (auto copy = dyn_cast<frisk::CopyOp>(op)) {
    access(node, copy.getSrcMemRef(), false); access(node, copy.getDstMemRef(), true);
    node.kind = space(copy.getSrcMemRef()) == 1 && space(copy.getDstMemRef()) == 3
                    ? "copy_gm_shm" : space(copy.getSrcMemRef()) == 3 &&
                      space(copy.getDstMemRef()) == 0 ? "shm_reg" : "copy";
  } else if (isa<frisk::WaitBufferReadyOp>(op)) {
    node.kind = "wait";
  } else if (isa<frisk::AllocBufferOp>(op)) {
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
        work |= isa<frisk::GemmOp, frisk::AsyncCopyGmToShmOp, frisk::CopyOp>(op);
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

LogicalResult analyze(ModuleOp module, StringRef selector, Problem &p) {
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
    auto alloc = b.value.getDefiningOp<frisk::AllocBufferOp>();
    if (!copy || !alloc || b.bytes <= 0 || b.space != 3 || copy.getDst() != b.value ||
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
        if (isa<frisk::AllocBufferOp>(def)) same = false;
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
JObject serialize(Problem &p) {
  JArray ops, edges, buffers;
  for (auto [i, n] : llvm::enumerate(p.nodes)) {
    JArray reads, writes;
    for (auto a : n.accesses)
      (a.write ? writes : reads).push_back(int64_t(p.bufferIDs.lookup(root(a.value))));
    ops.push_back(JObject{{"id", int64_t(i)}, {"name", n.op->getName().getStringRef()},
        {"kind", n.kind}, {"read_bytes", n.readBytes}, {"write_bytes", n.writeBytes},
        {"compute_ops", n.compute}, {"reads", std::move(reads)}, {"writes", std::move(writes)},
        {"pipeline_anchor", n.anchor}, {"opaque", n.opaque}});
  }
  for (auto &e : p.edges) edges.push_back(JObject{{"src", int64_t(e.src)},
      {"dst", int64_t(e.dst)}, {"same_stage", e.sameStage}, {"kind", e.kind}});
  for (auto [i, b] : llvm::enumerate(p.buffers)) {
    std::string type; llvm::raw_string_ostream os(type); b.value.getType().print(os);
    buffers.push_back(JObject{{"id", int64_t(i)}, {"type", type}, {"bytes", b.bytes},
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
  return JObject{{"schema", "deepgengraph.cpsat.problem"}, {"version", schemaVersion},
      {"fingerprint", p.fingerprint}, {"phase", "frisk-base-before-pipeline-schedule"},
      {"dma_summary", JObject{{"direct_issues", directDMA}, {"direct_waits", directWait},
          {"nested_issues", nestedDMA}, {"nested_waits", nestedWait}, {"synchronous_gm_shm", syncCopies}}},
      {"loop", std::move(loop)}, {"candidates", std::move(p.candidates)},
      {"operations", std::move(ops)}, {"dependencies", std::move(edges)},
      {"buffers", std::move(buffers)}, {"limits", JObject{{"max_slots", 2}, {"max_stages", 2}}},
      {"wait_semantics", "wait_all"}};
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
  Problem p;
  if (failed(analyze(module, options.loopSelector, p))) return failure();
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
    if (failed(analyze(*clone, p.id, cloneProblem)) || failed(apply(*clone, cloneProblem, *object))) return failure();
    module.getBodyRegion().takeBody(clone->getBodyRegion());
  }
  if (!options.dumpIRPath.empty()) {
    std::error_code error; llvm::raw_fd_ostream os(options.dumpIRPath, error);
    if (error) return module.emitError("cannot write CP-SAT IR snapshot");
    module.print(os); os << '\n';
  }
  return success();
}
