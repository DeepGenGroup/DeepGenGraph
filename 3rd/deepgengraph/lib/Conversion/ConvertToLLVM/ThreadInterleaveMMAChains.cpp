// Interleave independent register accumulators while matrix reads, DMA tasks
// and structured control flow are still explicit in Thread IR. No LLVM IR
// chain reconstruction or instruction motion is needed at export.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "ThreadSchedulePlan.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <map>

namespace mlir::frisk {
namespace {
constexpr StringLiteral Marker = thread_schedule::MMAInterleaveMarker;
using Ops = SmallVector<Operation *, 32>;

bool isMMA(Operation *op) {
  return thread_schedule::mma(op, /*allowShapeAliases=*/true);
}

enum class OperandKind { Register, LDS, Matrix };
struct OperandOrigin {
  OperandKind kind;
  Operation *packet = nullptr;
};

// Chain discovery depends on accumulator SSA. Operand origins only constrain
// source selection and read motion; register operands need no memory packet.
std::optional<OperandOrigin> operandOrigin(Value value, unsigned depth = 0) {
  if (depth > 16) return {};
  auto *op = value.getDefiningOp();
  if (!op) return OperandOrigin{OperandKind::Register};
  if (auto load = dyn_cast<LDSLoadOp>(op)) {
    if (load.getAlignment() < 8 || load.getResult().getType() !=
        VectorType::get({4}, Float16Type::get(op->getContext()))) return {};
    return OperandOrigin{OperandKind::LDS, op};
  }
  if (auto read = dyn_cast<ReadMatrixFromShmOp>(op)) {
    if (read.getIntrinsicName() != "llvm.hcu.ds.read.m32x16.b16" ||
        read.getResult().getType() != VectorType::get({4}, IntegerType::get(op->getContext(), 32)))
      return {};
    return OperandOrigin{OperandKind::Matrix, op};
  }
  if (auto cast = dyn_cast<vector::BitCastOp>(op))
    return operandOrigin(cast.getSource(), depth + 1);
  if (auto cast = dyn_cast<vector::ShapeCastOp>(op))
    return operandOrigin(cast.getSource(), depth + 1);
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return operandOrigin(extract.getSource(), depth + 1);
  if (isa<vector::FromElementsOp>(op)) {
    std::optional<OperandOrigin> origin;
    for (Value operand : op->getOperands()) {
      auto next = operandOrigin(operand, depth + 1);
      if (!next || (origin && (origin->packet != next->packet || origin->kind != next->kind)))
        return {};
      origin = next;
    }
    return origin;
  }
  // Already-produced register SSA is available without moving its producer.
  return OperandOrigin{OperandKind::Register};
}

bool matchesSource(OperandKind kind, StringRef source) {
  return source == "any" || (source == "register" && kind == OperandKind::Register) ||
         (source == "lds" && kind == OperandKind::LDS) ||
         (source == "matrix" && kind == OperandKind::Matrix);
}

bool pure(Operation *op) {
  return !isMMA(op) && thread_schedule::pure(op);
}

bool fixed(Operation *op) {
  // These operations retain their original MMAC ordinal, including task
  // dependencies. Additional loads/stores/calls and region control reject.
  return isa<gpu::BarrierOp, SyncThreadsInBlockOp>(op) ||
         op->getName().getStringRef() == "frisk.async_copy_gm_to_shm" ||
         op->getName().getStringRef() == "frisk.wait_buffer_ready";
}

struct Chain {
  Ops mmas, packets;
};

Chain getChain(Operation *root, StringRef source) {
  Chain chain;
  for (Operation *mma = root; mma;) {
    if (chain.mmas.size() == 32 || mma->hasAttr(Marker) ||
        mma->hasAttr("frisk.pv_interleave") || mma->hasAttr("frisk.qk_interleave")) return {};
    auto origin = operandOrigin(mma->getOperand(1));
    if (!origin || !matchesSource(origin->kind, source)) return {};
    auto *packet = origin->packet;
    if (packet && packet->getBlock() != root->getBlock()) return {};
    chain.mmas.push_back(mma);
    if (packet) chain.packets.push_back(packet);
    Operation *next = nullptr;
    for (Operation *user : mma->getResult(0).getUsers()) {
      if (!isMMA(user) || user->getOperand(2) != mma->getResult(0)) continue;
      if (next || !mma->getResult(0).hasOneUse() ||
          user->getBlock() != root->getBlock()) return {};
      next = user;
    }
    mma = next;
  }
  return chain;
}

// Plan the complete window before changing IR. Bounded read motion only
// moves supported ordinary LDS packets inside one memory/synchronization epoch.
struct Stats {
  unsigned regions = 0, hoisted = 0, guards = 0;
  std::map<std::string, unsigned> rejected;
};

bool schedule(ArrayRef<Chain *> chains, unsigned width, ModuleOp module,
              bool moveReads, int guardGroup, unsigned prefetchWords, Stats &stats) {
  auto reject = [&](StringRef reason) {
    ++stats.rejected[reason.str()];
    return false;
  };
  if (chains.size() < 2 || chains.size() > 16 || chains.size() % width) return false;
  Block *block = chains.front()->mmas.front()->getBlock();
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *block) positions[&op] = positions.size();
  Operation *lastRead = nullptr, *lastMMA = nullptr, *first = nullptr;
  unsigned firstMMA = positions.size();
  llvm::SmallPtrSet<Operation *, 32> mmas;
  llvm::SmallPtrSet<Operation *, 32> packets;
  for (Chain *chain : chains) {
    for (Operation *read : chain->packets) {
      packets.insert(read);
      if (!lastRead || positions[read] > positions[lastRead]) lastRead = read;
    }
    for (Operation *mma : chain->mmas) {
      mmas.insert(mma);
      firstMMA = std::min(firstMMA, positions[mma]);
      if (!first || positions[mma] < positions[first]) first = mma;
      if (!lastMMA || positions[mma] > positions[lastMMA]) lastMMA = mma;
    }
  }
  for (Operation *packet : packets)
    if ((!moveReads || isa<ReadMatrixFromShmOp>(packet)) && positions[packet] >= firstMMA)
      return reject("late operand packet");
  if (!moveReads && lastRead) first = lastRead->getNextNode();
  if (positions[lastMMA] - positions[first] > 2048) return reject("window limit");
  // Count every distinct B packet, including those already available. This
  // bounds the live operand footprint, not the kernel's total register usage.
  uint64_t words = 0;
  for (Operation *packet : packets) {
    auto type = cast<VectorType>(packet->getResult(0).getType());
    words += llvm::divideCeil(uint64_t(type.getNumElements()) * type.getElementTypeBitWidth(), uint64_t(32));
  }
  if (moveReads && words > prefetchWords) return reject("B operand word budget");
  Operation *end = lastMMA->getNextNode();
  Ops region;
  llvm::SmallPtrSet<Operation *, 32> members;
  SmallVector<Ops, 0> fixedSlots(mmas.size() + 1);
  unsigned ordinal = 0;
  for (Operation *op = first; op != end; op = op->getNextNode()) {
    region.push_back(op);
    members.insert(op);
    if (mmas.contains(op)) ++ordinal;
    else if (moveReads && packets.contains(op)) {
      // Do not dismantle a paired-load schedule owned by another pass.
      if (thread_schedule::protectedLDSPair(op))
        return reject("protected LDS pair");
    } else if (!pure(op)) {
      // Read motion needs a stronger proof than register-only interleaving.
      // An entire candidate must be in a single write/synchronization epoch.
      if (moveReads || !fixed(op)) return reject("effect or synchronization boundary");
      fixedSlots[ordinal].push_back(op);
    }
  }
  Ops desired, plan;
  for (unsigned group = 0; group < chains.size(); group += width)
    for (unsigned k = 0; k < chains.front()->mmas.size(); ++k)
      for (unsigned c = group; c < group + width; ++c)
        desired.push_back(chains[c]->mmas[k]);
  llvm::SmallPtrSet<Operation *, 32> emitted, visiting;
  std::function<bool(Operation *, bool)> emit = [&](Operation *op, bool explicitOp) {
    if (!members.contains(op) || emitted.contains(op)) return true;
    if ((!explicitOp && !pure(op)) || !visiting.insert(op).second) return false;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (!emit(def, false)) return false;
    visiting.erase(op);
    emitted.insert(op);
    plan.push_back(op);
    return true;
  };
  unsigned hoisted = 0;
  if (moveReads) {
    // Preserve original load order. Only their pure address dependencies may
    // move with them; an address depending on an MMA rejects the whole plan.
    for (Operation *op : region)
      if (packets.contains(op)) {
        if (!emit(op, true)) return reject("load dependency");
        ++hoisted;
      }
  }
  for (unsigned slot = 0; slot <= desired.size(); ++slot) {
    for (Operation *op : fixedSlots[slot]) if (!emit(op, true)) return false;
    if (slot < desired.size() && !emit(desired[slot], true)) return reject("MMA dependency");
  }
  for (Operation *op : region) if (!emit(op, false)) return false;
  if (plan.size() != region.size()) return false;

  thread_schedule::LocalSchedulePlan orderPlan{region, plan};
  if (!orderPlan.validate()) return reject("SSA order");
  orderPlan.apply();
  OpBuilder builder(module.getContext());
  for (auto [index, mma] : llvm::enumerate(desired)) {
    if (guardGroup < 0) {
      thread_schedule::guard(mma, Marker);
      ++stats.guards;
    }
    if (guardGroup < 0 || (guardGroup > 0 &&
        ((index + 1) % guardGroup == 0 || index + 1 == desired.size()))) {
      thread_schedule::guard(mma->getNextNode(), Marker);
      ++stats.guards;
    }
    mma->setAttr(Marker, builder.getI32IntegerAttr(width));
  }
  stats.hoisted += hoisted;
  ++stats.regions;
  return true;
}

class ThreadInterleaveMMAChainsPass
    : public PassWrapper<ThreadInterleaveMMAChainsPass, OperationPass<ModuleOp>> {
public:
  using Base = PassWrapper<ThreadInterleaveMMAChainsPass, OperationPass<ModuleOp>>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadInterleaveMMAChainsPass)
  ThreadInterleaveMMAChainsPass() = default;
  ThreadInterleaveMMAChainsPass(const ThreadInterleaveMMAChainsPass &other) : Base(other) {}
  explicit ThreadInterleaveMMAChainsPass(const MMAChainInterleaveOptions &options) {
    width = options.width; guardGroup = options.guardGroup;
    prefetchWords = options.prefetchWords; operandSource = options.operandSource;
    loadMotion = options.loadMotion;
  }
  Pass::Option<unsigned> width{*this, "width", llvm::cl::init(0),
                         llvm::cl::desc("0 disables; 1 serial; 2/4 interleave independent MMA chains")};
  Pass::Option<int> guardGroup{*this, "guard-group", llvm::cl::init(2),
      llvm::cl::desc("-1: legacy before/after; 0: none; 1..32: one guard per N MMAs")};
  Pass::Option<unsigned> prefetchWords{*this, "prefetch-words", llvm::cl::init(32),
      llvm::cl::desc("Distinct B packet footprint budget in 32-bit words per thread")};
  Pass::Option<std::string> operandSource{*this, "operand-source", llvm::cl::init("any"),
      llvm::cl::desc("Select any, register, lds, or matrix operands")};
  Pass::Option<std::string> loadMotion{*this, "load-motion", llvm::cl::init("none"),
      llvm::cl::desc("none requires ready operands; bounded may preload ordinary LDS operands")};
  StringRef getArgument() const final { return "frisk-thread-interleave-mma-chains"; }
  StringRef getDescription() const final { return "Interleave independent MMA accumulators in Thread IR"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<FriskDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    if (width != 0 && width != 1 && width != 2 && width != 4) {
      this->getOperation().emitError("MMA interleave width must be 0, 1, 2 or 4");
      return this->signalPassFailure();
    }
    if (guardGroup < -1 || guardGroup > 32) {
      this->getOperation().emitError("guard-group must be -1, 0 or 1..32");
      return this->signalPassFailure();
    }
    if (operandSource != "any" && operandSource != "register" &&
        operandSource != "lds" && operandSource != "matrix") {
      this->getOperation().emitError("operand-source must be any, register, lds, or matrix");
      return this->signalPassFailure();
    }
    if (loadMotion != "none" && loadMotion != "bounded") {
      this->getOperation().emitError("load-motion must be none or bounded");
      return this->signalPassFailure();
    }
    if (!width) return;
    ModuleOp module = this->getOperation();
    if (failed(thread_schedule::checkGuard(module))) return this->signalPassFailure();
    module.walk([&](func::FuncOp function) {
      if (function.isExternal() || thread_schedule::protectedFunction(function)) return;
      Stats stats;
      auto attempt = [&](ArrayRef<Chain *> group) {
        return schedule(group, width, module, loadMotion == "bounded", guardGroup, prefetchWords, stats);
      };
      function.walk([&](Block *block) {
        SmallVector<Chain, 0> candidates;
        for (Operation &op : *block) {
          if (!isMMA(&op) || isMMA(op.getOperand(2).getDefiningOp())) continue;
          Chain chain = getChain(&op, operandSource);
          if (chain.mmas.size() >= 2) candidates.push_back(std::move(chain));
        }
        llvm::SmallPtrSet<Chain *, 16> visited;
        for (Chain &seed : candidates) {
          if (!visited.insert(&seed).second) continue;
          SmallVector<Chain *> group{&seed};
          for (Chain &other : candidates)
            // Output columns need not use the same V packets. Combining all
            // equal-depth independent chains avoids rejecting a region just
            // because other output columns occur between two row fragments.
            // schedule() proves packet availability (or legal bounded preloading)
            // and checks the complete SSA/effect window before any movement.
            if (!visited.contains(&other) && seed.mmas.size() == other.mmas.size()) {
              group.push_back(&other);
              visited.insert(&other);
            }
          if (attempt(group)) {
            continue;
          }
          // Retry independently schedulable windows. Register-only chains
          // have no packet key, so packet grouping alone cannot separate them
          // when an unknown effect rejects a combined window.
          llvm::SmallPtrSet<Operation *, 32> selectedReads;
          for (Chain *chain : group)
            selectedReads.insert(chain->packets.begin(), chain->packets.end());
          DenseMap<Operation *, unsigned> epochs;
          unsigned epoch = 0;
          bool moveReads = loadMotion == "bounded";
          for (Operation &op : *block) {
            bool read = moveReads && selectedReads.contains(&op) &&
                        isa<LDSLoadOp>(op) && !thread_schedule::protectedLDSPair(&op);
            if (!isMMA(&op) && !pure(&op) && !read &&
                (moveReads || !fixed(&op))) ++epoch;
            epochs[&op] = epoch;
          }
          llvm::SmallPtrSet<Chain *, 16> epochVisited;
          for (Chain *windowSeed : group) {
            if (!epochVisited.insert(windowSeed).second) continue;
            unsigned key = epochs[windowSeed->mmas.front()];
            if (epochs[windowSeed->mmas.back()] != key) continue;
            SmallVector<Chain *> window{windowSeed};
            for (Chain *other : group)
              if (!epochVisited.contains(other) && epochs[other->mmas.front()] == key &&
                  epochs[other->mmas.back()] == key) {
                window.push_back(other);
                epochVisited.insert(other);
              }
            if (window.size() != group.size() && attempt(window)) continue;
            // Different ready packet sets can still impose incompatible
            // dependency windows; retain the previous packet-local fallback.
            llvm::SmallPtrSet<Chain *, 16> localVisited;
            for (Chain *localSeed : window) {
              if (!localVisited.insert(localSeed).second) continue;
              SmallVector<Chain *> local{localSeed};
              for (Chain *other : window)
                if (!localVisited.contains(other) && localSeed->packets == other->packets) {
                  local.push_back(other);
                  localVisited.insert(other);
                }
              if (local.size() != window.size()) attempt(local);
            }
          }
        }
      });
      llvm::errs() << "Thread MMA interleave: "
                   << stats.regions << " regions, width=" << width
                   << ", guard-group=" << guardGroup << ", guards=" << stats.guards
                   << ", hoisted-loads=" << stats.hoisted
                   << ", function=" << function.getName() << '\n';
      for (auto &[reason, count] : stats.rejected)
        llvm::errs() << "  rejected " << count << " candidate windows: " << reason << '\n';
    });
  }
};
} // namespace

std::unique_ptr<Pass> createThreadInterleaveMMAChainsPass(const MMAChainInterleaveOptions &options) {
  return std::make_unique<ThreadInterleaveMMAChainsPass>(options);
}
} // namespace mlir::frisk
