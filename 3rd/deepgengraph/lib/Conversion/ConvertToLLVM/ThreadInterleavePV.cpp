// Interleave independent register accumulators while matrix reads, DMA tasks
// and structured control flow are still explicit in Thread IR. No LLVM IR
// chain reconstruction or instruction motion is needed at export.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/PVInterleaveOptions.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "ThreadScheduleUtils.h"
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
constexpr StringLiteral Marker = "frisk.pv_interleave";
constexpr StringLiteral QKMarker = "frisk.qk_interleave";
constexpr StringLiteral Barrier = "llvm.amdgcn.sched.barrier";
using Ops = SmallVector<Operation *, 32>;

bool isMMA(Operation *op) {
  if (!op || op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  auto vector = [](Type type, unsigned bits, unsigned elements) {
    auto v = dyn_cast<VectorType>(type);
    return v && !v.isScalable() && v.getNumElements() == elements &&
           isa<FloatType>(v.getElementType()) &&
           v.getElementTypeBitWidth() == bits;
  };
  if (!vector(op->getOperand(0).getType(), 16, 4) ||
      !vector(op->getOperand(1).getType(), 16, 4) ||
      !vector(op->getOperand(2).getType(), 32, 4) ||
      op->getResult(0).getType() != op->getOperand(2).getType()) return false;
  if (auto mma = dyn_cast<WarpMmaRROp>(op)) {
    auto name = op->getAttrOfType<StringAttr>("inst_name");
    auto constraints = op->getAttrOfType<StringAttr>("inst_constraints");
    return name && constraints && isRegisterMMA(name, constraints);
  }
  // Thread legalization currently emits the validated register-only template
  // with has_side_effects. Recognize the complete instruction and constraints,
  // exactly as prepareRegisterMMAForLLVM does, without trusting a marker.
  auto assembly = dyn_cast<LLVM::InlineAsmOp>(op);
  return assembly && !assembly.getIsAlignStack() &&
         !assembly.getOperandAttrs() &&
         isRegisterMMA(assembly.getAsmString(), assembly.getConstraints());
}

// Only transparent register views of one explicit B packet are accepted.
// PV uses matrix packets; QK uses aligned ordinary LDS vector reads.
// Unlike the LLVM matcher, indices need not have folded to low/high shuffles:
// e.g. the high-half indices can still be affine expressions in thread_id.
// Every extraction and its index SSA value is preserved, never reinterpreted.
Operation *matrixPacket(Value value, bool qk, unsigned depth = 0) {
  if (depth > 16) return nullptr;
  auto *op = value.getDefiningOp();
  if (!op) return nullptr;
  if (qk) {
    if (auto load = dyn_cast<LDSLoadOp>(op))
      return load.getAlignment() >= 8 &&
                     load.getResult().getType() ==
                         VectorType::get({4}, Float16Type::get(op->getContext()))
                 ? op : nullptr;
  }
  if (auto read = dyn_cast<ReadMatrixFromShmOp>(op)) {
    auto type = dyn_cast<VectorType>(read.getResult().getType());
    return !qk && read.getIntrinsicName() == "llvm.hcu.ds.read.m32x16.b16" &&
                   type == VectorType::get({4}, IntegerType::get(op->getContext(), 32))
               ? op : nullptr;
  }
  if (auto cast = dyn_cast<vector::BitCastOp>(op))
    return matrixPacket(cast.getSource(), qk, depth + 1);
  if (auto cast = dyn_cast<vector::ShapeCastOp>(op))
    return matrixPacket(cast.getSource(), qk, depth + 1);
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return matrixPacket(extract.getSource(), qk, depth + 1);
  if (isa<vector::FromElementsOp>(op)) {
    Operation *packet = nullptr;
    for (Value operand : op->getOperands()) {
      auto *source = matrixPacket(operand, qk, depth + 1);
      if (!source || (packet && packet != source)) return nullptr;
      packet = source;
    }
    return packet;
  }
  return nullptr;
}

bool pure(Operation *op) {
  return !op->getNumRegions() && !op->hasTrait<OpTrait::IsTerminator>() &&
         !isMMA(op) && isMemoryEffectFree(op) && isSpeculatable(op);
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

Chain getChain(Operation *root, bool qk) {
  Chain chain;
  for (Operation *mma = root; mma;) {
    if (chain.mmas.size() == 32 || mma->hasAttr(Marker) || mma->hasAttr(QKMarker)) return {};
    auto *packet = matrixPacket(mma->getOperand(1), qk);
    if (!packet || packet->getBlock() != root->getBlock() ||
        llvm::is_contained(chain.packets, packet)) return {};
    chain.mmas.push_back(mma);
    chain.packets.push_back(packet);
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

LLVM::MemoryEffectsAttr guardEffects(MLIRContext *ctx) {
  using E = LLVM::ModRefInfo;
  return LLVM::MemoryEffectsAttr::get(ctx, {E::NoModRef, E::NoModRef,
      E::ModRef, E::NoModRef, E::NoModRef, E::NoModRef});
}

// Plan and validate the entire window before changing any operation. Only
// register computations may be pulled in as dependencies. QK additionally
// preloads selected B packets inside a proven write/synchronization-free epoch.
struct Stats {
  unsigned regions = 0, hoisted = 0, guards = 0;
  std::map<std::string, unsigned> rejected;
};

bool schedule(ArrayRef<Chain *> chains, unsigned width, ModuleOp module,
              bool qk, int guardGroup, unsigned prefetchWords, Stats &stats) {
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
  if (!lastRead) return false;
  if (!qk && positions[lastRead] >= firstMMA) return reject("late matrix packet");
  if (!qk) first = lastRead->getNextNode();
  if (positions[lastMMA] - positions[first] > 2048) return reject("window limit");
  // Count every distinct B packet, including those already available. This
  // bounds the live operand footprint, not the kernel's total register usage.
  if (qk && packets.size() * 2 > prefetchWords) return reject("B operand word budget");
  Operation *end = lastMMA->getNextNode();
  Ops region;
  llvm::SmallPtrSet<Operation *, 32> members;
  SmallVector<Ops, 0> fixedSlots(mmas.size() + 1);
  unsigned ordinal = 0;
  for (Operation *op = first; op != end; op = op->getNextNode()) {
    region.push_back(op);
    members.insert(op);
    if (mmas.contains(op)) ++ordinal;
    else if (qk && packets.contains(op)) {
      // Do not dismantle a paired-load schedule owned by another pass.
      if (op->hasAttr("frisk.k_lds_pair") || op->hasAttr("frisk.k_tail_pair"))
        return reject("protected LDS pair");
    } else if (!pure(op)) {
      // Unlike PV (whose packets are already available), QK moves reads.
      // An entire candidate must be in a single write/synchronization epoch.
      if (qk || !fixed(op)) return reject("effect or synchronization boundary");
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
  if (qk) {
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

  OpBuilder builder(module.getContext());
  auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(Barrier);
  if (guardGroup != 0 && !function) {
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(module.getContext()), {builder.getI32Type()}, false);
    function = builder.create<LLVM::LLVMFuncOp>(lastMMA->getLoc(), Barrier, type);
    function.setMemoryEffectsAttr(guardEffects(module.getContext()));
  }
  for (Operation *op : plan) op->moveBefore(end);
  for (auto [index, mma] : llvm::enumerate(desired)) {
    auto guard = [&]() {
      Value zero = builder.create<LLVM::ConstantOp>(mma->getLoc(), builder.getI32Type(),
                                                   builder.getI32IntegerAttr(0));
      auto call = builder.create<LLVM::CallOp>(mma->getLoc(), function, ValueRange{zero});
      call.setMemoryEffectsAttr(guardEffects(module.getContext()));
      ++stats.guards;
    };
    if (guardGroup < 0) {
      builder.setInsertionPoint(mma);
      guard();
    }
    if (guardGroup < 0 || (guardGroup > 0 &&
        ((index + 1) % guardGroup == 0 || index + 1 == desired.size()))) {
      builder.setInsertionPointAfter(mma);
      guard();
    }
    mma->setAttr(qk ? QKMarker : Marker, builder.getI32IntegerAttr(width));
  }
  stats.hoisted += hoisted;
  ++stats.regions;
  return true;
}

template<bool QK>
class ThreadInterleavePass
    : public PassWrapper<ThreadInterleavePass<QK>, OperationPass<ModuleOp>> {
public:
  using Base = PassWrapper<ThreadInterleavePass<QK>, OperationPass<ModuleOp>>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadInterleavePass)
  ThreadInterleavePass() = default;
  ThreadInterleavePass(const ThreadInterleavePass &other) : Base(other) {}
  ThreadInterleavePass(unsigned value, int guards, unsigned words = 32) {
    width = value; guardGroup = guards; prefetchWords = words;
  }
  Pass::Option<unsigned> width{*this, "width", llvm::cl::init(0),
                         llvm::cl::desc("0 disables; 1 serial; 2/4 interleave independent MMA chains")};
  Pass::Option<int> guardGroup{*this, "guard-group", llvm::cl::init(QK ? 2 : -1),
      llvm::cl::desc("-1: legacy before/after; 0: none; 1..32: one guard per N MMAs")};
  Pass::Option<unsigned> prefetchWords{*this, "prefetch-words", llvm::cl::init(32),
      llvm::cl::desc("QK distinct B operand budget in 32-bit words per thread")};
  StringRef getArgument() const final {
    return QK ? "frisk-thread-interleave-qk" : "frisk-thread-interleave-pv";
  }
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
    if (!width) return;
    ModuleOp module = this->getOperation();
    // Validate existing declarations before planning any region. A function
    // pass must not create module symbols, hence the module-scoped driver.
    if (Operation *symbol = module.lookupSymbol(Barrier)) {
      auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol);
      auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&this->getContext()),
          {IntegerType::get(&this->getContext(), 32)}, false);
      if (!function || !function.isExternal() || function.getFunctionType() != type) {
        module.emitError("conflicting declaration for ") << Barrier;
        return this->signalPassFailure();
      }
    }
    module.walk([&](func::FuncOp function) {
      if (function.isExternal() || thread_schedule::protectedFunction(function)) return;
      Stats stats;
      auto attempt = [&](ArrayRef<Chain *> group) {
        return schedule(group, width, module, QK, guardGroup, prefetchWords, stats);
      };
      function.walk([&](Block *block) {
        SmallVector<Chain, 0> candidates;
        for (Operation &op : *block) {
          if (!isMMA(&op) || isMMA(op.getOperand(2).getDefiningOp())) continue;
          Chain chain = getChain(&op, QK);
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
            // schedule() proves packet availability (or legal QK preloading)
            // and checks the complete SSA/effect window before any movement.
            if (!visited.contains(&other) && seed.mmas.size() == other.mmas.size()) {
              group.push_back(&other);
              visited.insert(&other);
            }
          if (attempt(group)) {
            continue;
          }
          // A large group or an intervening memory effect may prevent the
          // combined schedule. Preserve previously supported packet-local
          // opportunities rather than letting an unrelated chain disable them.
          llvm::SmallPtrSet<Chain *, 16> localVisited;
          for (Chain *localSeed : group) {
            if (!localVisited.insert(localSeed).second) continue;
            SmallVector<Chain *> local{localSeed};
            for (Chain *other : group)
              if (!localVisited.contains(other) && localSeed->packets == other->packets) {
                local.push_back(other);
                localVisited.insert(other);
              }
            if (local.size() != group.size()) attempt(local);
          }
        }
      });
      llvm::errs() << (QK ? "Thread QK interleave: " : "Thread PV interleave: ")
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

bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width, bool &consumed) {
  consumed = argument.starts_with("--pv-interleave");
  if (!consumed) return true;
  unsigned parsed;
  if (!argument.consume_front("--pv-interleave=") || argument.getAsInteger(10, parsed) ||
      (parsed != 0 && parsed != 1 && parsed != 2 && parsed != 4)) return false;
  width = parsed;
  return true;
}

bool parseMMAChainScheduleOption(llvm::StringRef argument,
                                MMAChainScheduleOptions &options, bool &consumed) {
  consumed = true;
  if (argument.starts_with("--qk-interleave")) {
    unsigned value;
    if (!argument.consume_front("--qk-interleave=") || argument.getAsInteger(10, value) ||
        (value != 0 && value != 1 && value != 2 && value != 4)) return false;
    options.qkWidth = value;
  } else if (argument.starts_with("--qk-prefetch-words")) {
    unsigned value;
    if (!argument.consume_front("--qk-prefetch-words=") || argument.getAsInteger(10, value))
      return false;
    options.qkPrefetchWords = value;
  } else if (argument.starts_with("--pv-guard-group") || argument.starts_with("--qk-guard-group")) {
    bool qk = argument.starts_with("--qk-");
    int value;
    if (!argument.consume_front(qk ? "--qk-guard-group=" : "--pv-guard-group=") ||
        argument.getAsInteger(10, value) || value < -1 || value > 32) return false;
    (qk ? options.qkGuardGroup : options.pvGuardGroup) = value;
  } else consumed = false;
  return true;
}

std::unique_ptr<Pass> createThreadInterleavePVPass(unsigned width, int guardGroup) {
  return std::make_unique<ThreadInterleavePass<false>>(width, guardGroup);
}
std::unique_ptr<Pass> createThreadInterleaveQKPass(unsigned width, int guardGroup,
                                              unsigned prefetchWords) {
  return std::make_unique<ThreadInterleavePass<true>>(width, guardGroup, prefetchWords);
}
} // namespace mlir::frisk
