// Interleave independent register accumulators while matrix reads, DMA tasks
// and structured control flow are still explicit in Thread IR. No LLVM IR
// chain reconstruction or instruction motion is needed at export.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/PVInterleaveOptions.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace mlir::frisk {
namespace {
constexpr StringLiteral Marker = "frisk.pv_interleave";
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

// Only transparent register views of one explicit matrix packet are accepted.
// Unlike the LLVM matcher, indices need not have folded to low/high shuffles:
// e.g. the high-half indices can still be affine expressions in thread_id.
// Every extraction and its index SSA value is preserved, never reinterpreted.
Operation *matrixPacket(Value value, unsigned depth = 0) {
  if (depth > 16) return nullptr;
  auto *op = value.getDefiningOp();
  if (!op) return nullptr;
  if (auto read = dyn_cast<ReadMatrixFromShmOp>(op)) {
    auto type = dyn_cast<VectorType>(read.getResult().getType());
    return read.getIntrinsicName() == "llvm.hcu.ds.read.m32x16.b16" &&
                   type == VectorType::get({4}, IntegerType::get(op->getContext(), 32))
               ? op : nullptr;
  }
  if (auto cast = dyn_cast<vector::BitCastOp>(op))
    return matrixPacket(cast.getSource(), depth + 1);
  if (auto cast = dyn_cast<vector::ShapeCastOp>(op))
    return matrixPacket(cast.getSource(), depth + 1);
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return matrixPacket(extract.getSource(), depth + 1);
  if (isa<vector::FromElementsOp>(op)) {
    Operation *packet = nullptr;
    for (Value operand : op->getOperands()) {
      auto *source = matrixPacket(operand, depth + 1);
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

Chain getChain(Operation *root) {
  Chain chain;
  for (Operation *mma = root; mma;) {
    if (chain.mmas.size() == 32 || mma->hasAttr(Marker)) return {};
    auto *packet = matrixPacket(mma->getOperand(1));
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
// register computations may be pulled in as dependencies; no load, collective
// or different MMAC can be speculated to satisfy a requested schedule.
bool schedule(ArrayRef<Chain *> chains, unsigned width, ModuleOp module) {
  if (chains.size() < 2 || chains.size() > 16 || chains.size() % width) return false;
  Block *block = chains.front()->mmas.front()->getBlock();
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *block) positions[&op] = positions.size();
  Operation *lastRead = nullptr, *lastMMA = nullptr;
  unsigned firstMMA = positions.size();
  llvm::SmallPtrSet<Operation *, 32> mmas;
  for (Operation *read : chains.front()->packets)
    if (!lastRead || positions[read] > positions[lastRead]) lastRead = read;
  for (Chain *chain : chains)
    for (Operation *mma : chain->mmas) {
      mmas.insert(mma);
      firstMMA = std::min(firstMMA, positions[mma]);
      if (!lastMMA || positions[mma] > positions[lastMMA]) lastMMA = mma;
    }
  if (!lastRead || positions[lastRead] >= firstMMA ||
      positions[lastMMA] - positions[lastRead] > 2048) return false;
  Operation *end = lastMMA->getNextNode();
  Ops region;
  llvm::SmallPtrSet<Operation *, 32> members;
  SmallVector<Ops, 0> fixedSlots(mmas.size() + 1);
  unsigned ordinal = 0;
  for (Operation *op = lastRead->getNextNode(); op != end; op = op->getNextNode()) {
    region.push_back(op);
    members.insert(op);
    if (mmas.contains(op)) ++ordinal;
    else if (!pure(op)) {
      if (!fixed(op)) return false;
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
  for (unsigned slot = 0; slot <= desired.size(); ++slot) {
    for (Operation *op : fixedSlots[slot]) if (!emit(op, true)) return false;
    if (slot < desired.size() && !emit(desired[slot], true)) return false;
  }
  for (Operation *op : region) if (!emit(op, false)) return false;
  if (plan.size() != region.size()) return false;

  OpBuilder builder(module.getContext());
  auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(Barrier);
  if (!function) {
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(module.getContext()), {builder.getI32Type()}, false);
    function = builder.create<LLVM::LLVMFuncOp>(lastMMA->getLoc(), Barrier, type);
    function.setMemoryEffectsAttr(guardEffects(module.getContext()));
  }
  for (Operation *op : plan) op->moveBefore(end);
  for (Operation *mma : desired) {
    auto guard = [&]() {
      Value zero = builder.create<LLVM::ConstantOp>(mma->getLoc(), builder.getI32Type(),
                                                   builder.getI32IntegerAttr(0));
      auto call = builder.create<LLVM::CallOp>(mma->getLoc(), function, ValueRange{zero});
      call.setMemoryEffectsAttr(guardEffects(module.getContext()));
    };
    builder.setInsertionPoint(mma);
    guard();
    builder.setInsertionPointAfter(mma);
    guard();
    mma->setAttr(Marker, builder.getI32IntegerAttr(width));
  }
  return true;
}

class ThreadInterleavePVPass
    : public PassWrapper<ThreadInterleavePVPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadInterleavePVPass)
  ThreadInterleavePVPass() = default;
  ThreadInterleavePVPass(const ThreadInterleavePVPass &other) : PassWrapper(other) {}
  explicit ThreadInterleavePVPass(unsigned value) { width = value; }
  Option<unsigned> width{*this, "width", llvm::cl::init(0),
                         llvm::cl::desc("0 disables; 1 guards; 2/4 interleave independent PV chains")};
  StringRef getArgument() const final { return "frisk-thread-interleave-pv"; }
  StringRef getDescription() const final { return "Interleave PV accumulators after thread loop unrolling"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<FriskDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    if (width != 0 && width != 1 && width != 2 && width != 4) {
      getOperation().emitError("PV interleave width must be 0, 1, 2 or 4");
      return signalPassFailure();
    }
    if (!width) return;
    auto module = getOperation();
    // Validate existing declarations before planning any region. A function
    // pass must not create module symbols, hence the module-scoped driver.
    if (Operation *symbol = module.lookupSymbol(Barrier)) {
      auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol);
      auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&getContext()),
          {IntegerType::get(&getContext(), 32)}, false);
      if (!function || !function.isExternal() || function.getFunctionType() != type) {
        module.emitError("conflicting declaration for ") << Barrier;
        return signalPassFailure();
      }
    }
    module.walk([&](func::FuncOp function) {
      unsigned changed = 0;
      function.walk([&](Block *block) {
        SmallVector<Chain, 0> candidates;
        for (Operation &op : *block) {
          if (!isMMA(&op) || isMMA(op.getOperand(2).getDefiningOp())) continue;
          Chain chain = getChain(&op);
          if (chain.mmas.size() >= 2) candidates.push_back(std::move(chain));
        }
        llvm::SmallPtrSet<Chain *, 16> visited;
        for (Chain &seed : candidates) {
          if (!visited.insert(&seed).second) continue;
          SmallVector<Chain *> group{&seed};
          for (Chain &other : candidates)
            if (!visited.contains(&other) && seed.packets == other.packets) {
              group.push_back(&other);
              visited.insert(&other);
            }
          changed += schedule(group, width, module);
        }
      });
      llvm::errs() << "Thread PV interleave: " << changed << " regions, width="
                   << width << ", function=" << function.getName() << '\n';
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

std::unique_ptr<Pass> createThreadInterleavePVPass(unsigned width) {
  return std::make_unique<ThreadInterleavePVPass>(width);
}
} // namespace mlir::frisk
