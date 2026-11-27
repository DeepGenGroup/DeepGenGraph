// Split explicitly opted-in local reductions before LLVM lowering. Keep the
// matcher deliberately limited to the scalar FP ops emitted by Reduce lowering.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/ThreadReductions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>

namespace mlir::frisk {
namespace {
constexpr StringLiteral Candidate = "frisk.try_multichain";
constexpr StringLiteral Rewritten = "frisk.local_reduction";
constexpr unsigned MaxNodes = 2048;

bool isProtected(func::FuncOp function) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none"})
    if (function->hasAttr(name)) return true;
  if (auto attrs = function->getAttrOfType<ArrayAttr>("passthrough"))
    for (Attribute attr : attrs) {
      if (auto pair = dyn_cast<ArrayAttr>(attr))
        if (!pair.empty()) attr = pair[0];
      if (auto name = dyn_cast<StringAttr>(attr))
        if (name.getValue() == "strictfp" || name.getValue() == "optnone")
          return true;
    }
  return false;
}

unsigned classify(Operation *op, const LocalReductionOptions &options) {
  if (!op || !op->hasAttr(Candidate) || op->hasAttr(Rewritten) ||
      op->getNumResults() != 1 || !isa<FloatType>(op->getResult(0).getType()))
    return 0;
  using O = LocalReductionOptions;
  unsigned kind = llvm::StringSwitch<unsigned>(op->getName().getStringRef())
      .Case("arith.addf", O::Add).Case("arith.mulf", O::Mul)
      .Case("arith.minnumf", O::Min).Case("arith.maxnumf", O::Max).Default(0);
  if (!(kind & options.operations)) return 0;
  if (!options.approximate) {
    auto attr = op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
    auto flags = attr ? attr.getValue() : arith::FastMathFlags::none;
    auto required = arith::FastMathFlags::nsz |
        ((kind == O::Min || kind == O::Max) ? arith::FastMathFlags::nnan
                                          : arith::FastMathFlags::reassoc);
    if ((flags & required) != required) return 0;
  }
  return kind;
}

bool hasIdentity(Operation *op, unsigned kind) {
  for (Value operand : op->getOperands()) {
    Attribute attr;
    if (!matchPattern(operand, m_Constant(&attr))) continue;
    auto constant = dyn_cast<FloatAttr>(attr);
    if (!constant) continue;
    const auto &value = constant.getValue();
    using O = LocalReductionOptions;
    if ((kind == O::Add && value.isZero()) ||
        (kind == O::Mul && value.isExactlyValue(1.0)) ||
        (kind == O::Min && value.isInfinity() && !value.isNegative()) ||
        (kind == O::Max && value.isInfinity() && value.isNegative()))
      return true;
  }
  return false;
}

unsigned rewrite(Operation *root, const LocalReductionOptions &options,
                 llvm::DenseSet<Operation *> &live) {
  unsigned kind = classify(root, options);
  if (!kind) return 0;
  SmallVector<Operation *> chain, positions;
  SmallVector<Value> leaves;
  Operation *current = root;
  while (true) {
    if (chain.size() == MaxNodes) return 0;
    chain.push_back(current);
    Operation *children[2] = {nullptr, nullptr};
    for (unsigned side = 0; side != 2; ++side) {
      auto *child = current->getOperand(side).getDefiningOp();
      if (child && child->getBlock() == root->getBlock() &&
          child->getNumResults() == 1 && child->getResult(0).hasOneUse() &&
          child->getName() == root->getName() && classify(child, options) == kind)
        children[side] = child;
    }
    if ((!children[0] && !children[1]) || (children[0] && children[1])) break;
    unsigned side = children[0] ? 0 : 1;
    leaves.push_back(current->getOperand(1 - side));
    positions.push_back(current);
    current = children[side];
  }
  // Retain the explicit identity operation once, as in the legacy LLVM pass.
  bool seed = hasIdentity(current, kind);
  if (seed) {
    leaves.push_back(current->getResult(0));
    positions.push_back(current);
  } else {
    leaves.push_back(current->getOperand(1));
    leaves.push_back(current->getOperand(0));
    positions.push_back(current);
    positions.push_back(current);
  }
  std::reverse(leaves.begin(), leaves.end());
  std::reverse(positions.begin(), positions.end());
  if (leaves.size() < options.minInputs) return 0;

  auto emit = [&](Value lhs, Value rhs, Operation *before) {
    OpBuilder builder(before);
    OperationState state(before->getLoc(), root->getName());
    state.addOperands({lhs, rhs});
    state.addTypes(lhs.getType());
    // Permission was checked on every original node. Do not carry assumptions
    // such as nnan/ninf onto newly regrouped partial sums/products.
    state.addAttribute("fastmath", arith::FastMathFlagsAttr::get(
        builder.getContext(), arith::FastMathFlags::none));
    state.addAttribute(Rewritten, builder.getUnitAttr());
    return builder.create(state)->getResult(0);
  };
  SmallVector<Value> partials(leaves.begin(), leaves.begin() + options.chains);
  for (unsigned i = options.chains; i < leaves.size(); ++i) {
    unsigned bank = i % options.chains;
    partials[bank] = emit(partials[bank], leaves[i], positions[i]);
  }
  while (partials.size() > 1) {
    SmallVector<Value> next;
    for (unsigned i = 0; i < partials.size(); i += 2)
      next.push_back(i + 1 == partials.size() ? partials[i]
                    : emit(partials[i], partials[i + 1], root));
    partials = std::move(next);
  }
  root->getResult(0).replaceAllUsesWith(partials.front());
  for (Operation *op : chain) {
    live.erase(op);
    if (op != current || !seed) op->erase();
  }
  if (seed) {
    current->removeAttr(Candidate);
    current->setAttr(Rewritten, UnitAttr::get(current->getContext()));
  }
  return leaves.size();
}

void splitLocalReductions(func::FuncOp function, const LocalReductionOptions &options) {
  if (isProtected(function)) return;
  SmallVector<Operation *> work;
  llvm::DenseSet<Operation *> live;
  function.walk([&](Operation *op) {
    if (classify(op, options)) {
      work.push_back(op);
      live.insert(op);
    }
  });
  unsigned groups = 0, inputs = 0;
  for (Operation *op : llvm::reverse(work)) {
    // Rewrites erase earlier nodes: never dereference stale worklist entries.
    if (!live.erase(op)) continue;
    if (unsigned count = rewrite(op, options, live)) {
      ++groups;
      inputs += count;
    }
  }
  // Preserve the existing tuning report format.
  llvm::errs() << "LocalReduction: " << function.getSymName() << ": "
               << groups << " groups, " << inputs << " inputs, " << options.chains
               << " chains (" << (options.approximate ? "approx" : "legal") << ")\n";
}

// Both pass arguments accept the same local chain settings. MLIR clones the
// CLI options; the C++ operation mask supplies the default when ops is omitted.
struct LocalChainSettings {
  explicit LocalChainSettings(Pass &pass, LocalReductionOptions options = {})
      : chains(pass, "chains", llvm::cl::init(options.chains)),
        minInputs(pass, "min-inputs", llvm::cl::init(options.minInputs)),
        operations(pass, "ops", llvm::cl::init("")),
        operationMask(options.operations) {}

  LogicalResult apply(func::FuncOp function, bool approximate) const {
    LocalReductionOptions options{approximate, chains, minInputs, operationMask};
    StringRef mode;
    bool consumed;
    if ((!operations.empty() && !parseLocalReductionOption(
             "--local-reduce-ops=" + operations.getValue(), mode, options, consumed)) ||
        !options.valid())
      return function.emitError("invalid thread local reduction options");
    splitLocalReductions(function, options);
    return success();
  }

  Pass::Option<unsigned> chains;
  Pass::Option<unsigned> minInputs;
  Pass::Option<std::string> operations;
  unsigned operationMask;
};

class ThreadLocalReductionPass
    : public PassWrapper<ThreadLocalReductionPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadLocalReductionPass)
  ThreadLocalReductionPass(LocalReductionOptions options = {}) : settings(*this, options) {
    approximate = options.approximate;
  }
  ThreadLocalReductionPass(const ThreadLocalReductionPass &other) : PassWrapper(other) {
    settings.operationMask = other.settings.operationMask;
  }
  StringRef getArgument() const final { return "frisk-thread-local-reduction"; }
  StringRef getDescription() const final {
    return "Split tagged scalar thread reductions into fixed-count chains";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

private:
  Option<bool> approximate{*this, "approximate", llvm::cl::init(false)};
  LocalChainSettings settings{*this};

  void runOnOperation() override {
    if (failed(settings.apply(getOperation(), approximate)))
      signalPassFailure();
  }
};

enum class Mode { Off, Legal, Approximate };

auto modeValues() {
  return llvm::cl::values(clEnumValN(Mode::Off, "off", "Skip this stage"),
                          clEnumValN(Mode::Legal, "legal", "Require legal rewrites"),
                          clEnumValN(Mode::Approximate, "approx", "Allow approximate FP rewrites"));
}

template <typename Options>
Mode modeFor(const std::optional<Options> &options) {
  if (!options) return Mode::Off;
  return options->approximate ? Mode::Approximate : Mode::Legal;
}

class ThreadReductionsPass
    : public PassWrapper<ThreadReductionsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadReductionsPass)
  ThreadReductionsPass(std::optional<LocalReductionOptions> local,
                       std::optional<DeferLaneSumOptions> lane)
      : settings(*this, local.value_or(LocalReductionOptions{})) {
    localMode = modeFor(local);
    laneMode = modeFor(lane);
    if (lane) waveSize = lane->waveSize;
  }
  ThreadReductionsPass(const ThreadReductionsPass &other) : PassWrapper(other) {
    settings.operationMask = other.settings.operationMask;
  }
  StringRef getArgument() const final { return "frisk-thread-reductions"; }
  StringRef getDescription() const final {
    return "Split local reduction chains and defer loop-carried lane sums";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, vector::VectorDialect>();
  }

private:
  Option<Mode> localMode{*this, "local-reduce", modeValues(), llvm::cl::init(Mode::Off)};
  Option<Mode> laneMode{*this, "defer-lane-sum", modeValues(), llvm::cl::init(Mode::Off)};
  LocalChainSettings settings{*this};
  Option<unsigned> waveSize{*this, "wave-size", llvm::cl::init(0)};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    DeferLaneSumOptions lane{laneMode == Mode::Approximate, waveSize};
    if (laneMode != Mode::Off && !lane.valid()) {
      function.emitError("wave-size must be 0, 32 or 64");
      return signalPassFailure();
    }
    if (localMode != Mode::Off && failed(settings.apply(function, localMode == Mode::Approximate)))
      return signalPassFailure();
    if (laneMode != Mode::Off)
      deferThreadLaneSums(function, lane);
  }
};
} // namespace

std::unique_ptr<Pass> createThreadLocalReductionPass(LocalReductionOptions options) {
  return std::make_unique<ThreadLocalReductionPass>(options);
}

std::unique_ptr<Pass> createThreadReductionsPass(
    std::optional<LocalReductionOptions> local,
    std::optional<DeferLaneSumOptions> lane) {
  return std::make_unique<ThreadReductionsPass>(local, lane);
}
} // namespace mlir::frisk
