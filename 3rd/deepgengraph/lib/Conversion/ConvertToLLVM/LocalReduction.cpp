#include "deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace mlir::frisk {
namespace {
constexpr llvm::StringLiteral Marker = "frisk.local_reduction";
constexpr unsigned MaxNodes = 2048;

struct Kind {
  unsigned opcode = 0;
  llvm::Intrinsic::ID intrinsic = llvm::Intrinsic::not_intrinsic;
  unsigned operation = 0;
  bool floating = false;
  bool operator==(const Kind &other) const {
    return opcode == other.opcode && intrinsic == other.intrinsic &&
           operation == other.operation;
  }
};

Kind classify(llvm::Instruction *inst, const LocalReductionOptions &options) {
  if (!inst || inst->getMetadata(Marker)) return {};
  auto *type = inst->getType();
  if (type->isVectorTy() && !llvm::isa<llvm::FixedVectorType>(type)) return {};
  Kind kind;
  kind.floating = type->isFPOrFPVectorTy();
  using O = LocalReductionOptions;
  if (auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(inst)) {
    kind.opcode = binary->getOpcode();
    switch (kind.opcode) {
    case llvm::Instruction::Add: case llvm::Instruction::FAdd: kind.operation = O::Add; break;
    case llvm::Instruction::Mul: case llvm::Instruction::FMul: kind.operation = O::Mul; break;
    case llvm::Instruction::And: kind.operation = O::And; break;
    case llvm::Instruction::Or: kind.operation = O::Or; break;
    case llvm::Instruction::Xor: kind.operation = O::Xor; break;
    default: return {};
    }
  } else if (auto *call = llvm::dyn_cast<llvm::IntrinsicInst>(inst)) {
    if (call->hasOperandBundles() || call->arg_size() != 2 ||
        call->isConvergent() || call->cannotDuplicate()) return {};
    kind.intrinsic = call->getIntrinsicID();
    switch (kind.intrinsic) {
    case llvm::Intrinsic::smin: case llvm::Intrinsic::umin:
    case llvm::Intrinsic::minnum: case llvm::Intrinsic::minimum: kind.operation = O::Min; break;
    case llvm::Intrinsic::smax: case llvm::Intrinsic::umax:
    case llvm::Intrinsic::maxnum: case llvm::Intrinsic::maximum: kind.operation = O::Max; break;
    default: return {};
    }
  } else return {};
  if (!(options.operations & kind.operation)) return {};
  if (kind.floating && !options.approximate) {
    auto flags = inst->getFastMathFlags();
    // Conservative legality: min/max needs no NaNs or signed-zero distinction;
    // arithmetic additionally requires explicit reassociation permission.
    if (!flags.noSignedZeros()) return {};
    if (kind.intrinsic ? !flags.noNaNs() : !flags.allowReassoc()) return {};
  }
  return kind;
}

bool hasIdentity(llvm::Instruction *inst, Kind kind, unsigned side) {
  auto *constant = llvm::dyn_cast<llvm::Constant>(inst->getOperand(side));
  if (!constant) return false;
  if (constant->getType()->isVectorTy()) constant = constant->getSplatValue();
  if (!constant) return false;
  using O = LocalReductionOptions;
  if (auto *fp = llvm::dyn_cast<llvm::ConstantFP>(constant)) {
    auto &v = fp->getValueAPF();
    if (kind.operation == O::Add) return v.isZero();
    if (kind.operation == O::Mul) return v.isExactlyValue(1.0);
    if (kind.operation == O::Max) return v.isInfinity() && v.isNegative();
    if (kind.operation == O::Min) return v.isInfinity() && !v.isNegative();
  }
  if (auto *integer = llvm::dyn_cast<llvm::ConstantInt>(constant)) {
    if (kind.operation == O::Add || kind.operation == O::Or || kind.operation == O::Xor)
      return integer->isZero();
    if (kind.operation == O::Mul) return integer->isOne();
    if (kind.operation == O::And) return integer->isMinusOne();
  }
  return false;
}

unsigned rewrite(llvm::Instruction *root, const LocalReductionOptions &options) {
  Kind kind = classify(root, options);
  if (!kind.operation) return 0;
  llvm::SmallVector<llvm::Instruction *> chain;
  llvm::SmallVector<llvm::Value *> leaves;
  llvm::SmallVector<llvm::Instruction *> positions;
  auto *current = root;
  while (true) {
    if (chain.size() == MaxNodes) return 0;
    chain.push_back(current);
    llvm::Instruction *children[2] = {nullptr, nullptr};
    for (unsigned side = 0; side != 2; ++side) {
      auto *child = llvm::dyn_cast<llvm::Instruction>(current->getOperand(side));
      if (child && child->getParent() == root->getParent() && child->hasOneUse() &&
          classify(child, options) == kind) children[side] = child;
    }
    // Don't flatten already branching trees or absorb shared intermediate uses.
    if ((!children[0] && !children[1]) || (children[0] && children[1])) break;
    unsigned side = children[0] ? 0 : 1;
    leaves.push_back(current->getOperand(1 - side));
    positions.push_back(current);
    current = children[side];
  }
  // Keep an explicit identity seed as the first partial. This reproduces the
  // tested -inf max initialization without any attention-specific matching.
  bool seed = hasIdentity(current, kind, 0) || hasIdentity(current, kind, 1);
  if (seed) {
    leaves.push_back(current);
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

  auto *marker = llvm::MDNode::get(root->getContext(), {});
  auto emit = [&](llvm::Value *lhs, llvm::Value *rhs, llvm::Instruction *before) {
    llvm::Instruction *inst;
    if (kind.intrinsic) {
      auto *callee = llvm::cast<llvm::CallInst>(root)->getCalledFunction();
      inst = llvm::CallInst::Create(callee, {lhs, rhs}, "local.reduce", before->getIterator());
    } else {
      inst = llvm::BinaryOperator::Create(
          static_cast<llvm::Instruction::BinaryOps>(kind.opcode), lhs, rhs,
          "local.reduce", before->getIterator());
    }
    // Never transfer poison-generating nowrap/fast-math flags to regrouped
    // partials. Integer operations use modular arithmetic; FP permission was
    // checked on every matched original operation, or explicitly opted into.
    inst->setDebugLoc(before->getDebugLoc());
    inst->setMetadata(Marker, marker);
    return inst;
  };
  llvm::SmallVector<llvm::Value *> partials(leaves.begin(), leaves.begin() + options.chains);
  for (unsigned i = options.chains; i < leaves.size(); ++i) {
    unsigned bank = i % options.chains;
    partials[bank] = emit(partials[bank], leaves[i], positions[i]);
  }
  while (partials.size() > 1) {
    llvm::SmallVector<llvm::Value *> next;
    for (unsigned i = 0; i < partials.size(); i += 2)
      next.push_back(i + 1 == partials.size() ? partials[i]
                    : emit(partials[i], partials[i + 1], root));
    partials = std::move(next);
  }
  auto *result = llvm::cast<llvm::Instruction>(partials.front());
  result->takeName(root);
  root->replaceAllUsesWith(result);
  // chain is ordered from root toward its inputs, so all deleted users go first.
  for (auto *inst : chain)
    if (inst != current || !seed) inst->eraseFromParent();
  if (seed) current->setMetadata(Marker, marker);
  return leaves.size();
}
} // namespace

bool LocalReductionOptions::valid() const {
  return chains >= 2 && chains <= 16 && minInputs >= chains * 2 &&
         minInputs <= MaxNodes && operations && !(operations & ~All);
}

bool parseLocalReductionOption(llvm::StringRef argument, llvm::StringRef &mode,
                              LocalReductionOptions &options, bool &consumed) {
  consumed = argument.starts_with("--local-reduce");
  if (!consumed) return true;
  if (mode.empty()) mode = "off";
  if (argument.consume_front("--local-reduce=")) {
    if (argument != "off" && argument != "legal" && argument != "approx") return false;
    mode = argument;
    options.approximate = mode == "approx";
    return true;
  }
  if (argument.consume_front("--local-reduce-chains="))
    return !argument.getAsInteger(10, options.chains);
  if (argument.consume_front("--local-reduce-min-inputs="))
    return !argument.getAsInteger(10, options.minInputs);
  if (argument.consume_front("--local-reduce-ops=")) {
    if (argument == "all") { options.operations = LocalReductionOptions::All; return true; }
    llvm::SmallVector<llvm::StringRef> names;
    argument.split(names, ',', -1, true);
    options.operations = 0;
    for (auto name : names) {
      unsigned bit = llvm::StringSwitch<unsigned>(name)
          .Case("add", LocalReductionOptions::Add).Case("mul", LocalReductionOptions::Mul)
          .Case("and", LocalReductionOptions::And).Case("or", LocalReductionOptions::Or)
          .Case("xor", LocalReductionOptions::Xor).Case("min", LocalReductionOptions::Min)
          .Case("max", LocalReductionOptions::Max).Default(0);
      if (!bit) return false;
      options.operations |= bit;
    }
    return options.operations;
  }
  return false;
}

llvm::PreservedAnalyses LocalReductionPass::run(
    llvm::Function &function, llvm::FunctionAnalysisManager &) {
  if (!options.valid() || function.isDeclaration() ||
      function.hasFnAttribute(llvm::Attribute::StrictFP) ||
      function.hasFnAttribute(llvm::Attribute::OptimizeNone)) return llvm::PreservedAnalyses::all();
  llvm::SmallVector<llvm::WeakTrackingVH> work;
  for (auto &block : function)
    for (auto &inst : block) if (classify(&inst, options).operation) work.emplace_back(&inst);
  unsigned groups = 0, inputs = 0;
  for (auto &handle : llvm::reverse(work)) {
    auto *inst = llvm::dyn_cast_or_null<llvm::Instruction>(handle);
    if (!inst) continue;
    if (unsigned count = rewrite(inst, options)) { ++groups; inputs += count; }
  }
  llvm::errs() << "LocalReduction: " << function.getName() << ": " << groups
               << " groups, " << inputs << " inputs, " << options.chains << " chains ("
               << (options.approximate ? "approx" : "legal") << ")\n";
  if (!groups) return llvm::PreservedAnalyses::all();
  auto preserved = llvm::PreservedAnalyses::none();
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

void optimizeLocalReductions(llvm::Module &module, LocalReductionOptions options) {
  llvm::FunctionAnalysisManager analyses;
  llvm::PassBuilder pb;
  pb.registerFunctionAnalyses(analyses);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(LocalReductionPass(options));
  for (auto &function : module)
    if (!function.isDeclaration()) pipeline.run(function, analyses);
}
} // namespace mlir::frisk
