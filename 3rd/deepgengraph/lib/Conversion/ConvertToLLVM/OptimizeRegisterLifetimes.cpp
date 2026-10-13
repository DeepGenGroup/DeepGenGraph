#include "deepgengraph/Conversion/ConvertToLLVM/OptimizeRegisterLifetimes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include <map>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

namespace mlir::frisk {
namespace {
constexpr llvm::StringLiteral marker = "frisk.register_lifetimes";

bool isAMDGPU(const llvm::Function &function) {
  const auto &triple = function.getParent()->getTargetTriple();
  return triple.isAMDGPU() || (triple.str().empty() &&
      function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL);
}

bool isThreadID(llvm::Value *value) {
  auto *call = llvm::dyn_cast<llvm::IntrinsicInst>(value);
  if (!call) return false;
  switch (call->getIntrinsicID()) {
  case llvm::Intrinsic::amdgcn_workitem_id_x:
  case llvm::Intrinsic::amdgcn_workitem_id_y:
  case llvm::Intrinsic::amdgcn_workitem_id_z: return true;
  default: return false;
  }
}

// Establish bounds BEFORE canonicalization and rematerialization. Otherwise
// lowering of affine floor divisions has already expanded signed correction
// DAGs, which are then needlessly recomputed at every memory access.
void exposeThreadBounds(llvm::Function &function) {
  auto *size = function.getMetadata("reqd_work_group_size");
  if (!size) return;
  for (llvm::Instruction &inst : llvm::instructions(function)) {
    if (!isThreadID(&inst)) continue;
    auto id = llvm::cast<llvm::IntrinsicInst>(inst).getIntrinsicID();
    unsigned axis = id == llvm::Intrinsic::amdgcn_workitem_id_x ? 0 :
                    id == llvm::Intrinsic::amdgcn_workitem_id_y ? 1 : 2;
    if (axis >= size->getNumOperands()) continue;
    auto *bound = llvm::mdconst::dyn_extract<llvm::ConstantInt>(size->getOperand(axis));
    if (!bound || bound->getValue().getActiveBits() > 11 ||
        bound->isZero() || bound->getZExtValue() > 1024) continue;
    auto range = llvm::computeConstantRange(
        &inst, false, llvm::SimplifyQuery(function.getDataLayout())).intersectWith(
        llvm::ConstantRange(llvm::APInt(32, 0), llvm::APInt(32, bound->getZExtValue())));
    if (!range.isEmptySet() && !range.isFullSet())
      inst.setMetadata(llvm::LLVMContext::MD_range,
          llvm::MDBuilder(function.getContext()).createRange(range.getLower(), range.getUpper()));
  }
}

// Integer/pointer expressions only. Loads, PHIs, calls, divisions and shifts
// retain their existing control dependencies: divisions/shifts are cloned only
// where the original already dominated the memory operation, never speculated.
bool isAddressExpression(llvm::Value *value) {
  auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
  if (!inst || (!inst->getType()->isIntegerTy() && !inst->getType()->isPointerTy()))
    return false;
  return llvm::isa<llvm::BinaryOperator, llvm::CastInst, llvm::ICmpInst,
                   llvm::SelectInst, llvm::GetElementPtrInst>(inst);
}

// Affine layout inverses often become long signed floor-divide chains. For a
// bounded workitem axis, prove a cheaper integer expression over its COMPLETE
// domain. This is not sampling: every legal ID is checked, including irregular
// workgroup sizes. Unknown inputs, poison, multiple axes and non-linear bit
// functions fall back. Never enumerate a kernel's data or specialize its shape.
unsigned simplifyThreadExpressions(llvm::Function &function) {
  using Table = llvm::SmallVector<llvm::APInt>;
  unsigned changed = 0;
  llvm::SmallVector<llvm::IntrinsicInst *> ids;
  for (auto &inst : llvm::instructions(function))
    if (isThreadID(&inst)) ids.push_back(llvm::cast<llvm::IntrinsicInst>(&inst));
  for (auto *id : ids) {
    auto range = llvm::computeConstantRange(
        id, false, llvm::SimplifyQuery(function.getDataLayout()));
    if (!range.getUnsignedMin().isZero() || range.isFullSet() ||
        range.getUnsignedMax().uge(1024)) continue;
    unsigned count = range.getUnsignedMax().getZExtValue() + 1;
    llvm::DenseMap<llvm::Value *, Table> tables;
    for (unsigned t = 0; t < count; ++t) tables[id].emplace_back(32, t);
    llvm::SmallVector<llvm::Instruction *> candidates;
    for (auto &inst : llvm::instructions(function)) {
      if (!isAddressExpression(&inst) || !inst.getType()->isIntegerTy() ||
          inst.getType()->getIntegerBitWidth() > 64) continue;
      bool dependent = false, known = true;
      for (auto *operand : inst.operand_values()) {
        dependent |= tables.contains(operand);
        known &= tables.contains(operand) || llvm::isa<llvm::ConstantInt>(operand);
      }
      if (!known || !dependent) continue;
      Table values;
      for (unsigned t = 0; t < count; ++t) {
        llvm::SmallVector<llvm::Constant *> operands;
        for (auto *operand : inst.operand_values())
          operands.push_back(tables.contains(operand)
              ? llvm::ConstantInt::get(function.getContext(), tables[operand][t])
              : llvm::cast<llvm::ConstantInt>(operand));
        auto *folded = llvm::dyn_cast_or_null<llvm::ConstantInt>(
            llvm::ConstantFoldInstOperands(&inst, operands, function.getDataLayout()));
        if (!folded) break;
        values.push_back(folded->getValue());
      }
      if (values.size() != count) continue;
      tables[&inst] = std::move(values);
      candidates.push_back(&inst);
    }
    // Replace largest expressions first; their now-dead subexpressions need
    // no replacement. Constants are also useful for removing signed corrections.
    for (auto *inst : llvm::reverse(candidates)) {
      if (inst->use_empty()) continue;
      const auto &values = tables[inst];
      auto base = values[0];
      if (llvm::all_of(values, [&](const auto &v) { return v == base; })) {
        inst->replaceAllUsesWith(llvm::ConstantInt::get(inst->getType(), base));
        ++changed;
        continue;
      }
      unsigned bits = base.getBitWidth();
      if (bits < 32) continue;
      llvm::SmallVector<llvm::APInt> coefficients;
      bool linear = true;
      for (unsigned bit = 0; (1u << bit) < count; ++bit)
        coefficients.push_back(values[1u << bit] - base);
      for (unsigned t = 0; t < count && linear; ++t) {
        auto predicted = base;
        for (unsigned bit = 0; bit < coefficients.size(); ++bit)
          if (t & (1u << bit)) predicted += coefficients[bit];
        linear &= predicted == values[t];
      }
      if (!linear) continue;
      // Merge bits that share a shift into one mask/shift operation. Reject
      // arbitrary multipliers rather than trading one expensive DAG for another.
      std::map<std::pair<int, bool>, uint32_t> groups;
      for (unsigned bit = 0; bit < coefficients.size(); ++bit) {
        auto coefficient = coefficients[bit];
        if (coefficient.isZero()) continue;
        bool negative = coefficient.isNegative();
        if (negative) coefficient = -coefficient;
        if (!coefficient.isPowerOf2()) { linear = false; break; }
        groups[{int(coefficient.logBase2()) - int(bit), negative}] |= 1u << bit;
      }
      if (!linear) continue;
      llvm::SmallPtrSet<llvm::Value *, 32> nodes;
      llvm::SmallVector<llvm::Value *> work{inst};
      while (!work.empty()) {
        auto *v = work.pop_back_val();
        if (v == id || !tables.contains(v) || !nodes.insert(v).second) continue;
        for (auto *operand : llvm::cast<llvm::Instruction>(v)->operand_values())
          work.push_back(operand);
      }
      unsigned estimated = 1;
      for (const auto &[key, mask] : groups)
        estimated += 2 + (key.first != 0);
      if (nodes.size() <= estimated + 1) continue;
      bool unsigned32 = llvm::all_of(values, [](const auto &v) { return v.isIntN(32); });
      bool signed32 = llvm::all_of(values, [](const auto &v) { return v.isSignedIntN(32); });
      unsigned width = unsigned32 || signed32 ? 32 : bits;
      // A high bit cannot be shifted in a narrower temporary. Keep its original
      // width even if cancellation makes the final value fit in 32 bits.
      for (const auto &coefficient : coefficients)
        if (!coefficient.isZero() &&
            (coefficient.isNegative() ? -coefficient : coefficient).getActiveBits() > width)
          width = bits;
      llvm::IRBuilder<> b(inst);
      b.SetCurrentDebugLocation(inst->getDebugLoc());
      auto *type = b.getIntNTy(width);
      llvm::Value *tid = b.CreateZExtOrTrunc(id, type);
      llvm::Value *value = llvm::ConstantInt::get(type, base.zextOrTrunc(width));
      for (const auto &[key, mask] : groups) {
        auto *part = b.CreateAnd(tid, llvm::ConstantInt::get(type, mask));
        if (key.first > 0) part = b.CreateShl(part, key.first);
        if (key.first < 0) part = b.CreateLShr(part, -key.first);
        value = key.second ? b.CreateSub(value, part) : b.CreateAdd(value, part);
      }
      value = unsigned32 ? b.CreateZExtOrTrunc(value, inst->getType(), "index.compact")
                         : b.CreateSExtOrTrunc(value, inst->getType(), "index.compact");
      inst->replaceAllUsesWith(value);
      ++changed;
    }
  }
  return changed;
}

bool isMathProducer(llvm::Instruction &inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
  if (!call || call->getNumOperandBundles() || !call->getCalledFunction()) return false;
  auto *fn = call->getCalledFunction();
  // OCML math has no errno/observable FP environment. Do not infer purity for
  // arbitrary external calls. No change to its implementation or FP accuracy.
  if (fn->getName() == "__ocml_exp2_f32")
    return call->arg_size() == 1 && call->getType()->isFloatTy() &&
           call->getArgOperand(0)->getType()->isFloatTy();
  if (!fn->isIntrinsic()) return false;
  switch (fn->getIntrinsicID()) {
  case llvm::Intrinsic::exp:
  case llvm::Intrinsic::exp2:
  case llvm::Intrinsic::log:
  case llvm::Intrinsic::log2:
  case llvm::Intrinsic::sqrt:
  case llvm::Intrinsic::amdgcn_exp2: return true;
  default: return false;
  }
}

bool isMathConsumer(llvm::Instruction *inst) {
  // Never move loads, calls (including convergent shuffles), constrained FP,
  // PHIs or stores. Ordinary FP instructions retain their exact operand order.
  return llvm::isa<llvm::FPTruncInst, llvm::FPExtInst, llvm::ExtractElementInst,
                   llvm::InsertElementInst, llvm::ShuffleVectorInst,
                   llvm::ExtractValueInst, llvm::InsertValueInst>(inst) ||
         (llvm::isa<llvm::BinaryOperator>(inst) && inst->getType()->isFPOrFPVectorTy());
}

bool canMaterialize(llvm::Type *type, const llvm::DataLayout &layout) {
  if (!type->isIntOrIntVectorTy() && !type->isFPOrFPVectorTy()) return false;
  auto bits = layout.getTypeSizeInBits(type);
  return !bits.isScalable() && (bits == 16 || bits == 32 || bits == 64 || bits == 128);
}

llvm::CallInst *materialize(llvm::Value *value, llvm::Instruction *before,
                            llvm::StringRef name) {
  auto *type = llvm::FunctionType::get(value->getType(), {value->getType()}, false);
  auto *assembly = llvm::InlineAsm::get(type, "", "=v,0", true);
  llvm::IRBuilder<> b(before);
  b.SetCurrentDebugLocation(before->getDebugLoc());
  auto *call = b.CreateCall(type, assembly, {value}, name);
  call->setDoesNotAccessMemory();
  call->setDoesNotThrow();
  call->addFnAttr(llvm::Attribute::WillReturn);
  call->addFnAttr(llvm::Attribute::Convergent);
  call->setMetadata(marker, llvm::MDNode::get(before->getContext(), {}));
  if (isThreadID(value)) {
    auto range = llvm::computeConstantRange(
        value, false, llvm::SimplifyQuery(before->getDataLayout()));
    // The identity must retain known thread bounds, otherwise constant-index
    // divisions may become much more expensive after hiding the intrinsic.
    if (auto *size = before->getFunction()->getMetadata("reqd_work_group_size")) {
      auto id = llvm::cast<llvm::IntrinsicInst>(value)->getIntrinsicID();
      unsigned axis = id == llvm::Intrinsic::amdgcn_workitem_id_x ? 0 :
                      id == llvm::Intrinsic::amdgcn_workitem_id_y ? 1 : 2;
      if (axis < size->getNumOperands())
        if (auto *bound = llvm::mdconst::dyn_extract<llvm::ConstantInt>(size->getOperand(axis)))
          if (bound->getZExtValue() > 0 && bound->getZExtValue() <= 1024)
            range = range.intersectWith(llvm::ConstantRange(
                llvm::APInt(32, 0), llvm::APInt(32, bound->getZExtValue())));
    }
    if (!range.isFullSet() && !range.isEmptySet())
      call->setMetadata(llvm::LLVMContext::MD_range,
          llvm::MDBuilder(before->getContext()).createRange(range.getLower(), range.getUpper()));
  }
  return call;
}

unsigned scheduleMathConsumers(llvm::Function &function) {
  unsigned packed = 0;
  const auto &layout = function.getDataLayout();
  for (llvm::BasicBlock &block : function) {
    llvm::SmallPtrSet<llvm::Instruction *, 32> consumers;
    llvm::SmallVector<llvm::Instruction *> work;
    for (llvm::Instruction &inst : block)
      if (isMathProducer(inst)) work.push_back(&inst);
    while (!work.empty()) {
      auto *producer = work.pop_back_val();
      for (llvm::User *user : producer->users()) {
        auto *inst = llvm::dyn_cast<llvm::Instruction>(user);
        if (inst && inst->getParent() == &block && isMathConsumer(inst) &&
            consumers.insert(inst).second) work.push_back(inst);
      }
    }
    llvm::SmallVector<llvm::Instruction *> ordered;
    for (llvm::Instruction &inst : block)
      if (consumers.contains(&inst)) ordered.push_back(&inst);
    for (llvm::Instruction *inst : ordered) {
      llvm::Instruction *latest = nullptr;
      for (llvm::Value *operand : inst->operands())
        if (auto *definition = llvm::dyn_cast<llvm::Instruction>(operand))
          if (definition->getParent() == &block && !llvm::isa<llvm::PHINode>(definition) &&
              (!latest || latest->comesBefore(definition))) latest = definition;
      if (latest && latest->getNextNode() != inst)
        inst->moveBefore(block, std::next(latest->getIterator()));
      auto *narrow = llvm::dyn_cast<llvm::FPTruncInst>(inst);
      if (!narrow || !canMaterialize(narrow->getOperand(0)->getType(), layout) ||
          !canMaterialize(narrow->getType(), layout)) continue;
      // Force the full-precision result once, instead of keeping the math
      // library's correction operands alive for a later mixed-precision FMA.
      auto *input = materialize(narrow->getOperand(0), narrow, "math.result");
      narrow->setOperand(0, input);
      auto *output = materialize(narrow, narrow->getNextNode(), "math.packed");
      narrow->replaceUsesWithIf(output, [&](llvm::Use &use) { return use.getUser() != output; });
      ++packed;
    }
  }
  return packed;
}

class AddressRematerializer {
  llvm::DenseMap<llvm::Value *, bool> divergent;
  llvm::DenseMap<llvm::Value *, llvm::Value *> local;
  llvm::SmallPtrSet<llvm::Value *, 8> retained;
  unsigned &clones;
  // Bound code growth for arbitrary generated expressions. The limit applies
  // to newly cloned nodes, sharing expressions already rebuilt in this block.
  bool affordable(llvm::Value *root) {
    llvm::SmallPtrSet<llvm::Value *, 32> seen;
    llvm::SmallVector<llvm::Value *> work{root};
    unsigned cost = 0;
    while (!work.empty()) {
      auto *value = work.pop_back_val();
      if (local.contains(value) || retained.contains(value) ||
          !dependsOnThread(value) || !seen.insert(value).second)
        continue;
      if (++cost > 96) return false;
      if (!isThreadID(value))
        for (llvm::Value *operand : llvm::cast<llvm::Instruction>(value)->operands())
          work.push_back(operand);
    }
    return true;
  }
  bool dependsOnThread(llvm::Value *value, unsigned depth = 0) {
    if (isThreadID(value)) return true;
    if (auto found = divergent.find(value); found != divergent.end()) return found->second;
    if (depth > 96 || !isAddressExpression(value)) return false;
    bool result = false;
    for (llvm::Value *operand : llvm::cast<llvm::Instruction>(value)->operands())
      result |= dependsOnThread(operand, depth + 1);
    divergent[value] = result;
    return result;
  }
  llvm::Value *rebuild(llvm::Value *value, llvm::Instruction *before) {
    if (!dependsOnThread(value)) return value;
    if (auto found = local.find(value); found != local.end()) return found->second;
    if (retained.contains(value))
      return local[value] = materialize(value, before, "address.base");
    if (isThreadID(value))
      return local[value] = materialize(value, before, "address.thread");
    auto *copy = llvm::cast<llvm::Instruction>(value)->clone();
    for (llvm::Use &operand : copy->operands()) operand.set(rebuild(operand.get(), before));
    copy->insertBefore(before->getIterator());
    copy->setName("address.local");
    ++clones;
    return local[value] = copy;
  }
public:
  explicit AddressRematerializer(unsigned &clones) : clones(clones) {}
  void chooseSharedBases(llvm::Function &function, llvm::LoopInfo &loops) {
    llvm::DenseMap<llvm::Instruction *, unsigned> savings;
    for (auto &block : function) {
      unsigned depth = loops.getLoopDepth(&block);
      if (!depth) continue;
      for (auto &inst : block) {
        llvm::Value *pointer = nullptr;
        if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) pointer = load->getPointerOperand();
        if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) pointer = store->getPointerOperand();
        if (!pointer) continue;
        llvm::SmallPtrSet<llvm::Value *, 32> seen;
        llvm::SmallVector<llvm::Value *> work{pointer};
        while (!work.empty() && seen.size() < 256) {
          auto *value = work.pop_back_val();
          if (!dependsOnThread(value) || !seen.insert(value).second || isThreadID(value)) continue;
          auto *expression = llvm::cast<llvm::Instruction>(value);
          if (expression->getParent() == &function.getEntryBlock() &&
              (expression->getOpcode() == llvm::Instruction::SDiv ||
               expression->getOpcode() == llvm::Instruction::SRem ||
               expression->getOpcode() == llvm::Instruction::UDiv ||
               expression->getOpcode() == llvm::Instruction::URem))
            savings[expression] += (1u << std::min(depth, 8u)) *
                expression->getType()->getIntegerBitWidth();
          for (auto *operand : expression->operand_values()) work.push_back(operand);
        }
      }
    }
    llvm::SmallVector<llvm::Instruction *> ranked;
    for (auto &inst : function.getEntryBlock()) if (savings.contains(&inst)) ranked.push_back(&inst);
    llvm::stable_sort(ranked, [&](auto *a, auto *b) { return savings[a] > savings[b]; });
    unsigned words = 0;
    // A bounded frontier of expensive shared offsets, never an unbounded set
    // of final pointers. It trades at most eight register words for avoiding
    // repeated division DAGs in nested loops; cheap offsets still rematerialize.
    for (auto *inst : ranked) {
      unsigned size = (inst->getType()->getIntegerBitWidth() + 31) / 32;
      if (words + size > 8) continue;
      retained.insert(inst);
      words += size;
    }
    llvm::errs() << "Address shared bases: " << retained.size() << " (" << words << " register words)\n";
  }
  void run(llvm::BasicBlock &block) {
    local.clear();
    // Snapshot accesses before inserting instructions. Per-block caching
    // shares nearby addresses, but never carries their expanded DAGs through
    // another block/loop. Uniform base pointers and loop PHIs remain shared.
    llvm::SmallVector<llvm::Instruction *> accesses;
    for (llvm::Instruction &inst : block)
      if (llvm::isa<llvm::LoadInst, llvm::StoreInst>(inst)) accesses.push_back(&inst);
    for (llvm::Instruction *inst : accesses) {
      unsigned pointer = llvm::isa<llvm::LoadInst>(inst) ? 0 : 1;
      auto *type = llvm::cast<llvm::PointerType>(inst->getOperand(pointer)->getType());
      if (type->getAddressSpace() != 1 && type->getAddressSpace() != 3) continue;
      if (!affordable(inst->getOperand(pointer))) continue;
      inst->setOperand(pointer, rebuild(inst->getOperand(pointer), inst));
    }
  }
};
} // namespace

llvm::PreservedAnalyses OptimizeRegisterLifetimesPass::run(
    llvm::Function &function, llvm::FunctionAnalysisManager &analyses) {
  if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP) ||
      function.hasFnAttribute(marker) || !isAMDGPU(function))
    return llvm::PreservedAnalyses::all();
  unsigned packed = scheduleMathConsumers(function), clones = 0;
  AddressRematerializer addresses(clones);
  addresses.chooseSharedBases(function, analyses.getResult<llvm::LoopAnalysis>(function));
  for (llvm::BasicBlock &block : function) addresses.run(block);
  function.addFnAttr(marker, "1");
  llvm::errs() << "OptimizeRegisterLifetimes: " << function.getName() << ": "
               << clones << " local address expressions, " << packed << " early math packs\n";
  auto preserved = llvm::PreservedAnalyses::none();
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

void optimizeRegisterLifetimes(llvm::Module &module) {
  llvm::FunctionAnalysisManager analyses;
  llvm::ModuleAnalysisManager modules;
  llvm::LoopAnalysisManager loops;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(modules);
  pb.registerLoopAnalyses(loops);
  pb.registerCGSCCAnalyses(cgscc);
  pb.registerFunctionAnalyses(analyses);
  pb.crossRegisterProxies(loops, analyses, cgscc, modules);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(llvm::InstCombinePass());
  pipeline.addPass(OptimizeRegisterLifetimesPass());
  pipeline.addPass(llvm::DCEPass());
  for (llvm::Function &function : module) {
    if (function.isDeclaration() || function.hasFnAttribute(marker) ||
        function.hasFnAttribute(llvm::Attribute::StrictFP) ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone) || !isAMDGPU(function)) continue;
    exposeThreadBounds(function);
    unsigned simplified = simplifyThreadExpressions(function);
    llvm::errs() << "Compact thread indices: " << simplified << " proven replacements\n";
    pipeline.run(function, analyses);
  }
}
} // namespace mlir::frisk
