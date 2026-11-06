// ===========================================================================
// Pass: optimizeRegisterLifetimes   优化汇总.py 第 23 项，LLVM 阶段
//
// 作用：缩短值在寄存器里的存活时间，降低 VGPR 压力。实际包含 5 个相互独立的优化：
//   exposeThreadBounds          给 workitem id 打上 !range，后续 InstCombine 能化简除法。
//   simplifyThreadExpressions   枚举所有线程号，把长的地址表达式换成等价的“掩码 + 移位”。
//   scheduleMathConsumers       把数学结果的消费者紧贴生产者，尽早 fptrunc 打包成 half。
//   AddressRematerializer       把依赖线程号的地址表达式在每个基本块内就地重建，不跨块存活。
//   compactIntegerComparisons   把一组 64-bit 比较拆成共享的高位比较 + 独立的低位比较。
//
// 执行顺序（文件末尾的 optimizeRegisterLifetimes）：
//   exposeThreadBounds -> simplifyThreadExpressions -> InstCombine -> OptimizeRegisterLifetimesPass
//   （math + address）-> DCE -> compactIntegerComparisons
//
// marker：函数属性 frisk.register_lifetimes，防止重复运行。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/OptimizeRegisterLifetimes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "LateIRPatterns.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include <map>
#include <tuple>
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
constexpr llvm::StringLiteral marker = detail::RegisterLifetimesMarker;

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
// 根据 reqd_work_group_size 给 workitem.id 添加 !range。必须在规范化之前做，否则 floordiv 的
// 符号修正已经展开，之后再化简就晚了。
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
// 对每个线程号指令，枚举 [0, count) 上所有依赖它的整数表达式的值，再找更便宜的等价表达式。
// 是对完整定义域的枚举证明，不是采样。
unsigned simplifyThreadExpressions(llvm::Function &function) {
  // Table：表达式在每个线程号上的取值。
  // tables：值 -> Table，线程号自己是恒等表。
  // candidates：能完整求值的指令，顺序是定义顺序。
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
    // 从大表达式往小处理。全部相同的直接替换成常量；否则尝试下面的按位线性形式。
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
      // coefficients[bit] = f(1 << bit) - f(0)。
      // linear 证明 f(t) = f(0) + sum(coefficients[bit] for t 的每个置位 bit)。
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
      // groups[(移位量, 是否为负)] = 线程号的掩码位。
      // 每组生成一次 (tid & mask) << shift。系数不是 2 的幂时放弃，不用乘法。
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
      // nodes：原表达式 DAG 的规模。estimated：新表达式的估计规模。不明显更小时放弃替换。
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
      // width：优先用 32-bit 计算（更省寄存器），但系数放不下时保持原宽度。
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

// 产生数学结果的指令：exp/log/sqrt 类。
// isMathConsumer：可以安全前移的消费者（fptrunc、向量拼装、浮点算术）。
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

// 生成一个空 inline asm：约束 "=v,0" 把值强制放进 VGPR，不产生任何指令。
// 作用是“钉住”值的生成位置，阻止 LLVM 把计算沉到或提到别处。
//   discriminator  非 0 时增加一个无用的立即数，避免相同的 asm 被 CSE 合并。
//   marker         标记这是本 Pass 生成的。
// 线程号的 identity 还要带上 !range，否则隐藏 intrinsic 后除法优化会失效。
llvm::CallInst *materialize(llvm::Value *value, llvm::Instruction *before,
                            llvm::StringRef name, unsigned discriminator = 0) {
  llvm::IRBuilder<> b(before);
  b.SetCurrentDebugLocation(before->getDebugLoc());
  llvm::SmallVector<llvm::Type *> types{value->getType()};
  llvm::SmallVector<llvm::Value *> args{value};
  if (discriminator) {
    types.push_back(b.getInt32Ty());
    args.push_back(b.getInt32(discriminator));
  }
  auto *type = llvm::FunctionType::get(value->getType(), types, false);
  auto *assembly = llvm::InlineAsm::get(type, "", discriminator ? "=v,0,i" : "=v,0", true);
  auto *call = b.CreateCall(type, assembly, args, name);
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

// Strip only operations that provably leave the upper word unchanged. Unlike
// add/sub, low-word OR/XOR cannot carry into it. This is independent of tensor
// shapes, launch dimensions, and the interpretation of the comparison.
// 去掉只改低 32 位的 or/xor 常量，返回高 32 位相同的根。add/sub 可能进位，不能去。
llvm::Value *upperWordRoot(llvm::Value *value) {
  for (unsigned depth = 0; depth < 8; ++depth) {
    auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(value);
    if (!binary || (binary->getOpcode() != llvm::Instruction::Or &&
                    binary->getOpcode() != llvm::Instruction::Xor)) break;
    auto *constant = llvm::dyn_cast<llvm::ConstantInt>(binary->getOperand(1));
    if (!constant || constant->getValue().getActiveBits() > 32) break;
    value = binary->getOperand(0);
  }
  return value;
}

// 值是否依赖线程号（在 VGPR 里）。纯标量的比较没有优化价值。
bool hasVectorInput(llvm::Value *value, llvm::SmallPtrSetImpl<llvm::Value *> &seen) {
  if (!seen.insert(value).second) return false;
  if (isThreadID(value)) return true;
  // Device-function arguments use the vector ABI unless explicitly inreg;
  // kernel arguments and workgroup IDs are uniform. Do not force scalar-only
  // index predicates into VGPRs. Unknown calls/loads conservatively opt out.
  if (auto *arg = llvm::dyn_cast<llvm::Argument>(value))
    return arg->getParent()->getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL &&
           !arg->hasAttribute(llvm::Attribute::InReg);
  auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
  if (!inst || !llvm::isa<llvm::BinaryOperator, llvm::CastInst,
                          llvm::SelectInst, llvm::PHINode>(inst)) return false;
  if (seen.size() > 96) return false;
  for (auto *operand : inst->operand_values())
    if (hasVectorInput(operand, seen)) return true;
  return false;
}

// Integer predicates often compare a collection of low-word offsets against
// the same bounds. Keeping all their i64 operands alive costs two VGPRs per
// element. Compare the shared upper words once, and construct each low word
// next to its comparison. The lexicographic decomposition is exact even for
// negative values, index wraparound and inputs larger than 2^32; no truncation
// of a pointer/index range is assumed. Signed predicates use SIGNED high-word
// and UNSIGNED low-word comparisons.
// 把一批 icmp i64 改写成：
//   equal = high(lhs) == high(rhs)                   只算一次
//   different = high(lhs) <pred> high(rhs)            只算一次
//   每条比较: select(equal, low(l) <unsigned pred> low(r), different)
// 这样 i64 操作数不用一直占两个 VGPR。按字典序分解，负数和回绕也成立。
unsigned compactIntegerComparisons(llvm::Function &function) {
  constexpr llvm::StringLiteral comparisonMarker = "frisk.register_comparisons";
  if (function.hasFnAttribute(comparisonMarker)) return 0;
  unsigned changed = 0, materializationID = 0;
  for (llvm::BasicBlock &block : function) {
    // Key：(谓词, lhs 根, rhs 根)，相同的比较分为一组。
    // groups：每组的比较指令。
    // order：保留出现顺序，让输出不受指针地址影响。
    using Key = std::tuple<unsigned, llvm::Value *, llvm::Value *>;
    std::map<Key, llvm::SmallVector<llvm::ICmpInst *>> groups;
    llvm::SmallVector<Key> order;
    for (llvm::Instruction &inst : block) {
      auto *compare = llvm::dyn_cast<llvm::ICmpInst>(&inst);
      if (!compare || !compare->getOperand(0)->getType()->isIntegerTy(64)) continue;
      auto *lhs = upperWordRoot(compare->getOperand(0));
      auto *rhs = upperWordRoot(compare->getOperand(1));
      Key key{compare->getPredicate(), lhs, rhs};
      auto [entry, inserted] = groups.try_emplace(key);
      if (inserted) order.push_back(key);
      entry->second.push_back(compare);
    }
    // Keep emitted names/discriminators independent of Value* allocation order.
    for (auto &key : order) {
      auto &comparisons = groups.at(key);
      // Amortize the high-word predicates. A single comparison or repeated
      // comparisons of identical values do not justify decomposition.
      // variants：组内实际不同的操作数。至少 4 条比较、3 个变体时才有收益。
      llvm::SmallPtrSet<llvm::Value *, 32> variants;
      auto [predicate, lhs, rhs] = key;
      for (auto *compare : comparisons)
        for (auto *operand : compare->operand_values())
          if (operand != lhs && operand != rhs) variants.insert(operand);
      if (comparisons.size() < 4 || variants.size() < 3) continue;
      llvm::SmallPtrSet<llvm::Value *, 32> seen;
      if (!llvm::any_of(variants, [&](llvm::Value *value) {
            seen.clear();
            return hasVectorInput(value, seen);
          })) continue;
      auto pred = static_cast<llvm::CmpInst::Predicate>(predicate);
      llvm::IRBuilder<> b(comparisons.front());
      b.SetCurrentDebugLocation(comparisons.front()->getDebugLoc());
      auto high = [&](llvm::Value *value) {
        return b.CreateTrunc(b.CreateLShr(value, 32), b.getInt32Ty(), "compare.high");
      };
      llvm::Value *highL = high(lhs), *highR = high(rhs);
      llvm::Value *equal = b.CreateICmpEQ(highL, highR, "compare.high.equal");
      llvm::Value *different = nullptr;
      if (!llvm::CmpInst::isEquality(pred))
        different = b.CreateICmp(llvm::CmpInst::getStrictPredicate(pred),
                                highL, highR, "compare.high.order");
      llvm::Value *lowL = b.CreateTrunc(lhs, b.getInt32Ty(), "compare.low.base");
      llvm::Value *lowR = b.CreateTrunc(rhs, b.getInt32Ty(), "compare.low.base");
      for (auto *compare : comparisons) {
        b.SetInsertPoint(compare);
        b.SetCurrentDebugLocation(compare->getDebugLoc());
        // 在比较点按 or/xor 路径重建低 32 位。base 用 materialize 钉住，避免 InstCombine 合回去。
        auto low = [&](llvm::Value *value, llvm::Value *root, llvm::Value *base) {
          llvm::SmallVector<llvm::BinaryOperator *> path;
          while (value != root) {
            auto *binary = llvm::cast<llvm::BinaryOperator>(value);
            path.push_back(binary);
            value = binary->getOperand(0);
          }
          if (path.empty()) return base;
          // Identical memory(none) inline asms can be CSE'd even with
          // sideeffect. The unused immediate distinguishes these boundaries
          // without emitting an instruction or using an extra register.
          llvm::Value *result = materialize(base, compare, "compare.local.base",
                                             ++materializationID);
          for (auto *binary : llvm::reverse(path)) {
            auto *constant = llvm::cast<llvm::ConstantInt>(binary->getOperand(1));
            result = b.CreateBinOp(binary->getOpcode(), result,
                b.getInt32(constant->getValue().getZExtValue()), "compare.local");
          }
          return result;
        };
        auto *left = low(compare->getOperand(0), lhs, lowL);
        auto *right = low(compare->getOperand(1), rhs, lowR);
        auto lowPred = llvm::ICmpInst::getUnsignedPredicate(pred);
        auto *lower = b.CreateICmp(lowPred, left, right, "compare.low");
        llvm::Value *result;
        if (pred == llvm::CmpInst::ICMP_EQ)
          result = b.CreateAnd(equal, lower, "compare.result");
        else if (pred == llvm::CmpInst::ICMP_NE)
          result = b.CreateOr(b.CreateNot(equal), lower, "compare.result");
        else
          result = b.CreateSelect(equal, lower, different, "compare.result");
        compare->replaceAllUsesWith(result);
        compare->eraseFromParent();
        ++changed;
      }
    }
  }
  function.addFnAttr(comparisonMarker, "1");
  if (changed)
    llvm::errs() << "Compact integer comparisons: " << function.getName()
                 << ": " << changed << " shared-high-word predicates\n";
  return changed;
}

// consumers：同一基本块内，由数学结果传递出去的消费者集合。
// ordered：按原始顺序排列的 consumers。
// 每个消费者移到它最晚操作数的后面；遇到 fptrunc 就用 materialize 把两侧钉住。
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

// 把依赖线程号的地址表达式在每个基本块内重建，这样长的地址 DAG 不会跨循环占用寄存器。
//   divergent  值 -> 是否依赖线程号的缓存。
//   local      本基本块内已重建的值，每块清空。
//   retained   不重建、继续共享的昂贵基址（含除法）。
//   clones     克隆的指令数，只用于日志。
class AddressRematerializer {
  llvm::DenseMap<llvm::Value *, bool> divergent;
  llvm::DenseMap<llvm::Value *, llvm::Value *> local;
  llvm::SmallPtrSet<llvm::Value *, 8> retained;
  unsigned &clones;
  // Bound code growth for arbitrary generated expressions. The limit applies
  // to newly cloned nodes, sharing expressions already rebuilt in this block.
  // 重建新增的节点不超过 96 个。已经重建或保留的值不计入。
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
  // 递归克隆到 before 之前。retained 和线程号本身用 materialize 钉在本块。
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
  // 选择保留的基址：入口块里被循环内地址用到的除法/取模。
  // savings[expr] 按循环深度加权：2^depth * 位宽。按收益排序，最多保留 8 个寄存器字。
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
  // 先记下要处理的访问，再逐个重建地址操作数。包括 GM/LDS 的 load/store、DMA 源地址、readfirstlane
  // 的输入和矩阵读指针。只重建地址，不移动访问本身。
  void run(llvm::BasicBlock &block) {
    local.clear();
    // Snapshot accesses before inserting instructions. Per-block caching
    // shares nearby addresses, but never carries their expanded DAGs through
    // another block/loop. Uniform base pointers and loop PHIs remain shared.
    // accesses：(指令, 地址操作数下标)。
    llvm::SmallVector<std::pair<llvm::Instruction *, unsigned>> accesses;
    for (llvm::Instruction &inst : block) {
      if (llvm::isa<llvm::LoadInst, llvm::StoreInst>(inst)) {
        unsigned operand = llvm::isa<llvm::LoadInst>(inst) ? 0 : 1;
        auto *type = llvm::cast<llvm::PointerType>(inst.getOperand(operand)->getType());
        if (type->getAddressSpace() == 1 || type->getAddressSpace() == 3)
          accesses.push_back({&inst, operand});
      } else if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        // DMA source addresses are memory operands too. Its LDS wave base is
        // passed through readfirstlane: rebuild that call's integer input in
        // place, never clone/move the convergent call or the DMA itself.
        if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(call))
          if (intrinsic->getIntrinsicID() == llvm::Intrinsic::amdgcn_global_load_async_lds ||
              intrinsic->getIntrinsicID() == llvm::Intrinsic::amdgcn_readfirstlane)
            accesses.push_back({&inst, 0});
        // HCU matrix reads have no upstream intrinsic ID, even though their
        // reserved LLVM names can still satisfy isa<IntrinsicInst>. Match the
        // ABI independently. Only rebuild the pointer, never the collective.
        auto *callee = call->getCalledFunction();
        if (callee && call->arg_size() == 2 &&
            (callee->getName() == detail::MatrixReadM32x16 ||
             callee->getName() == detail::MatrixReadM32x8) &&
            call->getArgOperand(0)->getType()->isPointerTy() &&
            call->getArgOperand(0)->getType()->getPointerAddressSpace() == 3)
          accesses.push_back({&inst, 0});
      }
    }
    for (auto [inst, operand] : accesses) {
      if (!affordable(inst->getOperand(operand))) continue;
      inst->setOperand(operand, rebuild(inst->getOperand(operand), inst));
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
    if (function.isDeclaration() ||
        function.hasFnAttribute(llvm::Attribute::StrictFP) ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone) || !isAMDGPU(function)) continue;
    if (!function.hasFnAttribute(marker)) {
      exposeThreadBounds(function);
      unsigned simplified = simplifyThreadExpressions(function);
      llvm::errs() << "Compact thread indices: " << simplified << " proven replacements\n";
      pipeline.run(function, analyses);
    }
    if (compactIntegerComparisons(function)) {
      analyses.invalidate(function, llvm::PreservedAnalyses::none());
      llvm::DCEPass().run(function, analyses);
    }
  }
}
} // namespace mlir::frisk
