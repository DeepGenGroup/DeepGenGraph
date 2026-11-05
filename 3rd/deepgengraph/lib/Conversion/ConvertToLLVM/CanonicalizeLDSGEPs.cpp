// ===========================================================================
// Pass: canonicalizeLDSGEPs（CanonicalizeLDSGEPsPass）   优化汇总.py 第 24 项，LLVM 阶段，开启 swizzle 后自动打开
//
// 作用：
//   把地址差恒定的 LDS load 改写成“公共基址 + 常量 GEP”，并提升 load 的 align。后端看到这个形式
//   才可能把两个 b64 读合并成 ds_read2_b64。本 Pass 只暴露这个机会，不强制合并。
//
// 证明差值恒定的两种手段（SCEV 优先）：
//   1. ScalarEvolution：适合线性表达式。
//   2. ThreadOffsets：枚举所有线程号精确求值。适合 SCEV 无法理解的 XOR / 位段，如 swizzle。
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/CanonicalizeLDSGEPs.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/Support/Alignment.h"
#include <map>
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Utils/Local.h"

namespace mlir::frisk {
namespace {
// Exact, bounded proof for nonlinear layouts. No data values, loop iterations,
// wave scheduling or sampled threads enter the proof. Unknown inputs fall back
// to SCEV. This complements SCEV, which treats XOR/bitfield extraction as opaque.
// 枚举线程号求 LDS 地址。
//   Table     表达式在每个线程号上的取值。
//   Pointer   指针 = base（LDS 全局变量）+ 每个线程的偏移 offsets。
//   count     线程数，来自 workitem.id.x 的范围；0 表示不可用，此时所有查询都失败。
//   integers / pointers  已求值的缓存。
//   unknown   正在求值或已失败的值，用来防环。
class ThreadOffsets {
  using Table = llvm::SmallVector<llvm::APInt>;
  struct Pointer { llvm::GlobalVariable *base; Table offsets; };
  llvm::Function &function;
  unsigned count = 0;
  std::map<llvm::Value *, Table> integers;
  std::map<llvm::Value *, Pointer> pointers;
  llvm::SmallPtrSet<llvm::Value *, 32> unknown;
  bool threadID(llvm::Value *v) {
    auto *id = llvm::dyn_cast<llvm::IntrinsicInst>(v);
    return id && id->getIntrinsicID() == llvm::Intrinsic::amdgcn_workitem_id_x;
  }
  const Table *integer(llvm::Value *v, unsigned depth = 0) {
    if (!count || depth > 128 || !v->getType()->isIntegerTy() ||
        v->getType()->getIntegerBitWidth() > 64) return nullptr;
    if (auto it = integers.find(v); it != integers.end()) return &it->second;
    if (!unknown.insert(v).second) return nullptr;
    Table values;
    if (auto *constant = llvm::dyn_cast<llvm::ConstantInt>(v))
      values.assign(count, constant->getValue());
    else if (threadID(v))
      for (unsigned t = 0; t < count; ++t) values.emplace_back(32, t);
    else if (auto *call = llvm::dyn_cast<llvm::CallInst>(v)) {
      // The register lifetime pass's exact tied empty asm is an identity.
      // Require its marker, spelling and constraint; keep the call itself.
      // OptimizeRegisterLifetimes 的 materialize 生成的空 asm 是恒等操作，可以穿透。
      auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand());
      if (!call->getMetadata("frisk.register_lifetimes") || !assembly ||
          !assembly->getAsmString().empty() || assembly->getConstraintString() != "=v,0" ||
          call->arg_size() != 1 || call->getArgOperand(0)->getType() != call->getType()) return nullptr;
      auto *input = integer(call->getArgOperand(0), depth + 1);
      if (!input) return nullptr;
      values = *input;
    } else {
      auto *inst = llvm::dyn_cast<llvm::Instruction>(v);
      if (!inst || !llvm::isa<llvm::BinaryOperator, llvm::CastInst,
                              llvm::ICmpInst, llvm::SelectInst>(inst)) return nullptr;
      llvm::SmallVector<const Table *> inputs;
      for (auto *operand : inst->operand_values()) {
        auto *input = integer(operand, depth + 1);
        if (!input) return nullptr;
        inputs.push_back(input);
      }
      for (unsigned t = 0; t < count; ++t) {
        llvm::SmallVector<llvm::Constant *> operands;
        for (auto *input : inputs)
          operands.push_back(llvm::ConstantInt::get(function.getContext(), (*input)[t]));
        auto *folded = llvm::dyn_cast_or_null<llvm::ConstantInt>(
            llvm::ConstantFoldInstOperands(inst, operands, function.getDataLayout()));
        if (!folded) return nullptr; // Includes poison, undef and unknown folds.
        values.push_back(folded->getValue());
      }
    }
    return &integers.emplace(v, std::move(values)).first->second;
  }
  const Pointer *pointer(llvm::Value *v, unsigned depth = 0) {
    if (!count || depth > 128) return nullptr;
    if (auto it = pointers.find(v); it != pointers.end()) return &it->second;
    const auto &dl = function.getDataLayout();
    unsigned width = dl.getIndexTypeSizeInBits(v->getType());
    if (width > 64) return nullptr;
    Pointer result;
    if (auto *global = llvm::dyn_cast<llvm::GlobalVariable>(v)) {
      result.base = global;
      result.offsets.assign(count, llvm::APInt(width, 0));
    } else if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(v)) {
      auto *base = pointer(gep->getPointerOperand(), depth + 1);
      if (!base) return nullptr;
      result.base = base->base;
      for (auto &index : gep->indices())
        if (!integer(index, depth + 1)) return nullptr;
      for (unsigned t = 0; t < count; ++t) {
        llvm::APInt delta(width, 0);
        if (!gep->accumulateConstantOffset(dl, delta, [&](llvm::Value &index, llvm::APInt &out) {
              out = (*integer(&index))[t].sextOrTrunc(width); return true;
            })) return nullptr;
        result.offsets.push_back(base->offsets[t] + delta);
      }
    } else return nullptr;
    return &pointers.emplace(v, std::move(result)).first->second;
  }
public:
  explicit ThreadOffsets(llvm::Function &f) : function(f) {
    for (auto &inst : llvm::instructions(f)) {
      if (!threadID(&inst)) continue;
      auto range = llvm::computeConstantRange(&inst, false,
          llvm::SimplifyQuery(f.getDataLayout()));
      if (range.getUnsignedMin().isZero() && !range.isFullSet() &&
          range.getUnsignedMax().ult(1024))
        count = std::max(count, unsigned(range.getUnsignedMax().getZExtValue() + 1));
      else { count = 0; return; }
    }
  }
  // 如果 a - b 对所有线程相同，返回这个差值。
  // alignment：所有线程偏移的公共对齐，再和全局变量的对齐取最小。
  std::optional<llvm::APInt> difference(llvm::Value *a, llvm::Value *b) {
    auto *pa = pointer(a), *pb = pointer(b);
    if (!pa || !pb || pa->base != pb->base) return std::nullopt;
    auto diff = pa->offsets[0] - pb->offsets[0];
    for (unsigned t = 1; t < count; ++t)
      if (pa->offsets[t] - pb->offsets[t] != diff) return std::nullopt;
    return diff;
  }
  llvm::Align alignment(llvm::Value *v) {
    auto *p = pointer(v);
    if (!p) return llvm::Align(1);
    auto align = p->base->getAlign().valueOrOne();
    for (auto offset : p->offsets)
      align = llvm::commonAlignment(align, offset.getZExtValue());
    return align;
  }
};
}
llvm::PreservedAnalyses CanonicalizeLDSGEPsPass::run(
    llvm::Function &f, llvm::FunctionAnalysisManager &am) {
  auto &se = am.getResult<llvm::ScalarEvolutionAnalysis>(f);
  auto &dt = am.getResult<llvm::DominatorTreeAnalysis>(f);
  auto &ac = am.getResult<llvm::AssumptionAnalysis>(f);
  ThreadOffsets exact(f);
  bool changedAlignment = false;
  // Anchor：可作为基址的指针，最多 128 个。
  // Rewrite：待改写的 load，以及它相对于 base 的 offset。
  // 先做完分析再改 IR，避免一边改 IR 一边使用 SCEV 结果。
  struct Anchor { llvm::Value *pointer; const llvm::SCEV *expr; };
  struct Rewrite { llvm::LoadInst *load; llvm::Value *base; llvm::APInt offset; };
  llvm::SmallVector<Anchor> anchors;
  llvm::SmallVector<Rewrite> rewrites;
  // Finish analysis before modifying IR. Keep the search bounded for kernels
  // with many unrelated shared allocations or address expressions.
  for (auto &inst : llvm::instructions(f)) {
    auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst);
    if (!load || load->getPointerAddressSpace() != 3 || load->isVolatile() ||
        load->isAtomic()) continue;
    auto *pointer = load->getPointerOperand();
    // Also handle arbitrary LDS pointer arguments and dynamic offsets without
    // a bounded workitem domain. LLVM proves alignment from pointer contracts,
    // assumptions and known bits; enumeration additionally covers nonlinear
    // thread layouts. Neither proof depends on a buffer name or attention role.
    // ds_read2 需要 load 的 align 足够大，这里按证明的结果提升。
    auto provenAlignment = std::max(exact.alignment(pointer),
        llvm::getKnownAlignment(pointer, f.getDataLayout(), load, &ac, &dt));
    if (provenAlignment > load->getAlign()) {
      load->setAlignment(provenAlignment);
      changedAlignment = true;
    }
    const auto *expr = se.getSCEV(pointer);
    bool found = false;
    // 与已有 anchor 比较。必须支配当前 load 才能作为基址。
    // 已经是“anchor + 同一常量”形式时跳过，保证重复运行结果稳定。
    // 找不到合适的 anchor 时，当前指针成为新的 anchor。
    for (const auto &anchor : anchors) {
      if (pointer == anchor.pointer) { found = true; break; }
      auto *definition = llvm::dyn_cast<llvm::Instruction>(anchor.pointer);
      if (definition && !dt.dominates(definition, load)) continue;
      auto *difference = llvm::dyn_cast<llvm::SCEVConstant>(
          se.getMinusSCEV(expr, anchor.expr));
      auto offset = difference ? std::optional<llvm::APInt>(difference->getAPInt())
                               : exact.difference(pointer, anchor.pointer);
      if (!offset) continue;
      if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer);
          gep && gep->getSourceElementType()->isIntegerTy(8) &&
          gep->getPointerOperand() == anchor.pointer && gep->getNumIndices() == 1) {
        if (auto *index = llvm::dyn_cast<llvm::ConstantInt>(gep->idx_begin()->get());
            index && index->getValue().sextOrTrunc(offset->getBitWidth()) == *offset) {
          found = true; break;
        }
      }
      rewrites.push_back({load, anchor.pointer, *offset});
      found = true; break;
    }
    if (!found && anchors.size() < 128) anchors.push_back({pointer, expr});
  }
  for (auto &rewrite : rewrites) {
    llvm::IRBuilder<> b(rewrite.load);
    // No inbounds promise: the proof is modular pointer arithmetic. Do not
    // speculate a load to make a pair possible. Alignment above is proven
    // over the full bounded domain, not inferred from a desired instruction.
    auto *gep = b.CreateGEP(b.getInt8Ty(), rewrite.base,
        llvm::ConstantInt::get(f.getContext(), rewrite.offset), "lds.const.gep");
    rewrite.load->setOperand(0, gep);
  }
  if (rewrites.empty() && !changedAlignment) return llvm::PreservedAnalyses::all();
  llvm::errs() << "CanonicalizeLDSGEPs: " << f.getName() << ": "
               << rewrites.size() << " proven constant offsets\n";
  auto result = llvm::PreservedAnalyses::none();
  result.preserveSet<llvm::CFGAnalyses>();
  return result;
}
void canonicalizeLDSGEPs(llvm::Module &module) {
  llvm::FunctionAnalysisManager functions;
  llvm::ModuleAnalysisManager modules;
  llvm::LoopAnalysisManager loops;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(modules);
  pb.registerLoopAnalyses(loops);
  pb.registerCGSCCAnalyses(cgscc);
  pb.registerFunctionAnalyses(functions);
  pb.crossRegisterProxies(loops, functions, cgscc, modules);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(CanonicalizeLDSGEPsPass());
  pipeline.addPass(llvm::DCEPass());
  for (auto &f : module)
    if (!f.isDeclaration()) pipeline.run(f, functions);
}
}
