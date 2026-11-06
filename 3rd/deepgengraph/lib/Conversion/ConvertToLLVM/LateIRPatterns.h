// ===========================================================================
// LLVM 阶段各 Pass 共用的 IR 形状定义和识别函数。
//
// 这些 Pass 靠 inline asm 文本和函数名识别 DMA、wait、矩阵读等指令。文本写错或改动时，
// 它们通常只会放弃优化、不会报错，所以生成方（TranslateToLLVM.cpp、NarrowKDMA.cpp）和
// 识别方必须使用这里的同一份定义。
//
// 使用方：DoubleBufferV、NarrowKDMA、PrefetchNextK、InterleavePV、PrefetchPairedLDS、
// ScheduleSoftmaxFragments、DeferLaneSum 等。
// ===========================================================================
#ifndef DEEPGENGRAPH_LATEIRPATTERNS_H
#define DEEPGENGRAPH_LATEIRPATTERNS_H
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <optional>
#include <string>

namespace mlir::frisk::detail {
using namespace llvm;
// 同时包含 MLIR 头文件的翻译单元里，这些名字会先找到 mlir:: 下的同名类型。
using llvm::FixedVectorType;
using llvm::Type;
using llvm::Value;
using Inst = Instruction;
using Call = CallInst;

// ---- 文本定义 -------------------------------------------------------------
// GM -> LDS 的 DMA。m0 是 wave 级 LDS 基址，每个 lane 写 m0 + lane * 包大小。
inline constexpr StringLiteral DMA128Asm =
    "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds";
inline constexpr StringLiteral DMA64Asm =
    "s_mov_b32 $0, $1\n\tbuffer_load_dwordx2 $2, $3, 0 offen lds";
inline constexpr StringLiteral DMAConstraints = "={m0},s,v,s,~{memory}";
// 等待本 wave 发出的 DMA 全部完成。
inline constexpr StringLiteral WaitAsm = "s_waitcnt vmcnt(0)";
inline constexpr StringLiteral WaitConstraints = "~{memory}";
// 矩阵读：一次读一个包，按指令语义分给多个寄存器。
inline constexpr StringLiteral MatrixReadM32x16 = "llvm.hcu.ds.read.m32x16.b16.v4i32.p3";
inline constexpr StringLiteral MatrixReadM32x8 = "llvm.hcu.ds.read.m32x8.b32.v4i32.p3";
inline constexpr StringLiteral HCUGlobalLoad128 = "llvm.hcu.global.load.dwordx4.lds";
// OptimizeRegisterLifetimes 给恒等空 asm 打的元数据，也是它的函数属性名。
inline constexpr StringLiteral RegisterLifetimesMarker = "frisk.register_lifetimes";

// ---- 单条指令的识别 --------------------------------------------------------
// i 是否对声明函数 name 的普通调用。i 可以为空。
inline bool isCallTo(Inst *i, StringRef name) {
  auto *c = dyn_cast_or_null<Call>(i);
  return c && !c->hasOperandBundles() && !c->isMustTailCall() &&
      c->getCalledFunction() && c->getCalledFunction()->isDeclaration() &&
      c->getCalledFunction()->getName() == name;
}

// i 调用 inline asm 时返回它，否则返回 nullptr。
inline InlineAsm *getInlineAsm(Inst *i) {
  auto *c = dyn_cast<Call>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall()) return nullptr;
  auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
  return a && !a->canThrow() && !a->isAlignStack() ? a : nullptr;
}

// 只比较文本，供只需要确认“这是 DMA / wait”的调用方使用。
inline bool isDMA128Asm(InlineAsm *a) {
  return a->getAsmString() == DMA128Asm && a->getConstraintString() == DMAConstraints;
}
inline bool isWaitAsm(InlineAsm *a) {
  return a->getAsmString() == WaitAsm && a->getConstraintString() == WaitConstraints;
}

// DMA 每个 lane 搬运的字节数：x2 为 8，x4 为 16，不是 DMA 返回 0。同时校验操作数类型。
inline unsigned dmaBytesPerLane(Inst *i) {
  auto *a = getInlineAsm(i);
  if (!a || !a->hasSideEffects() || a->getConstraintString() != DMAConstraints) return 0;
  auto *c = cast<Call>(i);
  if (c->arg_size() != 3 || !c->use_empty() || !c->getType()->isIntegerTy(32) ||
      !c->getArgOperand(0)->getType()->isIntegerTy(32) ||
      !c->getArgOperand(1)->getType()->isIntegerTy(32) ||
      c->getArgOperand(2)->getType() != FixedVectorType::get(Type::getInt32Ty(c->getContext()), 4))
    return 0;
  auto text = a->getAsmString();
  if (text == DMA64Asm) return 8;
  if (text == DMA128Asm) return 16;
  return 0;
}
inline bool isDMA128(Inst *i) { return dmaBytesPerLane(i) == 16; }

inline bool isWaitCall(Inst *i) {
  auto *a = getInlineAsm(i);
  return a && a->hasSideEffects() && isWaitAsm(a) && cast<Call>(i)->arg_empty();
}

// 无副作用的寄存器 MMAC（inline asm 形式）。
inline bool isRegisterMMACall(Inst *i) {
  auto *a = getInlineAsm(i);
  return a && !a->hasSideEffects() && cast<Call>(i)->doesNotAccessMemory() &&
      isRegisterMMA(a->getAsmString(), a->getConstraintString());
}

// 同上，并校验 A/B 是 <4 x half>、C 是 <4 x float>。i 可以为空。
inline bool isTypedRegisterMMACall(Inst *i) {
  if (!i || !isRegisterMMACall(i)) return false;
  auto *c = cast<Call>(i);
  auto &ctx = c->getContext();
  auto *ab = FixedVectorType::get(Type::getHalfTy(ctx), 4);
  auto *cd = FixedVectorType::get(Type::getFloatTy(ctx), 4);
  return c->arg_size() == 3 && c->getType() == cd && c->getArgOperand(0)->getType() == ab &&
      c->getArgOperand(1)->getType() == ab && c->getArgOperand(2)->getType() == cd;
}

// sched.barrier(0)：只约束编译器调度器，运行时没有代价。
inline bool isSchedHint(Inst *i) {
  if (!isCallTo(i, "llvm.amdgcn.sched.barrier")) return false;
  auto *c = cast<Call>(i);
  auto *v = c->arg_size() == 1 ? dyn_cast<ConstantInt>(c->getArgOperand(0)) : nullptr;
  return v && v->isZero();
}

// 可以安全地克隆或移动的纯计算。
inline bool isPureInst(Inst *i) {
  return !isa<CallBase, PHINode, AllocaInst>(i) && !i->isTerminator() &&
      !i->mayReadOrWriteMemory() && !i->mayHaveSideEffects() && isSafeToSpeculativelyExecute(i);
}

// 值是否对同一 workgroup 的所有线程相同。只接受常量、kernel 参数、workgroup id 和它们的纯运算。
inline bool isWorkgroupUniform(Value *v, SmallPtrSetImpl<Value *> &seen) {
  if (isa<Constant, Argument>(v)) return true;
  if (!seen.insert(v).second) return true;
  auto *i = dyn_cast<Inst>(v);
  if (!i) return false;
  if (isa<Call>(i))
    return isCallTo(i, "llvm.amdgcn.workgroup.id.x") || isCallTo(i, "llvm.amdgcn.workgroup.id.y") ||
           isCallTo(i, "llvm.amdgcn.workgroup.id.z");
  if (!isa<PHINode>(i) && !isPureInst(i)) return false;
  return llvm::all_of(i->operands(), [&](Value *op) { return isWorkgroupUniform(op, seen); });
}

inline bool isWorkgroupFence(Inst *i, AtomicOrdering order) {
  auto *f = dyn_cast_or_null<FenceInst>(i);
  return f && f->getOrdering() == order &&
      f->getSyncScopeID() == f->getContext().getOrInsertSyncScopeID("workgroup");
}

// 一个完整的同步序列：fence release + s_barrier + fence acquire，前面可能带 s_waitcnt。
// wait 非空表示“数据就绪”barrier，为空表示“读者交接”barrier。
struct Sync {
  Inst *release = nullptr, *barrier = nullptr, *acquire = nullptr, *wait = nullptr;
};

// 以 i 为 s_barrier，尝试匹配上面的序列。
inline std::optional<Sync> matchSync(Inst *i) {
  if (!isCallTo(i, "llvm.amdgcn.s.barrier") || !cast<Call>(i)->arg_empty() ||
      !isWorkgroupFence(i->getPrevNode(), AtomicOrdering::Release) ||
      !isWorkgroupFence(i->getNextNode(), AtomicOrdering::Acquire)) return {};
  Sync s{i->getPrevNode(), i, i->getNextNode(), nullptr};
  if (s.release->getPrevNode() && isWaitCall(s.release->getPrevNode()))
    s.wait = s.release->getPrevNode();
  return s;
}

// ---- 函数级前置检查 --------------------------------------------------------
// doubleBufferV、narrowKDMA、prefetchNextK 都按字节证明 LDS 的读写范围，需要相同的前提：
// 固定的 1D workgroup、wave64、唯一的静态 LDS 对象、所有分支对 workgroup uniform。
struct Wave64Kernel {
  unsigned threads = 0;           // workgroup 的线程数，是 64 的倍数。
  GlobalVariable *storage = nullptr;  // 唯一的 LDS 全局变量。
  uint64_t ldsBytes = 0;          // 这个对象的字节数。
};

// 不满足前提时返回空，并把原因写入 why。
inline std::optional<Wave64Kernel> analyzeWave64Kernel(Function &f, std::string &why) {
  auto fail = [&](StringRef reason) -> std::optional<Wave64Kernel> {
    why = reason.str();
    return std::nullopt;
  };
  auto *dims = f.getMetadata("reqd_work_group_size");
  if (!dims || dims->getNumOperands() != 3) return fail("requires fixed 1D workgroup size");
  Wave64Kernel kernel;
  for (unsigned d = 0; d < 3; ++d) {
    auto *size = mdconst::dyn_extract<ConstantInt>(dims->getOperand(d));
    if (!size || size->getValue().getActiveBits() > 32 || (d && !size->isOne()))
      return fail("requires fixed 1D workgroup size");
    if (!d) kernel.threads = size->getZExtValue();
  }
  if (!kernel.threads || kernel.threads > 1024 || kernel.threads % 64 ||
      f.getFnAttribute("target-features").getValueAsString().contains("+wavefrontsize32"))
    return fail("requires wave64 workgroups");
  for (auto &g : f.getParent()->globals())
    if (g.getAddressSpace() == 3) {
      if (kernel.storage || !g.hasInitializer()) return fail("requires one static LDS allocation");
      kernel.storage = &g;
    }
  if (!kernel.storage) return fail("missing LDS allocation");
  auto bytes = f.getParent()->getDataLayout().getTypeAllocSize(kernel.storage->getValueType());
  if (bytes.isScalable() || bytes.getFixedValue() > 65536)
    return fail("unsupported LDS allocation size");
  kernel.ldsBytes = bytes.getFixedValue();
  for (auto &block : f) {
    if (auto *branch = dyn_cast<CondBrInst>(block.getTerminator())) {
      SmallPtrSet<Value *, 32> seen;
      if (!isWorkgroupUniform(branch->getCondition(), seen))
        return fail("requires workgroup-uniform control flow");
    } else if (!isa<UncondBrInst, ReturnInst>(block.getTerminator())) {
      return fail("unsupported control flow");
    }
  }
  return kernel;
}
} // namespace mlir::frisk::detail
#endif
