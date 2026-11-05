#ifndef DEEPGENGRAPH_LDSADDRESSANALYSIS_H
#define DEEPGENGRAPH_LDSADDRESSANALYSIS_H
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include <map>
#include <optional>
#include <tuple>
namespace mlir::frisk::detail {
using namespace llvm;
using Inst = Instruction;
using Call = CallInst;
inline bool addressNamed(Inst *i, StringRef name) {
  auto *c = dyn_cast_or_null<Call>(i);
  return c && !c->hasOperandBundles() && !c->isMustTailCall() &&
      c->getCalledFunction() && c->getCalledFunction()->isDeclaration() &&
      c->getCalledFunction()->getName() == name;
}
inline InlineAsm *addressAssembly(Inst *i) {
  auto *c = dyn_cast<Call>(i);
  if (!c || c->hasOperandBundles() || c->isMustTailCall()) return nullptr;
  auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
  return a && !a->canThrow() && !a->isAlignStack() ? a : nullptr;
}
// LDSAddress：给定线程号和向量元素下标，把地址表达式精确求值成 LDS 内的字节偏移。
// doubleBufferV 和 prefetchNextK 靠它证明读写的字节范围。
// 注意：NarrowKDMA.cpp 里有一份几乎相同的副本。
//   storage  唯一的 LDS 全局变量，相对地址 0。
//   cache    (值, 线程号, 元素) -> 结果。先写入空值，所以环会判定失败。
// Exact evaluation of the restricted integer/thread-address DAG. Unknown
// values fail closed. A single LDS allocation is assigned relative base zero;
// actual rewrites add a byte displacement to the original pointer/base.
// This naturally handles arbitrary tile sizes, lane permutations and swizzles.
class LDSAddress {
  const DataLayout &dl;
  GlobalVariable *storage;
  std::map<std::tuple<Value *, unsigned, unsigned>, std::optional<uint64_t>> cache;
  // 参数 lane 实际是整个 workgroup 内的线程号，不是 wave 内 lane。
  std::optional<uint64_t> compute(Value *v, unsigned lane, unsigned element, unsigned depth) {
    if (depth > 128) return {};
    if (v == storage) return 0;
    if (auto *c = dyn_cast<ConstantInt>(v)) return c->getValue().getLimitedValue();
    if (auto *c = dyn_cast<Constant>(v)) {
      if (v->getType()->isVectorTy() && !isa<ConstantExpr>(v)) {
        auto *e = c->getAggregateElement(element);
        return e ? get(e, lane, 0, depth+1) : std::nullopt;
      }
    }
    if (auto *g = dyn_cast<GEPOperator>(v)) {
      auto base = get(g->getPointerOperand(), lane, element, depth+1);
      if (!base) return {};
      uint64_t offset = *base;
      for (auto it = gep_type_begin(g), end = gep_type_end(g); it != end; ++it) {
        auto ix = get(it.getOperand(), lane, element, depth+1);
        if (!ix) return {};
        if (auto *st = it.getStructTypeOrNull()) {
          if (*ix >= st->getNumElements()) return {};
          offset += dl.getStructLayout(st)->getElementOffset(*ix);
        } else {
          auto size = dl.getTypeAllocSize(it.getIndexedType());
          if (size.isScalable()) return {};
          unsigned bits = it.getOperand()->getType()->getScalarSizeInBits();
          if (!bits || bits > 64) return {};
          offset += APInt(bits, *ix).getSExtValue() * size.getFixedValue();
        }
      }
      return offset;
    }
    if (auto *e = dyn_cast<ExtractElementInst>(v)) {
      auto ix = get(e->getIndexOperand(), lane, 0, depth+1);
      if (!ix || *ix >= cast<FixedVectorType>(e->getVectorOperandType())->getNumElements()) return {};
      return get(e->getVectorOperand(), lane, *ix, depth+1);
    }
    if (auto *e = dyn_cast<InsertElementInst>(v)) {
      auto ix = get(e->getOperand(2), lane, 0, depth+1);
      if (!ix) return {};
      return get(e->getOperand(*ix == element ? 1 : 0), lane, *ix == element ? 0 : element, depth+1);
    }
    if (auto *s = dyn_cast<ShuffleVectorInst>(v)) {
      if (element >= s->getShuffleMask().size()) return {};
      int ix = s->getMaskValue(element);
      unsigned n = cast<FixedVectorType>(s->getOperand(0)->getType())->getNumElements();
      if (ix < 0) return {};
      return get(s->getOperand(ix >= int(n)), lane, ix % n, depth+1);
    }
    auto *op = dyn_cast<Operator>(v);
    if (op && Instruction::isCast(op->getOpcode())) {
      auto a = get(op->getOperand(0), lane, element, depth+1);
      if (!a) return {};
      unsigned bits = op->getOperand(0)->getType()->getScalarSizeInBits();
      switch (op->getOpcode()) {
      case Instruction::SExt: return bits && bits <= 64 ? std::optional<uint64_t>(APInt(bits,*a).getSExtValue()) : std::nullopt;
      case Instruction::ZExt: case Instruction::Trunc:
      case Instruction::PtrToInt: case Instruction::IntToPtr: return a;
      case Instruction::BitCast:
        if (dl.getTypeSizeInBits(op->getOperand(0)->getType()->getScalarType()) ==
            dl.getTypeSizeInBits(v->getType()->getScalarType())) return a;
        return {};
      default: return {};
      }
    }
    if (op && Instruction::isBinaryOp(op->getOpcode())) {
      auto a = get(op->getOperand(0), lane, element, depth+1);
      auto b = get(op->getOperand(1), lane, element, depth+1);
      unsigned bits = v->getType()->getScalarSizeInBits();
      if (!a || !b || !bits || bits > 64) return {};
      switch (op->getOpcode()) {
      case Instruction::Add: return *a + *b;
      case Instruction::Sub: return *a - *b;
      case Instruction::Mul: return *a * *b;
      case Instruction::And: return *a & *b;
      case Instruction::Or: return *a | *b;
      case Instruction::Xor: return *a ^ *b;
      case Instruction::Shl: return *b < bits ? std::optional<uint64_t>(*a << *b) : std::nullopt;
      case Instruction::LShr: return *b < bits ? std::optional<uint64_t>(*a >> *b) : std::nullopt;
      case Instruction::AShr: return *b < bits ? std::optional<uint64_t>(APInt(bits,*a).ashr(*b).getZExtValue()) : std::nullopt;
      case Instruction::UDiv: return *b ? std::optional<uint64_t>(*a / *b) : std::nullopt;
      case Instruction::URem: return *b ? std::optional<uint64_t>(*a % *b) : std::nullopt;
      case Instruction::SDiv: case Instruction::SRem: {
        APInt left(bits,*a), right(bits,*b);
        if (right.isZero() || (left.isMinSignedValue() && right.isAllOnes())) return {};
        return (op->getOpcode() == Instruction::SDiv ? left.sdiv(right) : left.srem(right)).getZExtValue();
      }
      default: return {};
      }
    }
    auto *c = dyn_cast<Call>(v);
    if (!c) return {};
    if (addressNamed(c, "llvm.amdgcn.workitem.id.x") && c->arg_empty()) return lane;
    if ((addressNamed(c, "llvm.amdgcn.workitem.id.y") || addressNamed(c, "llvm.amdgcn.workitem.id.z")) && c->arg_empty()) return 0;
    // readfirstlane 取的是本 wave 第一个 lane 的值，所以用线程号 lane/64*64 求值。
    // 这里写死了 wave64。
    if (addressNamed(c, "llvm.amdgcn.readfirstlane.i32") && c->arg_size() == 1)
      return get(c->getArgOperand(0), lane/64*64, 0, depth+1);
    auto *a = addressAssembly(c);
    if (a && (a->getConstraintString() == "=v,0" || a->getConstraintString() == "=v,0,i") &&
        a->getAsmString().empty() && c->arg_size() >= 1)
      return get(c->getArgOperand(0), lane, element, depth+1);
    return {};
  }
public:
  LDSAddress(const DataLayout &dl, GlobalVariable *storage) : dl(dl), storage(storage) {}
  std::optional<uint64_t> get(Value *v, unsigned lane, unsigned element = 0, unsigned depth = 0) {
    if (isa<ScalableVectorType>(v->getType()) || v->getType()->getScalarSizeInBits() > 64) return {};
    auto key = std::make_tuple(v,lane,element);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    cache[key] = {}; // Cyclic PHIs/unknown address loops fail closed.
    auto result = compute(v,lane,element,depth);
    unsigned bits = v->getType()->getScalarSizeInBits();
    if (result && bits && bits < 64) *result &= (uint64_t(1)<<bits)-1;
    cache[key] = result;
    return result;
  }
};

} // namespace mlir::frisk::detail
#endif
