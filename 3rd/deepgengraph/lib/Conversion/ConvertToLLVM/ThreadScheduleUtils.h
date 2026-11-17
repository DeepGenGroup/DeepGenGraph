// Shared legality and lowering contracts for late ThreadIR scheduling.
#ifndef DEEPGENGRAPH_THREAD_SCHEDULE_UTILS_H
#define DEEPGENGRAPH_THREAD_SCHEDULE_UTILS_H
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::frisk::thread_schedule {
using Ops = SmallVector<Operation *, 64>;
constexpr StringLiteral Barrier = "llvm.amdgcn.sched.barrier";

inline bool mma(Operation *op) {
  if (!op || op->getNumOperands() != 3 || op->getNumResults() != 1) return false;
  auto *ctx = op->getContext();
  auto ab = VectorType::get({4}, Float16Type::get(ctx));
  auto cd = VectorType::get({4}, Float32Type::get(ctx));
  if (op->getOperand(0).getType() != ab || op->getOperand(1).getType() != ab ||
      op->getOperand(2).getType() != cd || op->getResult(0).getType() != cd) return false;
  if (isa<WarpMmaRROp>(op)) {
    auto name = op->getAttrOfType<StringAttr>("inst_name");
    auto constraints = op->getAttrOfType<StringAttr>("inst_constraints");
    return name && constraints && isRegisterMMA(name, constraints);
  }
  auto assembly = dyn_cast<LLVM::InlineAsmOp>(op);
  return assembly && !assembly.getIsAlignStack() && !assembly.getOperandAttrs() &&
         isRegisterMMA(assembly.getAsmString(), assembly.getConstraints());
}

inline bool protectedFunction(func::FuncOp f) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none"})
    if (f->hasAttr(name)) return true;
  for (StringRef key : {"passthrough", "llvm.passthrough"})
    if (auto attrs = f->getAttrOfType<ArrayAttr>(key))
      for (Attribute attr : attrs) {
        if (auto pair = dyn_cast<ArrayAttr>(attr)) if (!pair.empty()) attr = pair[0];
        if (auto name = dyn_cast<StringAttr>(attr))
          if (name.getValue() == "strictfp" || name.getValue() == "optnone") return true;
      }
  return false;
}

inline bool pure(Operation *op) {
  return !op->getNumRegions() && !op->hasTrait<OpTrait::IsTerminator>() &&
         isMemoryEffectFree(op) && isSpeculatable(op);
}
// A planned next-iteration launch stays fixed while register-only softmax
// computations are scheduled around it. Check the region, not just its marker.
inline bool crossIterationIssue(Operation *op) {
  if (!isa<scf::IfOp>(op) || !op->hasAttr("frisk.cross_bn_k0")) return false;
  return !op->walk([&](Operation *nested) {
    if (nested == op || isa<scf::YieldOp, AsyncCopyGmToShmOp>(nested) || pure(nested))
      return WalkResult::advance();
    return WalkResult::interrupt();
  }).wasInterrupted();
}
inline bool wrapper(Operation *op) {
  return isa<vector::ExtractOp, vector::FromElementsOp, vector::InsertOp,
             vector::ShapeCastOp, vector::BitCastOp, vector::BroadcastOp,
             vector::ShuffleOp>(op) && pure(op);
}
inline bool address(Operation *op) {
  if (!pure(op)) return false;
  if (isa<affine::AffineApplyOp>(op)) return true;
  return op->getName().getDialectNamespace() == "arith" &&
         llvm::all_of(op->getResultTypes(), [](Type t) {
           return t.isIndex() || isa<IntegerType>(t);
         });
}

// Fold only an exact vector extract/reassemble identity, including all lanes.
// This exposes real MMAC operands without relying on LLVM InstCombine.
inline void foldVectorViews(func::FuncOp f) {
  SmallVector<vector::FromElementsOp> packs;
  f.walk([&](vector::FromElementsOp op) { packs.push_back(op); });
  for (auto pack : packs) {
    auto type = dyn_cast<VectorType>(pack.getType());
    if (!type || type.getRank() != 1 || type.isScalable()) continue;
    Value source;
    SmallVector<Operation *> extracts;
    bool valid = true;
    for (auto [n, value] : llvm::enumerate(pack->getOperands())) {
      auto extract = value.getDefiningOp<vector::ExtractOp>();
      if (!extract || extract.getSource().getType() != type ||
          extract.getStaticPosition().size() != 1 ||
          extract.getStaticPosition()[0] != int64_t(n) ||
          (source && source != extract.getSource())) { valid = false; break; }
      source = extract.getSource();
      extracts.push_back(extract);
    }
    if (!valid || !source) continue;
    pack.getResult().replaceAllUsesWith(source);
    pack.erase();
    for (Operation *extract : extracts) if (extract->use_empty()) extract->erase();
  }
}

inline LLVM::MemoryEffectsAttr guardEffects(MLIRContext *ctx) {
  using E = LLVM::ModRefInfo;
  return LLVM::MemoryEffectsAttr::get(ctx, {E::NoModRef, E::NoModRef,
      E::ModRef, E::NoModRef, E::NoModRef, E::NoModRef});
}
inline LogicalResult checkGuard(ModuleOp module) {
  if (Operation *symbol = module.lookupSymbol(Barrier)) {
    auto f = dyn_cast<LLVM::LLVMFuncOp>(symbol);
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()),
        {IntegerType::get(module.getContext(), 32)}, false);
    if (!f || !f.isExternal() || f.getFunctionType() != type)
      return module.emitError("conflicting declaration for ") << Barrier;
  }
  return success();
}
inline void guard(Operation *before, StringRef marker) {
  auto module = before->getParentOfType<ModuleOp>();
  OpBuilder b(module.getContext());
  auto f = module.lookupSymbol<LLVM::LLVMFuncOp>(Barrier);
  if (!f) {
    b.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()),
                                          {b.getI32Type()}, false);
    f = b.create<LLVM::LLVMFuncOp>(before->getLoc(), Barrier, type);
    f.setMemoryEffectsAttr(guardEffects(module.getContext()));
  }
  b.setInsertionPoint(before);
  auto zero = b.create<LLVM::ConstantOp>(before->getLoc(), b.getI32Type(), b.getI32IntegerAttr(0));
  auto call = b.create<LLVM::CallOp>(before->getLoc(), f, ValueRange{zero});
  call.setMemoryEffectsAttr(guardEffects(module.getContext()));
  call->setAttr(marker, b.getUnitAttr());
}

// A passthrough attribute survives Func-to-LLVM and LLVM IR translation. Late
// optimizations use it to respect the schedule, never to reconstruct it.
inline void markFunction(func::FuncOp f, StringRef marker, StringRef value) {
  OpBuilder b(f.getContext());
  f->setAttr(marker, b.getStringAttr(value));
  SmallVector<Attribute> attrs;
  if (auto old = f->getAttrOfType<ArrayAttr>("llvm.passthrough"))
    attrs.append(old.begin(), old.end());
  attrs.push_back(b.getArrayAttr({b.getStringAttr(marker), b.getStringAttr(value)}));
  f->setAttr("llvm.passthrough", b.getArrayAttr(attrs));
}
} // namespace mlir::frisk::thread_schedule
#endif
