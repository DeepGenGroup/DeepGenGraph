// Identity aliases of owned storage. This describes possible addresses, not
// equality of their contents: it must never be used to forward a stored value.
#ifndef DEEPGENGRAPH_SHARED_BUFFER_ALIASES_H
#define DEEPGENGRAPH_SHARED_BUFFER_ALIASES_H

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::frisk {

// A connected component includes every alternative of every selector. A
// physical permutation must be applied atomically to the whole component.
// Shape-changing casts, views, unknown origins and escaping users are left to
// the caller's conservative fallback; no pipeline annotations are consulted.
struct SharedBufferAliases {
  llvm::SetVector<Value> values;
  llvm::SetVector<Operation *> allocations;
  SmallVector<OpOperand *> accesses;

  static bool isAllocation(Value value) {
    return isa_and_nonnull<AllocBufferOp, memref::AllocOp>(value.getDefiningOp());
  }

  static bool incoming(Value value, SmallVectorImpl<Value> &inputs) {
    if (isAllocation(value))
      return true;
    if (auto copy = value.getDefiningOp<CopyOp>()) {
      inputs.push_back(copy.getDst());
      return true;
    }
    if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      inputs.append({select.getTrueValue(), select.getFalseValue()});
      return select.getCondition().getType().isInteger(1);
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      unsigned n = result.getResultNumber();
      if (auto branch = dyn_cast<scf::IfOp>(result.getOwner())) {
        if (branch.getElseRegion().empty())
          return false;
        inputs.append({branch.thenYield().getOperand(n),
                       branch.elseYield().getOperand(n)});
        return true;
      }
      if (auto loop = dyn_cast<scf::ForOp>(result.getOwner())) {
        inputs.append({loop.getInitArgs()[n], loop.getRegionIterArgs()[n],
                       loop.getBody()->getTerminator()->getOperand(n)});
        return true;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(result.getOwner())) {
        inputs.append({loop.getInits()[n], loop.getRegionIterArgs()[n],
                       loop.getBody()->getTerminator()->getOperand(n)});
        return true;
      }
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (!arg.getArgNumber())
        return false;
      unsigned n = arg.getArgNumber() - 1;
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        inputs.push_back(loop.getResult(n));
        return true;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(arg.getOwner()->getParentOp())) {
        inputs.push_back(loop.getResult(n));
        return true;
      }
    }
    return false;
  }

  // True only for uses which forward this exact descriptor. A copy also
  // writes storage, so its operand remains an access for client validation.
  static bool outgoing(OpOperand &use, SmallVectorImpl<Value> &outputs) {
    Operation *op = use.getOwner();
    unsigned n = use.getOperandNumber();
    if (auto select = dyn_cast<arith::SelectOp>(op); select && n > 0) {
      outputs.push_back(select.getResult());
      return true;
    }
    if (isa<scf::YieldOp, affine::AffineYieldOp>(op)) {
      Operation *parent = op->getParentOp();
      if (isa<scf::IfOp, scf::ForOp, affine::AffineForOp>(parent)) {
        outputs.push_back(parent->getResult(n));
        return true;
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(op);
        loop && n >= loop.getNumControlOperands()) {
      outputs.push_back(loop.getResult(n - loop.getNumControlOperands()));
      return true;
    }
    if (auto loop = dyn_cast<affine::AffineForOp>(op);
        loop && n >= loop.getNumControlOperands()) {
      outputs.push_back(loop.getResult(n - loop.getNumControlOperands()));
      return true;
    }
    if (auto copy = dyn_cast<CopyOp>(op);
        copy && n == 1 && copy.hasValueResult())
      outputs.push_back(copy->getResult(0));
    return false;
  }

  bool collect(Value seed) {
    auto type = dyn_cast<MemRefType>(seed.getType());
    if (!type || type.getMemorySpaceAsInt() != 3)
      return false;
    values.insert(seed);
    for (unsigned i = 0; i < values.size(); ++i) {
      Value value = values[i];
      if (value.getType() != type)
        return false;
      SmallVector<Value> neighbors;
      if (!incoming(value, neighbors))
        return false;
      if (isAllocation(value))
        allocations.insert(value.getDefiningOp());
      for (OpOperand &use : value.getUses())
        if (!outgoing(use, neighbors))
          accesses.push_back(&use);
      for (Value neighbor : neighbors)
        values.insert(neighbor);
    }
    return !allocations.empty();
  }

  // All possible allocations must have the same contract. Used while tiling
  // copies, before physical packing rewrites the selected descriptors.
  Attribute commonAttribute(StringRef name) const {
    Attribute attr;
    for (Operation *alloc : allocations) {
      auto next = alloc->getAttr(name);
      if (!next || (attr && attr != next))
        return {};
      attr = next;
    }
    return attr;
  }
};

} // namespace mlir::frisk
#endif
