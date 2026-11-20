#include "CheckedGlobalIO.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::frisk {
namespace {
struct Access {
  Operation *op;
  Value buffer, stored;
  SmallVector<Value> indices;
  AffineMap map;
};

Access getAccess(Operation *op) {
  Access a{op, {}, {}, {}, {}};
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    a.buffer = load.getMemref(); a.indices = load.getIndices();
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    a.buffer = store.getMemref(); a.stored = store.getValueToStore();
    a.indices = store.getIndices();
  } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    a.buffer = load.getMemRef(); a.indices = load.getMapOperands();
    a.map = load.getAffineMap();
  } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    a.buffer = store.getMemRef(); a.stored = store.getValueToStore();
    a.indices = store.getMapOperands(); a.map = store.getAffineMap();
  } else if (auto load = dyn_cast<vector::LoadOp>(op)) {
    a.buffer = load.getBase(); a.indices = load.getIndices();
  } else if (auto store = dyn_cast<vector::StoreOp>(op)) {
    a.buffer = store.getBase(); a.stored = store.getValueToStore();
    a.indices = store.getIndices();
  }
  return a;
}

Value inBounds(OpBuilder &b, Location loc, MemRefType type,
               ArrayRef<Value> indices, int64_t width = 1) {
  Value valid = b.create<arith::ConstantIntOp>(loc, 1, 1);
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    int64_t limit = type.getDimSize(dim);
    if (dim == type.getRank() - 1) limit -= width - 1;
    if (limit <= 0) return b.create<arith::ConstantIntOp>(loc, 0, 1);
    Value extent = b.create<arith::ConstantIndexOp>(loc, limit);
    // Unsigned comparison also rejects negative coordinates.
    Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                       indices[dim], extent);
    valid = b.create<arith::AndIOp>(loc, valid, pred);
  }
  return valid;
}

Value checkedScalar(OpBuilder &b, Location loc, Value buffer,
                    ArrayRef<Value> indices, Value stored) {
  auto type = cast<MemRefType>(buffer.getType());
  Value valid = inBounds(b, loc, type, indices);
  auto guard = b.create<scf::IfOp>(loc,
      stored ? TypeRange{} : TypeRange{type.getElementType()}, valid, !stored);
  {
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    if (stored) {
      b.create<memref::StoreOp>(loc, stored, buffer, indices);
    } else {
      Value value = b.create<memref::LoadOp>(loc, buffer, indices);
      b.create<scf::YieldOp>(loc, value);
      b.setInsertionPointToStart(&guard.getElseRegion().front());
      Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
      b.create<scf::YieldOp>(loc, zero);
    }
  }
  return stored ? Value{} : guard.getResult(0);
}
} // namespace

LogicalResult lowerCheckedGlobalIO(func::FuncOp kernel) {
  if (!kernel->hasAttr("frisk.checked_global_io") ||
      kernel->hasAttr("frisk.checked_global_io_lowered")) return success();
  SmallVector<Operation *> accesses;
  auto walk = kernel.walk([&](Operation *op) {
    if (isa<AsyncCopyGmToShmOp>(op)) {
      op->emitError("checked global IO requires synchronous copies; DMA has no logical bounds contract");
      return WalkResult::interrupt();
    }
    Access a = getAccess(op);
    if (!a.buffer) {
      if (!isMemoryEffectFree(op) && llvm::any_of(op->getOperandTypes(), [](Type t) {
            auto memref = dyn_cast<MemRefType>(t);
            return memref && memref.getMemorySpaceAsInt() == 1;
          })) {
        op->emitError("unsupported global memory operation in checked global IO kernel");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto type = cast<MemRefType>(a.buffer.getType());
    if (type.getMemorySpaceAsInt() != 1) return WalkResult::advance();
    // Views must have been composed before this point. Checking a tile-shaped
    // alias would miss crossings into adjacent heads/batches.
    auto arg = dyn_cast<BlockArgument>(a.buffer);
    Type valueType = a.stored ? a.stored.getType() : op->getResult(0).getType();
    auto vector = dyn_cast<VectorType>(valueType);
    if (!arg || arg.getOwner() != &kernel.getBody().front() ||
        !type.hasStaticShape() ||
        (vector && (vector.getRank() != 1 || vector.isScalable()))) {
      op->emitError("checked global IO requires resolved static kernel buffers and fixed rank-one vectors");
      return WalkResult::interrupt();
    }
    accesses.push_back(op);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) return failure();
  for (Operation *op : accesses) {
    // Earlier guarded loads may feed this access. Refresh operands after RAUW
    // instead of retaining Values owned by an erased load.
    Access a = getAccess(op);
    OpBuilder b(a.op);
    Location loc = a.op->getLoc();
    if (a.map) {
      SmallVector<Value> coordinates;
      for (unsigned d = 0; d < a.map.getNumResults(); ++d)
        coordinates.push_back(b.create<affine::AffineApplyOp>(
            loc, a.map.getSubMap({d}), a.indices));
      a.indices = std::move(coordinates);
    }
    Type valueType = a.stored ? a.stored.getType() : a.op->getResult(0).getType();
    Value result;
    if (auto vector = dyn_cast<VectorType>(valueType)) {
      // Keep native vector packets for fully valid rows. A packet straddling
      // the final axis is handled element by element, without issuing an OOB
      // vector load and trying to mask its result afterwards.
      int64_t width = vector.getDimSize(0);
      Value valid = inBounds(b, loc, cast<MemRefType>(a.buffer.getType()), a.indices, width);
      auto guard = b.create<scf::IfOp>(loc,
          a.stored ? TypeRange{} : TypeRange{valueType}, valid, true);
      {
        OpBuilder::InsertionGuard insertion(b);
        b.setInsertionPointToStart(&guard.getThenRegion().front());
        if (a.stored) b.create<vector::StoreOp>(loc, a.stored, a.buffer, a.indices);
        else {
          Value value = b.create<vector::LoadOp>(loc, vector, a.buffer, a.indices);
          b.create<scf::YieldOp>(loc, value);
        }
        b.setInsertionPointToStart(&guard.getElseRegion().front());
        Value packet;
        if (!a.stored) packet = b.create<arith::ConstantOp>(loc, b.getZeroAttr(vector));
        for (int64_t lane = 0; lane < width; ++lane) {
          auto indices = a.indices;
          Value offset = b.create<arith::ConstantIndexOp>(loc, lane);
          indices.back() = b.create<arith::AddIOp>(loc, indices.back(), offset);
          Value scalar;
          if (a.stored) scalar = b.create<vector::ExtractOp>(loc, a.stored, lane);
          Value value = checkedScalar(b, loc, a.buffer, indices, scalar);
          if (!a.stored) packet = b.create<vector::InsertOp>(loc, value, packet, lane);
        }
        if (!a.stored) b.create<scf::YieldOp>(loc, packet);
      }
      if (!a.stored) result = guard.getResult(0);
    } else result = checkedScalar(b, loc, a.buffer, a.indices, a.stored);
    if (result) a.op->getResult(0).replaceAllUsesWith(result);
    a.op->erase();
  }
  kernel->setAttr("frisk.checked_global_io_lowered", UnitAttr::get(kernel.getContext()));
  return success();
}
} // namespace mlir::frisk
