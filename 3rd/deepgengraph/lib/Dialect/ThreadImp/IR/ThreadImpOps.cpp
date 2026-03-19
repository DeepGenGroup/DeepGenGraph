#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"

#include "dbg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpOps.cpp.inc"
// #include "deepgengraph/Dialect/ThreadImp/IR/DeepgengraphTritonOps.cpp.inc"

#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.cpp.inc"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpEnums.cpp.inc"
// #include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.cpp.inc"
namespace mlir {
namespace threadimp {
void ThreadImpDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "deepgengraph/Dialect/ThreadImp/IR//ThreadImpOps.cpp.inc"
// #include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonOps.cpp.inc"
      >();

}

llvm::LogicalResult BlockCopyG2S::inferReturnTypes(::mlir::MLIRContext *context,
  std::optional<::mlir::Location> location,
  Adaptor adaptor,
  ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
  auto dstType = mlir::cast<TensorType>(adaptor.getDstTensor().getType());
  inferredReturnTypes.push_back(dstType);
  return mlir::success();  
}

// -- PreciseDotOp --
LogicalResult PreciseDotOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                             Adaptor adaptor,
                                             ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto lhs_type = cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs_type = cast<RankedTensorType>(adaptor.getRhs().getType());
  auto acc_type = adaptor.getAccType();

  auto lhs_elem_type = lhs_type.getElementType();
  auto rhs_elem_type = rhs_type.getElementType();
  if (lhs_elem_type != rhs_elem_type) {
    return emitOptionalError(location, "lhs and rhs elem_type mismatch");
  }

  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  if (lhs_rank < 2 || rhs_rank < 2) {
    return emitOptionalError(location, "lhs and rhs rank both need >= 2");
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();
  if (lhs_shape[lhs_rank - 1] != rhs_shape[rhs_rank - 2]) {
    return emitOptionalError(location, "lhs_shape[-2] and rhs_shape[-1] doesn't match");
  }

  if (lhs_rank != rhs_rank) {
    // batch dim broadcast
    if (lhs_rank > 2 && rhs_rank > 2) {
      return emitOptionalError(location, "operands broadcast batch dim differently, unaccepted");
    }

    if (lhs_rank > rhs_rank) {
      assert(rhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
      return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, acc_type);
      inferredReturnTypes.push_back(return_type);
    } else {
      assert(lhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(rhs_shape);
      return_shape[rhs_rank - 2] = lhs_shape[lhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, acc_type);
      inferredReturnTypes.push_back(return_type);
    }
  } else {
    llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
    return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
    auto return_type = RankedTensorType::get(return_shape, acc_type);
    inferredReturnTypes.push_back(return_type);
  }
  return success();
}


} // namespace deepgengraph::triton
} // namespace mlir

namespace mlir {


} // namespace mlir