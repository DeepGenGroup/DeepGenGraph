#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"

#include "dbg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <vector>

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

// llvm::LogicalResult CopyGlobalToShm::inferReturnTypes(
//   ::mlir::MLIRContext *context,
//   std::optional<::mlir::Location> location,
//   Adaptor adaptor,
//   ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
// {
//   auto blockShape = adaptor.getBlockShape();
//   auto elementType = mlir::cast<threadimp::PointerType>(adaptor.getSrcPointer().getType()).getElementType();
//   auto retType = mlir::RankedTensorType::get(blockShape, elementType);
//   inferredReturnTypes.push_back(retType);
//   return mlir::success();
// }


llvm::LogicalResult ThreadElementwiseBinaryOp::inferReturnTypes(
  ::mlir::MLIRContext *context,
  std::optional<::mlir::Location> location,
  Adaptor adaptor,
  ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
  auto ltensor = mlir::cast<TensorType>(adaptor.getLhs().getType());
  auto rtensor = mlir::cast<TensorType>(adaptor.getRhs().getType());
  auto lshape = ltensor.getShape();
  auto rshape = rtensor.getShape();
  std::vector<int64_t> inferShape;
  for(int i=0;i<lshape.size();++i){
    inferShape.push_back(std::max(lshape[i], rshape[i]));
  }
  Type retElementType;
  if(ltensor.getElementType().getIntOrFloatBitWidth() > rtensor.getElementTypeBitWidth()){
    retElementType = ltensor.getElementType();
  }
  else{
    retElementType = rtensor.getElementType();
  }
  auto retType = RankedTensorType::get(inferShape, retElementType);
  inferredReturnTypes.push_back(retType);
  return mlir::success();
}

// -- PreciseDotOp --
LogicalResult PreciseMatmulOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
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




LogicalResult PointerToEmptyGlobalOp::inferReturnTypes(
  ::mlir::MLIRContext *context, 
  std::optional<::mlir::Location> location,
  Adaptor adaptor,
  ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) 
{
  auto rt = threadimp::PointerType::get(mlir::cast<TensorType>(adaptor.getTensorType()), threadimp::MemSpace::GM);
  inferredReturnTypes.push_back(rt);
  return mlir::success(); 
}

LogicalResult PointerToOp::inferReturnTypes(
  ::mlir::MLIRContext *context, 
  std::optional<::mlir::Location> location,
  Adaptor adaptor,
  ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) 
{
  auto eType = mlir::cast<TensorType>(adaptor.getSrcTensor().getType()).getElementType();
  auto retType = threadimp::PointerType::get(eType, adaptor.getMemspace());
  inferredReturnTypes.push_back(retType);
  return mlir::success(); 
}



} // namespace deepgengraph::triton
} // namespace mlir

namespace mlir {


} // namespace mlir