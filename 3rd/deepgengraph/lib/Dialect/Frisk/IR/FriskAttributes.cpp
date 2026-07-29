#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.cpp.inc"

namespace mlir::frisk {
namespace {
void printShape(DenseI64ArrayAttr shape, llvm::raw_ostream &os) {
  os << "[";
  if (shape) {
    llvm::ArrayRef<int64_t> values = shape.asArrayRef();
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        os << ", ";
      os << values[i];
    }
  }
  os << "]";
}

void printAffineMapAttr(AffineMapAttr mapAttr, llvm::raw_ostream &os) {
  if (!mapAttr) {
    os << "<none>";
    return;
  }
  std::string buffer;
  llvm::raw_string_ostream valueStream(buffer);
  mapAttr.print(valueStream);
  valueStream.flush();
  os << valueStream.str();
}
} // namespace

void printLayoutDebug(LayoutAttr layout, llvm::raw_ostream &os) {
  if (!layout) {
    os << "<null layout>";
    return;
  }

  bool hasThreadMap = static_cast<bool>(layout.getForwardThread());
  os << (hasThreadMap ? "Fragment" : "Layout") << "(shape=";
  printShape(layout.getInputShape(), os);
  os << ", index=";
  printAffineMapAttr(layout.getForwardIndex(), os);
  if (hasThreadMap) {
    os << ", thread=";
    printAffineMapAttr(layout.getForwardThread(), os);
  }
  if (auto replicate = layout.getReplicateSize())
    os << ", replicate=" << replicate.getInt();
  os << ")";
}

std::string layoutDebugString(LayoutAttr layout) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  printLayoutDebug(layout, os);
  os.flush();
  return buffer;
}


AffineMap SharedSwizzleLayoutAttr::getAffineMap() const {
  return AffineMap::get(getContext());
}
LogicalResult SharedSwizzleLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape, function_ref<InFlightDiagnostic()> emitError) const {
  return success();
}

AffineMap DotOperandLayoutAttr::getAffineMap() const {
  return AffineMap::get(getContext());
}
LogicalResult DotOperandLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape, function_ref<InFlightDiagnostic()> emitError) const {
  return success();
}

AffineMap DotAccumLayoutAttr::getAffineMap() const {
  return AffineMap::get(getContext());
}
LogicalResult DotAccumLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape, function_ref<InFlightDiagnostic()> emitError) const {
  return success();
}

AffineMap NaiveLayoutAttr::getAffineMap() const {
  return AffineMap::get(getContext());
}
LogicalResult NaiveLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape, function_ref<InFlightDiagnostic()> emitError) const {
  return success();
}

AffineMap GPULayoutAttr::getAffineMap() const {
  return getBaseMap().getValue();
}

LogicalResult GPULayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape, function_ref<InFlightDiagnostic()> emitError) const {
  ArrayRef<int64_t> logicalShape = getLogicalShape().asArrayRef();
  ArrayRef<int64_t> paddedShape = getPaddedShape().asArrayRef();
  ArrayRef<int64_t> paddingOffsets = getPaddingOffsets().asArrayRef();

  if (logicalShape.size() != shape.size())
    return emitError() << "gpu layout logical rank (" << logicalShape.size()
                       << ") must match memref rank (" << shape.size() << ")";
  if (paddedShape.size() != shape.size())
    return emitError() << "gpu layout padded rank (" << paddedShape.size()
                       << ") must match memref rank (" << shape.size() << ")";
  if (paddingOffsets.size() != shape.size())
    return emitError() << "gpu layout padding offset rank ("
                       << paddingOffsets.size() << ") must match memref rank ("
                       << shape.size() << ")";

  for (auto [index, dims] :
       llvm::enumerate(llvm::zip(shape, logicalShape, paddedShape,
                                 paddingOffsets))) {
    auto [memrefDim, logicalDim, paddedDim, paddingOffset] = dims;
    if (memrefDim != ShapedType::kDynamic && logicalDim != memrefDim)
      return emitError() << "gpu layout logical shape at dim " << index
                         << " (" << logicalDim
                         << ") must match memref shape (" << memrefDim << ")";
    if (paddingOffset < 0)
      return emitError() << "gpu layout padding offset at dim " << index
                         << " must be non-negative";
    if (logicalDim != ShapedType::kDynamic &&
        paddedDim != ShapedType::kDynamic &&
        paddedDim < logicalDim + paddingOffset)
      return emitError() << "gpu layout padded shape at dim " << index
                         << " must cover logical shape plus padding offset";
  }

  return ::mlir::detail::verifyAffineMapAsLayout(getAffineMap(), shape,
                                                 emitError);
}

void FriskDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.cpp.inc"
      >();
}
} // namespace mlir::frisk
