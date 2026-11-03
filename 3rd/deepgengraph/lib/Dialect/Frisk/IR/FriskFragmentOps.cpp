#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::frisk;

static LogicalResult verifyTile(Operation *op, VectorType tile,
                                FragmentType fragment, ValueRange indices) {
  auto reg = fragment.getRegisterType();
  if (tile.isScalable() || tile.getRank() != reg.getRank() ||
      indices.size() != unsigned(tile.getRank()) ||
      tile.getElementType() != reg.getElementType())
    return op->emitOpError("incompatible tile/fragment rank, indices or element type");
  for (unsigned d = 0; d < indices.size(); ++d) {
    if (tile.getDimSize(d) % reg.getDimSize(d))
      return op->emitOpError("incomplete fragment: tile extent must be divisible by fragment extent");
    APInt index;
    if (matchPattern(indices[d], m_ConstantInt(&index)) &&
        (index.isNegative() || index.getLimitedValue() >=
         uint64_t(tile.getDimSize(d) / reg.getDimSize(d))))
      return op->emitOpError("fragment index out of bounds");
  }
  return success();
}

static LogicalResult verifyMemory(Operation *op, MemRefType memory,
                                  VectorType tile, FragmentType fragment,
                                  ValueRange indices) {
  if (failed(verifyTile(op, tile, fragment, indices))) return failure();
  if (!memory.hasStaticShape() || memory.getRank() != tile.getRank() ||
      memory.getElementType() != tile.getElementType())
    return op->emitOpError("requires a static memref matching the thread tile rank and element type");
  auto map = op->getAttrOfType<AffineMapAttr>("tile_layout");
  if (!map) {
    if (memory.getShape() != tile.getShape() || !fragment.getLayout().empty())
      return op->emitOpError("distributed memory access requires a serialized tile_layout");
    return success();
  }
  if (map.getValue().getNumDims() != 7 || map.getValue().getNumSymbols() ||
      map.getValue().getNumResults() != 2)
    return op->emitOpError("tile_layout must map seven coordinates to two logical indices");
  for (StringRef name : {"thread_widths", "warp_repeat", "warp_inst_unroll"}) {
    auto pair = op->getAttrOfType<DenseI64ArrayAttr>(name);
    if (!pair || pair.size() != 2 || pair[0] <= 0 || pair[1] <= 0)
      return op->emitOpError("missing or invalid layout field ") << name;
  }
  for (StringRef name : {"thread_creg_order", "warp_repeat_order"}) {
    auto pair = op->getAttrOfType<DenseI64ArrayAttr>(name);
    if (!pair || pair.size() != 2 ||
        !((pair[0] == 0 && pair[1] == 1) || (pair[0] == 1 && pair[1] == 0)))
      return op->emitOpError("invalid layout order ") << name;
  }
  auto ignored = op->getAttrOfType<IntegerAttr>("ignore_dim");
  if (!ignored || ignored.getInt() < -1 || ignored.getInt() > 1)
    return op->emitOpError("invalid ignore_dim");
  for (StringRef name : {"thread_widths", "warp_repeat", "warp_layout",
                         "warp_layout_order", "thread_creg_order", "warp_repeat_order",
                         "warp_threads"}) {
    Attribute field = fragment.getLayout().get(name);
    if (field && field != op->getAttr(name))
      return op->emitOpError("memory access and fragment distribution disagree on ") << name;
  }
  return success();
}

LogicalResult LoadFragOp::verify() {
  return verifyMemory(*this, cast<MemRefType>(getSource().getType()),
      cast<VectorType>(getTileType()), cast<FragmentType>(getResult().getType()), getIndices());
}
LogicalResult StoreFragOp::verify() {
  return verifyMemory(*this, cast<MemRefType>(getDestination().getType()),
      cast<VectorType>(getTileType()), cast<FragmentType>(getValue().getType()), getIndices());
}
LogicalResult ExtractFragOp::verify() {
  return verifyTile(*this, cast<VectorType>(getSource().getType()),
                    cast<FragmentType>(getResult().getType()), getIndices());
}
LogicalResult InsertFragOp::verify() {
  return verifyTile(*this, cast<VectorType>(getDestination().getType()),
                    cast<FragmentType>(getValue().getType()), getIndices());
}
LogicalResult PackFragOp::verify() {
  return getSource().getType() == cast<FragmentType>(getResult().getType()).getRegisterType()
      ? success() : emitOpError("register payload type mismatch");
}
LogicalResult UnpackFragOp::verify() {
  return getResult().getType() == cast<FragmentType>(getSource().getType()).getRegisterType()
      ? success() : emitOpError("register payload type mismatch");
}
LogicalResult GemmFragOp::verify() {
  for (auto [value, role] : llvm::zip_equal(getOperands(), ArrayRef<StringRef>{"A", "B", "C"})) {
    auto layout = cast<FragmentType>(value.getType()).getLayout();
    if (layout.getAs<StringAttr>("role") != StringAttr::get(getContext(), role) ||
        layout.getAs<StringAttr>("instruction") != getInstNameAttr())
      return emitOpError("MMA operand role/instruction layout mismatch");
  }
  auto a = cast<FragmentType>(getA().getType());
  auto b = cast<FragmentType>(getB().getType());
  auto c = cast<FragmentType>(getAcc().getType());
  if (a.getRegisterType().getElementType() != b.getRegisterType().getElementType() ||
      a.getLayout().get("warp_threads") != b.getLayout().get("warp_threads") ||
      a.getLayout().get("warp_threads") != c.getLayout().get("warp_threads"))
    return emitOpError("MMA operands must agree on input element type and wave size");
  return success();
}
LogicalResult ElementwiseFragOp::verify() {
  StringRef kind = getKind();
  unsigned count = (kind == "exp2" || kind == "cast") ? 1 : 2;
  if (kind != "add" && kind != "sub" && kind != "mul" && kind != "div" &&
      kind != "max" && kind != "exp2" && kind != "cast")
    return emitOpError("unsupported elementwise fragment operation");
  if (getInputs().size() != count) return emitOpError("wrong operand count");
  auto result = cast<FragmentType>(getResult().getType());
  for (Value input : getInputs()) {
    auto type = cast<FragmentType>(input.getType());
    if (type.getLayout() != result.getLayout() ||
        type.getRegisterType().getShape() != result.getRegisterType().getShape() ||
        (kind != "cast" && type != result) ||
        !isa<FloatType>(type.getRegisterType().getElementType()))
      return emitOpError("incompatible fragment layouts, shapes or element types");
  }
  if (!isa<FloatType>(result.getRegisterType().getElementType()))
    return emitOpError("requires floating point result");
  if (kind == "cast") {
    auto from = cast<FragmentType>(getInputs()[0].getType()).getRegisterType().getElementType();
    auto to = result.getRegisterType().getElementType();
    if (from != to && from.getIntOrFloatBitWidth() == to.getIntOrFloatBitWidth())
      return emitOpError("equal-width floating point format conversion is unsupported");
  }
  return success();
}
