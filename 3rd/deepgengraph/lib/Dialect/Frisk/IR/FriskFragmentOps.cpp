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

LogicalResult BroadcastFragOp::verify() {
  auto dst = getResult().getType().getRegisterType();
  if (auto src = dyn_cast<FragmentType>(getSource().getType())) {
    auto reg = src.getRegisterType();
    if (reg.getElementType() != dst.getElementType() || reg.getRank() != dst.getRank())
      return emitOpError("broadcast requires matching rank and element type");
    for (unsigned d = 0; d < reg.getRank(); ++d)
      if (reg.getDimSize(d) != 1 && reg.getDimSize(d) != dst.getDimSize(d))
        return emitOpError("only singleton fragment axes may be broadcast");
  } else if (getSource().getType() != dst.getElementType()) {
    return emitOpError("broadcast source must be a fragment or matching scalar");
  }
  return success();
}

LogicalResult ReduceFragOp::verify() {
  auto src = getSource().getType().getRegisterType();
  auto dst = getResult().getType().getRegisterType();
  if (!isa<FloatType>(src.getElementType()) || src.getElementType() != dst.getElementType() ||
      getDim() < 0 || getDim() >= src.getRank())
    return emitOpError("invalid reduction axis or floating-point element type");
  if (getKind() != "add" && getKind() != "mul" && getKind() != "min" && getKind() != "max")
    return emitOpError("unsupported reduction kind");
  SmallVector<int64_t> shape(src.getShape());
  if (dst.getRank() == src.getRank()) shape[getDim()] = 1;
  else shape.erase(shape.begin() + getDim());
  if (ArrayRef<int64_t>(shape) != dst.getShape())
    return emitOpError("result must retain the non-reduced axes");
  auto powerOfTwo = [](int64_t n) { return n > 0 && (n & (n - 1)) == 0; };
  if (!powerOfTwo(getLaneExtent()) || !powerOfTwo(getLaneStride()) ||
      !powerOfTwo(getWarpThreads()) || getWarpThreads() > 64 ||
      getLaneStride() > getWarpThreads() / getLaneExtent())
    return emitOpError("invalid reduction lane topology");
  if (getReductionWidth() <= 0 || src.getDimSize(getDim()) % getReductionWidth())
    return emitOpError("reduction width must divide the source axis");
  auto fragments = getOutputFragmentShape();
  if (fragments.size() != unsigned(dst.getRank()))
    return emitOpError("output fragment shape rank mismatch");
  for (unsigned d = 0; d < fragments.size(); ++d)
    if (fragments[d] <= 0 || dst.getDimSize(d) % fragments[d])
      return emitOpError("output fragment shape must divide the result");
  return success();
}

static LogicalResult verifySemanticLayout(Operation *op, DictionaryAttr layout,
    ShapedType block, VectorType registers) {
  auto map = layout.getAs<AffineMapAttr>("tile_layout");
  auto wave = layout.getAs<IntegerAttr>("warp_threads");
  auto ignored = layout.getAs<IntegerAttr>("ignore_dim");
  if (!map || map.getValue().getNumDims() != 7 || map.getValue().getNumSymbols() ||
      map.getValue().getNumResults() != 2 || !wave || wave.getInt() <= 0 ||
      !ignored || ignored.getInt() < -1 || ignored.getInt() > 1)
    return op->emitOpError("incomplete serialized layout");
  for (StringRef name : {"thread_widths", "warp_repeat", "warp_inst_unroll", "warp_layout",
                         "block_layout", "block_repeat"}) {
    auto pair = layout.getAs<DenseI64ArrayAttr>(name);
    if (!pair || pair.size() != 2 || pair[0] <= 0 || pair[1] <= 0)
      return op->emitOpError("invalid layout extent ") << name;
  }
  for (StringRef name : {"thread_creg_order", "warp_repeat_order", "warp_layout_order", "block_layout_order"}) {
    auto pair = layout.getAs<DenseI64ArrayAttr>(name);
    if (!pair || pair.size() != 2 ||
        !((pair[0] == 0 && pair[1] == 1) || (pair[0] == 1 && pair[1] == 0)))
      return op->emitOpError("invalid layout permutation ") << name;
  }
  auto lanes = layout.getAs<DenseI64ArrayAttr>("warp_layout");
  if (wave.getInt() % lanes[0] || wave.getInt() / lanes[0] != lanes[1])
    return op->emitOpError("layout lane extents do not match wave size");
  for (int64_t d = 0; d < registers.getRank(); ++d) {
    unsigned axis = registers.getRank() == 1 && ignored.getInt() == 0 ? 1 : d;
    int64_t extent = registers.getDimSize(d);
    if (block.getDimSize(d) == 1 || (registers.getRank() == 2 && ignored.getInt() == axis)) {
      if (extent != 1) return op->emitOpError("projected layout axis must have one register");
      continue;
    }
    for (StringRef name : {"thread_widths", "warp_repeat", "warp_inst_unroll", "block_repeat"}) {
      int64_t factor = layout.getAs<DenseI64ArrayAttr>(name)[axis];
      if (extent % factor) return op->emitOpError("register shape does not match serialized layout");
      extent /= factor;
    }
    if (extent != 1) return op->emitOpError("register shape does not match serialized layout");
  }
  if (auto divisor = layout.getAs<IntegerAttr>("warp_id_divisor"); divisor && divisor.getInt() <= 0)
    return op->emitOpError("warp_id_divisor must be positive");
  if (auto transform = layout.getAs<AffineMapAttr>("logical_transform"); transform &&
      (transform.getValue().getNumDims() != 2 || transform.getValue().getNumSymbols() ||
       transform.getValue().getNumResults() != 2))
    return op->emitOpError("invalid logical_transform");
  return success();
}

LogicalResult ConvertLayoutFragOp::verify() {
  auto src = getSource().getType().getRegisterType();
  auto dst = getResult().getType().getRegisterType();
  auto block = dyn_cast<ShapedType>(getBlockType());
  if (!block || !isa<MemRefType, VectorType>(block) || !block.hasStaticShape() ||
      block.getRank() != src.getRank() || block.getRank() != dst.getRank() ||
      block.getElementType() != src.getElementType() || src.getElementType() != dst.getElementType())
    return emitOpError("incompatible layout conversion block/register types");
  if (failed(verifySemanticLayout(*this, getSourceLayout(), block, src)) ||
      failed(verifySemanticLayout(*this, getTargetLayout(), block, dst))) return failure();
  if (auto plan = getShufflePlan()) {
    auto masks = plan->getAs<DenseI64ArrayAttr>("xor_masks");
    auto registers = plan->getAs<IntegerAttr>("registers");
    auto width = plan->getAs<IntegerAttr>("width");
    if (!masks || masks.empty() || masks.size() > 6 || !registers || !width ||
        width.getInt() <= 0 || width.getInt() > 64 ||
        (width.getInt() & (width.getInt() - 1)) ||
        registers.getInt() != (int64_t(1) << masks.size()) || src != dst ||
        dst.getRank() != 2 || dst.getDimSize(1) % registers.getInt())
      return emitOpError("invalid layout shuffle plan");
    for (int64_t mask : masks.asArrayRef())
      if (mask <= 0 || mask >= width.getInt() || (mask & (mask - 1)))
        return emitOpError("invalid layout shuffle XOR mask");
  }
  return success();
}

LogicalResult MaskFragOp::verify() {
  auto block = cast<MemRefType>(getBlockType());
  auto registers = getResult().getType().getRegisterType();
  if (!block.hasStaticShape() || block.getRank() != registers.getRank() ||
      block.getElementType() != registers.getElementType() ||
      getStarts().size() != unsigned(block.getRank()))
    return emitOpError("incompatible mask block/register types or starts");
  if (failed(verifySemanticLayout(*this, getLayout(), block, registers))) return failure();
  if (getAddend() && getAddend().getType() != getResult().getType())
    return emitOpError("mask addend must match the result fragment type");
  auto shape = getFragmentShape();
  if (shape.size() != unsigned(registers.getRank()))
    return emitOpError("mask fragment shape rank mismatch");
  for (unsigned d = 0; d < shape.size(); ++d)
    if (shape[d] <= 0 || registers.getDimSize(d) % shape[d])
      return emitOpError("mask fragment shape must divide the register shape");
  Block &body = getRegion().front();
  auto yield = dyn_cast<MaskYieldOp>(body.getTerminator());
  if (body.getNumArguments() != unsigned(block.getRank()) ||
      !llvm::all_of(body.getArgumentTypes(), [](Type t) { return t.isIndex(); }) ||
      !yield || yield->getNumOperands() != 1 ||
      yield->getOperand(0).getType() != registers.getElementType())
    return emitOpError("mask predicate must take logical indices and yield one element");
  return success();
}
