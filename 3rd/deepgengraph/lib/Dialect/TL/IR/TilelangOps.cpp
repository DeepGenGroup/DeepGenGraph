#include <cassert>
#include <cstdint>
#include <vector>
#include "dbg.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpAsmSupport.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// #include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/InitAllDialects.h"

#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"

// #define GET_ATTRDEF_CLASSES
// #include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphAttrs.cpp.inc"

// move dialect def in this file to make compiler happy
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.cpp.inc"
#include "deepgengraph/Dialect/TL/IR/TilelangEnums.cpp.inc"
#include "deepgengraph/Dialect/TL/IR/TilelangAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/TL/IR/TilelangTypes.cpp.inc"

namespace mlir {
namespace tilelang {
#include "deepgengraph/Dialect/TL/IR/TilelangOpInterfaces.cpp.inc"


// --- 打印逻辑 ---
// 这会让属性在 IR 中直接显示为 "GM", "L1" 等字符串
static void printMemSpace(AsmPrinter &p, Operation *op, MemSpaceAttr attr) {
  // stringifyMemSpace 是 TableGen 自动生成的函数
  p << stringifyMemSpace(attr.getValue());
}

// --- 解析逻辑 ---
// 从 IR 中的字符串解析回属性
static ParseResult parseMemSpace(AsmParser &p, MemSpaceAttr &attr) {
  StringRef keyword;
  // 1. 解析一个关键字 (e.g., "GM")
  if (p.parseKeyword(&keyword))
    return failure();

  // 2. 将字符串转换为枚举值 (symbolizeMemSpace 是自动生成的)
  auto loc = p.getCurrentLocation();
  auto enumValue = symbolizeMemSpace(keyword);

  if (!enumValue) {
    return p.emitError(loc, "invalid memory space: ") << keyword;
  }

  // 3. 构建 Attribute
  attr = MemSpaceAttr::get(p.getContext(), *enumValue);
  return success();
}



} // namespace tilelang
} // namespace mlir

#define GET_OP_CLASSES
#include "deepgengraph/Dialect/TL/IR/TilelangOps.cpp.inc"


namespace mlir {
namespace tilelang {

// Dialect Init -----------------
void TilelangDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "deepgengraph/Dialect/TL/IR/TilelangAttrs.cpp.inc"
  >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "deepgengraph/Dialect/TL/IR/TilelangTypes.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "deepgengraph/Dialect/TL/IR/TilelangOps.cpp.inc"
  >();
}

// PrimFuncOp ------------------------------------------

PrimFuncOp PrimFuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  PrimFuncOp::build(builder, state, name, type, attrs);
  return cast<PrimFuncOp>(Operation::create(state));
}
PrimFuncOp PrimFuncOp::create(Location location, StringRef name, FunctionType type,
                      Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}
PrimFuncOp PrimFuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  PrimFuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void PrimFuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult PrimFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void PrimFuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void WithKernelOp::getAsmBlockArgumentNames(::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn){
  if(&region != getBody()){
    return;
  }
  if (!region.empty() && region.front().getNumArguments() >= 3) {
    // 为第 0 个参数命名为 cid
    setNameFn(region.front().getArgument(0), "bz");
    // 为第 1 个参数命名为 vid
    setNameFn(region.front().getArgument(1), "by");
    setNameFn(region.front().getArgument(2), "bx");
  }
}

void WithScopeOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, bool is_cube){
  mlir::StringAttr kind;
  if(is_cube){
    kind = builder.getStringAttr("C");
  }
  else{
    kind = builder.getStringAttr("V");
  }
  state.addAttribute("executorKind", kind);
  auto r = state.addRegion();
  auto basicBlock = new mlir::Block();
  r->push_back(basicBlock);
}

// ---------------- AllocOp

void AllocOp::build(
  ::mlir::OpBuilder &builder, 
  ::mlir::OperationState &op, 
  MemSpace memspace, 
  ArrayRef<int64_t> shape, 
  Type elementType)
{
  // 将原生 C++ 类型转换为 MLIR 属性
  auto memspaceAttr = MemSpaceAttr::get(builder.getContext(), memspace);
  auto shapeAttr = builder.getI64ArrayAttr(shape);
  auto elementTypeAttr = TypeAttr::get(elementType);

  // 添加属性 elementType, memspace, shape
  op.addAttribute("memspace", memspaceAttr);
  op.addAttribute("shape", shapeAttr);
  op.addAttribute("elementType", elementTypeAttr);
  std::vector<Type> retTypes ;
  retTypes.push_back(MemRefType::get(shape,elementType, MemRefLayoutAttrInterface{}, memspaceAttr));
  op.addTypes(retTypes);
}

// ---------------- CopyOp
void CopyOp::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect> >& effects){
  effects.emplace_back(MemoryEffects::Read::get(), mlir::cast<MemRefType>(getSrc().getType()).getMemorySpace() );
  effects.emplace_back(MemoryEffects::Write::get());
}

::llvm::LogicalResult CopyOp::verify(){
  auto srcMem = mlir::cast<MemRefType>(getSrc().getType()).getMemorySpaceAsInt();
  auto dstMem = mlir::cast<MemRefType>(getDst().getType()).getMemorySpaceAsInt();
  MemSpace _src = MemSpace(srcMem);
  MemSpace _dst = MemSpace(dstMem);
  // self copy
  if(_src == _dst){
    return success();
  }
  // valid copy : GM->L0A, L0B, UB, L1; L1 -> L0A, L0B ;  L0C->GM, L1 ; UB -> GM; 
  if(_src == MemSpace::GM){
    if(_dst == MemSpace::L0C){
      return mlir::emitError(CopyOp::getLoc(), "error copy addr schema!");
    }
  }
  else if(_src == MemSpace::L1){
    if(_dst != MemSpace::L0A && _dst != MemSpace::L0B){
      return mlir::emitError(CopyOp::getLoc(), "error copy addr schema!");
    }
  }
  else if(_src == MemSpace::L0C){
    if(_dst != MemSpace::GM && _dst != MemSpace::L1){
      return mlir::emitError(CopyOp::getLoc(), "error copy addr schema!");
    }
  }
  else if(_src == MemSpace::UB){
    if(_dst != MemSpace::GM){
      return mlir::emitError(CopyOp::getLoc(), "error copy addr schema!");
    }
  }
  return success();
}

// ----------- BinaryElementwise Op

static LogicalResult binary_elementwise_inner_verify(Location loc, const Value& lhs, const Value& rhs){
  auto lhstype= mlir::cast<MemRefType>(lhs.getType());
  auto rhstype= mlir::cast<MemRefType>(rhs.getType());
  if(lhstype.getElementType() != rhstype.getElementType()){
    return emitError( loc, "err");
  }
  if(lhstype.getShape() != rhstype.getShape()){
    return emitError( loc, "err");
  }
  return success();
}

LogicalResult mlir::tilelang::TileAddOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}
LogicalResult mlir::tilelang::TileSubOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}
LogicalResult mlir::tilelang::TileMulOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}
LogicalResult mlir::tilelang::TileDivOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}
LogicalResult mlir::tilelang::TileMaxOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}
LogicalResult mlir::tilelang::TileMinOp::verify(){
  return binary_elementwise_inner_verify(getLoc(), getLhs(), getRhs());
}

// ------------- gemm_v0 op

void GemmV0Op::build(::mlir::OpBuilder &builder, 
  ::mlir::OperationState &state, 
  Value matrixA, 
  Value matrixB, 
  Value matrixC, 
  bool transA, 
  bool transB, 
  bool init)
{
  auto atype = mlir::cast<MemRefType>(matrixA.getType());
  auto btype = mlir::cast<MemRefType>(matrixB.getType());
  auto ctype = mlir::cast<MemRefType>(matrixC.getType());

  MemSpace aMemLoc = MemSpace(atype.getMemorySpaceAsInt()) ;
  MemSpace bMemLoc = MemSpace(btype.getMemorySpaceAsInt()) ;
  MemSpace cMemLoc = MemSpace(ctype.getMemorySpaceAsInt()) ;

  // assert(cMemLoc == MemSpace::L0C);  // a,b can be in l1 l0a l0b, c must in l0c

  std::vector<Value> vr = {matrixA, matrixB, matrixC};
  state.addOperands(vr);
  auto initAttr = builder.getBoolAttr(init);
  auto taAttr = builder.getBoolAttr(transA);
  auto tbAttr = builder.getBoolAttr(transB);
  state.addAttribute("init", initAttr);
  state.addAttribute("transA", taAttr);
  state.addAttribute("transB", tbAttr);
}

LogicalResult GemmV0Op::verify(){
  return success();
}

// ------------- SerialForOp

void SerialForOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, 
  Value lb, Value ub){
  std::vector<Value> vr = {lb, ub};
  state.addOperands(vr);
  auto region = state.addRegion();
  auto newBlock = new Block();
  region->push_back(newBlock);
  newBlock->addArgument(builder.getIndexType(), state.location );
  {
    OpBuilder::InsertionGuard guard(builder);  // guard dtor时，自动返回此前的插入点
    builder.setInsertionPointToEnd(newBlock);
    builder.create<YieldOp>(state.location);
  }
}

void SerialForOp::getAsmBlockArgumentNames(::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn){
  if(&region != getScopedBody()){
    return;
  }
  // 检查 Block 参数数量是否符合预期
  if (!region.empty() && region.front().getNumArguments() >= 1) {
    // 为第 0 个参数命名为 iter
    setNameFn(region.front().getArgument(0), "iter");
  }
}


} // namespace tilelang
} // namespace mlir