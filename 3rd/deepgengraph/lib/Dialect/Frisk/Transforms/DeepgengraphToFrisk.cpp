#include "deepgengraph/Common.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir::frisk {
  
#define GEN_PASS_DEF_KERNELOPTOFRISK
#define GEN_PASS_DEF_MEMANDCALCOPTOFRISK

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"
namespace dg = deepgengraph ;
namespace dgt = deepgengraph::triton;

// lower options
struct CalcOpToFriskOption {
  static bool useTensorCore ;
  static bool useTMA ;
};

bool CalcOpToFriskOption::useTensorCore = true;
bool CalcOpToFriskOption::useTMA = true;

namespace {

namespace dg = deepgengraph ;
namespace dgt = deepgengraph::triton;

// =================== static helper functions =================
static Value getKernelArgById(Operation *op, int64_t argId) {
  auto kernelOp = op->getParentOfType<frisk::KernelOp>();
  if (!kernelOp)
    return {};
  if (argId < 0 || argId >= static_cast<int64_t>(kernelOp.getNumArguments()))
    return {};
  return kernelOp.getArgument(argId);
}

static Type convertPointerType(deepgengraph::triton::PointerType ptrType) {
  auto tensorTy = ptrType.getPointeeType();
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{},  tensorTy.getEncoding());
}

static Type convertBlockPointerType(deepgengraph::triton::BlockPointerType blockPtrType) {
  auto tensorTy = blockPtrType.getPointeeType();
  SmallVector<int64_t> dynStrides(tensorTy.getRank(), ShapedType::kDynamic);
  // auto layout = StridedLayoutAttr::get(blockPtrType.getContext(), ShapedType::kDynamic, dynStrides);
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{}, tensorTy.getEncoding());
}

static void addMaterializations(TypeConverter &tc) {
  tc.addTargetMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
  tc.addSourceMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
}

static bool isTritonPointerLike(Type type) {
  return isa<deepgengraph::triton::PointerType, deepgengraph::triton::BlockPointerType>(type);
}

static void AppendMemspaceToMemrefValue(Value& v, int ms){
  if(mlir::isa<MemRefType>(v.getType())){
    auto _ty = mlir::cast<MemRefType>(v.getType());
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    v.setType(tA);
  }
}

static Type ModifyMemrefType(Type t, int ms){
  if(mlir::isa<MemRefType>(t)){
    auto _ty = mlir::cast<MemRefType>(t);
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    return tA;
  }
  else{
    return t;
  }
}

static Value stripMemrefTensorRoundTrip(Value v, MemRefType dstTy) {
  Value curr = v;
  for (int depth = 0; depth < 8; ++depth) {
    auto castOp = curr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getInputs().size() != 1) {
      break;
    }

    Value input = castOp.getInputs()[0];
    if (auto inputTy = dyn_cast<MemRefType>(input.getType())) {
      if (inputTy.getShape() == dstTy.getShape() &&
          inputTy.getElementType() == dstTy.getElementType()) {
        return input;
      }
    }
    curr = input;
  }
  return v;
}


// 从v开始，向上追溯其defOp，构建affine_expr表达式
static AffineExpr GetExprOfValue(
  mlir::Value v,  // 待分析的value
  std::map<std::string, AffineExpr>& dims,   // dims 容器
  std::map<int,Value>& arglist)  // 记录affinemap的参数的id与Value
{
  auto defOp = v.getDefiningOp();
  if(defOp == nullptr){
    llvm::outs() << "[def] null\n" ;llvm::outs().flush();
    if(auto blockarg = mlir::dyn_cast<BlockArgument>(v)){
      auto argId = blockarg.getArgNumber();
      auto parentOp = blockarg.getParentRegion()->getParentOp();
      if(auto concreteOp = mlir::dyn_cast<frisk::ParallelOp>(parentOp)){
        if(argId > 2){
          assert(false);
        }
        const char* labels[] = {"bz", "by", "bx"};
        if(dims.find(labels[argId]) == dims.end()){
          auto id = dims.size();
          arglist.insert(std::make_pair(id, blockarg));
          dims[labels[argId]] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims[labels[argId]];
      }
      else if(auto dgmaskOp = mlir::dyn_cast<dg::MaskOp>(parentOp)){
        if(argId > 2){
          assert(false);
        }
        auto v = dgmaskOp.getStart(argId);
        return GetExprOfValue(v, dims, arglist);
      }
      else if(auto affineforOp = mlir::dyn_cast<affine::AffineForOp>(parentOp)){
        if(argId == 0){
          // 为 iterVar
          auto id = dims.size();
          auto newDim = mlir::getAffineDimExpr(id, v.getContext());
          arglist.insert(std::make_pair(id, blockarg));
          dims.insert(std::make_pair(std::string("iv") + std::to_string(id), newDim));
          return newDim;
        }
        else{
          auto v = affineforOp.getBody()->getArgument(argId);
          return GetExprOfValue(v, dims, arglist);
        }
      }
      else{
        assert(false);
      }
    }
    else{
      assert(false);
    }
  }
  else{
    llvm::outs() << "[def] " << defOp->getName().getStringRef() << "\n";llvm::outs().flush();
  }
  if(mlir::isa<arith::AddIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) + GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::SubIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) - GetExprOfValue(rhs, dims, arglist);
  }
  if(mlir::isa<arith::MulIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) * GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::DivUIOp, arith::DivSIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist).floorDiv(GetExprOfValue(rhs, dims, arglist)) ;
  }
  else if(mlir::isa<arith::RemUIOp, arith::RemSIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) % GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp>(defOp)){
    int val = -999 ;
    auto constOp = mlir::dyn_cast<arith::ConstantOp>(defOp);
    if(constOp){
      val = mlir::cast<IntegerAttr>(constOp.getValue()).getInt();
    }
    return getAffineConstantExpr(val, v.getContext());
  }
  else if(mlir::isa<gpu::BlockIdOp>(defOp)){
    auto op = mlir::dyn_cast<gpu::BlockIdOp>(defOp);
    auto d = op.getDimension();
    const char* label[] = {"bx","by","bz"};
    size_t labelId = -1;
    switch (d) {
      case gpu::Dimension::x:
        labelId = 0; break;
      case gpu::Dimension::y:
        labelId = 1; break;
      case gpu::Dimension::z:
        labelId = 2; break;
      default:
        assert(false);
    }
    
    if(dims.find(label[labelId]) == dims.end()){
      auto id = dims.size();
      arglist[id] = op;
      dims[label[labelId]] = mlir::getAffineDimExpr(id, v.getContext());
    }
    return dims[label[labelId]] ;
  }
  else if(mlir::isa<gpu::ThreadIdOp>(defOp)){
    auto op = mlir::dyn_cast<gpu::BlockIdOp>(defOp);
    auto d = op.getDimension();
    AffineExpr ret;
    switch (d) {
      case gpu::Dimension::x:
        if(dims.find("tx") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["tx"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["tx"];
      case gpu::Dimension::y:
        if(dims.find("ty") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["ty"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["ty"];
      case gpu::Dimension::z:
        if(dims.find("tz") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["tz"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["tz"];
      default:
        assert(false);
    }
  }
  else if(auto forOp = mlir::dyn_cast<affine::AffineForOp>(defOp)){
    if(v == forOp.getInductionVar()){
      // 为 iterVar
      auto id = dims.size();
      auto newDim = mlir::getAffineDimExpr(id, v.getContext());
      arglist.insert(std::make_pair(id, v));
      dims.insert(std::make_pair(std::string("iv") + std::to_string(id), newDim));
    }
    else{
      assert(false && "不支持forOp带有返回值的expr推导");
    }
  }
  // not supported op
  assert(false && "not supported op");
}

// 
struct ArgIdViewBuffer {
  frisk::AllocBufferOp shmbuffer = nullptr;
  AffineMap baseLinearOffsetMap;
  AffineMap baseOffsetMap;
  std::vector<Value> baseOffsetMapOperands;
  std::vector<int64_t> blockShape;
  std::vector<int64_t> sourceStrides;
  std::vector<int64_t> blockStrides;
};

// 存放 argId : { arg对应的initView ， arg开辟view时建立的shm buffer }
static std::vector<ArgIdViewBuffer*>  s_argId_bufferInfo;

static std::vector<int64_t> getPhysicalStrides(ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

static std::vector<AffineExpr>
decomposePhysicalOffset(AffineExpr offset, ArrayRef<int64_t> shape,
                        ArrayRef<int64_t> strides, MLIRContext *ctx) {
  std::vector<AffineExpr> indices;
  indices.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) {
      indices.push_back(getAffineConstantExpr(0, ctx));
      continue;
    }

    AffineExpr idx = strides[i] == 1 ? offset : offset.floorDiv(strides[i]);
    if (!ShapedType::isDynamic(shape[i]) && shape[i] > 1) {
      idx = idx % shape[i];
    }
    indices.push_back(idx);
  }
  return indices;
}

static FailureOr<std::vector<int64_t>>
remapBlockStridesToPermutedLayout(ArrayRef<int64_t> blockStrides,
                                  ArrayRef<int64_t> originalShape,
                                  ArrayRef<int64_t> permute,
                                  ArrayRef<int64_t> permutedStrides,
                                  Operation *op, unsigned argId) {
  if (permute.size() != originalShape.size() ||
      permutedStrides.size() != originalShape.size()) {
    return op->emitError("arg_permutes rank mismatch while remapping strides for argument #")
           << argId;
  }

  auto originalStrides = getPhysicalStrides(originalShape);
  std::vector<int64_t> remappedStrides;
  remappedStrides.reserve(blockStrides.size());
  for (int64_t blockStride : blockStrides) {
    int64_t originalDim = -1;
    for (auto indexedStride : llvm::enumerate(originalStrides)) {
      if (indexedStride.value() == blockStride) {
        originalDim = static_cast<int64_t>(indexedStride.index());
        break;
      }
    }
    if (originalDim < 0) {
      return op->emitError("cannot map block pointer stride ")
             << blockStride << " to original argument #" << argId
             << " physical layout";
    }

    int64_t permutedDim = -1;
    for (auto indexedDim : llvm::enumerate(permute)) {
      if (indexedDim.value() == originalDim) {
        permutedDim = static_cast<int64_t>(indexedDim.index());
        break;
      }
    }
    if (permutedDim < 0) {
      return op->emitError("cannot map original dim ")
             << originalDim << " through arg_permutes for argument #"
             << argId;
    }

    remappedStrides.push_back(permutedStrides[permutedDim]);
  }
  return remappedStrides;
}

static FailureOr<MemRefType>
getPermutedMemRefType(MemRefType memrefTy, DenseI64ArrayAttr permuteAttr,
                      Operation *op, unsigned argId) {
  ArrayRef<int64_t> permute = permuteAttr.asArrayRef();
  ArrayRef<int64_t> oldShape = memrefTy.getShape();
  int64_t rank = memrefTy.getRank();
  if (static_cast<int64_t>(permute.size()) != rank) {
    return op->emitError("arg_permutes rank mismatch for argument #")
           << argId << ": expected " << rank << " dims, got "
           << permute.size();
  }

  SmallVector<bool> used(rank, false);
  SmallVector<int64_t> newShape;
  newShape.reserve(rank);
  for (int64_t dim : permute) {
    if (dim < 0 || dim >= rank) {
      return op->emitError("arg_permutes dim out of range for argument #")
             << argId << ": " << dim;
    }
    if (used[dim]) {
      return op->emitError("arg_permutes has duplicated dim for argument #")
             << argId << ": " << dim;
    }
    used[dim] = true;
    newShape.push_back(oldShape[dim]);
  }

  return MemRefType::get(newShape, memrefTy.getElementType(),
                         memrefTy.getLayout(), memrefTy.getMemorySpace());
}

// =============== Op Conversion Patterns =============


struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto gridAttr = dyn_cast_or_null<DenseI64ArrayAttr>(op->getAttr("grid"));
    auto permuteAttr = op->getAttr("arg_permutes");
    auto loc = op->getLoc();
    auto oldFuncType = op.getFunctionType();
    auto converter = getTypeConverter();
    if (!gridAttr)
      return op.emitError("missing or invalid `grid` attribute for kernel"),
             failure();

    llvm::SmallVector<Type> newInputs;
    llvm::SmallVector<Type> newOutputs;
    for (auto ty : oldFuncType.getInputs()) {
      auto newArgTy = converter->convertType(ty);
      if(mlir::isa<MemRefType>(newArgTy)){
        auto mem = mlir::cast<MemRefType>(newArgTy);
        auto newMem = MemRefType::get(mem.getShape(), mem.getElementType(), AffineMap{}, int(frisk::attr::MemorySpace::Global));
        newInputs.push_back(newMem);
      }
      else{
        newInputs.push_back(newArgTy);
      }
    }
    for(auto n : newInputs){
      s_argId_bufferInfo.push_back(nullptr);
    }
    // 1. build new function type
    auto newFuncType = rewriter.getFunctionType(newInputs, newOutputs);
    // 2. convert old region signature, inline it after new frisk.kernel
    TypeConverter::SignatureConversion sc{oldFuncType.getNumInputs()};
    for (int i = 0; i < oldFuncType.getNumInputs(); ++i) {
      sc.addInputs(i, newInputs[i]);
      // sc.addInputs(i, converter->convertType(oldFuncType.getInput(i)));
    }

    rewriter.convertRegionTypes(&op->getRegion(0), *converter, &sc);
    rewriter.applySignatureConversion(&op.getFunctionBody().front(), sc);

    auto newKernelOp = rewriter.create<frisk::KernelOp>(loc, op.getName(), newFuncType);
    newKernelOp->setAttr("grid", gridAttr);
    if (permuteAttr)
      newKernelOp->setAttr("arg_permutes", permuteAttr);
    rewriter.inlineRegionBefore(op->getRegion(0), newKernelOp.getRegion(), newKernelOp.getRegion().end());
    // 3. replace deepgengraph.return with frisk.end
    auto oldReturn = newKernelOp->getRegion(0).front().getOps<deepgengraph::ReturnOp>().begin();
    rewriter.setInsertionPoint(*oldReturn);
    auto newReturn = rewriter.create<frisk::EndOp>(op->getLoc());
    rewriter.replaceOp(*oldReturn, newReturn);
    
    // 4. insert frisk.parallel
    rewriter.setInsertionPointToStart(&newKernelOp->getRegion(0).front());
    auto ranges = gridAttr.asArrayRef();
    auto parallelOp = rewriter.create<frisk::ParallelOp>(loc, ranges, GetKernelConfig()->num_threads);
    auto parallelEntry = parallelOp.addEntryBlock();
    // move all ops expect frisk.end into frisk.parallel
    auto nextOp = parallelOp->getNextNode();
    while (nextOp != nullptr && !isa<frisk::EndOp>(nextOp)) {
      auto *next = nextOp->getNextNode();
      rewriter.moveOpBefore(nextOp, parallelEntry, parallelEntry->end());
      nextOp = next;
    }
    // find frisk.end for frisk.parallel, move it to the block end
    auto innerEndOp = parallelEntry->getOps<frisk::EndOp>().begin();
    rewriter.moveOpBefore(*innerEndOp, parallelEntry, parallelEntry->end());
    // replace gpu.bid with parallel block args
    llvm::SmallVector<gpu::BlockIdOp> bidOps;
    parallelEntry->walk([&](gpu::BlockIdOp bid) { bidOps.push_back(bid); });

    for (auto bidOp : bidOps) {
      int argId = -1;
      switch (bidOp.getDimension()) {
      case gpu::Dimension::x:
        argId = 2;
        break;
      case gpu::Dimension::y:
        argId = 1;
        break;
      case gpu::Dimension::z:
        argId = 0;
        break;
      default:
        assert(false && "unexpected block_id dim");
      }
      rewriter.replaceOp(bidOp, ValueRange{parallelEntry->getArgument(argId)});
    }

    rewriter.replaceOp(op, newKernelOp);

    return success();
  }
};

struct PointerOfConversionPattern : public OpConversionPattern<deepgengraph::triton::PointerOfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::PointerOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 删除
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto blockArg = getKernelArgById(op, argId);
    rewriter.replaceAllUsesWith(op, blockArg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BlockPointerOfConversionPattern
    : public OpConversionPattern<deepgengraph::triton::BlockPointerOfOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_ptr_of -> alloc_buffer[shared]
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockPointerOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto kernelOp = op->getParentOfType<frisk::KernelOp>();
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    // 获取参数列表中对应位置的原始memref 类型, 进而得到permute后的shape
    auto argMemrefType = mlir::cast<MemRefType>( kernelOp.getArgument(argId).getType());
    auto permutedShape = argMemrefType.getShape();
    llvm::outs() << "argId["<<argId<<"] " ;
    llvm::outs() << "permuted : " << permutedShape[0] << "," << permutedShape[1] << "," << permutedShape[2] << "," << permutedShape[3] << ";\n";
    llvm::outs().flush();

    // 根据是否有read属性，建立 allocBufferOp （read：后续有loadOp读取指针指向的数据。 write：后续有storeOp 向指针指向的内存写入数据）
    auto info = new ArgIdViewBuffer{};
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    auto memTy = mlir::dyn_cast<MemRefType>(resTy);
    frisk::AllocBufferOp newOp = nullptr;
    if(op->hasAttr("read")){
      newOp = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), memTy.getShape(), memTy.getElementType(), 16, int64_t(frisk::attr::MemorySpace::Shared));
      info->shmbuffer = newOp;
    }

    // 根据 baseOffset, order, stride, 得到 baseOffset的计算map 以及 mapOperands. 
    std::map<std::string, AffineExpr> dims;
    std::map<int, Value> arglist;
    auto expr_baseOffset= GetExprOfValue(op.getBaseOffset(), dims, arglist);
    std::vector<Value> vr_baseOffset;
    for(auto [k,v] : arglist){
      vr_baseOffset.push_back(v);
    }
    auto stride = op.getStride();

    auto sourceStrides = getPhysicalStrides(permutedShape);
    llvm::outs() << "sourceStrides:" << sourceStrides[0] << "," << sourceStrides[1] << "," << sourceStrides[2] << "," << sourceStrides[3] << "\n";llvm::outs().flush();
    llvm::outs() << "expr_baseOffset:" << expr_baseOffset << "\n";llvm::outs().flush();

    std::vector<int64_t> blockStrides(stride.begin(), stride.end());
    if (auto argPermutes = kernelOp->getAttrOfType<ArrayAttr>("arg_permutes")) {
      if (argId >= static_cast<int64_t>(argPermutes.size())) {
        return op->emitError("arg_permutes missing entry for argument #")
               << argId;
      }
      auto densePermute = mlir::dyn_cast<DenseI64ArrayAttr>(argPermutes[argId]);
      if (!densePermute) {
        return op->emitError("arg_permutes entry for argument #")
               << argId << " must be a dense i64 array attribute";
      }

      SmallVector<int64_t> originalShape(permutedShape.size(),
                                         ShapedType::kDynamic);
      auto permute = densePermute.asArrayRef();
      for (auto indexedDim : llvm::enumerate(permute)) {
        originalShape[indexedDim.value()] = permutedShape[indexedDim.index()];
      }

      auto remappedStrides = remapBlockStridesToPermutedLayout(
          blockStrides, originalShape, permute, sourceStrides,
          op.getOperation(), argId);
      if (failed(remappedStrides)) {
        return failure();
      }
      blockStrides = std::move(*remappedStrides);
    }

    auto resExprArray = decomposePhysicalOffset(expr_baseOffset, permutedShape,
                                                sourceStrides, op->getContext());

    auto baseOffsetMap = AffineMap::get(dims.size(), 0, resExprArray, op->getContext());
    auto baseLinearOffsetMap =
        AffineMap::get(dims.size(), 0, ArrayRef<AffineExpr>{expr_baseOffset},
                       op->getContext());

    // save info
    info->baseLinearOffsetMap = baseLinearOffsetMap;
    info->baseOffsetMap = baseOffsetMap;
    info->baseOffsetMapOperands = vr_baseOffset;
    info->blockShape = op.getBlockShape();
    info->sourceStrides = std::move(sourceStrides);
    info->blockStrides = std::move(blockStrides);
    s_argId_bufferInfo[argId] = info;
    if(newOp){
      // 含read，需要创建buffer存数据
      rewriter.replaceOp(op, newOp);
    }
    else{
      // write，将所有使用op结果的位置替换为 globalMem，之后删除op
      auto kernelOp = op->getParentOfType<frisk::KernelOp>();
      rewriter.replaceAllUsesExcept(op, kernelOp.getArgument(argId), op);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct BlockLoadConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_load -> buffer_view + frisk.copy(view, dstMem)  之后用dstMem替换 blockLoad的结果. 如果有 block_advance, 删除之
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // find info
    auto loc = op->getLoc();
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    if(s_argId_bufferInfo[argId] == nullptr){
      assert(false);
    }
    auto info = s_argId_bufferInfo[argId];

    // 检查该 blockPtr 是否会move
    auto parentKernel = op->getParentOfType<frisk::KernelOp>();
    std::vector<Value> globalBuffers;
    dgt::BlockAdvanceOp ptrAdvance = nullptr;
    if(parentKernel != nullptr){
      for(auto arg : parentKernel.getBody()->getArguments()){
        globalBuffers.push_back(arg);
      }
      parentKernel->walk([&](dgt::BlockAdvanceOp advanceOp){
        auto id = advanceOp->getAttrOfType<IntegerAttr>("argId").getInt();
        if(argId == id){
          ptrAdvance = advanceOp;
        }
      });
    }
    if(!ptrAdvance){
      // 没有 ptr_advance, 索引只依赖于 block_ptr_of 的 baseOffset 索引
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange indices, AffineMap indexMap, ArrayRef<int64_t> ranges);
      auto map = info->baseOffsetMap;
      auto indice = info->baseOffsetMapOperands;
      auto indexExprs = map.getResults();
      // 比较 GM 的rank和newExpr的个数. 保证维度对齐. TODO:此处需要重新考虑 GM permute之后的布局.如何从 block_ptr_of 推断出前序的 affineExpr
      // 本质原因 : asuka block_ptr_of 中没有包含 permute 的信息. <1,4096,32,128> 四维 != attr中的[128, 128] 二维信息
      auto globalMemTy = mlir::cast<MemRefType>(globalBuffers[argId].getType());
      std::vector<AffineExpr> newExprs;
      
      for(int i=0; i < (globalMemTy.getShape().size() - indexExprs.size()); ++i){
        newExprs.push_back(mlir::getAffineConstantExpr(0, op.getContext()));
      }
      for(auto expr : map.getResults()){
        newExprs.push_back(expr);
      }
      auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), newExprs, op->getContext());
      auto view = rewriter.create<frisk::BufferViewOp>(loc, globalBuffers[argId], indice, newMap, info->blockShape);
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst);
      rewriter.create<frisk::CopyOp>(loc, view, info->shmbuffer);
      rewriter.replaceOp(op, info->shmbuffer);
    }
    else{
      // 需要基于 baseOffset + advance 逻辑,进一步计算索引
      // 遍历所有parent for，拿到ivs 和 step ，(ivs / step) 为 当前循环次数
      /**
      for(int i=0;i<3;++i){
        for(int j=0;j< bx ; ++j){
          view = someView(arg0);
          blockAdvance(view, offset = (32,128));
          累计循环次数 = i * bx + j
          // offset 计算 ：
          如果 view 的 初始索引为 [0,0, bx*32, by * 512]
          那么 advance 后 view = [0,0, bx*32 + 32, by * 512 + 128]
          // 表达式构建 ：loop = iv0/step0 + iv1/step1 * ub0 + iv2/step2 * ub0 * ub1 + ...
          // [base_x + loop * offset_x, base_y + loop * offset_y] 
        }
      }
      */
      std::vector<Value> ivs;
      std::vector<AffineExpr > loopCountExprs;
      std::vector<AffineExpr> ubs;
      std::vector<ValueRange> ubsOperands;
      Operation* currOp = ptrAdvance;
      int oldDimCount = info->baseOffsetMap.getNumDims();
      int newdimCount = oldDimCount;  //  从旧 dimCount开始,新增dim
      // 从ptradvance 开始, 递归地遍历 所有父级forOp
      while(currOp != nullptr){
        if(auto parentLoop = currOp->getParentOfType<affine::AffineForOp>()){
          ivs.push_back(parentLoop.getInductionVar());
          auto step = parentLoop.getStepAsInt();
          auto ubMapExpr = parentLoop.getUpperBoundMap().getResult(0);
          auto ubVals = parentLoop.getUpperBoundOperands();
          // 新建dim
          auto newDim = mlir::getAffineDimExpr(newdimCount, op->getContext());
          // iv / step = 当前循环的次数
          newDim = newDim.floorDiv(step);
          loopCountExprs.push_back(newDim);
          ubs.push_back(ubMapExpr);
          ubsOperands.push_back(ubVals);
          
          newdimCount++;
          currOp = parentLoop;
        }
        else{
          break;
        }
      }
      // 累乘 ubs : {u0, u1 * u0, u2*u1*u0, ...}
      for(int i=1;i<ubs.size();++i){
        ubs[i] = ubs[i] * ubs[i-1];
      }
      // 构建 iv 的expr 
      AffineExpr loop_expr = mlir::getAffineConstantExpr(0, op->getContext());
      std::vector<Value> loop_expr_values;

      for(int i=0;i < loopCountExprs.size() ; ++i){
        auto loopIv = loopCountExprs[i];
        loop_expr_values.push_back(ivs[i]);
        if(i-1 >= 0){
          loopIv = loopIv * ubs[i-1];
          for(auto v : ubsOperands[i-1]){
            loop_expr_values.push_back(v);
          }
        }
        // (iv0 / step0) + (iv1 / step1) * ub0 + (iv2 / step2) * (ub0*ub1) + ...
        loop_expr = loop_expr + loopIv;
      }

      auto loc = op->getLoc();
      auto indices = info->baseOffsetMapOperands;
      auto offset = ptrAdvance.getOffsets();
      // affineMap的操作数 value = 原有 + 新收集的ivs
      std::vector<Value> newIndices;
      for(auto v : indices){
        newIndices.push_back(v);
      }
      for(auto v : loop_expr_values){
        newIndices.push_back(v);
      }
      AffineExpr linearOffset = info->baseLinearOffsetMap.getResult(0);
      // block_advance 的 offset 属于 block_ptr 的二维逻辑轴。
      // 先按 permute 后的 GM stride 合成线性 offset，再统一反解为 GM 多维坐标。
      for(int i=0; i < offset.size(); ++i){
        if(offset[i] == 0){
          continue;
        }
        assert(i < info->blockStrides.size() &&
               "block advance offset rank must match block pointer strides");
        linearOffset = linearOffset + (offset[i] * info->blockStrides[i]) * loop_expr;
      }
      auto globalMemTy = mlir::cast<MemRefType>(globalBuffers[argId].getType());
      auto newExprs = decomposePhysicalOffset(linearOffset, globalMemTy.getShape(),
                                              info->sourceStrides, op->getContext());
      // newMap dim增加，symbol不变，expr重建
      auto newMap = AffineMap::get(newdimCount , info->baseOffsetMap.getNumSymbols(), newExprs, op->getContext());
      // newView的indices为newMap的操作数
      llvm::outs() << "newMap=" << newMap << "  newIndices.size=" << newIndices.size() << " indices.size() = " << indices.size() <<"\n";llvm::outs().flush();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange indices, AffineMap indexMap, ArrayRef<int64_t> ranges);
      auto view = rewriter.create<frisk::BufferViewOp>(loc, globalBuffers[argId], newIndices, newMap, info->blockShape);
      rewriter.create<frisk::CopyOp>(loc, view, info->shmbuffer);
      
      rewriter.replaceOp(op, info->shmbuffer);
      rewriter.eraseOp(ptrAdvance);
    }

    return success();
  }
};

struct BlockStoreConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_store %14, %24 :  直接替换
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto src = adaptor.getValue();
    auto temp = mlir::cast<MemRefType>(src.getType());
    auto newSrcTy = MemRefType::get(temp.getShape(), temp.getElementType(), AffineMap{}, int(friskMs::Shared));
    src.setType(newSrcTy);
    auto dst = adaptor.getDstPointer();
    auto dstMemref = mlir::dyn_cast<MemRefType>(dst.getType());
    auto srcMemref = mlir::dyn_cast<MemRefType>(src.getType());

    if(dstMemref.getRank() > srcMemref.getRank() || dstMemref.getShape() != srcMemref.getShape()){
      // 从shm拷贝到global
      auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
      auto& offsetMap = s_argId_bufferInfo[argId]->baseOffsetMap;
      auto& mapOperands = s_argId_bufferInfo[argId]->baseOffsetMapOperands;
      auto newOp = rewriter.create<frisk::CopyOp>(op->getLoc(), src,dst, mapOperands, offsetMap);
      rewriter.replaceOp(op, newOp);
    }
    else{
      auto newOp = rewriter.create<frisk::CopyOp>(op->getLoc(), src, dst);
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

struct BlockAdvanceConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
  using OpConversionPattern::OpConversionPattern;
  // 直接删除
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ZeroOpConversionPattern : public OpConversionPattern<dg::ZeroOp> {
  using OpConversionPattern::OpConversionPattern;


  LogicalResult matchAndRewrite(dg::ZeroOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // %16 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef()[0];

    auto buffer = rewriter.create<frisk::AllocBufferOp>(loc, op.getShape(), op.getElementType(), 16, outMs);
    AppendNameToLoc(buffer);
    mlir::Attribute valueAttr;
    auto eleTy = op.getElementType();
    if(eleTy.isFloat()){
      valueAttr = rewriter.getFloatAttr(eleTy, 0.0);
    }
    else if(eleTy.isInteger()){
      valueAttr = rewriter.getIntegerAttr(eleTy, 0);
    }
    else{
      assert(false);
    }
    auto fillOp = rewriter.create<frisk::FillOp>(loc, buffer, valueAttr);
    AppendNameToLoc(fillOp);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};


struct ConvertOpConversionPattern : public OpConversionPattern<dg::ConvertOp> {
  using OpConversionPattern::OpConversionPattern;
  // %23 = deepgengraph.convert %22, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16> 替换为 allocBuffer + frisk.copy 
  LogicalResult matchAndRewrite(dg::ConvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    llvm::outs() << "enter ConvertOpConversionPattern : " << op << "\n"; llvm::outs().flush();
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef()[0];
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef()[0];
    auto loc = op->getLoc();
    auto operand = adaptor.getOperand();
    AppendMemspaceToMemrefValue( operand , inMs);
    auto dstType = adaptor.getDstType();
    ModifyMemrefType(dstType, outMs);
    
    auto convertedTy = getTypeConverter()->convertType(op.getResult().getType());
    if(mlir::isa<MemRefType>(convertedTy)){
      auto newMemTy = mlir::dyn_cast<MemRefType>(convertedTy);
      auto outerMostFor = getOuterMostOp<affine::AffineForOp>(op);
      frisk::AllocBufferOp allocBuffer {};
      {
        RewriterBase::InsertionGuard ig{rewriter};
        rewriter.setInsertionPoint(outerMostFor);
        allocBuffer = rewriter.create<frisk::AllocBufferOp>(loc, newMemTy.getShape(), newMemTy.getElementType(), 16, outMs);
      }
      auto copyOp = rewriter.create<frisk::CopyOp>(loc, adaptor.getOperand(), allocBuffer);
      
      rewriter.replaceOp(op, allocBuffer);
    }
    else{
      // %16 = "deepgengraph.convert"(%5) <{dst_type = f16}> {inMs = array<i32: 0>, outMs = array<i32: 0>} : (tensor<1xf32>) -> tensor<1xf16>
      // 转换为 arith.truncf %
      auto srcWidth = mlir::cast<FloatType>(adaptor.getOperand().getType()).getWidth();
      auto dstWitdth = mlir::cast<FloatType>(convertedTy).getWidth();
      mlir::Operation* newOp {};
      if(srcWidth > dstWitdth){
        newOp = rewriter.create<arith::TruncFOp>(op->getLoc(), adaptor.getDstType(), adaptor.getOperand());
      }
      else{
        newOp = rewriter.create<arith::ExtFOp>(op->getLoc(), adaptor.getDstType(), adaptor.getOperand());
      }
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

struct ForTypeConversionPattern : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    // Use the already-converted init args from the adaptor as the new iter_args
    SmallVector<Value> newIterArgs(adaptor.getInits().begin(), adaptor.getInits().end());

    bool needConvert = false;
    for(auto [oldIter, newIter] : llvm::zip(op.getInits(), adaptor.getInits())){
      if(oldIter.getType() != newIter.getType()){
        needConvert = true;
        break;
      }
    }
    if (!needConvert) return failure();

    // 2. 用转换后的 newIterArgs 创建新 ForOp
    auto newForOp = rewriter.create<affine::AffineForOp>(
        op.getLoc(),
        adaptor.getLowerBoundOperands(), op.getLowerBoundMap(),
        adaptor.getUpperBoundOperands(), op.getUpperBoundMap(),
        op.getStepAsInt(),
        newIterArgs);  // ✅ 关键：传入转换后的 args

    // 3. 构建 SignatureConversion
    //    旧 Block 参数: [iv: index, arg1: BlockPtrType, arg2: ...]
    //    新 Block 参数: [iv: index, arg1: memref, arg2: ...]
    TypeConverter::SignatureConversion sigConv(op.getBody()->getNumArguments());

    // IV 不变
    sigConv.addInputs(0, rewriter.getIndexType());

    // iter_args: 用 newIterArgs 的类型替换旧参数类型
    for (unsigned i = 0; i < newIterArgs.size(); ++i) {
      sigConv.addInputs(i + 1, newIterArgs[i].getType());
    }

    // 4. 移动旧 Region 到新 ForOp，并应用参数类型转换
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), 
                                 newForOp.getRegion(), 
                                 newForOp.getRegion().end());
    
    // applySignatureConversion 会在 block 入口插入 cast 处理类型不匹配
    if (failed(rewriter.convertRegionTypes(&newForOp.getRegion(), 
                                            *getTypeConverter(), &sigConv))) {
      return failure();
    }

    // 5. 替换旧 Op 的结果
    rewriter.modifyOpInPlace(newForOp, [&](){
      newForOp->setAttr(IN_MEMSPACE, op->getAttr(IN_MEMSPACE));
      newForOp->setAttr(OUT_MEMSPACE, op->getAttr(OUT_MEMSPACE));
    });
    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

struct YieldTypeConversionPattern : public OpConversionPattern<affine::AffineYieldOp> {
  using OpConversionPattern<affine::AffineYieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineYieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    bool needConvert = false;
    for(auto [oldV, newV] : llvm::zip(op->getOperands(), adaptor.getOperands())){
      if(oldV.getType() != newV.getType()){
        needConvert = true;
        break;
      }
    }
    if(!needConvert){
      return failure();
    }
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(op, adaptor.getOperands());
    return success();
  }
};

// %cst = arith.constant dense<0.127531052> : tensor<1xf32> loc(#loc) 转换到 arith.constant 0.127531052 : f32
struct ArithSingleElementTensorConversionPattern : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto retType = op.getResult().getType();
    if(mlir::isa<TensorType>(retType)){
      auto tensorTy = mlir::dyn_cast<TensorType>(retType);
      auto shape = tensorTy.getShape();
      int len = 1;
      for(auto s : shape){
        len *= s;
      }
      if(len > 1){
        return failure();  // 暂不支持以长度大于1的denseElements数组进行赋值
      }
      auto val = mlir::cast<DenseFPElementsAttr>(op.getValue());
      float v = 0;
      if(!val){
        return failure();
      }
      auto vals = val.getValues<APFloat>();
      for(auto it : vals){
        v = it.convertToFloat();
      }
      auto constVal = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getF32FloatAttr(v));
      rewriter.replaceOp(op, constVal);
      return success();
    }
    else{
      return failure();
    }
  }
};

// 清空affineFor的Inits和 yield 返回值
struct AffineForEmptyInitsAndYieldPattern : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor, 
                                ConversionPatternRewriter &rewriter) const override 
  {
    // 如果本来就没有 inits，说明不需要这个 pattern 处理
    if (op.getInits().empty()) {
      return failure();
    }

    auto loc = op->getLoc();
    std::vector<Value> newResults;
    int initIdx = 0;
    for(auto initVal : op.getInits()){
      auto defOp = initVal.getDefiningOp();
      rewriter.modifyOpInPlace(defOp, [&](){
        defOp->setAttr("initIdx", rewriter.getIndexAttr(initIdx));
      });
      // 标记initVal的所有消费者
      for(auto use : initVal.getUsers()){
        use->setAttr("initIdx", rewriter.getIndexAttr(initIdx));
      }
      // 标记affineYield的所有消费者
      for(auto u : op->getResults()[initIdx].getUsers()){
        u->setAttr("initIdx", rewriter.getIndexAttr(initIdx));
      }
      initIdx++;
      newResults.push_back(defOp->getResult(0));
    }
    llvm::outs() << "op.getNumIterOperands() = " << op.getNumIterOperands() << "\n";llvm::outs().flush();
    for(int i=0;i<op.getNumIterOperands();++i){
      auto to = newResults[i];
      auto from = op.getRegionIterArgs()[i];
      rewriter.replaceAllUsesExcept(from, to, op);
    }

    auto vr = ValueRange{};

    // 1. 创建新的 AffineForOp，并为其生成带有正确参数（仅感应变量 Index）的空 Body 
    auto newForOp = rewriter.create<affine::AffineForOp>(
      loc, 
      op.getLowerBoundOperands(), op.getLowerBoundMap(),
      op.getUpperBoundOperands(), op.getUpperBoundMap(),
      op.getStepAsInt(),
      /*iterArgs=*/vr, // 清空最后的 iterArgs
      [&](OpBuilder &b, Location nestedLoc, Value iv, ValueRange args) {
        // 此时新 Block 的参数只有 iv（索引变量）
      }
    );

    // 2. 将旧循环体中的所有操作克隆/移动到新循环体的末尾
    // 注意：此时 newForOp 已经拥有一个合法的、带有一个 iv 参数的 Block
    Block *oldBlock = op.getBody();
    Block *newBlock = newForOp.getBody();

    // 3. 设置参数映射：旧循环体的第一个参数（IV）映射到新循环体的 IV
    // 旧循环体的后续 iter_args 参数在最终结果中会被废弃（因为我们要删掉它们）
    IRMapping mapping;
    mapping.map(oldBlock->getArgument(0), newBlock->getArgument(0));

    // 4. 将旧 Block 中的操作（除了最后的 Terminator 以外）全部克隆到新 Block 中
    rewriter.setInsertionPointToStart(newBlock);
    for (auto &nestedOp : oldBlock->without_terminator()) {
      rewriter.clone(nestedOp, mapping);
    }

    // 5. 单独处理旧的 Terminator (AffineYieldOp)。
    // 清空 iter_args/yield 后，原来的 loop-carried memref 结果需要显式写回
    // 对应 init buffer，才能保留 SSA iter_arg 的累加语义。
    auto oldYieldOp = mlir::cast<affine::AffineYieldOp>(oldBlock->getTerminator());
    for (auto [idx, initVal] : llvm::enumerate(op.getInits())) {
      auto initAlloc = initVal.getDefiningOp<frisk::AllocBufferOp>();
      if (!initAlloc) {
        continue;
      }
      if (idx >= oldYieldOp.getNumOperands()) {
        continue;
      }
      auto dstTy = dyn_cast<MemRefType>(initVal.getType());
      auto src = mapping.lookupOrDefault(oldYieldOp.getOperand(idx));
      if (dstTy) {
        src = stripMemrefTensorRoundTrip(src, dstTy);
      }
      auto srcTy = dyn_cast<MemRefType>(src.getType());
      if (!dstTy || !srcTy || dstTy.getShape() != srcTy.getShape()) {
        continue;
      }
      rewriter.create<frisk::CopyOp>(oldYieldOp.getLoc(), src, initVal);
    }
    rewriter.create<affine::AffineYieldOp>(oldYieldOp.getLoc());

    // 6. 用新 Op 替代旧 Op，并返回成功
    
    rewriter.replaceAllUsesWith(op->getResults(), newResults);
    rewriter.eraseOp(op);
    return success();
  }
};


// scf.for -> affine.for
struct SCFForToAffineFor : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    std::map<std::string, AffineExpr> dims_lb; 
    std::map<int, Value> arglist_lb;
    std::map<std::string, AffineExpr> dims_ub; 
    std::map<int, Value> arglist_ub;
    
    auto lbexpr = GetExprOfValue(op.getLowerBound(), dims_lb, arglist_lb);
    auto ubexpr = GetExprOfValue(op.getUpperBound(), dims_ub, arglist_ub);
    
    std::vector<Value> lbvr, ubvr;
    for(int i=0;i<arglist_lb.size();++i){
      lbvr.push_back(arglist_lb[i]);
    }
    for(int i=0;i<arglist_ub.size();++i){
      ubvr.push_back(arglist_ub[i]);
    }
    
    auto lbMap = AffineMap::get(dims_lb.size(), 0, lbexpr);
    auto ubMap = AffineMap::get(dims_ub.size(), 0, ubexpr);

    int stepNum;
    auto stepOp = op.getStep().getDefiningOp<arith::ConstantOp>();
    if(stepOp){
      stepNum = mlir::dyn_cast<IntegerAttr>(stepOp.getValue()).getInt();
    }
    
    auto affineFor = rewriter.create<affine::AffineForOp>(op->getLoc(), lbvr, lbMap, ubvr, ubMap, stepNum, op.getInitArgs());
        
    rewriter.inlineRegionBefore(op.getRegion(), affineFor.getRegion(), affineFor.getRegion().end());
    Block* contentBlock = &affineFor->getRegion(0).back();
    Block* entryBlock = &affineFor->getRegion(0).front();
    rewriter.mergeBlocks(contentBlock, entryBlock, entryBlock->getArguments());
    // 5. 对移入的 Block 执行“签名转换 (Signature Conversion)”
    // 这一步是让 MLIR 框架安全地将 Block 参数从 block_ptr 转换成 memref，
    // 并且会自动在内部插入 "unrealized_conversion_cast"，保证内部尚未被转换的 block_load 不会因为类型校验崩溃！
    TypeConverter::SignatureConversion sigConversion(affineFor.getBody()->getNumArguments());
    
    // 第 0 个参数是归纳变量 (Induction Variable)，保持为 index 类型
    sigConversion.addInputs(0, rewriter.getIndexType());
    
    // 剩下的参数是 iter_args，转换为 adaptor 中对应的已转换类型
    for (auto [idx, arg] : llvm::enumerate(op.getInitArgs())) {
      sigConversion.addInputs(idx + 1, arg.getType());
    }
    
    // 应用签名转换
    
    rewriter.applySignatureConversion(&affineFor.getRegion().front(), sigConversion, nullptr);

    // 6. 替换 Op (Yield 的替换可以交给独立的 SCFYieldTypeConversionPattern 处理)
    auto oldTerm = affineFor.getBody()->getTerminator();
    rewriter.setInsertionPoint(oldTerm);
    auto newTerm = rewriter.create<affine::AffineYieldOp>(op->getLoc(), oldTerm->getOperands());
    rewriter.replaceOp(oldTerm, newTerm);
    rewriter.replaceOp(op, affineFor);
    return success();
  }
};

struct MatmulOpConversionPattern : public OpConversionPattern<dg::PreciseDotOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(dg::PreciseDotOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override 
  {
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef();
    auto memA = adaptor.getLhs();
    AppendMemspaceToMemrefValue(memA, inMs[0]);
    auto memB = adaptor.getRhs();
    AppendMemspaceToMemrefValue(memB, inMs[1]);
    auto shapeA = mlir::cast<MemRefType>(memA.getType()).getShape();
    auto shapeB = mlir::cast<MemRefType>(memB.getType()).getShape();
    int sizeM = shapeA[0];
    int sizeN = shapeB[1];
    int sizeK = shapeB[0];

    std::vector<int64_t> cshape = {sizeM, sizeN};

    // 找到父级最外层的forOp(如果没有,就直接在前面插入)
    mlir::Operation* currOp = getOuterMostOp<affine::AffineForOp>(op);
    frisk::AllocBufferOp memC {} ;
    {
      RewriterBase::InsertionGuard ig{rewriter};
      rewriter.setInsertionPoint(currOp);
      memC = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), cshape, op.getAccType(), 16, outMs[0]);
    }
    if(CalcOpToFriskOption::useTensorCore){
      // tensorcore 计算 gemm
      auto friskGEMM = rewriter.create<frisk::GemmOp>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs(), memC, false,false);
    }
    else{
      // cudacore 计算 gemm
      std::vector<int64_t> ranges = {sizeM, sizeN};
      auto block = rewriter.create<frisk::BlockOp>(op->getLoc(), ranges, nullptr);
      auto loc = block->getLoc();
      RewriterBase::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToStart(block.getBody(0));
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lowerBound, Value upperBound, Value step, ValueRange initArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> odsArg4 = nullptr);
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc,0);
      auto step_one = rewriter.create<arith::ConstantIndexOp>(loc,1);
      auto k = rewriter.create<arith::ConstantIndexOp>(loc, sizeK);

      auto forOp = rewriter.create<affine::AffineForOp>(block->getLoc(), 0, sizeK, 1);
      rewriter.setInsertionPointToStart(forOp.getBody(0));
      auto iter_k = forOp.getInductionVar();
      auto i = block.getBody(0)->getArgument(0);
      auto j = block.getBody(0)->getArgument(1);
      std::vector<Value> indices = {i,j,iter_k};

      // {i,j,k} : [i,k] [k,j] [i,j]
      auto ctx = op->getContext();
      auto dimI = mlir::getAffineDimExpr(0, ctx);
      auto dimJ = mlir::getAffineDimExpr(1, ctx);
      auto dimK = mlir::getAffineDimExpr(2, ctx);
      auto affineMapA= AffineMap::get(3, 0, {dimI, dimK}, ctx); 
      auto affineMapB= AffineMap::get(3, 0, {dimK, dimJ}, ctx); 
      auto affineMapC= AffineMap::get(3, 0, {dimI, dimJ}, ctx); 
      auto a = rewriter.create<affine::AffineLoadOp>(loc, memA, affineMapA, indices);
      auto b = rewriter.create<affine::AffineLoadOp>(loc, memB, affineMapB, indices);
      auto acc = rewriter.create<affine::AffineLoadOp>(loc, memC, affineMapC, indices);

      Value prod = rewriter.create<arith::MulFOp>(loc, a, b);
      if (prod.getType() != acc.getType()) {
        if (!isa<FloatType>(prod.getType()) || !isa<FloatType>(acc.getType()))
          return failure();
        auto prodFloatTy = cast<FloatType>(prod.getType());
        auto accFloatTy = cast<FloatType>(acc.getType());
        if (prodFloatTy.getWidth() < accFloatTy.getWidth()) {
          prod = rewriter.create<arith::ExtFOp>(loc, acc.getType(), prod);
        } else if (prodFloatTy.getWidth() > accFloatTy.getWidth()) {
          prod = rewriter.create<arith::TruncFOp>(loc, acc.getType(), prod);
        }
      }
      auto added = rewriter.create<arith::AddFOp>(loc, prod, acc);
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, AffineMap map, ValueRange mapOperands);
      rewriter.create<affine::AffineStoreOp>(loc, added, memC, affineMapC, indices);
    }
    rewriter.replaceOp(op, memC);
    return success();
  }
};

struct BinaryOpConversionPattern : public OpInterfaceConversionPattern<dg::BroadcastableBinaryOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  
  virtual LogicalResult
  matchAndRewrite(dg::BroadcastableBinaryOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (operands.size() != 2){
      return failure();
    }
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef();
    Value memLhs = operands[0];
    AppendMemspaceToMemrefValue(memLhs, inMs[0]);
    Value memRhs = operands[1];
    AppendMemspaceToMemrefValue(memRhs, inMs[1]);

    auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType)
      return failure();
    auto resultShape = resultTensorType.getShape();

    // 找到父级最外层的forOp(如果没有,就直接在前面插入), 插入结果buffer的alloc
    mlir::Operation* currOp = getOuterMostOp<affine::AffineForOp>(op);
    frisk::AllocBufferOp alloc {} ;
    {
      RewriterBase::InsertionGuard ig{rewriter};
      rewriter.setInsertionPoint(currOp);
      alloc = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), resultShape, resultTensorType.getElementType(), 16, outMs[0]);
    }
    std::vector<int64_t> ranges(resultShape.begin(), resultShape.end());
    auto blockOp = rewriter.create<frisk::BlockOp>(op->getLoc(), ranges, nullptr);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockOp.getBody(0));
      SmallVector<Value, 4> indices(blockOp.getBody(0)->getArguments().begin(),
                                    blockOp.getBody(0)->getArguments().end());
      auto zero = rewriter.create<arith::ConstantIndexOp>(blockOp->getLoc(), 0);
      auto buildOperandIndices = [&](Value mem, Value originalTensorVal) -> FailureOr<SmallVector<Value, 4>> {
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!memTy)
          return failure();
        auto srcTy = dyn_cast<RankedTensorType>(originalTensorVal.getType());
        if (!srcTy)
          return failure();
        int64_t operandRank = memTy.getRank();
        int64_t resultRank = static_cast<int64_t>(indices.size());
        if (operandRank > resultRank)
          return failure();
        if (srcTy.getRank() != operandRank)
          return failure();

        int64_t offset = resultRank - operandRank;
        SmallVector<Value, 4> operandIndices;
        operandIndices.reserve(operandRank);
        for (int64_t i = 0; i < operandRank; ++i) {
          int64_t dim = offset + i;
          // Broadcasted dimensions always read index 0 from the source tensor.
          if (srcTy.getShape()[i] == 1) {
            operandIndices.push_back(zero);
          } else {
            operandIndices.push_back(indices[dim]);
          }
        }
        return operandIndices;
      };
      Value lhs {}, rhs {};
      if(mlir::isa<MemRefType>(memLhs.getType())){
        auto lhsIndicesOr = buildOperandIndices(memLhs, op.getLhs());
        if (failed(lhsIndicesOr)){
          return failure();
        }
        lhs = rewriter.create<affine::AffineLoadOp>(blockOp->getLoc(), memLhs, *lhsIndicesOr);
      }
      else{
        lhs = memLhs;
      }
      if(mlir::isa<MemRefType>(memRhs.getType())){
        auto rhsIndicesOr = buildOperandIndices(memRhs, op.getRhs());
        if (failed(rhsIndicesOr)){
          return failure();
        }
        rhs = rewriter.create<affine::AffineLoadOp>(blockOp->getLoc(), memRhs, *rhsIndicesOr);
      }
      else{
        rhs = memRhs;
      }

      Value ret;
      Type lhsType = lhs.getType();
      if (isa<dg::AddOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::AddFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::AddIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::SubOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::SubFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::SubIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::MulOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::MulFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::MulIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::DivOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::DivFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::DivSIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::PowOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<math::PowFOp>(blockOp->getLoc(), lhs, rhs);
        else if (isa<IntegerType>(lhsType))
          ret = rewriter.create<math::IPowIOp>(blockOp->getLoc(), lhs, rhs);
        else
          return failure();
      } else if (auto cmpOp = dyn_cast<dg::CmpOp>(op.getOperation())) {
        Value pred;
        if (isa<FloatType>(lhsType)) {
          arith::CmpFPredicate fpred =
              cmpOp.getCmpType() == dg::CmpType::GT ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OGE;
          pred = rewriter.create<arith::CmpFOp>(blockOp->getLoc(), fpred, lhs, rhs);
        } else if (isa<IntegerType, IndexType>(lhsType)) {
          arith::CmpIPredicate ipred =
              cmpOp.getCmpType() == dg::CmpType::GT ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sge;
          pred = rewriter.create<arith::CmpIOp>(blockOp->getLoc(), ipred, lhs, rhs);
        } else {
          return failure();
        }

        Type outElemTy = resultTensorType.getElementType();
        if (pred.getType() == outElemTy) {
          ret = pred;
        } else if (isa<IntegerType>(outElemTy)) {
          ret = rewriter.create<arith::ExtUIOp>(blockOp->getLoc(), outElemTy, pred);
        } else {
          return failure();
        }
      } else {
        return failure();
      }

      rewriter.create<affine::AffineStoreOp>(blockOp->getLoc(), ret, alloc, indices);
    }
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

struct Exp2OpConversionPattern : public OpConversionPattern<dg::Exp2Op> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dg::Exp2Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
  {
    auto loc = op->getLoc();
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef();
    auto operandType = mlir::dyn_cast<MemRefType>(adaptor.getOperand().getType());

    // 找到父级最外层的forOp(如果没有,就直接在前面插入)
    mlir::Operation* currOp = getOuterMostOp<affine::AffineForOp>(op);
    frisk::AllocBufferOp buffer {};
    {
      RewriterBase::InsertionGuard ig{rewriter};
      rewriter.setInsertionPoint(currOp);
      buffer = rewriter.create<frisk::AllocBufferOp>(loc, operandType.getShape(), operandType.getElementType(), 16, outMs[0]);
    }

    auto blockOp = rewriter.create<frisk::BlockOp>(loc, operandType.getShape(), nullptr);
    {
      RewriterBase::InsertionGuard g{rewriter};
      rewriter.setInsertionPointToStart(blockOp.getBody(0));
      std::vector<Value> indices = {blockOp.getBody(0)->getArguments().begin(), blockOp.getBody(0)->getArguments().end()};
      auto operand = adaptor.getOperand();
      AppendMemspaceToMemrefValue(operand, inMs[0]);
      auto val = rewriter.create<affine::AffineLoadOp>(loc, operand, indices);
      auto ret = rewriter.create<math::Exp2Op>(loc, val);
      auto store = rewriter.create<affine::AffineStoreOp>(loc, ret, buffer, indices);
    }
    rewriter.replaceOp(op, buffer);
    return success();
  }
};

struct ReduceOpConversionPattern : public OpConversionPattern<dg::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dg::ReduceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
  {
    // %42 = deepgengraph.reduce(%40, init = %32), dim = 1, op =  ADD, keep_dim = true : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef();
    auto outMemTy = mlir::dyn_cast<MemRefType>( getTypeConverter()->convertType(op.getType()));
    auto inMemTy = mlir::dyn_cast<MemRefType>( adaptor.getOperand().getType());
    // 找到父级最外层的forOp(如果没有,就直接在前面插入)
    mlir::Operation* currOp = getOuterMostOp<affine::AffineForOp>(op);
    frisk::AllocBufferOp buffer {};
    {
      RewriterBase::InsertionGuard ig{rewriter};
      rewriter.setInsertionPoint(currOp);
      buffer = rewriter.create<frisk::AllocBufferOp>(loc, outMemTy.getShape(), outMemTy.getElementType(), 16, outMs[0]);
    }
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::StringAttr kind, ::mlir::IntegerAttr dim);
    std::string kind;
    switch (op.getReduceType()) {
      case dg::ReduceType::ADD: kind = "add";break;
      case dg::ReduceType::MUL: kind = "mul";break;
      case dg::ReduceType::ANY: kind = "any";break;
      default: assert(false); break;
    }
    auto operand = adaptor.getOperand();
    AppendMemspaceToMemrefValue(operand, inMs[0]);
    auto reduce = rewriter.create<frisk::ReduceOp>(loc, operand, buffer, rewriter.getStringAttr(kind), op.getReduceDimension());

    if (auto init = adaptor.getInit()) {
      if (op.getReduceType() != dg::ReduceType::ADD &&
          op.getReduceType() != dg::ReduceType::MUL) {
        return failure();
      }
      AppendMemspaceToMemrefValue(init, inMs[1]);
      auto initMemTy = dyn_cast<MemRefType>(init.getType());
      if (!initMemTy || initMemTy.getShape() != outMemTy.getShape()) {
        return failure();
      }

      auto block = rewriter.create<frisk::BlockOp>(loc, outMemTy.getShape(), nullptr);
      {
        RewriterBase::InsertionGuard guard{rewriter};
        rewriter.setInsertionPointToStart(block.getBody(0));
        SmallVector<Value, 4> indices(block.getBody(0)->getArguments().begin(),
                                      block.getBody(0)->getArguments().end());
        auto reducedVal = rewriter.create<affine::AffineLoadOp>(loc, buffer, indices);
        auto initVal = rewriter.create<affine::AffineLoadOp>(loc, init, indices);
        Value combined;
        if (op.getReduceType() == dg::ReduceType::ADD) {
          if (isa<FloatType>(outMemTy.getElementType())) {
            combined = rewriter.create<arith::AddFOp>(loc, initVal, reducedVal);
          } else {
            combined = rewriter.create<arith::AddIOp>(loc, initVal, reducedVal);
          }
        } else {
          if (isa<FloatType>(outMemTy.getElementType())) {
            combined = rewriter.create<arith::MulFOp>(loc, initVal, reducedVal);
          } else {
            combined = rewriter.create<arith::MulIOp>(loc, initVal, reducedVal);
          }
        }
        rewriter.create<affine::AffineStoreOp>(loc, combined, buffer, indices);
      }
    }

    SmallVector<UnrealizedConversionCastOp, 4> memrefCastUsers;
    for (Operation *user : op.getResult().getUsers()) {
      auto castOp = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!castOp || castOp->getNumResults() != 1) {
        continue;
      }
      auto castMemTy = dyn_cast<MemRefType>(castOp.getResult(0).getType());
      if (!castMemTy || castMemTy.getShape() != outMemTy.getShape() ||
          castMemTy.getElementType() != outMemTy.getElementType()) {
        continue;
      }
      memrefCastUsers.push_back(castOp);
    }
    for (auto castOp : memrefCastUsers) {
      rewriter.replaceAllUsesWith(castOp.getResult(0), buffer.getResult());
      rewriter.eraseOp(castOp);
    }

    rewriter.replaceOp(op, buffer);
    return success();
  }
};


/*
  %27 = deepgengraph.mask starts = [%8, %arg4], sizes = [128, 128], type = f32 {
  ^bb0(%arg9: index, %arg10: index):
    %36 = arith.addi %arg9, %c1 : index
    %37 = arith.cmpi ule, %36, %arg10 : index
    %38 = scf.if %37 -> (f32) {
      scf.yield %cst_0 : f32
    } else {
      scf.yield %cst : f32
    }
    deepgengraph.mask_yield %38 : f32
  } {inMs = array<i32>, outMs = array<i32: 0>} : (index, index) -> tensor<128x128xf32>
  将yield换为 affine.store, starts 解析为affineExpr
*/
struct MaskOpConversionPattern : public OpConversionPattern<dg::MaskOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dg::MaskOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
  {
    auto loc = op->getLoc();
    auto inMs = getOpInputMemspaceAttr(op).asArrayRef();
    auto outMs = getOpOutputMemspaceAttr(op).asArrayRef();
    frisk::AllocBufferOp buffer = nullptr;
    {
      RewriterBase::InsertionGuard ig{rewriter};
      auto outerMostFor = getOuterMostOp<affine::AffineForOp>(op);
      rewriter.setInsertionPoint(outerMostFor);
      buffer = rewriter.create<frisk::AllocBufferOp>(loc, op.getSizes(), op.getElementType(), 16, outMs[0]);
    }
    auto starts = op.getStarts();
    
    auto newOp = rewriter.create<frisk::BlockOp>(loc, op.getSizes(), nullptr);
    auto *newBody = newOp.getBody(0);
    auto ivs = newBody->getArguments();
    if (starts.size() != ivs.size()){
      return failure();
    }
    
    // starts 转为 affineExpr，与 blockOp的arg结合，构成 shiftedIndices
    rewriter.setInsertionPointToStart(newBody);
    SmallVector<Value, 2> shiftedIndices;
    for(int i=0;i<starts.size();++i){
      std::map<std::string, AffineExpr> dims{}; std::map<int, Value> arglist {};
      AffineExpr indiceExpr = GetExprOfValue(starts[i], dims, arglist);
      int dimCount = dims.size();
      indiceExpr = indiceExpr + getAffineDimExpr(dimCount, op->getContext());
      arglist[dimCount] = ivs[i];
      std::vector<AffineExpr> exprs = {indiceExpr};

      SmallVector<Value,4> mapOperands {};
      for(int i=0;i<arglist.size();++i){
        mapOperands.push_back(arglist.at(i));
      }
      auto newIndex = rewriter.create<affine::AffineApplyOp>(op->getLoc(), exprs, mapOperands);
      shiftedIndices.push_back(newIndex);
    }

    // Replace source block arguments at inline time, avoiding RAUW on IVs.
    // 用全局的shiftIndice 替换原有的blockArg
    rewriter.inlineBlockBefore(op.getBody(0), newBody, newBody->getTerminator()->getIterator(), shiftedIndices);

    // 查找mask 内的scf.if else 语句块，获取true false两个值
    std::vector<scf::IfOp> ifOPs{};
    newBody->walk([&](scf::IfOp ifOp){
      ifOPs.push_back(ifOp);
    });
    // 替换 scf.if else 为 arith.select
    for(auto ifOp : ifOPs){
      mlir::Value cond{};
      mlir::Value thenYield {};
      mlir::Value elseYield {};
      cond = ifOp.getCondition();
      ifOp.getThenRegion().walk([&](scf::YieldOp yield){
        thenYield = yield->getOperand(0);
      });
      ifOp.getElseRegion().walk([&](scf::YieldOp yield){
        elseYield = yield->getOperand(0);
      });
      rewriter.setInsertionPoint(ifOp);
      auto select = rewriter.create<arith::SelectOp>(op->getLoc(), cond, thenYield, elseYield);
      rewriter.replaceOp(ifOp, select);
    }

    SmallVector<dg::MaskYieldOp, 2> yields;
    newOp->walk([&](dg::MaskYieldOp yield) { yields.push_back(yield); });
    for (dg::MaskYieldOp yield : yields) {
      RewriterBase::InsertionGuard guard{rewriter};
      rewriter.setInsertionPoint(yield);
      if(outMs[0] == int(friskMs::Local)){
        // 对maskOp，若 dst为local，只需要考虑线程自己持有的数据即可。不需要全局的shiftIndice
        rewriter.create<affine::AffineStoreOp>(loc, yield->getOperand(0), buffer, newBody->getArguments());
      }
      else{
        assert(false && "dg::maskOp 的dst只能是local!检查前面的推断代码是否有错");
      }
      rewriter.eraseOp(yield);
    }

    rewriter.replaceOp(op, buffer);
    return success();
  }
};

}  // namespace end
// =================== Pass Implement ===============

class ConvertMemAndCalcOpToFrisk : public impl::MemAndCalcOpToFriskBase<ConvertMemAndCalcOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();
    
    // ===== 1. Lower mem相关的op ==========
    {
      TypeConverter tc;
      // typeconversion rules :
      // tensor -> memref ; dgt.ptr -> memref ; dgt.block_ptr -> memref
      tc.addConversion([](Type type) { return type; });
      tc.addConversion([](deepgengraph::triton::PointerType ptrType) { 
        auto tensorTy = ptrType.getPointeeType();
        return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{});
      });
      tc.addConversion(
      [](deepgengraph::triton::BlockPointerType blockPtrType) { 
        auto tensorTy = blockPtrType.getPointeeType();
        return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{});
      });
      tc.addConversion([](TensorType ty) -> Type {
        int64_t len = 1;
        for(auto s : ty.getShape()){
          len *= s;
        }
        if(len > 1){
          return MemRefType::get(ty.getShape(), ty.getElementType(), AffineMap{});
        }
        else{
          return ty.getElementType();
        }
      });
      addMaterializations(tc);
      ConversionTarget target(*ctx);
      target.addLegalDialect<FriskDialect, memref::MemRefDialect, func::FuncDialect, deepgengraph::DeepgengraphDialect,
        deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, scf::SCFDialect, affine::AffineDialect,
        tensor::TensorDialect>();
  
      // stage 1 : 转化指针定义op -> memref buffer
      ConversionTarget t0 = target;
      t0.addIllegalOp<dgt::PointerOfOp, dgt::BlockPointerOfOp>();
  
      RewritePatternSet ps0(ctx);
      ps0.add<PointerOfConversionPattern, BlockPointerOfConversionPattern>(tc, ctx);
      applyPartialConversion(op, t0, std::move(ps0));
      
      // stage 2 : 指针读写op -> memref 读写
      RewritePatternSet ps1(ctx);
      ps1.add<BlockLoadConversionPattern,
        BlockStoreConversionPattern,ZeroOpConversionPattern ,ConvertOpConversionPattern,
        ForTypeConversionPattern, YieldTypeConversionPattern
      >(tc, ctx);
      ConversionTarget t1 = target;
      t1.addIllegalOp<dgt::PointerOfOp, dgt::BlockPointerOfOp,
        dgt::BlockLoadOp, dgt::BlockStoreOp,
        dgt::TensorFromOp, dg::ZeroOp, dg::ConvertOp,
        dgt::BlockAdvanceOp >();
      t1.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp forOp) {
        for (Value initArg : forOp.getInits()) {
          if (isTritonPointerLike(initArg.getType())){
            return false;
          }
        }
        for (Type resultType : forOp.getResultTypes()) {
          if (isTritonPointerLike(resultType)){
            return false;
          }
        }
        return true;
      });
  
      t1.addDynamicallyLegalOp<affine::AffineYieldOp>([](affine::AffineYieldOp yieldOp) {
        for (Value operand : yieldOp.getOperands()) {
          if (isTritonPointerLike(operand.getType())){
            return false;
          }
        }
        return true;
      });
  
      applyPartialConversion(op, t1, std::move(ps1));
  
      // stage 3 ：constant 分配的tensor 改为 分配memref
      ConversionTarget t2(*ctx);
      t2.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op){
        return !mlir::isa<TensorType>(op.getResult().getType());
      });
      t2.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});
      RewritePatternSet p2(ctx);
      p2.add<ArithSingleElementTensorConversionPattern>(tc,ctx);
      applyPartialConversion(op, t2, std::move(p2));
      
      llvm::outs() << " ---- before AffineForEmptyInitsAndYieldPattern :\n" << getOperation() << "\n"; llvm::outs().flush();
      // stage 4 : 删除 affineFor 的 initArgs 和 yield
      ConversionTarget t3(*ctx);
      t3.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp op){
        return op.getInits().empty();
      });
      t3.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});
      RewritePatternSet p3(ctx);
      p3.add<AffineForEmptyInitsAndYieldPattern>(tc,ctx);
      applyPartialConversion(op, t3, std::move(p3));
      // 改为在Pass中分析，直接给op setattr。在pattern使用前
    }
    llvm::outs() << " ---- after lower memref Op :\n" << getOperation() << "\n"; llvm::outs().flush();

    // ========== 2. lower calcuate op
    {
      TypeConverter tc;
      tc.addConversion([](Type type) -> std::optional<Type> {
        auto memSpaceInt = int(frisk::attr::MemorySpace::Local);
        if (auto rankedTensorTy = dyn_cast<RankedTensorType>(type)) {
          int64_t len = 1;
          for(auto s : rankedTensorTy.getShape()){
            len *= s;
          }
          if(len > 1){
            return MemRefType::get(rankedTensorTy.getShape(), rankedTensorTy.getElementType(), AffineMap{}, memSpaceInt);
          }
          else{
            return rankedTensorTy.getElementType();
          }
        }
        if (auto unrankedTensorTy = dyn_cast<UnrankedTensorType>(type)) {
          return UnrankedMemRefType::get(unrankedTensorTy.getElementType(), memSpaceInt);
        }
        if (auto memref = dyn_cast<MemRefType>(type)) {
          int64_t len = 1;
          for(auto s : memref.getShape()){
            len *= s;
          }
          if(len > 1){
            if(memref.getMemorySpaceAsInt() <= 0){
              return MemRefType::get(memref.getShape(), memref.getElementType(), AffineMap{}, memSpaceInt ) ;
            }
          }
          else{
            return memref.getElementType();
          }
        }
        return type;
      });
      addMaterializations(tc);

      ConversionTarget target(*ctx);
      target.addLegalDialect<FriskDialect, affine::AffineDialect, memref::MemRefDialect, func::FuncDialect, dg::DeepgengraphDialect,
                            dgt::DeepgengraphTritonDialect, arith::ArithDialect, math::MathDialect,
                            scf::SCFDialect, tensor::TensorDialect>();

      target.addIllegalOp<dg::AddOp, dg::SubOp, dg::MulOp, 
        dg::DivOp, dg::PowOp, dg::CmpOp, dg::PreciseDotOp, dg::MaskOp,
        dg::Exp2Op, dg::ReduceOp
      >();

      RewritePatternSet ps(ctx);
      ps.add<
        MatmulOpConversionPattern,BinaryOpConversionPattern, 
        MaskOpConversionPattern,Exp2OpConversionPattern,
        ReduceOpConversionPattern
      >(tc, ctx);

      if (failed(applyPartialConversion(op, target, std::move(ps)))) {
        signalPassFailure();
      }

      // ======== 3. remove unused ops
      while(true){
        bool hasChanged = false;
        SmallVector<Operation*> dumpOps {};
        op->walk<WalkOrder::PostOrder>([&](Operation* childOp){
          if(!childOp->getResults().empty()){
            bool isDump = true;
            for(auto res : childOp->getResults()){
              isDump = isDump && res.getUsers().empty();
            }
            if(isDump){
              dumpOps.push_back(childOp);
            }
          }
          else{
            if(auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(childOp)){
              if(storeOp.getMemref().getUsers().empty()){
                dumpOps.push_back(childOp);
              }
            }
          }
        });
        for(auto unused : dumpOps){
          unused->erase();
          hasChanged = true;
        }
        if(!hasChanged){
          break;
        }
      }
    }
  }
};

// scf.for -> affine.for
struct ConvertSCFForToAffineForPass 
    : public PassWrapper<ConvertSCFForToAffineForPass, OperationPass<deepgengraph::KernelOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSCFForToAffineForPass)

    StringRef getArgument() const final { return "add-tensor-memspace"; }
    StringRef getDescription() const final { return "Add memspace encoding to tensors based on their position."; }

    void runOnOperation() override {
      auto ctx = getOperation()->getContext();
      RewritePatternSet ps(ctx);

      ps.add<SCFForToAffineFor>(ctx);
      ConversionTarget tar(*ctx);
      tar.addIllegalOp<scf::ForOp>();
      tar.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});

      applyPartialConversion(getOperation(), tar, std::move(ps));

    }
};

// dg.kernel -> frisk.kernel
class ConvertKernelOpToFrisk : public impl::KernelOpToFriskBase<ConvertKernelOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();

    TypeConverter tc;
    tc.addConversion([](Type type) { return type; });
    tc.addConversion([](TensorType tensorTy) {
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    });
    tc.addConversion([](deepgengraph::triton::PointerType ptrType) { return convertPointerType(ptrType); });
    tc.addConversion(
        [](deepgengraph::triton::BlockPointerType blockPtrType) { return convertBlockPointerType(blockPtrType); });
    addMaterializations(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<FriskDialect, memref::MemRefDialect, func::FuncDialect, deepgengraph::DeepgengraphDialect,
                           deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, 
                           tensor::TensorDialect>();
    target.addIllegalOp<deepgengraph::KernelOp>();

    RewritePatternSet ps(ctx);
    ps.add<KernelOpConversionPattern>(tc, ctx);

    if (failed(applyPartialConversion(op, target, std::move(ps)))) {
      signalPassFailure();
      return;
    }

    // Step 2: 根据 arg_permutes 属性，结合 argId 对 kernel 参数的 memref
    // 类型进行 permute，使后续 GM buffer_view 按融合后的逻辑 layout 反解索引。
    bool hasFailure = false;
    getOperation()->walk([&](frisk::KernelOp kernelOp) {
      if (hasFailure)
        return;

      auto argPermutes = kernelOp->getAttrOfType<ArrayAttr>("arg_permutes");
      if (!argPermutes)
        return;

      auto oldFuncType = kernelOp.getFunctionType();
      SmallVector<Type> newInputs(oldFuncType.getInputs().begin(),
                                  oldFuncType.getInputs().end());
      bool changed = false;

      for (auto [idx, permuteAttr] : llvm::enumerate(argPermutes)) {
        if (idx >= newInputs.size())
          break;

        auto memrefTy = dyn_cast<MemRefType>(newInputs[idx]);
        if (!memrefTy)
          continue;

        auto densePermute = dyn_cast<DenseI64ArrayAttr>(permuteAttr);
        if (!densePermute) {
          kernelOp->emitError("arg_permutes entry for argument #")
              << idx << " must be a dense i64 array attribute";
          hasFailure = true;
          return;
        }

        FailureOr<MemRefType> newMemrefTy =
            getPermutedMemRefType(memrefTy, densePermute, kernelOp, idx);
        if (failed(newMemrefTy)) {
          hasFailure = true;
          return;
        }

        newInputs[idx] = *newMemrefTy;
        kernelOp.getBody(0)->getArgument(idx).setType(*newMemrefTy);
        changed = true;
      }

      if (changed) {
        auto newFuncType =
            FunctionType::get(ctx, newInputs, oldFuncType.getResults());
        kernelOp.setFunctionType(newFuncType);
      }
    });

    if (hasFailure)
      signalPassFailure();

  }
};


// ============ Pass Creator ============

std::unique_ptr<Pass> createConvertScfForOpPass() {
  return std::make_unique<ConvertSCFForToAffineForPass>();
}

std::unique_ptr<Pass> createConvertKernelOpToFriskPass() {
  return std::make_unique<ConvertKernelOpToFrisk>();
}

std::unique_ptr<Pass> createConvertMemAndCalcOpPass() {
  return std::make_unique<ConvertMemAndCalcOpToFrisk>();
}


} // namespace mlir::frisk
