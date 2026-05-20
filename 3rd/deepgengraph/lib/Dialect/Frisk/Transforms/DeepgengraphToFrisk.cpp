#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mlir::frisk {

#define GEN_PASS_DEF_KERNELOPTOFRISK
#define GEN_PASS_DEF_MEMOPTOFRISK
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

namespace dg = deepgengraph ;
namespace dgt = deepgengraph::triton;


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

static Value getKernelArgById(Operation *op, int64_t argId) {
  auto kernelOp = op->getParentOfType<frisk::KernelOp>();
  if (!kernelOp)
    return {};
  if (argId < 0 || argId >= static_cast<int64_t>(kernelOp.getNumArguments()))
    return {};
  return kernelOp.getArgument(argId);
}

static bool isTritonPointerLike(Type type) {
  return isa<deepgengraph::triton::PointerType, deepgengraph::triton::BlockPointerType>(type);
}

static void AppendMemspaceToMemrefValue(Value& v, frisk::attr::MemorySpace ms){
  if(mlir::isa<MemRefType>(v.getType())){
    auto _ty = mlir::cast<MemRefType>(v.getType());
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    v.setType(tA);
  }
}

static Type ModifyMemrefType(Type t, frisk::attr::MemorySpace ms){
  if(mlir::isa<MemRefType>(t)){
    auto _ty = mlir::cast<MemRefType>(t);
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    return tA;
  }
  else{
    return t;
  }
}


// 从v开始，向上追溯其defOp，构建affine_expr表达式
static AffineExpr GetExprOfValue(
  mlir::Value v,  // 待分析的value
  std::map<std::string, AffineExpr>& dims,   // dims 容器
  std::map<int,Value>& arglist)  // 记录affinemap的参数的id与Value
{
  auto defOp = v.getDefiningOp();
  if(defOp == nullptr){
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
      else{
        assert(false);
      }
    }
    else{
      assert(false);
    }
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
  // not supported op

  assert(false);
}

// 
struct ArgIdViewBuffer {
  frisk::AllocBufferOp shmbuffer = nullptr;
  AffineMap baseOffsetMap;
  std::vector<Value> baseOffsetMapOperands;
  std::vector<int64_t> blockShape;
};

// 存放 argId : { arg对应的initView ， arg开辟view时建立的shm buffer }
static std::vector<ArgIdViewBuffer*>  s_argId_bufferInfo;

// ----------------- Patterns ----------

static std::vector<std::vector<int64_t>> permuteInfo;

struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto gridAttr = op->getAttr("grid");
    auto permuteAttr = op->getAttr("arg_permutes");
    
    // Parse `arg_permutes` attribute into `permuteInfo`.
    // Expected form: [array<i64: 0, 2, 1, 3>, array<i64: 0, 2, 3, 1>, ...]
    permuteInfo.clear();
    if (permuteAttr) {
      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(permuteAttr)) {
        for (auto a : arr.getValue()) {
          if (auto darr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(a)) {
            std::vector<int64_t> v;
            for (auto x : darr.asArrayRef())
              v.push_back(x);
            permuteInfo.push_back(std::move(v));
          } else if (auto de = mlir::dyn_cast<mlir::DenseIntElementsAttr>(a)) {
            std::vector<int64_t> v;
            for (auto ap : de.getValues<llvm::APInt>())
              v.push_back(ap.getSExtValue());
            permuteInfo.push_back(std::move(v));
          } else if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
            permuteInfo.push_back(std::vector<int64_t>{iattr.getInt()});
          }
        }
      } else if (auto darr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(permuteAttr)) {
        std::vector<int64_t> v;
        for (auto x : darr.asArrayRef()){
          v.push_back(x);
        }
        permuteInfo.push_back(std::move(v));
      }
    }
    auto loc = op->getLoc();
    auto oldFuncType = op.getFunctionType();
    auto converter = getTypeConverter();

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
    newKernelOp->setAttr("arg_permutes", permuteAttr);
    rewriter.inlineRegionBefore(op->getRegion(0), newKernelOp.getRegion(), newKernelOp.getRegion().end());
    // 3. replace deepgengraph.return with frisk.end
    auto oldReturn = newKernelOp->getRegion(0).front().getOps<deepgengraph::ReturnOp>().begin();
    rewriter.setInsertionPoint(*oldReturn);
    auto newReturn = rewriter.create<frisk::EndOp>(op->getLoc());
    rewriter.replaceOp(*oldReturn, newReturn);
    
    // 4. insert frisk.parallel
    rewriter.setInsertionPointToStart(&newKernelOp->getRegion(0).front());
    auto ranges = cast<DenseI64ArrayAttr>(gridAttr).asArrayRef();
    auto parallelOp = rewriter.create<frisk::ParallelOp>(loc, ranges, 128);
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
    // 建立 allocBufferOp
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto info = new ArgIdViewBuffer{};
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    auto memTy = mlir::dyn_cast<MemRefType>(resTy);
    auto newOp = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), memTy.getShape(), memTy.getElementType(), 16, int64_t(frisk::attr::MemorySpace::Shared));
    info->shmbuffer = newOp;

    // 根据 baseOffset, order, stride, 得到 baseOffset的计算map 以及 mapOperands. 
    std::map<std::string, AffineExpr> dims;
    std::map<int, Value> arglist;
    auto expr_baseOffset= GetExprOfValue(op.getBaseOffset(), dims, arglist);
    std::vector<Value> vr_baseOffset;
    for(auto [k,v] : arglist){
      vr_baseOffset.push_back(v);
    }
    auto stride = op.getStride();
    auto order = op.getOrder();

    std::vector<int64_t>* permute = nullptr;
    if(!permuteInfo.empty()){
      permute = &permuteInfo[argId];
    }
    
    auto basePtrType = mlir::dyn_cast<MemRefType>(adaptor.getBasePointer().getType());
    auto basePtrOldShape = basePtrType.getShape();  // basePtr 名义上的形状（即参数列表里的形状）
    std::vector<int64_t> basePtrPermutedShape;
    for(int i=0;i<basePtrOldShape.size();++i){
      int id = i;
      if(permute){
        id = permute->at(i);
      }
      basePtrPermutedShape.push_back(basePtrOldShape[id]);
    }

    // auto baseoffset_x = expr_baseOffset.floorDiv(stride[order[0]]);
    // auto baseoffset_y = expr_baseOffset.floorDiv(stride[order[1]]);
    // TODO : 存疑。先按照底层存储方式 <1,32,4096,128> 计算offsetxy. 如果遇到转置的，再做讨论 
    auto baseoffset_x = expr_baseOffset.floorDiv(basePtrPermutedShape.back());
    auto baseoffset_y = expr_baseOffset % basePtrPermutedShape.back();
    
    std::vector<AffineExpr> resExprArray = { baseoffset_y, baseoffset_x};
    int32_t product = 1;
    for(int i=basePtrPermutedShape.size()-1;i>=0;--i){
      product *= basePtrPermutedShape[i];
      if(i < basePtrPermutedShape.size() - 2){
        resExprArray.push_back(expr_baseOffset.floorDiv(product));
      }
    }
    std::reverse(resExprArray.begin(), resExprArray.end());

    auto baseOffsetMap = AffineMap::get(dims.size(), 0, resExprArray, op->getContext());

    // save info
    info->baseOffsetMap = baseOffsetMap;
    info->baseOffsetMapOperands = vr_baseOffset;
    info->blockShape = op.getBlockShape();
    s_argId_bufferInfo[argId] = info;
    rewriter.replaceOp(op, newOp);
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
      auto indexExprs = info->baseOffsetMap.getResults();
      // affineMap的操作数 value = 原有 + 新收集的ivs
      std::vector<Value> newIndices;
      for(auto v : indices){
        newIndices.push_back(v);
      }
      for(auto v : loop_expr_values){
        newIndices.push_back(v);
      }
      std::vector<AffineExpr> newExprs;
      // 比较 GM 的rank和newExpr的个数. 保证维度对齐. TODO:此处需要重新考虑 GM permute之后的布局.如何从 block_ptr_of 推断出前序的 affineExpr
      // 本质原因 : asuka block_ptr_of 中没有包含 permute 的信息. <1,4096,32,128> 四维 != attr中的[128, 128] 二维信息
      auto globalMemTy = mlir::cast<MemRefType>(globalBuffers[argId].getType());
      for(int i=0; i < (globalMemTy.getShape().size() - indexExprs.size()); ++i){
        newExprs.push_back(mlir::getAffineConstantExpr(0, op.getContext()));
      }
      // 表达式构建 ：loop = iv0/step0 + iv1/step1 * ub0 + iv2/step2 * (ub0 * ub1) + (iv3/step3) * (ub0*ub1*ub2)
      // [base_x + loop * offset_x, base_y + loop * offset_y] 
      for(int i=0;i < indexExprs.size(); ++i){
        AffineExpr newexpr;
        if(i >= indexExprs.size() - 2){
          newexpr = indexExprs[i] + offset[i-(indexExprs.size() - 2)] * loop_expr; 
        }
        else{
          newexpr = indexExprs[i]; 
        }
        newExprs.push_back(newexpr);
      }
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
    auto dst = adaptor.getDstPointer();
    AppendMemspaceToMemrefValue(src, frisk::attr::MemorySpace::Shared);
    AppendMemspaceToMemrefValue(dst, frisk::attr::MemorySpace::Global);
    auto newOp = rewriter.create<frisk::CopyOp>(op->getLoc(), src, dst);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};



struct BlockAdvanceConversionPattern
    : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
  using OpConversionPattern::OpConversionPattern;
  // 直接删除
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ZeroOpConversionPattern
    : public OpConversionPattern<dg::ZeroOp> {
  using OpConversionPattern::OpConversionPattern;


  LogicalResult matchAndRewrite(dg::ZeroOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // %16 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto buffer = rewriter.create<frisk::AllocBufferOp>(loc, op.getShape(), op.getElementType(), 16, int(frisk::attr::MemorySpace::Shared));
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
    rewriter.create<frisk::FillOp>(loc, buffer, valueAttr);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};


struct ConvertOpConversionPattern
    : public OpConversionPattern<dg::ConvertOp> {
  using OpConversionPattern::OpConversionPattern;


  LogicalResult matchAndRewrite(dg::ConvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // %16 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto operand = adaptor.getOperand();
    AppendMemspaceToMemrefValue( operand , frisk::attr::MemorySpace::Shared);
    auto dstType = adaptor.getDstType();
    ModifyMemrefType(dstType, frisk::attr::MemorySpace::Shared);
    auto newOp = rewriter.create<frisk::ConvertOp>(loc, operand, dstType);
    rewriter.replaceOp(op, newOp);
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


// %cst = arith.constant dense<0.127531052> : tensor<1xf32> loc(#loc)
struct ArithTensorConversionPattern : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto retType = op.getResult().getType();
    if(mlir::isa<TensorType>(retType)){
      auto tensorTy = mlir::dyn_cast<TensorType>(retType);
      MemRefType memrefTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{}, int(frisk::attr::MemorySpace::Shared));

      if(!memrefTy){
        return failure();
      }
      auto allocOp = rewriter.create<memref::AllocOp>(op->getLoc(), memrefTy);
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
      auto zero = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexAttr(0));
      std::vector<Value> indices;
      for(auto dim : memrefTy.getShape()){
        indices.push_back(zero);
      }
      auto newOp = rewriter.create<affine::AffineStoreOp>(op->getLoc(), constVal ,allocOp, indices);
      rewriter.replaceOp(op, allocOp);
      return success();
    }
    else{
      return failure();
    }
  }
};

// 
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
    for(auto initVal : op.getInits()){
      auto defOp = initVal.getDefiningOp();
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

    // 5. 单独处理旧的 Terminator (AffineYieldOp)，创建没有任何操作数的新 YieldOp
    auto oldYieldOp = mlir::cast<affine::AffineYieldOp>(oldBlock->getTerminator());
    rewriter.create<affine::AffineYieldOp>(oldYieldOp.getLoc());

    // 6. 用新 Op 替代旧 Op，并返回成功
    
    rewriter.replaceAllUsesWith(op->getResults(), newResults);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

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
    }
  }
};




class ConvertMemOpToFrisk : public impl::MemOpToFriskBase<ConvertMemOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();

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
    tc.addConversion([](TensorType ty){
      return MemRefType::get(ty.getShape(), ty.getElementType());
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
    p2.add<ArithTensorConversionPattern>(tc,ctx);
    applyPartialConversion(op, t2, std::move(p2));
    
    // stage 4 : 删除 affineFor 的 initArgs 和 yield
    ConversionTarget t3(*ctx);
    t3.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp op){
      return op.getInits().empty();
    });
    t3.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});
    RewritePatternSet p3(ctx);
    p3.add<AffineForEmptyInitsAndYieldPattern>(tc,ctx);
    applyPartialConversion(op, t3, std::move(p3));

  }
};




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


// 定义 Pass，继承自 OperationPass 并且作用于 func::FuncOp
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

std::unique_ptr<Pass> createConvertScfForOpPass() {
  return std::make_unique<ConvertSCFForToAffineForPass>();
}

std::unique_ptr<Pass> createConvertKernelOpToFriskPass() {
  return std::make_unique<ConvertKernelOpToFrisk>();
}

std::unique_ptr<Pass> createConvertMemOpPass() {
  return std::make_unique<ConvertMemOpToFrisk>();
}




} // namespace mlir::frisk
