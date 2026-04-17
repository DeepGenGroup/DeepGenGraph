#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <memory>
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
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
}

static Type convertBlockPointerType(deepgengraph::triton::BlockPointerType blockPtrType) {
  auto tensorTy = blockPtrType.getPointeeType();
  SmallVector<int64_t> dynStrides(tensorTy.getRank(), ShapedType::kDynamic);
  // auto layout = StridedLayoutAttr::get(blockPtrType.getContext(), ShapedType::kDynamic, dynStrides);
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
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

static std::map<int, frisk::BufferViewOp > s_map_argId_initBufferView;

// ----------------- Patterns ----------

struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto gridAttr = op->getAttr("grid");
    auto loc = op->getLoc();
    auto oldFuncType = op.getFunctionType();
    auto converter = getTypeConverter();

    llvm::SmallVector<Type> newInputs;
    llvm::SmallVector<Type> newOutputs;
    for (auto ty : oldFuncType.getInputs()) {
      newInputs.push_back(converter->convertType(ty));
    }
    // 1. build new function type
    auto newFuncType = rewriter.getFunctionType(newInputs, newOutputs);
    // 2. convert old region signature, inline it after new frisk.kernel
    TypeConverter::SignatureConversion sc{oldFuncType.getNumInputs()};
    for (int i = 0; i < oldFuncType.getNumInputs(); ++i) {
      sc.addInputs(i, converter->convertType(oldFuncType.getInput(i)));
    }

    rewriter.convertRegionTypes(&op->getRegion(0), *converter, &sc);
    rewriter.applySignatureConversion(&op.getFunctionBody().front(), sc);

    auto newKernelOp = rewriter.create<frisk::KernelOp>(loc, op.getName(), newFuncType);
    newKernelOp->setAttr("grid", gridAttr);
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
  // block_ptr_of base=%ptr. 将ptr绕过，直接绑定到 kernelOp的 mem 参数上
  // 新建 frisk.bufferview 建立 入参mem的 view， 替换 result
  // 删除对应的 dg.pointerOfOp

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockPointerOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto convertedType = dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!convertedType){
      return failure();
    }
    // 
    Value source = adaptor.getBasePointer();
    int argId = -1;
    if (auto argIdAttr = op->getAttrOfType<IntegerAttr>("argId")) {
      argId = argIdAttr.getInt();
      if (Value kernelArg = getKernelArgById(op, argId)){
        source = kernelArg;
      }
    }
    if (!source || !isa<MemRefType>(source.getType())){
      return failure();
    }

    Value baseoffset = adaptor.getBaseOffset();
    auto memref =  mlir::cast<MemRefType>(adaptor.getBasePointer().getType());
    auto rank = memref.getRank();
    auto offset = op.getOffset();
    auto stride = op.getStride();
    auto order = op.getOrder();
    auto s0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), stride[order[0]]);
    auto s1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), stride[order[1]]);
    // 根据 baseOffset 和 stride，计算 base x,y 坐标偏移
    auto base_x = rewriter.create<arith::DivUIOp>(op->getLoc(), baseoffset, s0);
    auto base_y = rewriter.create<arith::DivUIOp>(op->getLoc(), baseoffset, s1);
    
    
    std::vector<Value> indices = {}; 
    auto zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for(int i=0;i<rank-2;++i){
      indices.push_back(zero);
    }
    indices.push_back(base_x);
    indices.push_back(base_y);

    auto map2dim = AffineMap::getMultiDimIdentityMap(rank, op->getContext());
    std::vector<int64_t> ranges = {1,1,1,1};
    ranges[2] = op.getBlockShape()[0];
    ranges[3] = op.getBlockShape()[1];
    auto view = rewriter.create<frisk::BufferViewOp>(op->getLoc(), source, indices, map2dim, ranges);
    view->setAttr("argId", op->getAttr("argId"));
    if(op->hasAttr("move")){
      s_map_argId_initBufferView[argId] = view;
      view->setAttr("move", op->getAttr("move"));
    }
    rewriter.replaceOp(op, view);
    return success();
  }
};

struct BlockLoadConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_load ：先添加 %mem = frisk.alloc_buffer,
  // 之后 frisk.copy %memview, %mem
  // 使用 %mem 替换 op的结果 
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto newRetType = mlir::dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto shape = newRetType.getShape();
    auto eleTy = newRetType.getElementType();
    auto allocOp = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),shape, eleTy);
    auto copyOp = rewriter.create<frisk::CopyOp>(op->getLoc(), adaptor.getSrcPointer(), allocOp);
    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

struct BlockStoreConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_store %14, %24 :  直接替换
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto newOp = rewriter.create<frisk::CopyOp>(op->getLoc(), adaptor.getValue(), adaptor.getDstPointer());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct BlockAdvanceConversionPattern
    : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
  using OpConversionPattern::OpConversionPattern;
  // 本质上是将一个 buffer_view 圈出的窗口在baseMem上滑动。滑动距离= offsets. 累计滑动距离需要根据 offsets 和 上一次距离 计算得到
  // 替换为 %next = frisk.buffer_view, scf.yield %next.  将op结果替换为 %next

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto baseMem = getKernelArgById(op, argId);
    
    auto initView = s_map_argId_initBufferView[argId];
    auto loc = op->getLoc();
    auto indices = initView.getIndices();
    auto offset = op.getOffsets();
    std::vector<Value> offsetValues;
    for(auto offset : op.getOffsets()){
      offsetValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, offset)) ;
    }
    std::vector<Value> newIndices;
    for(int i=0;i < indices.size(); ++i){
      if(i < indices.size() - 2){
        newIndices.push_back(indices[i]);
      }
      else{
        auto newcoord = rewriter.create<arith::AddIOp>(loc, indices[i], offsetValues[i-(indices.size() - 2)]);
        newIndices.push_back(newcoord);
      }
    }

    auto newView = rewriter.create<BufferViewOp>(loc, initView.getSource(), newIndices, initView.getIndexMap(), initView.getRanges());
    rewriter.replaceOp(op, newView);
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
    auto buffer = rewriter.create<frisk::AllocBufferOp>(loc, op.getShape(), op.getElementType());
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
    auto newOp = rewriter.create<frisk::ConvertOp>(loc, adaptor.getOperand(), adaptor.getDstType());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};



struct SCFForTypeConversionPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto newFor = rewriter.create<scf::ForOp>(op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
                                              adaptor.getStep(), adaptor.getInitArgs());

    rewriter.mergeBlocks(op.getBody(), newFor.getBody(), newFor.getBody()->getArguments());
    rewriter.replaceOp(op, newFor.getResults());
    return success();
  }
};

struct SCFYieldTypeConversionPattern : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
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
                           deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, scf::SCFDialect,
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
    tc.addConversion([](deepgengraph::triton::PointerType ptrType) { return convertPointerType(ptrType); });
    tc.addConversion(
        [](deepgengraph::triton::BlockPointerType blockPtrType) { return convertBlockPointerType(blockPtrType); });
    tc.addConversion([](TensorType ty){
      return MemRefType::get(ty.getShape(), ty.getElementType());
    });
    addMaterializations(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<FriskDialect, memref::MemRefDialect, func::FuncDialect, deepgengraph::DeepgengraphDialect,
                           deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, scf::SCFDialect,
                           tensor::TensorDialect>();

    target.addIllegalOp<dgt::PointerOfOp, dgt::BlockPointerOfOp,
                        dgt::BlockLoadOp, dgt::BlockStoreOp,
                        dgt::TensorFromOp, dg::ZeroOp, dg::ConvertOp,
                        dgt::BlockAdvanceOp
    >();

    RewritePatternSet ps(ctx);
    ps.add<PointerOfConversionPattern, BlockPointerOfConversionPattern, BlockLoadConversionPattern,
      BlockStoreConversionPattern,ZeroOpConversionPattern ,ConvertOpConversionPattern,
      BlockAdvanceConversionPattern, SCFForTypeConversionPattern, SCFYieldTypeConversionPattern
    >(tc, ctx);

    target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp forOp) {
      for (Value initArg : forOp.getInitArgs()) {
        if (isTritonPointerLike(initArg.getType()))
          return false;
      }
      for (Type resultType : forOp.getResultTypes()) {
        if (isTritonPointerLike(resultType))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp yieldOp) {
      for (Value operand : yieldOp.getOperands()) {
        if (isTritonPointerLike(operand.getType()))
          return false;
      }
      return true;
    });

    if(failed(applyPartialConversion(op, target, std::move(ps)))){
      signalPassFailure();
    }

  }
};

std::unique_ptr<Pass> createConvertKernelOpToFriskPass() {
  return std::make_unique<ConvertKernelOpToFrisk>();
}

std::unique_ptr<Pass> createConvertMemOpPass() {
  return std::make_unique<ConvertMemOpToFrisk>();
}

} // namespace mlir::frisk
