#include "deepgengraph/Common.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <dlfcn.h>
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {

// ===================================================================
//                  amend memerf alloca Addrspace       
// ===================================================================
// 将memref.alloca的地址空间进行修改，local=0为cuda/loacl=5为rocm
struct AmendAllocaOpAddrSpace : public OpRewritePattern<memref::AllocaOp> {
  AmendAllocaOpAddrSpace(MLIRContext *context, Target target)
    : OpRewritePattern(context), target(target) {}

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp, PatternRewriter &rewriter) const override {
    MemRefType originalType = allocaOp.getType();
    int requiredSpace = (target == Target::ROCm) ? 5 : 0;
    if (static_cast<int>(originalType.getMemorySpaceAsInt()) == requiredSpace) {
      return failure();
    }
    MLIRContext *ctx = allocaOp.getContext();
    Attribute memorySpaceAttr = IntegerAttr::get(IntegerType::get(ctx, 64), requiredSpace);
    MemRefType newType = MemRefType::get(originalType.getShape(), originalType.getElementType(), 
                                         originalType.getLayout(),memorySpaceAttr);

    rewriter.setInsertionPoint(allocaOp);
    auto newAlloca = rewriter.create<memref::AllocaOp>(allocaOp.getLoc(), newType);
    rewriter.replaceOp(allocaOp, newAlloca.getResult());
    return success();
  }
  private:
    Target target;
};

struct AmendAllocaOpAddrSpacePass : public PassWrapper<AmendAllocaOpAddrSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AmendAllocaOpAddrSpacePass)

  AmendAllocaOpAddrSpacePass(Target target) : target(target) {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AmendAllocaOpAddrSpace>(&getContext(), target);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  private:
    Target target;
};



// ===================================================================
//                 gpu index to nvvm/rocdl index 
// ===================================================================
// 将 GUP 的IdOp转成 rocdl/nvvm的IdOp，读取func的attr加到新的IdOp上
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public OpConversionPattern<Op> {
private:
  unsigned indexBitwidth;
  StringRef boundsAttrName;

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       MLIRContext *context)
      : OpConversionPattern<Op>(typeConverter, context),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName("") {}

  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       MLIRContext *context,
                                       StringRef boundsAttrName)
      :  OpConversionPattern<Op>(typeConverter, context),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName(boundsAttrName) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    }

    Operation *function;
    if (auto Func = op->template getParentOfType<func::FuncOp>())
      function = Func;
    if (auto llvmFunc = op->template getParentOfType<LLVM::LLVMFuncOp>())
      function = llvmFunc;
    if (!boundsAttrName.empty() && function) {
      if (auto attr = function->template getAttrOfType<DenseI32ArrayAttr>(
              boundsAttrName)) {
        int32_t maximum = attr[static_cast<uint32_t>(op.getDimension())];
        newOp->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
    }

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

Value getLaneId(ConversionPatternRewriter &rewriter, Location loc,
                const unsigned indexBitwidth) {
  auto int32Type = IntegerType::get(rewriter.getContext(), 32);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value minus1 = rewriter.create<arith::ConstantIntOp>(loc, -1, 32);
  Value mbcntLo = rewriter.create<ROCDL::MbcntLoOp>(loc, int32Type,
                                                    ValueRange{minus1, zero});
  Value laneId = rewriter.create<ROCDL::MbcntHiOp>(loc, int32Type,
                                                  ValueRange{minus1, mbcntLo});
  return laneId;
}

static unsigned getBitWidth(Type type) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  auto vec = cast<VectorType>(type);
  assert(!vec.isScalable() && "scalable vectors are not supported");
  return vec.getNumElements() * getBitWidth(vec.getElementType());
}

static Value createI32Constant(OpBuilder &builder, Location loc,
                               int32_t value) {
  Type i32 = builder.getI32Type();
  return builder.create<LLVM::ConstantOp>(loc, i32, value);
}

SmallVector<Value> decomposeValue(OpBuilder &builder, Location loc,
                                              Value src, Type dstType) {
  Type srcType = src.getType();
  if (srcType == dstType)
    return {src};

  unsigned srcBitWidth = getBitWidth(srcType);
  unsigned dstBitWidth = getBitWidth(dstType);
  if (srcBitWidth == dstBitWidth) {
    Value cast = builder.create<LLVM::BitcastOp>(loc, dstType, src);
    return {cast};
  }

  if (dstBitWidth > srcBitWidth) {
    auto smallerInt = builder.getIntegerType(srcBitWidth);
    if (srcType != smallerInt)
      src = builder.create<LLVM::BitcastOp>(loc, smallerInt, src);

    auto largerInt = builder.getIntegerType(dstBitWidth);
    Value res = builder.create<LLVM::ZExtOp>(loc, largerInt, src);
    return {res};
  }
  assert(srcBitWidth % dstBitWidth == 0 &&
         "src bit width must be a multiple of dst bit width");
  int64_t numElements = srcBitWidth / dstBitWidth;
  auto vecType = VectorType::get(numElements, dstType);

  src = builder.create<LLVM::BitcastOp>(loc, vecType, src);

  SmallVector<Value> res;
  for (auto i : llvm::seq(numElements)) {
    Value idx = createI32Constant(builder, loc, i);
    Value elem = builder.create<LLVM::ExtractElementOp>(loc, src, idx);
    res.emplace_back(elem);
  }

  return res;
}


Value composeValue(OpBuilder &builder, Location loc, ValueRange src,
                               Type dstType) {
  assert(!src.empty() && "src range must not be empty");
  if (src.size() == 1) {
    Value res = src.front();
    if (res.getType() == dstType)
      return res;

    unsigned srcBitWidth = getBitWidth(res.getType());
    unsigned dstBitWidth = getBitWidth(dstType);
    if (dstBitWidth < srcBitWidth) {
      auto largerInt = builder.getIntegerType(srcBitWidth);
      if (res.getType() != largerInt)
        res = builder.create<LLVM::BitcastOp>(loc, largerInt, res);

      auto smallerInt = builder.getIntegerType(dstBitWidth);
      res = builder.create<LLVM::TruncOp>(loc, smallerInt, res);
    }

    if (res.getType() != dstType)
      res = builder.create<LLVM::BitcastOp>(loc, dstType, res);

    return res;
  }

  int64_t numElements = src.size();
  auto srcType = VectorType::get(numElements, src.front().getType());
  Value res = builder.create<LLVM::PoisonOp>(loc, srcType);
  for (auto &&[i, elem] : llvm::enumerate(src)) {
    Value idx = createI32Constant(builder, loc, i);
    res = builder.create<LLVM::InsertElementOp>(loc, srcType, res, elem, idx);
  }

  if (res.getType() != dstType)
    res = builder.create<LLVM::BitcastOp>(loc, dstType, res);

  return res;
}

struct GPUShuffleOpToROCDLLowering : public OpConversionPattern<gpu::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override 
  {
    Location loc = op->getLoc();
    Value initShflValue = adaptor.getValue();

    const unsigned indexBitwidth = 32;
    Value srcLaneId = getLaneId(rewriter, loc, indexBitwidth);

    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto boolType = mlir::IntegerType::get(rewriter.getContext(), 1);
    Value width = adaptor.getWidth();
    Value trueVal = rewriter.create<LLVM::ConstantOp>(loc, boolType, 1);

    Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, width);  // selfLane + width

    Value dstLane;

    switch (op.getMode()) {
    case gpu::ShuffleMode::UP:
      dstLane = rewriter.create<LLVM::SubOp>(loc, int32Type, srcLaneId,
                                             adaptor.getOffset());  // read from lane[srcLaneId - offs]
      break;
    case gpu::ShuffleMode::DOWN:
      dstLane = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId,
                                             adaptor.getOffset());  // read from lane[srcLaneId + offs]
      break;
    case gpu::ShuffleMode::XOR:
      dstLane = rewriter.create<LLVM::XOrOp>(loc, int32Type, srcLaneId,
                                             adaptor.getOffset());
      break;
    case gpu::ShuffleMode::IDX:
      // width 代表了划分的组长度。 offset 表示组内偏移
      auto offset = adaptor.getOffset();
      auto rem = rewriter.create<LLVM::SRemOp>(loc, int32Type, srcLaneId, width);
      auto sub = rewriter.create<LLVM::SubOp>(loc, int32Type, srcLaneId, rem);  
      dstLane = rewriter.create<LLVM::AddOp>(loc, int32Type, sub, offset);  // read from lane[srcLaneId - srcLaneId % width + offset]
      break;
    }
    Value selectDstLane;
    Value isActiveSrcLane;
    if(op.getMode() != gpu::ShuffleMode::IDX){
      Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 0);
      Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, zero, width);
      Value widthOrZeroIfOutside = rewriter.create<LLVM::AndOp>(loc, int32Type, add, negwidth);  // (selfLane + width) & (-width)
      isActiveSrcLane = rewriter.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::slt, dstLane, widthOrZeroIfOutside);
      selectDstLane = rewriter.create<LLVM::SelectOp>(loc, isActiveSrcLane,
                                                            dstLane, srcLaneId);
    }
    else{
      // 对于 shuffle idx，不必检查 dstLane 和 width的关系
      selectDstLane = dstLane;
      isActiveSrcLane = trueVal;
    }
    Value two = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 2);
    Value dwordAlignedDstLane =
        rewriter.create<LLVM::ShlOp>(loc, int32Type, selectDstLane, two);

    SmallVector<Value> decomposed =
        decomposeValue(rewriter, loc, initShflValue, int32Type);
    SmallVector<Value> swizzled;
    for (Value v : decomposed) {
      Value res = rewriter.create<ROCDL::DsBpermuteOp>(loc, int32Type,
                                                       dwordAlignedDstLane, v);
      swizzled.emplace_back(res);
    }
    Value shflValue =
        composeValue(rewriter, loc, swizzled, initShflValue.getType());
    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    return success();
  }
};

static NVVM::ShflKind convertShflKind(gpu::ShuffleMode mode) {
  switch (mode) {
  case gpu::ShuffleMode::XOR:
    return NVVM::ShflKind::bfly;
  case gpu::ShuffleMode::UP:
    return NVVM::ShflKind::up;
  case gpu::ShuffleMode::DOWN:
    return NVVM::ShflKind::down;
  case gpu::ShuffleMode::IDX:
    return NVVM::ShflKind::idx;
  }
  llvm_unreachable("unknown shuffle mode");
}

struct GPUShuffleOpToNVVMLowering : public OpConversionPattern<gpu::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto f32Type = Float32Type::get(rewriter.getContext());
    auto predTy = IntegerType::get(rewriter.getContext(), 1);

    // NVVM shfl.sync only supports 32-bit types (i32/f32).
    // For sub-32-bit types like f16, promote to f32, shuffle, then truncate.
    bool needF16Cast = isa<Float16Type>(valueTy);
    Value shflInput = adaptor.getValue();
    Type shflValueTy = valueTy;
    if (needF16Cast) {
      shflInput = rewriter.create<LLVM::FPExtOp>(loc, f32Type, shflInput);
      shflValueTy = f32Type;
    }

    // Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, int32Type, -1);
    // Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 32);
    // Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    // Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne, numLeadInactiveLane);
    // Value maskAndClamp;
    // if (op.getMode() == gpu::ShuffleMode::UP) {
    //   // Clamp lane: `32 - activeWidth`
    //   maskAndClamp = numLeadInactiveLane;
    // } else {
    //   // Clamp lane: `activeWidth - 1`
    //   maskAndClamp = rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);
    // }
    Value segmaskAndClamp;
    auto constOp = adaptor.getWidth().getDefiningOp<arith::ConstantOp>();
    auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
    auto width = intAttr.getInt();
    // llvm::outs() << "witdh: " << width;
    if (width < 32) {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, ((32 - width) << 8) + 31);
    } else {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, width - 1);;
    }
    
    bool predIsUsed = !op->getResult(1).use_empty();
    UnitAttr returnValueAndIsValidAttr = nullptr;
    Type resultTy = shflValueTy;
    if (predIsUsed) {
      returnValueAndIsValidAttr = rewriter.getUnitAttr();
      resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {shflValueTy, predTy});
    }
    NVVM::ShflKind nvvmMode;
    switch (op.getMode()) {
      case gpu::ShuffleMode::XOR:
        nvvmMode = NVVM::ShflKind::bfly;
        break;
      case gpu::ShuffleMode::UP:
        nvvmMode =  NVVM::ShflKind::up;
        break;
      case gpu::ShuffleMode::DOWN:
        nvvmMode =  NVVM::ShflKind::down;
        break;
      case gpu::ShuffleMode::IDX:
        nvvmMode =  NVVM::ShflKind::idx;
        break;
      default:
        return failure();
    }
    // Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        // maskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, minusOne, shflInput, adaptor.getOffset(),
        segmaskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    if (predIsUsed) {
      Value shflValue = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 0);
      if (needF16Cast)
        shflValue = rewriter.create<LLVM::FPTruncOp>(loc, valueTy, shflValue);
      Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 1);
      rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    } else {
      Value result = shfl;
      if (needF16Cast)
        result = rewriter.create<LLVM::FPTruncOp>(loc, valueTy, result);
      rewriter.replaceOp(op, {result, nullptr});
    }
    return success();
  }
};


LLVM::LLVMFuncOp getOrDefineFunction(mlir::ModuleOp moduleOp,
                                           Location loc, OpBuilder &b,
                                           StringRef name,
                                           LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(moduleOp.getBody());
    ret = b.create<LLVM::LLVMFuncOp>(loc, name, type, LLVM::Linkage::External);
  }
  return ret;
}

SmallString<16> getUniqueSymbolName(mlir::ModuleOp moduleOp,
                                           StringRef prefix) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (prefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));
  return stringConstName;
}

LLVM::GlobalOp getOrCreateStringConstant(OpBuilder &b, Location loc,
                                mlir::ModuleOp moduleOp, Type llvmI8,
                                StringRef namePrefix, StringRef str,
                                uint64_t alignment = 0, unsigned addrSpace = 0) {
  llvm::SmallString<20> nullTermStr(str);
  nullTermStr.push_back('\0'); // Null terminate for C
  auto globalType =
      LLVM::LLVMArrayType::get(llvmI8, nullTermStr.size_in_bytes());
  StringAttr attr = b.getStringAttr(nullTermStr);

  // Try to find existing global.
  for (auto globalOp : moduleOp.getOps<LLVM::GlobalOp>())
    if (globalOp.getGlobalType() == globalType && globalOp.getConstant() &&
        globalOp.getValueAttr() == attr &&
        globalOp.getAlignment().value_or(0) == alignment &&
        globalOp.getAddrSpace() == addrSpace)
      return globalOp;

  // Not found: create new global.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(moduleOp.getBody());
  SmallString<16> name = getUniqueSymbolName(moduleOp, namePrefix);
  return b.create<LLVM::GlobalOp>(loc, globalType,
                                  /*isConstant=*/true, LLVM::Linkage::Internal,
                                  name, attr, alignment, addrSpace);
}



// 将gpu barrier转成rocdl的barrier
struct GPUBarrierToROCDLLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(brOp);
    return success();
  }
};

// 将gpu barrier转成NVVM的barrier0
struct GPUBarrierToNVVMLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(brOp);
    return success();
  }
};

// 将上述 3 个重写加到这个pass中
struct GPUToROCDLOrNVVMPass : public PassWrapper<GPUToROCDLOrNVVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToROCDLOrNVVMPass)

  explicit GPUToROCDLOrNVVMPass(Target target_, unsigned indexBitwidth_) : 
                                target(target_), indexBitwidth(indexBitwidth_) {};
  private:
    Target target;
    unsigned indexBitwidth;
    StringRef amdgcnDataLayout =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:"
    "32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:"
    "64-S32-A5-G1-ni:7:8:9";

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget Codetarget(getContext());
    LowerToLLVMOptions options(&getContext());
    if (target == Target::ROCm) {
      options.dataLayout = llvm::DataLayout(amdgcnDataLayout);
    }
    options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter typeConverter(&getContext(), options);

    Codetarget.addIllegalDialect<gpu::GPUDialect>();
    Codetarget.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect, NVVM::NVVMDialect>();

    if (target == Target::ROCm) {
      Codetarget.addLegalDialect<arith::ArithDialect>();
      // populateGpuToROCDLConversionPatterns(typeConverter, patterns, gpu::amd::Runtime::HIP);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, 
                                               ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(typeConverter, &getContext(), "range");
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                               ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(typeConverter, &getContext(), "range");
      // patterns.add<GPUPrintfOpToLLVMCallLowering>(typeConverter);
      patterns.add<GPUShuffleOpToROCDLLowering>(typeConverter, &getContext());
      patterns.add<GPUBarrierToROCDLLowering>(&getContext());
      populateMathToROCDLConversionPatterns(typeConverter, patterns);  // exp仅支持fp64/16
    } else if (target == Target::CUDA) {
      // populateGpuToNVVMConversionPatterns(typeConverter, patterns, 10);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp, 
                                               NVVM::BlockIdYOp, NVVM::BlockIdZOp>>(typeConverter, &getContext(), "range");
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                               NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>>(typeConverter, &getContext(), "range");
      // patterns.add<GPUPrintfOpToVPrintfLowering>(typeConverter);
      patterns.add<GPUShuffleOpToNVVMLowering>(typeConverter, &getContext());
      patterns.add<GPUBarrierToNVVMLowering>(&getContext());
      populateLibDeviceConversionPatterns(typeConverter, patterns, /*benefit*/10);  // 大部分只支持fp32/fp16
    }

    if (failed(applyPartialConversion(getOperation(), Codetarget, std::move(patterns)))){
      return signalPassFailure();
    }

  }
};


// ===================================================================
//         LLVMFuncOp add attribute (nvvm.kernel, nvvm.maxnid)
// ===================================================================
// LLVMFunc 添加 nvvm.kernel 和 nvvm.maxnid 属性给func
struct LLVMFuncOpAddGPUAttrPass : public PassWrapper<LLVMFuncOpAddGPUAttrPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMFuncOpAddGPUAttrPass)
  explicit LLVMFuncOpAddGPUAttrPass(Target target_) : target(target_) {};
  private:
    Target target;
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(module);
    module.walk<WalkOrder::PreOrder>([&](LLVM::LLVMFuncOp funcOp) {
      auto blockdims = funcOp->getAttrOfType<IntegerAttr>(THREAD_NUM);
      if (!blockdims) {
        llvm::errs() << "[LLVMFuncOpAddGPUAttrPass] missing attr " << THREAD_NUM
                     << " on function: " << funcOp.getName() << "\n";
        return;
      }
      int32_t flatSize = blockdims.getInt();
      // auto len = blockdims.asArrayRef().size();
      // for (int32_t size : blockdims.asArrayRef()) {
      //   flatSize *= size;
      // }
      auto reqdAttr =  DenseI32ArrayAttr::get(funcOp->getContext(), llvm::ArrayRef<int32_t>({flatSize,1,1}));
      if (target == Target::CUDA) {
        funcOp->setAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName(), builder.getIntegerAttr(builder.getI1Type(), 1));
        funcOp->setAttr(mlir::NVVM::NVVMDialect::getMaxntidAttrName(), reqdAttr);
      } else {
        funcOp->setAttr(ROCDL::ROCDLDialect::getKernelFuncAttrName(), builder.getIntegerAttr(builder.getI1Type(), 1));
        funcOp->setAttr(ROCDL::ROCDLDialect::getReqdWorkGroupSizeAttrName(), reqdAttr);
        StringAttr flatSizeAttr = StringAttr::get(funcOp->getContext(), Twine(flatSize) + "," + Twine(flatSize));
        funcOp->setAttr(ROCDL::ROCDLDialect::getFlatWorkGroupSizeAttrName(),flatSizeAttr);
      }
    });
  }
};

// ###################################################################################################
}  // end namespace


namespace mlir::frisk {
  std::unique_ptr<OperationPass<ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target) {
    return std::make_unique<AmendAllocaOpAddrSpacePass>(target);
  }
  
  std::unique_ptr<Pass> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth) {
    return std::make_unique<GPUToROCDLOrNVVMPass>(target, indexBitwidth);
  }
  
  std::unique_ptr<OperationPass<ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target) {
    return std::make_unique<LLVMFuncOpAddGPUAttrPass>(target);
  }
}



// ================================================================



