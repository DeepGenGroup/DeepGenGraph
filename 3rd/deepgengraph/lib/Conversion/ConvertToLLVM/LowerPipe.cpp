#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>
#include <optional>

namespace mlir::frisk {

static constexpr unsigned kLLVMIndexBitwidth = 64;

// LLVM represents an n-D vector as nested arrays of 1-D vectors.  Array
// indices on llvm.extractvalue/insertvalue have to be constants, so the
// standard Vector-to-LLVM patterns cannot lower a scalar access with a
// dynamic index in any non-trailing dimension.  Flatten those accesses while
// they are still in the Vector dialect; the resulting 1-D dynamic access maps
// directly to llvm.extractelement/insertelement.
static Value linearizeVectorPosition(PatternRewriter &rewriter, Location loc,
                                     VectorType vectorType,
                                     ArrayRef<OpFoldResult> position) {
  assert(static_cast<int64_t>(position.size()) == vectorType.getRank() &&
         "expected a full-rank vector position");

  Value linear = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto [dim, index] : llvm::enumerate(position)) {
    Value dimSize = rewriter.create<arith::ConstantIndexOp>(
        loc, vectorType.getDimSize(dim));
    linear = rewriter.create<arith::MulIOp>(loc, linear, dimSize);
    linear = rewriter.create<arith::AddIOp>(
        loc, linear,
        getValueOrCreateConstantIndexOp(rewriter, loc, index));
  }
  return linear;
}

struct FlattenDynamicVectorExtract
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern<vector::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = op.getSourceVectorType();
    if (sourceType.getRank() < 2 || !op.hasDynamicPosition() ||
        op.getNumIndices() != static_cast<unsigned>(sourceType.getRank()) ||
        isa<VectorType>(op.getType()))
      return failure();

    Location loc = op.getLoc();
    VectorType flatType = VectorType::get(
        {sourceType.getNumElements()}, sourceType.getElementType());
    Value flatVector = rewriter.create<vector::ShapeCastOp>(
        loc, flatType, op.getVector());
    Value linearIndex = linearizeVectorPosition(
        rewriter, loc, sourceType, op.getMixedPosition());
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, flatVector,
                                                    linearIndex);
    return success();
  }
};

struct FlattenDynamicVectorInsert
    : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern<vector::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp op,
                                PatternRewriter &rewriter) const override {
    VectorType destType = op.getDestVectorType();
    if (destType.getRank() < 2 || !op.hasDynamicPosition() ||
        op.getNumIndices() != static_cast<unsigned>(destType.getRank()) ||
        isa<VectorType>(op.getValueToStoreType()))
      return failure();

    Location loc = op.getLoc();
    VectorType flatType = VectorType::get(
        {destType.getNumElements()}, destType.getElementType());
    Value flatVector =
        rewriter.create<vector::ShapeCastOp>(loc, flatType, op.getDest());
    Value linearIndex = linearizeVectorPosition(
        rewriter, loc, destType, op.getMixedPosition());
    Value flatResult = rewriter.create<vector::InsertOp>(
        loc, op.getValueToStore(), flatVector, linearIndex);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, destType, flatResult);
    return success();
  }
};

static std::optional<SmallVector<int64_t, 4>>
computeContiguousStrides(MemRefType memRefType) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(memRefType.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (!strides.empty() && strides.back() != 1)
    return std::nullopt;
  if (memRefType.getLayout().isIdentity())
    return strides;

  auto sizes = memRefType.getShape();
  for (int index = 0, e = strides.size() - 1; index < e; ++index) {
    if (ShapedType::isDynamic(sizes[index + 1]) ||
        ShapedType::isDynamic(strides[index]) ||
        ShapedType::isDynamic(strides[index + 1]))
      return std::nullopt;
    if (strides[index] != strides[index + 1] * sizes[index + 1])
      return std::nullopt;
  }
  return strides;
}

struct VectorTypeCastOpIndexBitwidthConversion
    : public ConvertOpToLLVMPattern<vector::TypeCastOp> {
  using ConvertOpToLLVMPattern<vector::TypeCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::TypeCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = castOp->getLoc();
    MemRefType sourceMemRefType =
        cast<MemRefType>(castOp.getOperand().getType());
    MemRefType targetMemRefType = castOp.getType();

    if (!sourceMemRefType.hasStaticShape() ||
        !targetMemRefType.hasStaticShape())
      return failure();

    auto llvmSourceDescriptorTy =
        dyn_cast<LLVM::LLVMStructType>(adaptor.getOperands()[0].getType());
    if (!llvmSourceDescriptorTy)
      return failure();
    MemRefDescriptor sourceMemRef(adaptor.getOperands()[0]);

    auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(targetMemRefType));
    if (!llvmTargetDescriptorTy)
      return failure();

    auto sourceStrides = computeContiguousStrides(sourceMemRefType);
    if (!sourceStrides)
      return failure();
    auto targetStrides = computeContiguousStrides(targetMemRefType);
    if (!targetStrides)
      return failure();
    if (llvm::any_of(*targetStrides, ShapedType::isDynamic))
      return failure();

    Type indexType = getIndexType();
    auto desc =
        MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);
    desc.setAllocatedPtr(rewriter, loc,
                         sourceMemRef.allocatedPtr(rewriter, loc));
    desc.setAlignedPtr(rewriter, loc, sourceMemRef.alignedPtr(rewriter, loc));
    desc.setOffset(rewriter, loc,
                   createIndexAttrConstant(rewriter, loc, indexType, 0));

    for (const auto &indexedSize :
         llvm::enumerate(targetMemRefType.getShape())) {
      int64_t index = indexedSize.index();
      desc.setSize(rewriter, loc, index,
                   createIndexAttrConstant(rewriter, loc, indexType,
                                           indexedSize.value()));
      desc.setStride(rewriter, loc, index,
                     createIndexAttrConstant(rewriter, loc, indexType,
                                             (*targetStrides)[index]));
    }

    rewriter.replaceOp(castOp, {desc});
    return success();
  }
};


// =====================================================================
//                  Vecotr Dialect To LLVM Dialect
// =====================================================================
// 将memref lowering到llvm上，因为 passes.h.inc中的base类没有提供可以选择indexBitWidth的options，所以自己写了一个
struct VectorToLLVMPass : public PassWrapper<VectorToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToLLVMPass)

  VectorToLLVMPass(unsigned indexBitWidth_=kLLVMIndexBitwidth) : indexBitWidth(indexBitWidth_) {};

  unsigned indexBitWidth;

  void runOnOperation() override {
    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(indexBitWidth);
    bool force32BitVectorIndices = indexBitWidth == 32;

    {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      mlir::vector::populateVectorBitCastLoweringPatterns(patterns);
      mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
      mlir::vector::populateVectorContractLoweringPatterns(
          patterns, mlir::vector::VectorContractLowering::Dot);
      mlir::vector::populateVectorMaskOpLoweringPatterns(patterns);
      mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
      mlir::vector::populateVectorInterleaveLoweringPatterns(patterns);
      mlir::vector::populateVectorTransposeLoweringPatterns(
          patterns, mlir::vector::VectorTransposeLowering::EltWise);
      mlir::vector::populateVectorTransferLoweringPatterns(
          patterns, /*maxTransferRank=*/1);
      mlir::vector::populateVectorMaskMaterializationPatterns(
          patterns, force32BitVectorIndices);
      mlir::vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
      mlir::vector::populateVectorStepLoweringPatterns(patterns);
      mlir::vector::populateVectorRankReducingFMAPattern(patterns);
      mlir::vector::populateVectorGatherLoweringPatterns(patterns);
      patterns.add<FlattenDynamicVectorExtract, FlattenDynamicVectorInsert>(
          &getContext(), PatternBenefit(2));
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    LLVMConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<vector::VectorDialect>();

    LLVMTypeConverter converter(&getContext(), options);
    RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorTransferLoweringPatterns(patterns);
    patterns.add<VectorTypeCastOpIndexBitwidthConversion>(
        converter, PatternBenefit(2));
    mlir::populateVectorToLLVMConversionPatterns(
        converter, patterns, false, force32BitVectorIndices);
    // mlir::populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createVectorToLLVMPass(int indexBitwidth) {
  return std::make_unique<VectorToLLVMPass>(indexBitwidth);
}

bool firstLowering(mlir::ModuleOp &mod, mlir::MLIRContext *context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLowerAffinePass());                     // affine -> scf/vector
  // pm.addPass(mlir::createParallelLoopToGpuPass());               // scf.parallelOp -> gpu...
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;
}

bool secondLowering(mlir::ModuleOp &mod, mlir::MLIRContext *context,
                    Target target) {
  mlir::PassManager pm(context);

  // 1. 结构与高阶方言转换 (SCF, Affine, Vector -> SCF)
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::frisk::createAmendAllocaOpAddrSpacePass(target));
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSCFToControlFlowPass()); // scf -> cf

  // 2. 基础控制流、MemRef、Func 降级到 LLVM
  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));

  ConvertFuncToLLVMPassOptions funcOptions;
  funcOptions.indexBitwidth = kLLVMIndexBitwidth;
  funcOptions.useBarePtrCallConv = true;
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));

  // 3. GPU / ROCDL / NVVM 降级
  pm.addPass(mlir::frisk::createLLVMFuncOpAddGPUAttrPass(target));
  pm.addPass(mlir::frisk::createGPUToROCDLOrNVVMPass(target, kLLVMIndexBitwidth));

  // 4. Vector lowering can create arith constants and UB poison values, so it
  // must run before the scalar dialects are finalized.
  pm.addPass(createVectorToLLVMPass(kLLVMIndexBitwidth));

  // 5. 标量类型与计算降级 (Arith, UB, Index)
  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));

  UBToLLVMConversionPassOptions ubOptions;
  ubOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createUBToLLVMConversionPass(ubOptions));

  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = kLLVMIndexBitwidth; // 修复：统一使用 kLLVMIndexBitwidth
  pm.addPass(mlir::createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));

  // 6. 清理多余的类型转换 Cast 及优化
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  if (mlir::failed(pm.run(mod))) {
    return false;
  }

  return true;
}
}  // namespace mlir::frisk
