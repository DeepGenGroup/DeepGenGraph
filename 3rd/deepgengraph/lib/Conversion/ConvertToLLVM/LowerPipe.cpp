#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>
#include <optional>

namespace mlir::frisk {

static constexpr unsigned kLLVMIndexBitwidth = 64;

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
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    LLVMConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

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
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addNestedPass<mlir::func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::frisk::createAmendAllocaOpAddrSpacePass(
      target)); // ROCm local alloca -> addrspace(5)
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSCFToControlFlowPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = kLLVMIndexBitwidth;

  pm.addPass(mlir::createConvertControlFlowToLLVMPass(
      cfOptions)); // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  // pm.addPass(createVectorToLLVMPass(kLLVMIndexBitwidth)); // 自定义 vector to llvm pass
  pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = kLLVMIndexBitwidth;              // 使用 i64 index，避免 malloc 参数/ptrtoint 生成 i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(
      memrefOptions)); // memref -> llvm

  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = kLLVMIndexBitwidth;              // func lowering 到 llvm 时，其 index 转成 i64
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions)); // func -> llvm

  pm.addPass(mlir::frisk::createLLVMFuncOpAddGPUAttrPass(
      target)); // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
  pm.addPass(mlir::frisk::createGPUToROCDLOrNVVMPass(
      target, kLLVMIndexBitwidth)); // GPU indexOp to rocdl/nvvm indexOp

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createArithToLLVMConversionPass(
      arithOptions)); // arith -> llvm
  UBToLLVMConversionPassOptions ubOptions;
  ubOptions.indexBitwidth = kLLVMIndexBitwidth;
  pm.addPass(mlir::createUBToLLVMConversionPass(ubOptions)); // ub -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());                         // 手动写的去除多余UnrealizedCast
  pm.addPass(
      mlir::createReconcileUnrealizedCastsPass()); // 内置去除多余cast的pass
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(createMallocFuncOpArgTypeI32ToI64Pass());                      // 将malloc 的 func 的函数签名换成 i64，ptrtointOp/callOp跟着换（因为如果强制使用malloci32，后续llvmtranslation报错，llvm malloc只支持i64）
  // pm.addPass(mlir::createLowerGpuOpsToROCDLOpsPass());
  // pm.addPass(createConvertGPUPrintToLLVMPass());

  // pm.addPass(mlir::createGpuToLLVMConversionPass());
  if (mlir::failed(pm.run(mod))) {
    return false;
  }

  return true;
}

}  // namespace mlir::frisk
