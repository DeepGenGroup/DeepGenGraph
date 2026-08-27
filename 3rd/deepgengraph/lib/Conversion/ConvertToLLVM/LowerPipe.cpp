#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include <memory>

namespace mlir::frisk {


// =====================================================================
//                  Vecotr Dialect To LLVM Dialect
// =====================================================================
// 将memref lowering到llvm上，因为 passes.h.inc中的base类没有提供可以选择indexBitWidth的options，所以自己写了一个
struct VectorToLLVMPass : public PassWrapper<VectorToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToLLVMPass)

  VectorToLLVMPass(unsigned indexBitWidth_=32) : indexBitWidth(indexBitWidth_) {};

  unsigned indexBitWidth;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(indexBitWidth);
    bool force32BitVectorIndices = false;
    if (indexBitWidth == 32) {
      force32BitVectorIndices = true;
    }

    LLVMTypeConverter converter(&getContext(), options);
    mlir::populateVectorToLLVMConversionPatterns(converter, patterns, false, force32BitVectorIndices);
    // mlir::populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createVectorToLLVMPass(int indexBitwidth){
  return std::make_unique<VectorToLLVMPass>(indexBitwidth);
}

bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context) {
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

bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context, Target target) {
  mlir::PassManager pm(context);
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addNestedPass<mlir::func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = 32;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  pm.addPass(createVectorToLLVMPass(32));                    // 自定义 vector to llvm pass
  // pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = 32;                              // 这个32会将malloc func的参数也定义为i32，以及将ptrtointOp的返回也是i32，llvm malloc func不支持i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm

  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = 32;                              // func loewring 到 llvm 时，其index转到llvm上是使用i32类型
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm

  pm.addPass(mlir::frisk::createLLVMFuncOpAddGPUAttrPass(target));                       // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
  pm.addPass(mlir::frisk::createGPUToROCDLOrNVVMPass(target, 32));          // GPU indexOp to rocdl/nvvm indexOp

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = 32;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));            // arith -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());                         // 手动写的去除多余UnrealizedCast
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());                // 内置去除多余cast的pass
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(createMallocFuncOpArgTypeI32ToI64Pass());                      // 将malloc 的 func 的函数签名换成 i64，ptrtointOp/callOp跟着换（因为如果强制使用malloci32，后续llvmtranslation报错，llvm malloc只支持i64）
  // pm.addPass(mlir::createLowerGpuOpsToROCDLOpsPass());
  // pm.addPass(createConvertGPUPrintToLLVMPass());
  
  // pm.addPass(mlir::createGpuToLLVMConversionPass());
  if (mlir::failed(pm.run(mod))){
    return false;
  }

  return true;  
}

}  // namespace mlir::frisk
