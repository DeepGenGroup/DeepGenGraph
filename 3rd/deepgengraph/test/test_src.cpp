#include "deepgengraph/Common.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "deepgengraph/Conversion/DeepgengraphToLinalgOnTensor/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include <cassert>
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"

#include "deepgengraph/Analysis/ThreadAnalysis.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"       // 包含 translateModuleToLLVMIR
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;

int readDeepgenGraphIRAndConvertToFriskPipeline(int argc, char ** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);

  // 首先，注册需要的 dialect
  ctx->loadDialect<
    func::FuncDialect, 
    arith::ArithDialect,
    tensor::TensorDialect,
    linalg::LinalgDialect,
    memref::MemRefDialect,
    scf::SCFDialect,
    affine::AffineDialect,
    math::MathDialect,
    deepgengraph::DeepgengraphDialect,
    deepgengraph::triton::DeepgengraphTritonDialect,
    frisk::FriskDialect,
    LLVM::LLVMDialect,
    vector::VectorDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  if (!src) {
    llvm::errs() << "Failed to parse input MLIR file: " << argv[1] << "\n";
    return 1;
  }
  // 简单的输出，在 debug 的时候常用
  analyze::PointerTracer::getPointerInfo(*src);
  src->dump();
  mlir::PassManager pm(ctx.get());

  auto AddPass = [&](std::unique_ptr<Pass> pass){
    PassManager pm(ctx.get());
    pm.addPass(std::move(pass));
    pm.run(src->getOperation());
  };

  auto AddPassNested = [&](std::unique_ptr<Pass> pass){
    PassManager pm(ctx.get());
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    pm.run(src->getOperation());
  };

  pm.addNestedPass<deepgengraph::KernelOp>(frisk::createDeepgenGraphSimplifyPass());
  pm.addPass(frisk::createAddKernelargPermuteInfoPass());
  pm.run(src->getOperation());


  llvm::outs() << "\n---------- after simplifyPass ---------\n"; llvm::outs().flush();src->dump();
  pm.addNestedPass<deepgengraph::KernelOp>(frisk::createConvertScfForOpPass());
  pm.addNestedPass<deepgengraph::KernelOp>(frisk::createMemspaceAnalyzePass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after scfForConversion ---------\n"; llvm::outs().flush();src->dump();

  pm.addPass(frisk::createConvertKernelOpToFriskPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createConvertKernelOpToFriskPass ---------\n"; llvm::outs().flush();src->dump();
  
  pm.addNestedPass<frisk::KernelOp>(frisk::createConvertMemAndCalcOpPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createConvertMemAndCalcOpPass ---------\n"; llvm::outs().flush();src->dump();
  
  pm.addNestedPass<frisk::KernelOp>(frisk::createFriskFuseBlockOpsPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createFriskFuseBlockOpsPass ---------\n"; llvm::outs().flush();src->dump();

  pm.addNestedPass<frisk::KernelOp>(frisk::createFuseBlockOpWithDTypeConvertOpPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createFuseBlockOpWithDTypeConvertOpPass ---------\n"; llvm::outs().flush();src->dump();

  pm.addPass(frisk::createConvertFriskToBasePass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createConvertFriskToBasePass ---------\n"; llvm::outs().flush();src->dump();

  
  // pm.addNestedPass<func::FuncOp>(frisk::createFriskLayoutInferPass());
  // pm.run(src->getOperation());
  // llvm::outs() << "\n---------- after createFriskLayoutInferPass ---------\n"; llvm::outs().flush();src->dump();

  pm.addNestedPass<func::FuncOp>(mlir::frisk::createConvertFriskBaseToThreadLevelIRPass());
  // pm.addNestedPass<func::FuncOp>(mlir::affine::createAffineLoopNormalizePass(true));
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // pm.addPass(mlir::createSymbolDCEPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createConvertFriskBaseToThreadLevelIRPass ---------\n"; llvm::outs().flush();src->dump();
  
  AddPass(frisk::createThreadLevelIRLegalizePass());
  llvm::outs() << "\n---- after threadIR legalize -----\n"; llvm::outs().flush(); src->dump();
  
  mlir::ModuleOp mod = *src;
  frisk::firstLowering(mod, src->getContext());
  frisk::secondLowering(mod, src->getContext(), frisk::Target::ROCm);
  llvm::outs() << "\n---- after secondLowering -----\n"; llvm::outs().flush(); src->dump();
  
  // ------- convert to llvmir text
  //  创建真正的 LLVM 上下文
  llvm::LLVMContext llvmContext;

  // 3. 将 MLIR ModuleOp 转换为 llvm::Module
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(mod, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR ModuleOp to LLVM IR.\n";
    return 1;
  }

  // 4. 将 llvm::Module 打印为文本
  std::string llvmIrStr;
  std::error_code ec;
  llvm::raw_fd_ostream os("finalLLVMText.ll",ec);
  if(!ec){
    llvmModule->print(os, /*AssemblyAnnotationWriter=*/nullptr);
    llvm::outs() << "[d] llvmIR 已输出到 finalLLVMText.ll\n" ; llvm::outs().flush();
  }
  return 0;
}


// 结论：不能直接逆map，得手动写affineMap的逆。
int testAffineMapCaluclate(){
  MLIRContext ctx;
  ctx.loadDialect<affine::AffineDialect>();
  OpBuilder builder(&ctx);

  auto x = builder.getAffineDimExpr(0);
  auto y = builder.getAffineDimExpr(1);
  auto affineMap = AffineMap::get(2, 0, ArrayRef<AffineExpr>{x * 10 + y}, &ctx);

  constexpr int64_t xValue = 3;
  constexpr int64_t yValue = 7;
  constexpr int64_t expected = xValue * 10 + yValue;

  SmallVector<Attribute, 2> operands = {
      builder.getIndexAttr(xValue),
      builder.getIndexAttr(yValue),
  };
  SmallVector<Attribute, 1> results;
  if (failed(affineMap.constantFold(operands, results)) || results.size() != 1) {
    llvm::errs() << "failed to calculate affine map\n";
    return 1;
  }

  auto result = cast<IntegerAttr>(results.front()).getInt();
  llvm::outs() << "affine map: " << affineMap << "\n";
  llvm::outs() << "x = " << xValue << ", y = " << yValue << ", z = " << result << "\n";

  if (result != expected) {
    llvm::errs() << "unexpected result, expected " << expected << "\n";
    return 1;
  }

  if (!affineMap.isPermutation())
    llvm::outs() << "forward map is not a permutation, skip inversePermutation\n";

  auto z = builder.getAffineDimExpr(0);
  auto inverseMap =
      AffineMap::get(1, 0, ArrayRef<AffineExpr>{z.floorDiv(10), z % 10}, &ctx);
  SmallVector<OpFoldResult, 1> zOperands = {builder.getIndexAttr(result)};
  SmallVector<OpFoldResult> inverseResults =
      affine::makeComposedFoldedMultiResultAffineApply(
          builder, builder.getUnknownLoc(), inverseMap, zOperands);

  auto inverseValues = getConstantIntValues(inverseResults);
  if (!inverseValues || inverseValues->size() != 2) {
    llvm::errs() << "failed to calculate inverse affine map\n";
    return 1;
  }

  int64_t recoveredX = (*inverseValues)[0];
  int64_t recoveredY = (*inverseValues)[1];
  llvm::outs() << "inverse affine map: " << inverseMap << "\n";
  llvm::outs() << "z = " << result << ", x = " << recoveredX
               << ", y = " << recoveredY << "\n";

  if (recoveredX != xValue || recoveredY != yValue) {
    llvm::errs() << "unexpected inverse result\n";
    return 1;
  }
  return 0;
}

void testGPULayout(){
  MLIRContext ctx;
  ctx.loadDialect<
    func::FuncDialect,
    arith::ArithDialect,
    tensor::TensorDialect,
    linalg::LinalgDialect,
    memref::MemRefDialect,
    scf::SCFDialect,
    affine::AffineDialect,
    math::MathDialect,
    deepgengraph::DeepgengraphDialect,
    deepgengraph::triton::DeepgengraphTritonDialect,
    frisk::FriskDialect
  >();

  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto func = builder.create<func::FuncOp>(loc, "gpu_layout_memref_test", funcType);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  constexpr int64_t kRows = 128;
  constexpr int64_t kCols = 32;
  constexpr int64_t kPaddedCols = 64;

  auto shapeAttr = builder.getDenseI64ArrayAttr({kRows, kCols});
  auto paddedShapeAttr = builder.getDenseI64ArrayAttr({kRows, kPaddedCols});
  auto paddingOffsets = builder.getDenseI64ArrayAttr({0, 0});
  auto row = builder.getAffineDimExpr(0);
  auto col = builder.getAffineDimExpr(1);
  auto baseMap = AffineMapAttr::get(
      AffineMap::get(2, 0, ArrayRef<AffineExpr>{row * kPaddedCols + col}, &ctx));
  auto swizzle = frisk::GPUSwizzleAttr::get(&ctx, 3, 4, 5);
  auto gpuLayout = frisk::GPULayoutAttr::get(
      &ctx, shapeAttr, paddedShapeAttr, paddingOffsets,
      frisk::attr::GPUTargetSpace::SHM, baseMap, swizzle);

  if (!isa<MemRefLayoutAttrInterface>(gpuLayout)) {
    llvm::errs() << "GPULayoutAttr does not implement MemRefLayoutAttrInterface\n";
    return;
  }

  auto gpuMemrefType = MemRefType::get(
      {kRows, kCols}, builder.getF32Type(),
      cast<MemRefLayoutAttrInterface>(gpuLayout),
      builder.getI64IntegerAttr(
          static_cast<int64_t>(frisk::attr::MemorySpace::Shared)));
  auto alloc = builder.create<memref::AllocaOp>(loc, gpuMemrefType);
  (void)alloc;
  builder.create<func::ReturnOp>(loc);

  llvm::outs() << "gpu memref type: " << gpuMemrefType << "\n";
  llvm::outs() << "layout affine map: " << gpuMemrefType.getLayout().getAffineMap()
               << "\n";
  mod.dump();
}


void testCompareAffinemap() {
  MLIRContext ctx;
  OpBuilder builder(&ctx);
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto map1 =
      AffineMap::get(2, 0, ArrayRef<AffineExpr>{(d0 + d1 * 2) * 3,  (d0 + d1 * 2) % 3}, &ctx);
  auto map2 =
      AffineMap::get(2, 0, ArrayRef<AffineExpr>{d0 * 3 + d1 * 6,  (d0 - d1) % 3}, &ctx);
  
  mlir::FlatLinearValueConstraints(2,0);

  if(simplifyAffineMap(map1) == simplifyAffineMap(map2)){
    llvm::outs() << "ok\n";
  }
  else{
    llvm::outs() << "error\n";
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0] << " <input.mlir>\n";
    return 1;
  }
  return readDeepgenGraphIRAndConvertToFriskPipeline(argc, argv);
  // if (testAffineMapCaluclate())
  //   return 1;
  // testGPULayout();
  // testCompareAffinemap();
}
