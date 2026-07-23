#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"

#include "deepgengraph/Analysis/ThreadAnalysis.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LowerInfo.h"

using namespace mlir;

int readDeepgenGraphIRAndConvertToFriskPipeline(int argc, char ** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);

  // 首先，注册需要的 dialect
  ctx->loadDialect<
    func::FuncDialect, 
    arith::ArithDialect,
    tensor::TensorDialect,
    linalg::LinalgDialect,
    scf::SCFDialect,
    affine::AffineDialect,
    math::MathDialect,
    deepgengraph::DeepgengraphDialect,
    deepgengraph::triton::DeepgengraphTritonDialect,
    frisk::FriskDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  // 简单的输出，在 debug 的时候常用
  analyze::PointerTracer::getPointerInfo(*src);
  src->dump();
  mlir::PassManager pm(ctx.get());

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
  pm.addNestedPass<func::FuncOp>(mlir::frisk::createConvertFriskBaseToThreadLevelIRPass());
  // pm.addPass(mlir::createSymbolDCEPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createConvertFriskBaseToThreadLevelIRPass ---------\n"; llvm::outs().flush();src->dump();

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

  auto kernelType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto kernel = builder.create<frisk::KernelOp>(loc, "gpu_layout_test", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);

  constexpr int64_t kRows = 128;
  constexpr int64_t kCols = 32;
  auto alloc = builder.create<frisk::AllocBufferOp>(
      loc, ArrayRef<int64_t>{kRows, kCols}, builder.getF32Type(), 0,
      int64_t(frisk::attr::MemorySpace::Shared));

  auto shapeAttr = builder.getDenseI64ArrayAttr({kRows, kCols});
  auto paddingOffsets = builder.getDenseI64ArrayAttr({0, 0});
  auto row = builder.getAffineDimExpr(0);
  auto col = builder.getAffineDimExpr(1);
  auto baseMap = AffineMapAttr::get(
      AffineMap::get(2, 0, ArrayRef<AffineExpr>{row * kCols + col}, &ctx));
  auto gpuLayout = frisk::GPULayoutAttr::get(
      &ctx, shapeAttr, shapeAttr, paddingOffsets,
      frisk::attr::GPUTargetSpace::SHM, baseMap, frisk::GPUSwizzleAttr());

  alloc->setAttr("gpu_layout", gpuLayout);
  mod.dump();
}

int main(int argc, char** argv) {
  // readDeepgenGraphIRAndConvertToFriskPipeline(argc, argv);
  // if (testAffineMapCaluclate())
  //   return 1;
  testGPULayout();
  return 0;
}
