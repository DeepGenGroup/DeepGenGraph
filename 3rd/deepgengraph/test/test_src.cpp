#include "deepgengraph/Common.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

using namespace mlir;

namespace {

bool isUnknownLocation(Location loc) { return isa<UnknownLoc>(loc); }

bool isCandidateOperationLine(llvm::StringRef line) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("//") || trimmed.starts_with("#") ||
      trimmed.starts_with("^") || trimmed.starts_with("}"))
    return false;

  if (trimmed.starts_with("module") || trimmed.starts_with("func.func") ||
      trimmed.starts_with("return") || trimmed.starts_with("scf.yield") ||
      trimmed.starts_with("scf.if") || trimmed.starts_with("scf.for") ||
      trimmed.starts_with("deepgengraph.") ||
      trimmed.starts_with("deepgengraph_") || trimmed.starts_with("arith."))
    return true;

  return trimmed.contains(" = ");
}

SmallVector<Location> collectSourceOperationLocations(StringRef filename,
                                                      MLIRContext *ctx) {
  SmallVector<Location> locs;
  auto fileOrErr = llvm::MemoryBuffer::getFile(filename);
  if (!fileOrErr) {
    llvm::errs() << "warning: failed to read source for debug locations: "
                 << filename << "\n";
    return locs;
  }

  llvm::StringRef buffer = (*fileOrErr)->getBuffer();
  size_t start = 0;
  unsigned lineNo = 1;
  while (start <= buffer.size()) {
    size_t end = buffer.find('\n', start);
    llvm::StringRef line =
        buffer.slice(start, end == llvm::StringRef::npos ? buffer.size() : end);
    if (isCandidateOperationLine(line)) {
      size_t col = line.find_first_not_of(" \t");
      locs.push_back(FileLineColLoc::get(
          ctx, filename, lineNo,
          static_cast<unsigned>(col == llvm::StringRef::npos ? 1 : col + 1)));
    }
    if (end == llvm::StringRef::npos)
      break;
    start = end + 1;
    ++lineNo;
  }
  return locs;
}

void setUnknownBlockArgLocs(Operation *op, Location loc) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments()) {
        if (isUnknownLocation(arg.getLoc()))
          arg.setLoc(loc);
      }
    }
  }
}

void attachSourceLocationsFromText(ModuleOp module, StringRef filename) {
  SmallVector<Location> sourceLocs =
      collectSourceOperationLocations(filename, module.getContext());
  if (sourceLocs.empty())
    return;

  unsigned locIndex = 0;
  module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    Location sourceLoc =
        sourceLocs[std::min<unsigned>(locIndex, sourceLocs.size() - 1)];
    if (isUnknownLocation(op->getLoc()))
      op->setLoc(sourceLoc);
    setUnknownBlockArgLocs(op, sourceLoc);
    ++locIndex;
  });
}

void fillUnknownLocationsFromParents(Operation *op, Location inheritedLoc) {
  Location currentLoc = op->getLoc();
  if (isUnknownLocation(currentLoc)) {
    op->setLoc(inheritedLoc);
    currentLoc = inheritedLoc;
  }
  setUnknownBlockArgLocs(op, currentLoc);

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block)
        fillUnknownLocationsFromParents(&nestedOp, currentLoc);
    }
  }
}

FileLineColLoc findFileLineColLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return findFileLineColLoc(nameLoc.getChildLoc());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location nestedLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = findFileLineColLoc(nestedLoc))
        return fileLoc;
    }
  }
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto fileLoc = findFileLineColLoc(callLoc.getCaller()))
      return fileLoc;
    return findFileLineColLoc(callLoc.getCallee());
  }
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return findFileLineColLoc(opaqueLoc.getFallbackLocation());
  return {};
}

FileLineColLoc findFileLineColLoc(Operation *op) {
  if (auto fileLoc = findFileLineColLoc(op->getLoc()))
    return fileLoc;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (auto fileLoc = findFileLineColLoc(&nestedOp))
          return fileLoc;
      }
    }
  }
  return {};
}

void attachLLVMDebugScopes(ModuleOp module, StringRef inputFilename) {
  MLIRContext *ctx = module.getContext();
  Builder builder(ctx);
  llvm::SmallString<128> directory;
  llvm::SmallString<64> basename;
  llvm::sys::path::append(directory, llvm::sys::path::parent_path(inputFilename));
  llvm::sys::path::append(basename, llvm::sys::path::filename(inputFilename));
  if (directory.empty())
    directory = ".";

  auto diFile = LLVM::DIFileAttr::get(ctx, basename.str(), directory.str());
  auto compileUnit = LLVM::DICompileUnitAttr::get(
      ctx, DistinctAttr::create(UnitAttr::get(ctx)), llvm::dwarf::DW_LANG_C,
      diFile, builder.getStringAttr("DeepGenGraph MLIR"), false,
      LLVM::DIEmissionKind::LineTablesOnly, LLVM::DINameTableKind::Default);
  auto subroutineType =
      LLVM::DISubroutineTypeAttr::get(ctx, ArrayRef<LLVM::DITypeAttr>{});

  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.getBody().empty())
      return;

    if (func.getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    FileLineColLoc fileLoc = findFileLineColLoc(func.getOperation());
    Location funcLoc = fileLoc ? Location(fileLoc) : func.getLoc();
    unsigned line = fileLoc ? fileLoc.getLine() : 1;
    auto name = builder.getStringAttr(func.getName());
    auto subprogram = LLVM::DISubprogramAttr::get(
        ctx, DistinctAttr::create(UnitAttr::get(ctx)), compileUnit, diFile,
        name, name, diFile, line, line, LLVM::DISubprogramFlags::Definition,
        subroutineType, {}, {});
    func->setLoc(FusedLoc::get(ctx, {funcLoc}, subprogram));
  });

  module.walk([&](LLVM::LLVMFuncOp func) {
    auto fusedSubprogram =
        func.getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>();
    if (!fusedSubprogram)
      return;

    auto subprogram = fusedSubprogram.getMetadata();
    func.walk([&](Operation *op) {
      if (op == func.getOperation())
        return;

      FileLineColLoc fileLoc = findFileLineColLoc(op->getLoc());
      if (!fileLoc)
        return;

      StringRef locFilename = fileLoc.getFilename().getValue();
      if (locFilename == inputFilename)
        return;

      auto opDiFile = LLVM::DIFileAttr::get(
          ctx, llvm::sys::path::filename(locFilename),
          llvm::sys::path::parent_path(locFilename));
      auto lexicalBlock = LLVM::DILexicalBlockFileAttr::get(
          ctx, subprogram, opDiFile, /*discriminator=*/0);
      op->setLoc(FusedLoc::get(ctx, {op->getLoc()}, lexicalBlock));
    });
  });
}

} // namespace

frisk::KernelConfig* frisk::GetKernelConfig() {
  static frisk::KernelConfig cfg;
  cfg.num_threads = 64;
  cfg.gridDimXYZ = {64,32,1};
  return &cfg;
}


void frisk::AppendNameToLoc(mlir::Operation* targetOp){
  // 假设 op 是已有的 Operation*，newOpName 是你要追加的信息
  auto origLoc = targetOp->getLoc();
  mlir::MLIRContext *ctx = targetOp->getContext();
  llvm::StringRef extraInfo = targetOp->getName().getStringRef();
  mlir::Location newLoc = origLoc; // 默认降级为原 Loc

  // 1. 匹配并提取 FileLineColLoc
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(origLoc)) {
    // 获取原 fileName 字符串
    llvm::StringRef origFileName = fileLoc.getFilename().getValue();
    // 拼接新的 fileName
    std::string newFileName = (origFileName + "_" + extraInfo).str();
    // 2. 重新构建包含新 fileName 的 FileLineColLoc
    newLoc = mlir::FileLineColLoc::get(
        mlir::StringAttr::get(ctx, newFileName),
        fileLoc.getLine(),
        fileLoc.getColumn()
    );
  }
  targetOp->setLoc(newLoc);
}

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
  attachSourceLocationsFromText(*src, argv[1]);
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

  auto AddKernelPass = [&](std::unique_ptr<Pass> pass){
    PassManager pm(ctx.get());
    pm.addNestedPass<frisk::KernelOp>(std::move(pass));
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

  AddPass(frisk::createConvertKernelOpToFriskPass());
  llvm::outs() << "\n---------- after createConvertKernelOpToFriskPass ---------\n"; llvm::outs().flush();src->dump();
  
  AddKernelPass(frisk::createConvertMemAndCalcOpPass());
  AddPass(mlir::createReconcileUnrealizedCastsPass());
  llvm::outs() << "\n---------- after createConvertMemAndCalcOpPass ---------\n"; llvm::outs().flush();src->dump();
  
  AddKernelPass(frisk::createFriskFuseBlockOpsPass());
  llvm::outs() << "\n---------- after createFriskFuseBlockOpsPass ---------\n"; llvm::outs().flush();src->dump();

  AddKernelPass(frisk::createFuseBlockOpWithDTypeConvertOpPass());
  llvm::outs() << "\n---------- after createFuseBlockOpWithDTypeConvertOpPass ---------\n"; llvm::outs().flush();src->dump();

  AddPass(frisk::createConvertFriskToBasePass());
  llvm::outs() << "\n---------- after createConvertFriskToBasePass ---------\n"; llvm::outs().flush();src->dump();

  
  // pm.addNestedPass<func::FuncOp>(frisk::createFriskLayoutInferPass());
  // pm.run(src->getOperation());
  // llvm::outs() << "\n---------- after createFriskLayoutInferPass ---------\n"; llvm::outs().flush();src->dump();

  AddPassNested(mlir::frisk::createConvertFriskBaseToThreadLevelIRPass());
  AddPassNested(mlir::affine::createAffineLoopNormalizePass(true));
  AddPass(mlir::createCSEPass());
  AddPassNested(mlir::bufferization::createBufferLoopHoistingPass());
  AddPassNested(mlir::bufferization::createBufferHoistingPass());
  AddPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  AddPass(mlir::createCanonicalizerPass());
  mlir::affine::AffineVectorizeOptions opt;  opt.vectorSizes = {4};
  AddPassNested(mlir::affine::createAffineVectorize(opt));
  AddPassNested( mlir::affine::createLoopUnrollPass() );
  AddPassNested( mlir::affine::createLoopFusionPass( int(frisk::friskMs::Local) ) );

  AddPassNested(mlir::createMem2Reg());

  AddPass(mlir::createCSEPass());
  AddPass(mlir::createCanonicalizerPass());

  // pm.addPass(mlir::createSymbolDCEPass());
  llvm::outs() << "\n---------- after createConvertFriskBaseToThreadLevelIRPass ---------\n"; llvm::outs().flush();src->dump();
  
  AddPass(frisk::createThreadLevelIRLegalizePass());
  llvm::outs() << "\n---- after threadIR legalize -----\n"; llvm::outs().flush(); src->dump();
  
  mlir::ModuleOp mod = *src;
  frisk::firstLowering(mod, src->getContext());
  frisk::secondLowering(mod, src->getContext(), frisk::Target::ROCm);
  llvm::outs() << "\n---- after secondLowering -----\n"; llvm::outs().flush(); src->dump();
  fillUnknownLocationsFromParents(mod.getOperation(), mod.getLoc());
  attachLLVMDebugScopes(mod, argv[1]);
  
  // ------- convert to llvmir text
  //  创建真正的 LLVM 上下文
  llvm::LLVMContext llvmContext;

  // 3. 将 MLIR ModuleOp 转换为 llvm::Module
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(mod, llvmContext, argv[1]);

  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR ModuleOp to LLVM IR.\n";
    return 1;
  }
  llvmModule->setSourceFileName(argv[1]);

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

// TODO : 实现linalg.copy 实现两个 memref<4x3xf32> 之间的copy
void testLinalgCopy() {
  MLIRContext ctx;
  ctx.loadDialect<affine::AffineDialect, arith::ArithDialect,
                  func::FuncDialect, gpu::GPUDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, scf::SCFDialect>();

  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  constexpr int64_t kRows = 64;
  constexpr int64_t kCols = 32;
  constexpr int64_t kThreads = 32;
  constexpr int64_t kElements = kRows * kCols;
  auto memrefType = MemRefType::get({64, 32}, builder.getF32Type());
  auto funcType =
      builder.getFunctionType(TypeRange{memrefType, memrefType}, TypeRange{});
  auto func = builder.create<func::FuncOp>(loc, "linalg_copy_memref_test",
                                           funcType);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  SmallVector<Value, 1> inputs{entry->getArgument(0)};
  SmallVector<Value, 1> outputs{entry->getArgument(1)};
  builder.create<linalg::CopyOp>(loc, inputs, outputs);
  builder.create<func::ReturnOp>(loc);

  if (failed(verify(mod))) {
    llvm::errs() << "failed to verify linalg.copy test module before pass\n";
    mod.print(llvm::errs());
    llvm::errs() << "\n";
    return;
  }

  llvm::outs() << "\n---------- before convert-linalg-to-parallel-loops "
                  "---------\n";
  mod.print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();

  // MLIR 自带的 linalg pass 可以把 buffer 语义的 linalg op 降成
  // scf.parallel。对 linalg.copy 来说，效果就是把三维 copy 切成
  // parallel loop nest，loop body 里变成一次 load + store。
  //
  // 注意：scf.parallel 只是目标无关的并行循环，还没有绑定到具体
  // gpu.thread_id / threadIdx.x。如果要降到真实线程，后面还需要接
  // gpu-map-parallel-loops、gpu.launch lowering，或者接入 Frisk 自己的
  // thread-level lowering 规则。
  PassManager pm(&ctx);
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  if (failed(pm.run(mod))) {
    llvm::errs() << "failed to run convert-linalg-to-parallel-loops\n";
    mod.print(llvm::errs());
    llvm::errs() << "\n";
    return;
  }

  if (failed(verify(mod))) {
    llvm::errs() << "failed to verify linalg.copy test module after pass\n";
    mod.print(llvm::errs());
    llvm::errs() << "\n";
    return;
  }

  llvm::outs() << "\n---------- after convert-linalg-to-parallel-loops "
                  "----------\n";
  mod.print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();

  // 具体切到 32 个线程的一种直接方案：
  //   linear = threadIdx.x; linear < 64 * 32; linear += 32
  //   row = linear / 32
  //   col = linear % 32
  // 也就是第 tid 个线程负责线性下标 tid, tid + 32, tid + 64, ...
  // 每个线程处理 (64 * 32) / 32 = 64 个元素。
  //
  // 这里没有再从 scf.parallel 自动 lower，而是手工构造目标形态，
  // 方便观察后续 Frisk pass 可以生成什么 IR。
  builder.setInsertionPointToEnd(mod.getBody());
  auto threadFunc = builder.create<func::FuncOp>(
      loc, "thread_mapped_copy_32_threads", funcType);
  Block *threadEntry = threadFunc.addEntryBlock();
  builder.setInsertionPointToStart(threadEntry);

  auto cElements = builder.create<arith::ConstantIndexOp>(loc, kElements);
  auto cThreads = builder.create<arith::ConstantIndexOp>(loc, kThreads);
  auto tidx = builder.create<gpu::ThreadIdOp>(
      loc, gpu::Dimension::x, builder.getIndexAttr(kThreads));
  auto forOp = builder.create<scf::ForOp>(loc, tidx.getResult(), cElements,
                                          cThreads);

  builder.setInsertionPointToStart(forOp.getBody());
  Value linear = forOp.getInductionVar();
  auto d0 = builder.getAffineDimExpr(0);
  auto rowMap = AffineMap::get(1, 0, d0.floorDiv(kCols), &ctx);
  auto colMap = AffineMap::get(1, 0, d0 % kCols, &ctx);
  Value row = builder.create<affine::AffineApplyOp>(loc, rowMap, linear);
  Value col = builder.create<affine::AffineApplyOp>(loc, colMap, linear);
  Value value = builder.create<memref::LoadOp>(
      loc, threadEntry->getArgument(0), ValueRange{row, col});
  builder.create<memref::StoreOp>(loc, value, threadEntry->getArgument(1),
                                  ValueRange{row, col});

  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  if (failed(verify(mod))) {
    llvm::errs() << "failed to verify thread mapped copy example\n";
    mod.print(llvm::errs());
    llvm::errs() << "\n";
    return;
  }

  llvm::outs() << "\n---------- explicit thread mapping, 32 threads "
                  "----------\n";
  threadFunc.print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();

}

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0] << " <input.mlir>\n";
    return 1;
  }
  return readDeepgenGraphIRAndConvertToFriskPipeline(argc, argv);

  // testLinalgCopy();
  // return 0;
}
