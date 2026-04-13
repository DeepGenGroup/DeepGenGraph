#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

#include "deepgengraph/Conversion/DeepgengraphTritonToThreadImp/Passes.h"
#include "deepgengraph/Dialect/ThreadImp/Transforms/Passes.h"
#include "deepgengraph/Analysis/ThreadAnalysis.h"

using namespace mlir;

int lowerDeepgengraphToAffine(int argc, char ** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);
  mlir::Builder builder(ctx.get());

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
    deepgengraph::triton::DeepgengraphTritonDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  // 简单的输出，在 debug 的时候常用
  src->dump();

  mlir::PassManager pm(ctx.get());

  // // pm.addNestedPass<deepgengraph::KernelOp>(deepgengraph::createConvertDeepgengraphToLinalgOnTensorPass());
  // pm.addPass(deepgengraph::createConvertDeepgengraphToLinalgOnTensorPass());
  // pm.addPass( mlir::createLinalgGeneralizeNamedOpsPass());
  // pm.addPass( mlir::bufferization::createEmptyTensorEliminationPass()) ;
  // pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  // pm.run(src->getOperation());
  // llvm::outs() << "\n====== after lower to linalg on tensor =====\n" ; llvm::outs().flush();
  // src->dump();
  
  // llvm::outs() << "\n====== after lower to affine =====\n" ; llvm::outs().flush();

  // pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  // pm.addPass(mlir::createConvertTensorToLinalgPass());
  // pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

  // mlir::affine::AffineParallelizeOptions opt;
  // opt.maxNested = 2;  // bz.by- simple parallel is OK.  bx need tile & para with BM
  // pm.addNestedPass<func::FuncOp>(mlir::affine::createAffineParallelize(opt));
  // // pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());  // to scf.parallel

  // // pm.addPass(mlir::affine::createLoopFusionPass(0,0,true,affine::FusionMode::Greedy)) ;
  
  // pm.addPass(mlir::createLowerAffinePass()) ;
  // pm.addPass(createParallelLoopFusionPass());


  
  // // pm.addNestedPass<func::FuncOp>(mlir::affine::createLoopTilingPass(255*4)); 
  
  // pm.run(src->getOperation());

  src->dump();
  return 0;
}

int readDeepgenGraphIRAndConvert(int argc, char ** argv) {
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
    tilelang::TilelangDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  // 简单的输出，在 debug 的时候常用
  src->dump();

  mlir::PassManager pm(ctx.get());
  pm.addPass(deepgengraph::createConvertDeepgenGraphToTilelangPass());

  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after conversion ---------\n"; llvm::outs().flush();
  src->dump();
  return 0;
}

#if 0  // deprecated
int testUseTilelangDialect(){
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
    memref::MemRefDialect,
    tilelang::TilelangDialect
    >();
  
  mlir::OpBuilder builder = mlir::OpBuilder(ctx.get());
  mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto &s = mod.getBodyRegion().getBlocks().front();
  builder.setInsertionPointToStart(&s);
  std::vector<int64_t> shape1 = {1,32,1024,128};
  auto tensorType = mlir::RankedTensorType::get(shape1, builder.getF32Type());
  std::vector<Type> insType = {tensorType};
  std::vector<Type> outType = {};
  auto funcType = builder.getFunctionType( insType, outType);
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> argAttrs = {});
  auto primFuncOp = builder.create<tilelang::PrimFuncOp>(builder.getUnknownLoc(), "testPrim", funcType);
  auto entry = primFuncOp.addEntryBlock();
  auto args = entry->getArguments();
  
  builder.setInsertionPointToEnd(entry);
  auto block_num = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getI32IntegerAttr(128));
  auto withKernelOp = builder.create<tilelang::WithKernelOp>(builder.getUnknownLoc(), block_num->getResult(0), true ,nullptr);

  // builder.setInsertionPointToEnd( newEntry );
  builder.setInsertionPointToStart(&withKernelOp.getScopedBody().front());
  builder.create<linalg::AddOp>(withKernelOp->getLoc(), mlir::ValueRange{args[0], args[0]}, mlir::ValueRange{args[0]} );
  builder.create<arith::AddIOp>(withKernelOp->getLoc(), withKernelOp.getScopedBody().getArgument(0), withKernelOp.getScopedBody().getArgument(1));
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemSpace memspace, ArrayRef<int64_t> shape, Type elementType);
  
  std::vector<int64_t> gm_shape = {1,2,3,4};
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemSpace memspace, ArrayRef<int64_t> shape, Type elementType);

  auto memAttr= tilelang::MemSpaceAttr::get(builder.getContext(), tilelang::MemSpace::GM);
  auto shapeAttr = builder.getI64ArrayAttr(gm_shape);
  auto allocop = builder.create<tilelang::AllocOp>(withKernelOp->getLoc(), tilelang::MemSpace::GM, gm_shape, builder.getF16Type());
  auto allocop2 = builder.create<tilelang::AllocOp>(withKernelOp->getLoc(), tilelang::MemSpace::L0A, gm_shape, builder.getF16Type());
  auto allocop3 = builder.create<tilelang::AllocOp>(withKernelOp->getLoc(), tilelang::MemSpace::L0B, gm_shape, builder.getF16Type());
  auto allocop4 = builder.create<tilelang::AllocOp>(withKernelOp->getLoc(), tilelang::MemSpace::L0C, gm_shape, builder.getF16Type());
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::AffineMapAttr src_map, ::mlir::Value dst, ::mlir::AffineMapAttr dst_map);
  
  builder.create<tilelang::TileAddOp>(withKernelOp->getLoc(), allocop2,allocop3);
  builder.create<tilelang::TileMaxOp>(withKernelOp->getLoc(), allocop2,allocop3);
  builder.create<tilelang::TileMinOp>(withKernelOp->getLoc(), allocop2,allocop3);
  builder.create<tilelang::TileSubOp>(withKernelOp->getLoc(), allocop2,allocop3);
  builder.create<tilelang::TileDivOp>(withKernelOp->getLoc(), allocop2,allocop3);
  builder.create<tilelang::TileMulOp>(withKernelOp->getLoc(), allocop2,allocop3);
  

  builder.create<tilelang::CopyOp>(withKernelOp->getLoc(), allocop, mlir::AffineMap::get(builder.getContext()), allocop2, mlir::AffineMap::get(builder.getContext()));
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::tilelang::CrossFlagAttr flag, ::mlir::IntegerAttr id);
  auto flag = builder.create<tilelang::SetCrossFlagOp>(
    withKernelOp->getLoc(),
    tilelang::CrossFlagAttr::get(builder.getContext(), tilelang::CrossFlag::MTE1), 
    builder.getI32IntegerAttr(1));
  
  builder.create<tilelang::WaitCrossFlagOp>(withKernelOp->getLoc(), flag);


  auto c_scope = builder.create<tilelang::WithScopeOp>(withKernelOp->getLoc(), true );
  auto v_scope = builder.create<tilelang::WithScopeOp>(withKernelOp->getLoc(), false );
  
  builder.setInsertionPointToStart(&c_scope.getScopedBody().front());
  builder.create<arith::AddIOp>(c_scope->getLoc(), withKernelOp.getBodyArgs(0), withKernelOp.getBodyArgs(1));
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value matrixA, Value matrixB, Value matrixC, bool transA, bool transB, bool init);
  builder.create<tilelang::GemmV0Op>(c_scope->getLoc(), allocop2, allocop3, allocop4, false, false, true);
  auto lb = builder.create<arith::ConstantIntOp>(c_scope->getLoc(), 0 , 32);
  auto ub = builder.create<arith::ConstantIntOp>(c_scope->getLoc(), 88 , 32);
  builder.create<tilelang::SerialForOp>(c_scope->getLoc(), lb, ub);
  builder.setInsertionPointToStart(&v_scope.getScopedBody().front());
  builder.create<arith::AddIOp>(v_scope->getLoc(), withKernelOp.getBodyArgs(1), withKernelOp.getBodyArgs(0));

  builder.setInsertionPointToEnd(entry);
  builder.create<tilelang::ReturnOp>(primFuncOp->getLoc());

  llvm::outs() << mod << "\n"; llvm::outs().flush();
  return 0;
}
#endif


int readDeepgenGraphIRAndConvertToThreadsImp(int argc, char ** argv) {
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
    tilelang::TilelangDialect,
    mlir::threadimp::ThreadImpDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  // 简单的输出，在 debug 的时候常用
  analyze::PointerTracer::getPointerInfo(*src);
  src->dump();
  const auto& infoMap = analyze::PointerTracer::getMap();
  mlir::PassManager pm(ctx.get());
  // pm.addPass(deepgengraph::createConvertDeepgengraphTritonToThreadImpPass());
  pm.addNestedPass<deepgengraph::KernelOp>(mlir::threadimp::createInlineDevicekernelOpPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after createInlineDevicekernelOpPass ---------\n"; llvm::outs().flush();src->dump();
  pm.addPass(mlir::threadimp::createConvertMemOpPass());
  pm.addPass(threadimp::createConvertBlockCalcOpToThreadImpPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n---------- after conversion ---------\n"; llvm::outs().flush();src->dump();
  return 0;
}


int main(int argc, char** argv) {
  // readDeepgenGraphIRAndConvert(argc, argv);
  // testUseTilelangDialect();
  // lowerDeepgengraphToAffine(argc,argv);
  readDeepgenGraphIRAndConvertToThreadsImp(argc, argv);
  return 0;
}