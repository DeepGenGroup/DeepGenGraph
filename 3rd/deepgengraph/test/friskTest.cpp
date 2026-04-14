#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"

using namespace mlir;

int main() {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);
  
  ctx->loadDialect<mlir::frisk::FriskDialect,
                  mlir::affine::AffineDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();

  mlir::OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  // create module
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  // module -> frisk.kernel
  auto kernelType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto kernel = builder.create<mlir::frisk::KernelOp>(loc, "frisk_test_kernel", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);

  // kernel body -> frisk.parallel
  auto parallel = builder.create<mlir::frisk::ParallelOp>(loc, ArrayRef<int64_t>{16, 16}, 256);
  Block *parallelEntry = parallel.addEntryBlock();
  builder.setInsertionPointToStart(parallelEntry);

  // Common test values.
  auto f32 = builder.getF32Type();
  // auto mem2D = MemRefType::get({16, 16}, f32);
  // auto mem1D = MemRefType::get({16}, f32);

  auto a = builder.create<mlir::frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{16, 16}, f32, 0, 3);
  auto b = builder.create<mlir::frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{16, 16}, f32, 0, 3);
  auto c = builder.create<mlir::frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{16, 16}, f32, 0, 0);
  auto r = builder.create<mlir::frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{16}, f32, 0, 0);

  // Dynamic slice metadata for copy/reduce/buffer_view testing.
  auto id2D = AffineMap::getMultiDimIdentityMap(2, ctx.get());
  auto range2D = builder.getDenseI64ArrayAttr({16, 16});
  auto map2DAttr = AffineMapAttr::get(id2D);
  llvm::SmallVector<Value, 2> idx2DVec{parallelEntry->getArgument(0), parallelEntry->getArgument(1)};
  ValueRange idx2D(idx2DVec);

  // buffer_view test: take a 2D sub-view from A with shape [8, 16].
  auto subRange2D = builder.getDenseI64ArrayAttr({8, 16});
  auto aView = builder.create<mlir::frisk::BufferViewOp>(
      loc, a.getResult(), idx2D, map2DAttr, subRange2D);
  builder.create<mlir::frisk::FillOp>(loc, aView.getView(), builder.getF32FloatAttr(3.0f));
  builder.create<mlir::frisk::GemmWaitOp>(loc, aView.getView());

  // buffer_view test: take a 1D sub-view from R with shape [8].
  auto id1D = AffineMap::getMultiDimIdentityMap(1, ctx.get());
  auto map1DAttr = AffineMapAttr::get(id1D);
  auto subRange1D = builder.getDenseI64ArrayAttr({8});
  llvm::SmallVector<Value, 1> idx1DVec{parallelEntry->getArgument(0)};
  ValueRange idx1D(idx1DVec);
  auto rView = builder.create<mlir::frisk::BufferViewOp>(
      loc, r.getResult(), idx1D, map1DAttr, subRange1D);
  builder.create<mlir::frisk::GemmWaitOp>(loc, rView.getView());

  // gemm / gemmwait / fill / copy / reduce.
  builder.create<mlir::frisk::GemmOp>(loc, a.getResult(), b.getResult(), c.getResult(), false, false);
  builder.create<mlir::frisk::GemmWaitOp>(loc, c.getResult());
  builder.create<mlir::frisk::FillOp>(loc, c.getResult(), builder.getF32FloatAttr(0.0f));
  builder.create<mlir::frisk::CopyOp>(loc, a.getResult(), b.getResult());
  builder.create<mlir::frisk::ReduceOp>(loc, c.getResult(), r.getResult(), "add", 1);

  // forOp test.
  builder.create<mlir::frisk::ForOp>(loc, 0, 8, 1, [&](Value iv) {
    (void)iv;
    builder.create<mlir::frisk::GemmWaitOp>(loc, c.getResult());
  });

  // blockOp test.
  builder.create<mlir::frisk::BlockOp>(loc, ArrayRef<int64_t>{2, 2}, [&](ValueRange blockIvs) {
    (void)blockIvs;
    builder.create<mlir::frisk::FillOp>(loc, b.getResult(), builder.getF32FloatAttr(1.0f));
  });

  // ifOp test (IntegerSet embedded condition).
  auto d0 = getAffineDimExpr(0, ctx.get());
  auto d1 = getAffineDimExpr(1, ctx.get());
  auto s0 = getAffineSymbolExpr(0, ctx.get());
  IntegerSet condSet = IntegerSet::get(/*dimCount=*/2, /*symbolCount=*/1,
                                       ArrayRef<AffineExpr>{d0 + s0 - d1},
                                       ArrayRef<bool>{false});
  auto symN = builder.create<arith::ConstantIndexOp>(loc, 16);
  auto ifOp = builder.create<mlir::frisk::IfOp>(
      loc, condSet, ValueRange{parallelEntry->getArgument(0), parallelEntry->getArgument(1), symN});

  {
    auto *thenBlock = new Block();
    ifOp.getThenRegion().push_back(thenBlock);
    OpBuilder thenBuilder = OpBuilder::atBlockBegin(thenBlock);
    thenBuilder.create<mlir::frisk::GemmWaitOp>(loc, c.getResult());
    thenBuilder.create<mlir::frisk::EndOp>(loc);
  }
  {
    auto *elseBlock = new Block();
    ifOp.getElseRegion().push_back(elseBlock);
    OpBuilder elseBuilder = OpBuilder::atBlockBegin(elseBlock);
    elseBuilder.create<mlir::frisk::FillOp>(loc, c.getResult(), builder.getF32FloatAttr(2.0f));
    elseBuilder.create<mlir::frisk::EndOp>(loc);
  }

  mod->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
