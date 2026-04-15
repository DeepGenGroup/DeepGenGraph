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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"

#include <cmath>
#include <math.h>
#include <stdio.h>

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"

using namespace mlir;

int main() {
  DialectRegistry registry;
  registerAllExtensions(registry);
  registerAllDialects(registry);
  auto ctx = std::make_unique<MLIRContext>(registry);
  
  ctx->loadDialect<frisk::FriskDialect,
                  affine::AffineDialect,
                  func::FuncDialect,
                  arith::ArithDialect,
                  math::MathDialect>();

  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  // create module
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  // Common test values.
  auto f32 = builder.getF32Type();
  auto f16 = builder.getF16Type();
  int64_t b = 1, h = 32, s = 2048, d = 128;
  int64_t BM = 128, BN = 128;
  auto arg_type = MemRefType::get({b, h, s, d}, f16, {}, static_cast<unsigned>(1));

  // module -> frisk.kernel
  auto kernelType = builder.getFunctionType(TypeRange{arg_type, arg_type, arg_type, arg_type}, TypeRange{});
  auto kernel = builder.create<frisk::KernelOp>(loc, "attn", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);
  auto Q = kernelEntry->getArgument(0);
  auto K = kernelEntry->getArgument(1);
  auto V = kernelEntry->getArgument(2);
  auto O = kernelEntry->getArgument(3);

  // kernel body -> frisk.parallel
  auto parallel = builder.create<frisk::ParallelOp>(loc, ArrayRef<int64_t>{1, 32, s/BM}, 128);
  Block *parallelEntry = parallel.addEntryBlock();
  builder.setInsertionPointToStart(parallelEntry);
  auto bz = parallelEntry->getArgument(0);
  auto by = parallelEntry->getArgument(1);
  auto bx = parallelEntry->getArgument(2);

  // shared memory
  auto smemQ = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f16, 128, 3);
  auto smemK = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BN, d}, f16, 128, 3);
  auto smemV = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BN, d}, f16, 128, 3);
  auto smemO = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f16, 128, 3);

  // regsiters memory
  auto acc_s = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BN}, f32, 0, 0);
  auto acc_s_cast = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BN}, f16, 0, 0);
  auto acc_o = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f32, 0, 0);
  auto scores_max = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_max_prev = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_scale = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_sum = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto logsum = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);

  // init
  auto zore = builder.getAffineConstantExpr(0);
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);
  auto mapQO = AffineMap::get(3, 0, ArrayRef<AffineExpr>{d0, d1, d2 * BM, zore}, ctx.get());
  auto mapKV = AffineMap::get(3, 0, ArrayRef<AffineExpr>{d0, d1, d2, zore}, ctx.get());
  auto Q_view = builder.create<frisk::BufferViewOp>(loc, Q, ValueRange({bz, by, bx}), mapQO, ArrayRef<int64_t>{1, 1, BM, d});
  builder.create<frisk::CopyOp>(loc, Q_view, smemQ.getResult());
  builder.create<frisk::FillOp>(loc, acc_o.getResult(), builder.getF32FloatAttr(0.0f));
  builder.create<frisk::FillOp>(loc, logsum.getResult(), builder.getF32FloatAttr(0.0f));
  builder.create<frisk::FillOp>(loc, scores_max.getResult(), builder.getF32FloatAttr(-INFINITY));

  // for
  builder.create<frisk::ForOp>(loc, 0, s, BN, [&](Value iv) {
    auto K_view = builder.create<frisk::BufferViewOp>(loc, K, ValueRange({bz, by, iv}), mapKV, ArrayRef<int64_t>{1, 1, BN, d});
    builder.create<frisk::CopyOp>(loc, K_view, smemK.getResult());
    // gemm
    builder.create<frisk::GemmOp>(loc, smemQ, smemK, acc_s, false, true);
    // ====== softmax ======
    // copy perv_max to max
    builder.create<frisk::CopyOp>(loc, scores_max_prev, scores_max);
    // fill max
    builder.create<frisk::FillOp>(loc, scores_max, builder.getF32FloatAttr(-INFINITY));
    // reduce max
    builder.create<frisk::ReduceOp>(loc, acc_s, scores_max, "max", 1);
    // m = max(score_max, score_max_prev)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, scores_max, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max_prev, blockIvs);
      auto m_elem = builder.create<arith::MaxNumFOp>(loc, elem1, elem2);
      builder.create<affine::AffineStoreOp>(loc, m_elem, scores_max, blockIvs);
    });
    // scale = exp(scores_max_prev - scores_max)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, scores_max_prev, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max, blockIvs);
      auto sub_elem = builder.create<arith::SubFOp>(loc, elem1, elem2);
      auto exp1 = builder.create<math::ExpOp>(loc, sub_elem);
      builder.create<affine::AffineStoreOp>(loc, exp1, scores_max, blockIvs);
    });
    // scc_s = exp(scc_s - scores_max)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, BN}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_s, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max, ValueRange({blockIvs[0]}));
      auto sub_elem = builder.create<arith::SubFOp>(loc, elem1, elem2);
      auto exp2 = builder.create<math::ExpOp>(loc, sub_elem);
      builder.create<affine::AffineStoreOp>(loc, exp2, scores_max, ValueRange({blockIvs[0]}));
    });
    // reduce sum
    builder.create<frisk::ReduceOp>(loc, acc_s, scores_sum, "add", 1);
    // logsum = logsum * scores_scale + scores_sum
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, logsum, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_scale, blockIvs);
      auto elem3 = builder.create<affine::AffineLoadOp>(loc, scores_sum, blockIvs);
      auto mul_elem = builder.create<arith::MulFOp>(loc, elem1, elem2);
      auto add_elem = builder.create<arith::AddFOp>(loc, mul_elem, elem3);
      builder.create<affine::AffineStoreOp>(loc, add_elem, logsum, blockIvs);
    });
    // copy acc_s to acc_s_cast
    builder.create<frisk::CopyOp>(loc, acc_s, acc_s_cast);
    // ====== rescale ======
    // acc_o = acc_o * scale
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, d}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_o, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_scale, ValueRange({blockIvs[0]}));
      auto mul_elem = builder.create<arith::MulFOp>(loc, elem1, elem2);
      builder.create<affine::AffineStoreOp>(loc, mul_elem, acc_o, blockIvs);
    });
    // copy v to smemv
    auto V_view = builder.create<frisk::BufferViewOp>(loc, V, ValueRange({bz, by, iv}), mapKV, ArrayRef<int64_t>{1, 1, BN, d});
    builder.create<frisk::CopyOp>(loc, V_view, smemV);
    // gemm
    builder.create<frisk::GemmOp>(loc, acc_s_cast, smemV, acc_o, false, false);
  });
  // acc_o = acc_o / logsum
  builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, d}, [&](ValueRange blockIvs) {
    auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_o, blockIvs);
    auto elem2 = builder.create<affine::AffineLoadOp>(loc, logsum, ValueRange({blockIvs[0]}));
    auto div_elem = builder.create<arith::DivFOp>(loc, elem1, elem2);
    builder.create<affine::AffineStoreOp>(loc, div_elem, acc_o, blockIvs);
  });
  // copy acc_o to smemo
  builder.create<frisk::CopyOp>(loc, acc_o, smemO);
  // copy smemo to O
  auto O_view = builder.create<frisk::BufferViewOp>(loc, O, ValueRange({bz, by, bx}), mapQO, ArrayRef<int64_t>{1, 1, BM, d});
  builder.create<frisk::CopyOp>(loc, smemO, O_view);



  
  // // ifOp test (IntegerSet embedded condition).
  // auto d0 = getAffineDimExpr(0, ctx.get());
  // auto d1 = getAffineDimExpr(1, ctx.get());
  // auto s0 = getAffineSymbolExpr(0, ctx.get());
  // IntegerSet condSet = IntegerSet::get(/*dimCount=*/2, /*symbolCount=*/1, ArrayRef<AffineExpr>{d0 + s0 - d1}, ArrayRef<bool>{false});
  // auto symN = builder.create<arith::ConstantIndexOp>(loc, 16);
  // auto ifOp = builder.create<frisk::IfOp>(loc, condSet, ValueRange{parallelEntry->getArgument(0), parallelEntry->getArgument(1), symN});

  // {
  //   auto *thenBlock = new Block();
  //   ifOp.getThenRegion().push_back(thenBlock);
  //   OpBuilder thenBuilder = OpBuilder::atBlockBegin(thenBlock);
  //   thenBuilder.create<frisk::GemmWaitOp>(loc, c.getResult());
  //   thenBuilder.create<frisk::EndOp>(loc);
  // }
  // {
  //   auto *elseBlock = new Block();
  //   ifOp.getElseRegion().push_back(elseBlock);
  //   OpBuilder elseBuilder = OpBuilder::atBlockBegin(elseBlock);
  //   elseBuilder.create<frisk::FillOp>(loc, c.getResult(), builder.getF32FloatAttr(2.0f));
  //   elseBuilder.create<frisk::EndOp>(loc);
  // }

  mod->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
