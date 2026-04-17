#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir::frisk {

#define GEN_PASS_DEF_CALCULATEOPTOFRISK

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

namespace dg = deepgengraph ;
namespace dgt = deepgengraph::triton;

static void addMaterializations(TypeConverter &tc) {
  tc.addTargetMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
  tc.addSourceMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
}

static Value getKernelArgById(Operation *op, int64_t argId) {
  auto kernelOp = op->getParentOfType<frisk::KernelOp>();
  if (!kernelOp)
    return {};
  if (argId < 0 || argId >= static_cast<int64_t>(kernelOp.getNumArguments()))
    return {};
  return kernelOp.getArgument(argId);
}

// ---------- patterns -----------------

struct MatmulOpConversionPattern : public OpConversionPattern<dg::PreciseDotOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(dg::PreciseDotOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override 
  {
    auto memA = adaptor.getLhs();
    auto memB = adaptor.getRhs();
    auto shapeA = mlir::cast<MemRefType>(memA.getType()).getShape();
    auto shapeB = mlir::cast<MemRefType>(memB.getType()).getShape();
    int sizeM = shapeA[0];
    int sizeN = shapeB[1];
    int sizeK = shapeB[0];

    std::vector<int64_t> cshape = {sizeM, sizeN};
    auto memC = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), cshape, op.getAccType());
    std::vector<int64_t> ranges = {sizeM, sizeN};
    auto block = rewriter.create<frisk::BlockOp>(op->getLoc(), ranges, nullptr);
    {
      auto loc = block->getLoc();
      RewriterBase::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToStart(block.getBody(0));
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lowerBound, Value upperBound, Value step, ValueRange initArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> odsArg4 = nullptr);
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc,0);
      auto step_one = rewriter.create<arith::ConstantIndexOp>(loc,1);
      auto k = rewriter.create<arith::ConstantIndexOp>(loc, sizeK);
      auto forOp = rewriter.create<scf::ForOp>(block->getLoc(), zero,k, step_one );
      rewriter.setInsertionPointToStart(forOp.getBody(0));
      auto iter_k = forOp.getInductionVar();
      auto i = block.getBody(0)->getArgument(0);
      auto j = block.getBody(0)->getArgument(1);
      std::vector<Value> indiceA = {i,iter_k};
      std::vector<Value> indiceB = {iter_k, j};
      std::vector<Value> indiceC = {i, j};
      auto a = rewriter.create<memref::LoadOp>(loc, memA, indiceA);
      auto b = rewriter.create<memref::LoadOp>(loc, memB, indiceB);
      auto acc  = rewriter.create<memref::LoadOp>(loc, memC, indiceC);
      Value prod = rewriter.create<arith::MulFOp>(loc, a, b);
      if (prod.getType() != acc.getType()) {
        if (!isa<FloatType>(prod.getType()) || !isa<FloatType>(acc.getType()))
          return failure();
        auto prodFloatTy = cast<FloatType>(prod.getType());
        auto accFloatTy = cast<FloatType>(acc.getType());
        if (prodFloatTy.getWidth() < accFloatTy.getWidth()) {
          prod = rewriter.create<arith::ExtFOp>(loc, acc.getType(), prod);
        } else if (prodFloatTy.getWidth() > accFloatTy.getWidth()) {
          prod = rewriter.create<arith::TruncFOp>(loc, acc.getType(), prod);
        }
      }
      auto added = rewriter.create<arith::AddFOp>(loc, prod, acc);
      auto store = rewriter.create<memref::StoreOp>(loc, added, memC, indiceC);
    }
    rewriter.replaceOp(op, memC);
    return success();
  }
};


struct BinaryOpConversionPattern : public OpInterfaceConversionPattern<dg::BroadcastableBinaryOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  
  virtual LogicalResult
  matchAndRewrite(dg::BroadcastableBinaryOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (operands.size() != 2)
      return failure();

    auto materializeRankedTensorToMemRef = [&](Value v) -> FailureOr<Value> {
      if (isa<MemRefType>(v.getType()))
        return v;
      auto rankedTensorTy = dyn_cast<RankedTensorType>(v.getType());
      if (!rankedTensorTy)
        return failure();
      auto memTy = MemRefType::get(rankedTensorTy.getShape(), rankedTensorTy.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), memTy, v).getResult(0);
    };

    auto memLhsOr = materializeRankedTensorToMemRef(operands[0]);
    if (failed(memLhsOr))
      return failure();
    Value memLhs = *memLhsOr;
    auto memRhsOr = materializeRankedTensorToMemRef(operands[1]);
    if (failed(memRhsOr))
      return failure();
    Value memRhs = *memRhsOr;

    auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType)
      return failure();
    auto resultShape = resultTensorType.getShape();

    auto convertedResultType = MemRefType::get(resultShape, resultTensorType.getElementType());
    auto alloc = rewriter.create<frisk::AllocBufferOp>(op->getLoc(), convertedResultType.getShape(), convertedResultType.getElementType());
    std::vector<int64_t> ranges(resultShape.begin(), resultShape.end());
    auto blockOp = rewriter.create<frisk::BlockOp>(op->getLoc(), ranges, nullptr);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockOp.getBody(0));
      SmallVector<Value, 4> indices(blockOp.getBody(0)->getArguments().begin(),
                                    blockOp.getBody(0)->getArguments().end());
      auto zero = rewriter.create<arith::ConstantIndexOp>(blockOp->getLoc(), 0);
      auto buildOperandIndices = [&](Value mem, Value originalTensorVal) -> FailureOr<SmallVector<Value, 4>> {
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!memTy)
          return failure();
        auto srcTy = dyn_cast<RankedTensorType>(originalTensorVal.getType());
        if (!srcTy)
          return failure();
        int64_t operandRank = memTy.getRank();
        int64_t resultRank = static_cast<int64_t>(indices.size());
        if (operandRank > resultRank)
          return failure();
        if (srcTy.getRank() != operandRank)
          return failure();

        int64_t offset = resultRank - operandRank;
        SmallVector<Value, 4> operandIndices;
        operandIndices.reserve(operandRank);
        for (int64_t i = 0; i < operandRank; ++i) {
          int64_t dim = offset + i;
          // Broadcasted dimensions always read index 0 from the source tensor.
          if (srcTy.getShape()[i] == 1) {
            operandIndices.push_back(zero);
          } else {
            operandIndices.push_back(indices[dim]);
          }
        }
        return operandIndices;
      };

      auto lhsIndicesOr = buildOperandIndices(memLhs, op.getLhs());
      if (failed(lhsIndicesOr))
        return failure();
      auto rhsIndicesOr = buildOperandIndices(memRhs, op.getRhs());
      if (failed(rhsIndicesOr))
        return failure();

      auto lhs = rewriter.create<memref::LoadOp>(blockOp->getLoc(), memLhs, *lhsIndicesOr);
      auto rhs = rewriter.create<memref::LoadOp>(blockOp->getLoc(), memRhs, *rhsIndicesOr);

      Value ret;
      Type lhsType = lhs.getType();
      if (isa<dg::AddOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::AddFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::AddIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::SubOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::SubFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::SubIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::MulOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::MulFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::MulIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::DivOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<arith::DivFOp>(blockOp->getLoc(), lhs, rhs);
        else
          ret = rewriter.create<arith::DivSIOp>(blockOp->getLoc(), lhs, rhs);
      } else if (isa<dg::PowOp>(op.getOperation())) {
        if (isa<FloatType>(lhsType))
          ret = rewriter.create<math::PowFOp>(blockOp->getLoc(), lhs, rhs);
        else if (isa<IntegerType>(lhsType))
          ret = rewriter.create<math::IPowIOp>(blockOp->getLoc(), lhs, rhs);
        else
          return failure();
      } else if (auto cmpOp = dyn_cast<dg::CmpOp>(op.getOperation())) {
        Value pred;
        if (isa<FloatType>(lhsType)) {
          arith::CmpFPredicate fpred =
              cmpOp.getCmpType() == dg::CmpType::GT ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OGE;
          pred = rewriter.create<arith::CmpFOp>(blockOp->getLoc(), fpred, lhs, rhs);
        } else if (isa<IntegerType, IndexType>(lhsType)) {
          arith::CmpIPredicate ipred =
              cmpOp.getCmpType() == dg::CmpType::GT ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sge;
          pred = rewriter.create<arith::CmpIOp>(blockOp->getLoc(), ipred, lhs, rhs);
        } else {
          return failure();
        }

        Type outElemTy = convertedResultType.getElementType();
        if (pred.getType() == outElemTy) {
          ret = pred;
        } else if (isa<IntegerType>(outElemTy)) {
          ret = rewriter.create<arith::ExtUIOp>(blockOp->getLoc(), outElemTy, pred);
        } else {
          return failure();
        }
      } else {
        return failure();
      }

      rewriter.create<memref::StoreOp>(blockOp->getLoc(), ret, alloc, indices);
    }
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class CalcOpToFrisk : public impl::CalculateOpToFriskBase<CalcOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();

    TypeConverter tc;
    tc.addConversion([](Type type) -> std::optional<Type> {
      if (auto rankedTensorTy = dyn_cast<RankedTensorType>(type)) {
        return MemRefType::get(rankedTensorTy.getShape(), rankedTensorTy.getElementType());
      }
      if (auto unrankedTensorTy = dyn_cast<UnrankedTensorType>(type)) {
        return UnrankedMemRefType::get(unrankedTensorTy.getElementType(), 0);
      }
      return type;
    });
    addMaterializations(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<FriskDialect, affine::AffineDialect, memref::MemRefDialect, func::FuncDialect, dg::DeepgengraphDialect,
                           dgt::DeepgengraphTritonDialect, arith::ArithDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect>();

    target.addIllegalOp<dg::AddOp, dg::SubOp, dg::MulOp, dg::DivOp, dg::PowOp, dg::CmpOp, dg::PreciseDotOp>();

    RewritePatternSet ps(ctx);
    ps.add<
      MatmulOpConversionPattern,
      BinaryOpConversionPattern
    >(tc, ctx);

    if (failed(applyPartialConversion(op, target, std::move(ps)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertCalcOpPass() {
  return std::make_unique<CalcOpToFrisk>();
}

} // namespace mlir::frisk
