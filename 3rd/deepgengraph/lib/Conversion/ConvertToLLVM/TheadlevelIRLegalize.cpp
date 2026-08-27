#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"

namespace mlir::frisk {

namespace {
#define GEN_PASS_DEF_THREADLEVELIRLEGALIZE
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h.inc"


using friskMs = frisk::attr::MemorySpace;

static std::string GetShmVarName(){
  static int i = 0;
  auto ret = llvm::Twine("shm_") + llvm::Twine(i++);
  return ret.str();
}

static int ComputeShmUsageBytes(std::optional<MemRefType> type){
  static int usage = 0;
  if(!type){
    return usage;
  }
  int dims = 1;
  for(auto i : type->getShape()){
    dims *= i;
  }
  usage += (dims * type->getElementTypeBitWidth() / 8);
  return usage;
}

static std::optional<int64_t> GetBlockIdRange(gpu::BlockIdOp bid) {
  if (auto range = bid->getAttrOfType<IntegerAttr>("range")) {
    return range.getInt();
  }
  if (auto upperBound = bid.getUpperBoundAttr()) {
    return upperBound.getInt();
  }
  return std::nullopt;
}

// memref.alloc 分配shm 改为定义 global shm var
struct MemAllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto memTy = mlir::cast<MemRefType>(op.getResult().getType());
    if(memTy.getMemorySpaceAsInt() != (int)friskMs::Shared){
      return failure();
    }
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod != nullptr);
    
    rewriter.setInsertionPointToStart(mod.getBody());
    auto shmVarName = GetShmVarName();
    IntegerAttr alignAttr;
    if(auto align = op.getAlignment()){
      alignAttr = rewriter.getI64IntegerAttr(align.value());
    }
    auto globalOp = rewriter.create<memref::GlobalOp>(
      op->getLoc(),
      shmVarName,
      rewriter.getStringAttr("public"),
      memTy,
      rewriter.getUnitAttr(),
      false,
      alignAttr
      );
    rewriter.setInsertionPoint(op);
    // rewriter.replaceAllUsesWith(op.getResult(), )
    ComputeShmUsageBytes(memTy);
    auto getGlobal = rewriter.create<memref::GetGlobalOp>(
          op->getLoc(),op.getResult().getType(),shmVarName);
    rewriter.replaceOp(op, getGlobal);
    return success();
  }
};

// inner-block sync op
struct FriskSyncOpConversion : public OpConversionPattern<frisk::SyncThreadsInBlockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::SyncThreadsInBlockOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<gpu::BarrierOp>(op->getLoc());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// warp mma op conversion
struct FriskWarpMMAOpConversion : public OpConversionPattern<frisk::WarpMmaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::WarpMmaOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs);
    auto memA = mlir::cast<MemRefType>(adaptor.getA().getType());
    auto memB = mlir::cast<MemRefType>(adaptor.getB().getType());
    auto memC = mlir::cast<MemRefType>(adaptor.getC().getType());
    auto m = memA.getShape()[0];
    auto k = memA.getShape()[1];
    auto n = memB.getShape()[1];
    auto asm_string = op->getAttrOfType<StringAttr>("inst_name").data();
    auto constraints = op->getAttrOfType<StringAttr>("inst_constraints").data();
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
    auto zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    std::vector<Value> indices = {zero, zero};
    auto convertMemToVectorPack  = [&](Value buffer) -> vector::LoadOp {
      auto memTy = mlir::cast<MemRefType>(buffer.getType());
      int count = 1;
      for(int i : memTy.getShape()){
        count*=i;
      } 
      auto vecPack = VectorType::get({count}, memTy.getElementType());
      auto vectorLoad = rewriter.create<vector::LoadOp>(op->getLoc(), vecPack, buffer, indices);
      return vectorLoad;
    };
    
    auto vA = convertMemToVectorPack(adaptor.getA());
    auto vB = convertMemToVectorPack(adaptor.getB());
    auto vC = convertMemToVectorPack(adaptor.getC());
    
    std::vector<Value> vr = {vA.getResult(), vB.getResult(), vC.getResult()};
    // auto retTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto retTy = VectorType::get({4}, memC.getElementType());
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),retTy , vr, asm_string, constraints,
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);

    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value valueToStore, ::mlir::Value base, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
    rewriter.create<vector::StoreOp>(op->getLoc(), asmOp->getResult(0), adaptor.getC(), indices);
    rewriter.eraseOp(op);
    return success();
  }
};


class ThreadLevelIRLegalizePass : public impl::ThreadLevelIRLegalizeBase<ThreadLevelIRLegalizePass> {
public:
  void runOnOperation(){
    // ----- step 1 : module中收集所有function。检查是否为kernel，不是的话去掉. 之后加上 gridDIm 属性
    auto module = getOperation();
    std::vector<mlir::func::FuncOp> funcToRemove{};
    std::vector<mlir::func::FuncOp> validKernels{};
    module->walk([&funcToRemove, &validKernels](mlir::func::FuncOp f){
      if(!f->hasAttr("thread_num")){
        funcToRemove.push_back(f);
      }
      else{
        validKernels.push_back(f);
      }
    });

    for(auto f : funcToRemove){
      f->erase();
    }
    
    for(auto kernel : validKernels){
      std::array<int32_t, 3> gridDims = {1,1,1};
      auto walkResult = kernel->walk([&](mlir::gpu::BlockIdOp bid){
        auto dim = bid.getDimension();
        auto ub = GetBlockIdRange(bid);
        if (!ub) {
          bid.emitOpError("requires a static 'range' or 'upper_bound' attribute");
          return WalkResult::interrupt();
        }
        switch (dim) {
          case mlir::gpu::Dimension::x : gridDims[2] = *ub; break;
          case mlir::gpu::Dimension::y : gridDims[1] = *ub; break;
          case mlir::gpu::Dimension::z : gridDims[0] = *ub; break;
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        return signalPassFailure();
      }
      OpBuilder b{module->getContext()};
      kernel->setAttr("gridDim", b.getI32ArrayAttr(gridDims));
    }
    
    // --------- step 2 ： AllocOp创建shm 改为分配 global shm var 并 get_global
    MLIRContext *context = module->getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      LLVM::LLVMDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect>();

    target.addIllegalOp<memref::AllocOp>();
    RewritePatternSet patterns(context);
    patterns.add<MemAllocOpConversion>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))){
      return signalPassFailure();
    }
    
    // --- step 3 : 将剩下的frisk op 转为对应的底层op
    ConversionTarget t2(*context);

    t2.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      LLVM::LLVMDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect,
      vector::VectorDialect
      >();

    t2.addIllegalOp< frisk::WarpMmaOp, frisk::SyncThreadsInBlockOp>();
    RewritePatternSet p2(context);
    p2.add<FriskSyncOpConversion, FriskWarpMMAOpConversion>(context);
    if (failed(applyPartialConversion(module, t2, std::move(p2)))){
      return signalPassFailure();
    }
  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createThreadLevelIRLegalizePass() {
  return std::make_unique<ThreadLevelIRLegalizePass>();
}

}  // end namespace frisk 
