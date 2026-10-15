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
#include "deepgengraph/Common.h"
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
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
      // 出现了 memref.alloc 分配local mem的句子。需要转为 alloca
      assert(memTy.getMemorySpaceAsInt() == (int)friskMs::Local);
      auto newAlloca = rewriter.create<memref::AllocaOp>(op->getLoc(), memTy );
      rewriter.replaceOp(op, newAlloca);
      return success();
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

// Exact, unmodified templates selected by HardwareSpecification.cpp. Do not
// translate arbitrary assembly/modifiers by replacing underscores with dots:
// the f32 instruction names and the scalar x4 ABI are special cases.
static StringRef getHCUMMACName(StringRef assembly) {
  if (assembly == "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3")
    return "llvm.hcu.mmac.f32.16x16x16.f16";
  if (assembly == "v_mmac_f32_16x16x8_f32 $0, $1, $2, $3" ||
      assembly == "v_mmac_16x16x8_f32 $0, $1, $2, $3")
    return "llvm.hcu.mmac.16x16x8.f32";
  if (assembly == "v_mmac_f32_16x16x4_f32 $0, $1, $2, $3" ||
      assembly == "v_mmac_16x16x4_f32 $0, $1, $2, $3")
    return "llvm.hcu.mmac.16x16x4.f32";
  return {};
}

static LogicalResult lowerHCUMMAC(frisk::WarpMmaRROp op,
                                 ValueRange operands, StringRef name,
                                 ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  bool scalarAB = name == "llvm.hcu.mmac.16x16x4.f32";
  bool halfAB = name == "llvm.hcu.mmac.f32.16x16x16.f16";
  Type element = halfAB ? rewriter.getF16Type() : rewriter.getF32Type();
  int64_t count = halfAB ? 4 : (scalarAB ? 1 : 2);
  auto abVector = VectorType::get({count}, element);
  auto cdVector = VectorType::get({4}, rewriter.getF32Type());
  auto matches = [](Type type, VectorType expected) {
    auto vector = dyn_cast<VectorType>(type);
    return vector && !vector.isScalable() &&
           vector.getNumElements() == expected.getNumElements() &&
           vector.getElementType() == expected.getElementType();
  };
  if (!matches(operands[0].getType(), abVector) ||
      !matches(operands[1].getType(), abVector) ||
      !matches(operands[2].getType(), cdVector) ||
      !matches(op.getResult().getType(), cdVector))
    return op.emitOpError("fragment types do not match HCU MMAC ABI for ") << name;

  // IntrinsicsHCU.td: HCUMMopIntrinsic(A, B, C, lit, lts), or
  // HCUMMopVstepIntrinsic(A, B, C, vstep, lts). No modifiers => all zero.
  Type abType = scalarAB ? element : Type(abVector);
  Type controlType = scalarAB ? rewriter.getI32Type() : rewriter.getI1Type();
  auto functionType = LLVM::LLVMFunctionType::get(
      cdVector, {abType, abType, cdVector, controlType, rewriter.getI1Type()}, false);
  auto symbol = module.lookupSymbol(name);
  auto function = dyn_cast_or_null<LLVM::LLVMFuncOp>(symbol);
  if (symbol && (!function || !function.isExternal() ||
                 function.getFunctionType() != functionType))
    return op.emitOpError("conflicting declaration for ") << name;

  // The build-host LLVM may not register these DTK names. As for HCU matrix
  // loads, emit a typed declaration/call, leaving intrinsic recognition to DTK.
  // Explicit effects keep host optimization correct even without that registry.
  auto noMemory = LLVM::MemoryEffectsAttr::get(rewriter.getContext(),
      {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
       LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
       LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef});
  if (!function) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    function = rewriter.create<LLVM::LLVMFuncOp>(loc, name, functionType);
    function.setConvergent(true);
    function.setWillReturn(true);
    function.setMemoryEffectsAttr(noMemory);
  }
  auto flatten = [&](Value value, VectorType type) -> Value {
    if (value.getType() == type)
      return value;
    return rewriter.create<vector::ShapeCastOp>(loc, type, value);
  };
  Value a = flatten(operands[0], abVector);
  Value b = flatten(operands[1], abVector);
  Value c = flatten(operands[2], cdVector);
  if (scalarAB) {
    a = rewriter.create<vector::ExtractOp>(loc, a, ArrayRef<int64_t>{0});
    b = rewriter.create<vector::ExtractOp>(loc, b, ArrayRef<int64_t>{0});
  }
  Value control = rewriter.create<LLVM::ConstantOp>(
      loc, controlType, rewriter.getIntegerAttr(controlType, 0));
  Value lts = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
  auto call = rewriter.create<LLVM::CallOp>(
      loc, function, ValueRange{a, b, c, control, lts});
  call.setConvergent(true);
  call.setWillReturn(true);
  call.setMemoryEffectsAttr(noMemory);
  Value result = call.getResult();
  if (result.getType() != op.getResult().getType())
    result = rewriter.create<vector::ShapeCastOp>(
        loc, op.getResult().getType(), result);
  rewriter.replaceOp(op, result);
  return success();
}

// warp mma op conversion
struct FriskWarpMMAOpConversion : public OpConversionPattern<frisk::WarpMmaRROp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(frisk::WarpMmaRROp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto nameAttr = op->getAttrOfType<StringAttr>("inst_name");
    auto constraintsAttr = op->getAttrOfType<StringAttr>("inst_constraints");
    if (!nameAttr || !constraintsAttr)
      return op.emitOpError("requires string inst_name and inst_constraints attributes");
    auto instName = nameAttr.getValue();
    auto useIntrinsic = op->getParentOfType<ModuleOp>()
                            ->getAttrOfType<BoolAttr>("frisk.use_hcu_mmac");
    StringRef intrinsic = getHCUMMACName(instName);
    if (useIntrinsic && useIntrinsic.getValue() && !intrinsic.empty() &&
        constraintsAttr.getValue() == "=v,v,v,0")
      return lowerHCUMMAC(op, adaptor.getOperands(), intrinsic, rewriter);
    std::string asm_string = instName.str();
    if (instName.starts_with("v_mmac_f32_16x16x16_f16 ")) {
      // The DTK backend does not schedule the MMAC hidden in this inline asm.
      // On gfx936, the unpadded dependent-MMAC test reads stale results; the
      // padded test verifies both zero and nonzero accumulator inputs.
      // Keep input and output spacing INSIDE the same asm so register moves,
      // result stores and register reuse cannot enter the protected interval.
      // This is the conservative spacing validated on hardware, not a claim
      // about the instruction's minimum latency. Shorten only after measuring,
      // or replace the opaque asm with a target intrinsic with hazard handling.
      std::string padding;
      padding += "s_nop 1\n\t"; // 经过测试，1个 s_nop 1 已经能够满足正确性。但不能不加
      asm_string = padding + asm_string + "\n\t" + padding;
    }
    auto constraints = constraintsAttr.getValue();
    
    auto getCollapsed = [&](mlir::Value oldValue){
      auto typeA = mlir::cast<ShapedType>(oldValue.getType());
      std::vector<int64_t> shapeAfterCollapse{};
      bool needConvert = false;
      for(auto s : typeA.getShape()){
        if(s != 1){
          shapeAfterCollapse.push_back(s);
        }
        else{
          needConvert = true;
        }
      }
      if(shapeAfterCollapse.empty()){
        shapeAfterCollapse.push_back(1);
      }
      mlir::Value newVal;
      if(needConvert){
        auto collapsedType = VectorType::get(shapeAfterCollapse, typeA.getElementType());
        newVal = rewriter.create<vector::ShapeCastOp>(op->getLoc(), collapsedType, oldValue);
      }
      else{
        newVal = oldValue;
      }
      return newVal;
    };

    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source);
    auto typeB = mlir::cast<ShapedType>(adaptor.getB().getType());
    auto typeC = mlir::cast<ShapedType>(adaptor.getC().getType());
    auto newA = getCollapsed(adaptor.getA());
    auto newB = getCollapsed(adaptor.getB());
    auto newC = getCollapsed(adaptor.getC());
    std::vector<Value> vr = {newA, newB, newC};
    auto retTy = mlir::cast<VectorType>(op.getResult().getType());
    std::vector<int64_t> collapsedResShape{};
    bool isConverted = false;
      for(auto s : retTy.getShape()){
        if(s != 1){
          collapsedResShape.push_back(s);
        }
        else{
          isConverted = true;
        }
      }
    if(collapsedResShape.empty()){
      collapsedResShape.push_back(1);
    }
    auto newRetTy = VectorType::get(collapsedResShape, retTy.getElementType());
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),newRetTy , vr, asm_string, constraints,
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, /*convergent=*/true,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);
    auto ret = asmOp.getResult(0);

    if(isConverted){
      ret = rewriter.create<vector::ShapeCastOp>(op->getLoc(), retTy, asmOp->getResult(0));
    }
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct FlattenDynamicVectorInsert : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern<vector::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp op,
                                PatternRewriter &rewriter) const override {
    VectorType destType = op.getDestVectorType();
    // 仅处理 2D 向量且包含动态索引的情况
    if (destType.getRank() != 2) return failure();

    Location loc = op.getLoc();
    Value val = op.getValueToStore();
    Value destVec = op.getDest();
    auto offsets = op.getDynamicPosition(); // 获取索引列表

    int64_t dim0 = destType.getDimSize(0);
    int64_t dim1 = destType.getDimSize(1);
    int64_t totalSize = dim0 * dim1;

    // 1. 展平向量类型
    VectorType flatType = VectorType::get({totalSize}, destType.getElementType());
    Value flatVec = rewriter.create<vector::ShapeCastOp>(loc, flatType, destVec);

    // 2. 计算 1D 线性偏移量 (offset0 * dim1 + offset1)
    Value dim1Const = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
    Value rowOff = rewriter.create<arith::MulIOp>(loc, offsets[0], dim1Const);
    Value flatIdx = rewriter.create<arith::AddIOp>(loc, rowOff, offsets[1]);

    // 3. 替换为 1D 动态 insert 与 shape_cast
    Value flatInserted = rewriter.create<vector::InsertOp>(loc, val, flatVec, flatIdx);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, destType, flatInserted);

    return success();
  }
};

class ThreadLevelIRLegalizePass : public impl::ThreadLevelIRLegalizeBase<ThreadLevelIRLegalizePass> {
public:
  void runOnOperation(){
    // ----- step 1 : module中收集所有function。检查是否为kernel，不是的话去掉. 之后加上 gridDIm 属性
    auto module = getOperation();
    if (module->hasAttr("frisk.use_hcu_mmac") &&
        !module->getAttrOfType<BoolAttr>("frisk.use_hcu_mmac")) {
      module.emitError("frisk.use_hcu_mmac must be a boolean");
      return signalPassFailure();
    }
    std::vector<mlir::func::FuncOp> funcToRemove{};
    std::vector<mlir::func::FuncOp> validKernels{};
    module->walk([&funcToRemove, &validKernels](mlir::func::FuncOp f){
      if(!f->hasAttr(THREAD_NUM)){
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
    // --- step 3 : vector 去除退化维度
    {
      ConversionTarget t2(*context);
      RewritePatternSet patterns(&getContext());
      vector::populateDropUnitDimWithShapeCastPatterns(patterns);

      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
      }
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

    t2.addIllegalOp< frisk::WarpMmaRROp, frisk::SyncThreadsInBlockOp>();
    RewritePatternSet p2(context);
    p2.add<FriskSyncOpConversion, FriskWarpMMAOpConversion>(context);
    if (failed(applyPartialConversion(module, t2, std::move(p2)))){
      return signalPassFailure();
    }
    {
      // RewritePatternSet patterns(context);
      // patterns.add<FlattenDynamicVectorInsert>(context);
      // if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      //   signalPassFailure();
      // }
    }
  }
};

}  // end namespace 

std::unique_ptr<mlir::Pass> createThreadLevelIRLegalizePass() {
  return std::make_unique<ThreadLevelIRLegalizePass>();
}

}  // end namespace frisk 
