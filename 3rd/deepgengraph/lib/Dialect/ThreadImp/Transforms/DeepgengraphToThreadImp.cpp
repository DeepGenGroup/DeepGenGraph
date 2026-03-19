#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "deepgengraph/Analysis/ThreadAnalysis.h"

namespace mlir::threadimp {

#define GEN_PASS_DEF_DEEPGENGRAPHTOTHREADIMP
#include "deepgengraph/Dialect/ThreadImp/Transforms/Passes.h.inc"

} // namespace mlir::deepgengraph

namespace mlir::threadimp{ 


class ThreadImpTypeConverter : public mlir::TypeConverter {
public:
  ThreadImpTypeConverter() {
    // 1. 默认保留标准类型（如 index, i32, f16 等）
    addConversion([](Type type) { return type; });

    // 2. 转换 deepgengraph_triton.ptr 类型
    addConversion([&](deepgengraph::triton::PointerType type) -> Type {
      // 提取原始的 pointee_type (通常是 RankedTensorType)
      auto pointee = mlir::dyn_cast<RankedTensorType>(type.getPointeeType());
      if (!pointee) return nullptr;
      
      // 构造你的 ThreadImp_PtrType
      // 假设默认映射到 GM (Global Memory)
      return threadimp::PointerType::get(pointee, threadimp::MemSpace::GM);
    });

    // 3. 转换 deepgengraph_triton.block_ptr 类型
    addConversion([&](deepgengraph::triton::BlockPointerType type) -> Type {
      auto pointee = mlir::dyn_cast<RankedTensorType>(type.getPointeeType());
      if (!pointee) return nullptr;

      // 同样映射到你的指针类型
      return threadimp::PointerType::get(pointee, MemSpace::GM);
    });
  }
};


class DeviceKernelOpRewritePattern : public OpConversionPattern<mlir::deepgengraph::triton::DeviceKernelOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
    mlir::deepgengraph::triton::DeviceKernelOp op, 
    OpAdaptor adaptor, // adaptor 中的 operands 已经是转换后的类型了！,
    ConversionPatternRewriter &rewriter) 
  {
    op->getParentOp()->setAttr("grid", op.getGridAttr());
    auto loc = op->getLoc();
    int bx = 0 ;
    for(auto arg : op->getBlock()->getArguments()){
      if(auto ty = mlir::dyn_cast<IndexType>(arg.getType())){
        // block arg is bid
        auto bid = rewriter.create<threadimp::GetBlockIdOp>(loc, rewriter.getI32IntegerAttr(0));
        arg.replaceAllUsesWith(bid);
        bx++;
      }
      else if(auto ty = mlir::dyn_cast<deepgengraph::triton::PointerType>(arg.getType())){
        // block arg is ptr
        if(auto outerDef = arg.getDefiningOp<deepgengraph::triton::PointerOfOp>()){
          // replace blockarg with its outer define
          arg.replaceAllUsesWith(outerDef);
        }
      }
    }
    // 假设 op 是你的 Container Op，destBlock 是外部 Block（如 func 的 EntryBlock）
    auto source = op->getBlock();
    auto *destBlock = op.getParentOp()->getBlock(); // 获取当前 Op 所在的 Block
    auto insertPoint = op->getIterator(); // 记录当前位置

    // 将 Region 里的 Block 直接移动到外部 Block 之前/之后
    rewriter.inlineBlockBefore(source, destBlock,insertPoint);
    rewriter.eraseOp(op);
    // 注意：搬运后通常需要处理：
    // 1. Block 合并（使用 mergeBlocks）
    // 2. 将原本 Region 的参数（Block Arguments）映射为外部的 Value
    return mlir::success();
  }
};


// %9 = deepgengraph_triton.block_ptr_of base = %arg6, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
// %10 = deepgengraph_triton.block_load %9 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16> loc(#loc)
// // 转换: 发现 block_load 时，找到其ptr的属性，作为memcpy的参数。在其前面插入一个 tensor.empty 操作，构建一个空的shm buffer。
// // %shm = tensor.empty : tensor<128x128xf16, #shm>
// // %shm_filled = threadimpl.memcpy(dst = %shm, base = %arg6, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0], therads = 128)
class BlockLoadOpConverionPattern : public OpConversionPattern<deepgengraph::triton::BlockLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp op, 
      OpAdaptor adaptor, // adaptor 中的 operands 已经是转换后的类型了！
      ConversionPatternRewriter &rewriter) const override 
  {
    using namespace deepgengraph::triton;
    // 获取转换后的源指针（现在它的类型应该是 threadimp::PointerType）
    Value srcPtr = adaptor.getSrcPointer();
    
    // 获取结果类型转换后的目标类型
    TensorType retTensorType = mlir::cast<TensorType>(getTypeConverter()->convertType(op.getType()));
    auto loc = op->getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, retTensorType.getShape(), retTensorType.getElementType());
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value dst_tensor, Value src_pointer, Value base_offset, ArrayRef<int64_t> shape, ArrayRef<int64_t> stride, ArrayRef<int64_t> offset, ArrayRef<int64_t> block_shape);
    const auto& infoMap = analyze::PointerTracer::getMap();
    auto blockPtrOfOp = infoMap.at(op).m_blockPtrOfOp;

    auto newop = rewriter.create<threadimp::BlockCopyG2S>(loc, emptyOp.getResult(),srcPtr, 
      blockPtrOfOp.getBaseOffset(), 
      blockPtrOfOp.getShape(), 
      blockPtrOfOp.getStride(), 
      blockPtrOfOp.getOffset(), 
      blockPtrOfOp.getBlockShape());
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class BlockAdvanceOpConverionPattern : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, 
      OpAdaptor adaptor, // adaptor 中的 operands 已经是转换后的类型了！
      ConversionPatternRewriter &rewriter) const override 
  {
    using namespace deepgengraph::triton;
    // 获取转换后的源指针（现在它的类型应该是 threadimp::PointerType）
    Value srcPtr = adaptor.getSrcPointer();

    auto offsets = adaptor.getOffsets();
    auto loc = op->getLoc();
    auto nextPtrType = getTypeConverter()->convertType(op.getNextPointer().getType());
    auto newop = rewriter.create<threadimp::PointerAdvanceOp>(loc, nextPtrType ,srcPtr, op.getOffsets());
    rewriter.replaceOp(op, newop);
    rewriter.replaceAllUsesWith(op, newop);
    return success();
  }
};


// 假设我们要将类型 !old_type 转换为 !new_type
// 在实际代码中，这可能是从 deepgengraph_triton.block_ptr 转换到 threadimp.ptr

struct SCFForTypeConversion : public OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        
        // 1. 获取转换后的 Result 类型（即循环返回值的类型）
        SmallVector<Type> newResultTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
            return failure();
        }

        // 2. 创建新的 ForOp
        // 注意：adaptor.getInitArgs() 已经自动包含了转换后的初始值
        auto newForOp = rewriter.create<scf::ForOp>(
            op.getLoc(), 
            adaptor.getLowerBound(), 
            adaptor.getUpperBound(),
            adaptor.getStep(), 
            adaptor.getInitArgs()
        );

        // 3. 移动 Region（循环体）
        // 我们将旧 ForOp 的 Region 整个移到新 ForOp 中
        rewriter.inlineRegionBefore(op.getRegion(), newForOp.getRegion(), newForOp.getRegion().end());

        // 4. **关键步骤**：转换 Block 参数类型
        // 这一步会更新循环内部 ^bb0 的参数签名（iv, iter_args...）
        // 它会自动处理从 OldType 到 NewType 的映射
        if (failed(rewriter.convertRegionTypes(&newForOp.getRegion(), *getTypeConverter()))) {
            return failure();
        }

        // 5. 替换原有的 Op
        rewriter.replaceOp(op, newForOp.getResults());
        return success();
    }
};

// 同样需要处理 scf.yield，因为它的操作数必须与新的 scf.for 返回类型匹配
struct SCFYieldTypeConversion : public OpConversionPattern<scf::YieldOp> {
    using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // 直接使用 adaptor 提供的、已经转换好的 operands 创建新的 yield
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
        return success();
    }
};

// --- Pass 注册部分 ---




class BlockLevelIRToThreadImpPass : public impl::DeepgengraphToThreadImpBase< BlockLevelIRToThreadImpPass >
{
public:
  void runOnOperation() {
    using namespace deepgengraph::triton;
    MLIRContext *context = &getContext();
    ThreadImpTypeConverter typeConverter; // 实例化转换器
    
    RewritePatternSet patterns(context);
    patterns.add<
    DeviceKernelOpRewritePattern,
      BlockLoadOpConverionPattern,
      BlockAdvanceOpConverionPattern,
      SCFForTypeConversion,
      SCFYieldTypeConversion
    >(typeConverter, context);
    ConversionTarget target(*context);
    
    // 定义合法性：如果一个 Op 使用了旧指针类型，它就是不合法的
    target.addDynamicallyLegalOp<deepgengraph::triton::BlockLoadOp>(
        [&](deepgengraph::triton::BlockLoadOp op) {
          return typeConverter.isLegal(op.getType()) && 
                typeConverter.isLegal(op.getSrcPointer().getType());
        });
    target.addDynamicallyLegalOp<deepgengraph::triton::BlockAdvanceOp>(
        [&](deepgengraph::triton::BlockAdvanceOp op) {
          return typeConverter.isLegal(op.getType()) && 
                typeConverter.isLegal(op.getSrcPointer().getType()) &&
                typeConverter.isLegal(op.getNextPointer().getType())
                ;
        });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp forOp){
      return typeConverter.isLegal(forOp.getResultTypes()) && typeConverter.isLegal(forOp->getOperandTypes());
    });
    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op){
      return typeConverter.isLegal(op->getResultTypes()) && typeConverter.isLegal(op.getOperandTypes());
    });

    // 必须告知 Target 哪些 Dialect 是合法的
    target.addLegalDialect<
      threadimp::ThreadImpDialect, 
      tensor::TensorDialect,
      deepgengraph::triton::DeepgengraphTritonDialect,
      deepgengraph::DeepgengraphDialect,
      scf::SCFDialect,
      arith::ArithDialect
      >();
    
    // 关键：将非法 Op 标记出来
    target.addIllegalOp<
      BlockLoadOp, BlockAdvanceOp, DeviceKernelOp
    >();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createConvertDeepgengraphTritonToThreadImpPass(){
  return std::make_unique<BlockLevelIRToThreadImpPass>(); 
}

}
