#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
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

static TypeConverter* GetThreadImpTypeConverter(){
  static TypeConverter converter;
  // ptr & block_ptr -> threadimp::pointer
  converter.addConversion([](Type normalTy){return normalTy;});
  converter.addConversion([](deepgengraph::triton::PointerType ty){
    auto eleTy = ty.getPointeeType().getElementType();
    auto newTy = threadimp::PointerType::get(eleTy, threadimp::MemSpace::GM);
    return newTy;
  });
  converter.addConversion([](deepgengraph::triton::BlockPointerType ty){
    auto eleTy = ty.getPointeeType().getElementType();
    auto newTy = threadimp::PointerType::get(eleTy, MemSpace::SHM);
    return newTy;
  });
  // materialization：类型还没转换时，插入临时 cast
  converter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
  return builder.create<UnrealizedConversionCastOp>(
      loc, resultType, inputs).getResult(0);
  });
  converter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
  return builder.create<UnrealizedConversionCastOp>(
      loc, resultType, inputs).getResult(0);
  });

  return &converter;
}

// argId -> 转换后的 threadImp::pointer
static DenseMap<int, mlir::Value> s_map_argId_newGlobalPtr;
static DenseMap<int, mlir::Value> s_map_argId_newShmPtr;

// deepgengraph_triton.block_load 加入详细信息，以免后续op转换后丢失
class BlockLoadOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp blockLoadOp,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    auto blockPtrOf = blockLoadOp.getSrcPointer().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(blockLoadOp.getSrcPointer())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    blockLoadOp->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    blockLoadOp->setAttr("stride", blockPtrOf.getStrideAttr());
    blockLoadOp->setAttr("order", blockPtrOf.getOrderAttr());
    blockLoadOp->setAttr("offset", blockPtrOf.getOffsetAttr());
    blockLoadOp->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    blockPtrOf->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    pointerOfOp->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    return mlir::success();
  }
};

// deepgengraph_triton.block_store 加入详细信息，以免后续op转换后丢失
class BlockStoreOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockStoreOp blockStoreOp,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    auto blockPtrOf = blockStoreOp.getDstPointer().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(blockStoreOp.getDstPointer())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    blockStoreOp->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    blockStoreOp->setAttr("stride", blockPtrOf.getStrideAttr());
    blockStoreOp->setAttr("order", blockPtrOf.getOrderAttr());
    blockStoreOp->setAttr("offset", blockPtrOf.getOffsetAttr());
    blockStoreOp->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    blockPtrOf->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    pointerOfOp->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    return mlir::success();
  }
};

// deepgengraph_triton.block_store 加入详细信息，以免后续op转换后丢失
class BlockAdvanceOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockAdvanceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    auto blockPtrOf = op.getOperand().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(op.getOperand())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    op->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    op->setAttr("stride", blockPtrOf.getStrideAttr());
    op->setAttr("order", blockPtrOf.getOrderAttr());
    op->setAttr("offset", blockPtrOf.getOffsetAttr());
    op->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    return mlir::success();
  }
};


// deepgengraph_triton.ptr_of -> ptr_to_global
class PointerOfOpConverionPattern : public OpConversionPattern<deepgengraph::triton::PointerOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    deepgengraph::triton::PointerOfOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
  {
    auto newRetType = getTypeConverter()->convertType(op.getType());
    auto newOp = rewriter.create<threadimp::PointerToOp>(
      op->getLoc(), newRetType, adaptor.getOperand(), threadimp::MemSpace::GM);
    int argId = mlir::cast<mlir::IntegerAttr>(op->getAttr("argId")).getInt();
    s_map_argId_newGlobalPtr[argId] = newOp.getResult();
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};



// deepgengraph_triton.block_ptr_of : 包含了 globalPtr 和隐含的一块shm。意为 从global读取数据后，写入shm，或者从shm写入 global的指定位置。
// deepgengraph_triton.block_advance ： 移动 globalPtr 偏移指定offset，之后再读写数据到 shm
// 可利用 argId 属性，直接访问 func的arg，
// 需要处理 类型转换时 pointer 指向的tensor形状< 全局形状 vs blockshape>

class BlockPtrOfOpConverionPattern : public OpConversionPattern<deepgengraph::triton::BlockPointerOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    deepgengraph::triton::BlockPointerOfOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
  {
    auto ptrType = this->getTypeConverter()->convertType(op.getType());
    auto eleTy = mlir::cast<threadimp::PointerType>(ptrType).getElementType();
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> shape, Type elementType);
    auto shmBuffer = rewriter.create<AllocSharedOp>(op->getLoc(), adaptor.getBlockShape(), eleTy);
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type pointer, ::mlir::Value srcTensor, ::mlir::threadimp::MemSpace memspace);
    auto ptrOp = rewriter.create<threadimp::PointerToOp>(op->getLoc(), shmBuffer, MemSpace::SHM );
    ptrOp->setAttr("argId", op->getAttr("argId"));
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    s_map_argId_newShmPtr[argId] = ptrOp;
    rewriter.replaceOp(op, ptrOp);
    return mlir::success(); 
  }
};




// deepgengraph_triton.block_load -> copyGlobalToShm
class BlockLoadOpConversionPatern : public OpConversionPattern<deepgengraph::triton::BlockLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    deepgengraph::triton::BlockLoadOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const 
  {
    auto newRetTy = this->getTypeConverter()->convertType(op.getType());
    DenseI64ArrayAttr offset = mlir::cast<DenseI64ArrayAttr>(op->getAttr("offset")) ;
    DenseI64ArrayAttr block_shape = mlir::cast<DenseI64ArrayAttr>(op->getAttr("block_shape")) ;
    DenseI64ArrayAttr order = mlir::cast<DenseI64ArrayAttr>(op->getAttr("order")) ;
    DenseI64ArrayAttr stride = mlir::cast<DenseI64ArrayAttr>(op->getAttr("stride")) ;
    auto shmPtr = adaptor.getSrcPointer();
    int argId = mlir::cast<IntegerAttr>(op->getAttr("argId")).getInt();
    auto globalPtr = s_map_argId_newGlobalPtr[argId]; 
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src_pointer, ::mlir::Value dst_pointer, ::llvm::ArrayRef<int64_t> offset, ::llvm::ArrayRef<int64_t> block_shape, ::llvm::ArrayRef<int64_t> src_stride, ::llvm::ArrayRef<int64_t> src_order);

    auto newOp = rewriter.create<threadimp::CopyGlobalToShm>(op->getLoc(), 
      globalPtr,shmPtr, offset , block_shape, stride, order);
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultData, ::mlir::Value shm_ptr);
    auto loadOp = rewriter.create<threadimp::ReadShmOp>(op->getLoc(), newRetTy, shmPtr);
    newOp->setAttr("argId", op->getAttr("argId"));
    loadOp->setAttr("argId", op->getAttr("argId"));
    rewriter.replaceOp(op, loadOp);
    return mlir::success(); 
  }
};

class BlockStoreOpConversionPatern : public OpConversionPattern<deepgengraph::triton::BlockStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    deepgengraph::triton::BlockStoreOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const 
  {
    DenseI64ArrayAttr offset = mlir::cast<DenseI64ArrayAttr>(op->getAttr("offset")) ;
    DenseI64ArrayAttr block_shape = mlir::cast<DenseI64ArrayAttr>(op->getAttr("block_shape")) ;
    DenseI64ArrayAttr order = mlir::cast<DenseI64ArrayAttr>(op->getAttr("order")) ;
    DenseI64ArrayAttr stride = mlir::cast<DenseI64ArrayAttr>(op->getAttr("stride")) ;
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto globalPtr = s_map_argId_newGlobalPtr[argId];
    auto shmPtr = adaptor.getDstPointer();
    auto newOp = rewriter.create<threadimp::CopyShmToGlobal>(op->getLoc(), 
    shmPtr, globalPtr ,offset , block_shape, stride, order);
    newOp->setAttr("argId", op->getAttr("argId"));
    rewriter.replaceOp(op, newOp);
    return mlir::success(); 
  }
};

// deepgengraph_triton.block_advance -> ptr_advance 移动 glboal pointer
class BlockAdvanceOpConverionPattern : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, 
      OpAdaptor adaptor, // adaptor 中的 operands 已经是转换后的类型了！
      ConversionPatternRewriter &rewriter) const override 
  {
    using namespace deepgengraph::triton;
    Value srcPtr = adaptor.getSrcPointer();

    auto offsets = adaptor.getOffsets();
    auto loc = op->getLoc();
    auto nextPtrType = getTypeConverter()->convertType(op.getNextPointer().getType());
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto globalPtr = s_map_argId_newGlobalPtr[argId];
    auto newop = rewriter.create<threadimp::PointerAdvanceOp>(loc, nextPtrType, adaptor.getSrcPointer() , op.getOffsets());
    newop->setAttr("argId", op->getAttr("argId"));
    rewriter.replaceOp(op, newop);
    return success();
  }
};

// deepgengraph_triton.block_advance -> ptr_advance 移动 glboal pointer
class ZeroOpConverionPattern : public OpConversionPattern<deepgengraph::ZeroOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::ZeroOp op, 
      OpAdaptor adaptor, // adaptor 中的 operands 已经是转换后的类型了！
      ConversionPatternRewriter &rewriter) const override 
  {
    auto newOp = rewriter.create<threadimp::AllocSharedOp>(op->getLoc(), adaptor.getShape(), adaptor.getElementType(), rewriter.getF32FloatAttr(0.0));
    // auto ptr = rewriter.create<threadimp::PointerToOp>(op->getLoc(), newOp, MemSpace::SHM);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};


// 假设我们要将类型 !old_type 转换为 !new_type
// 在实际代码中，这可能是从 deepgengraph_triton.block_ptr 转换到 threadimp.ptr

struct SCFForTypeConversion : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      scf::ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // 1. 获取转换后的结果类型 (基于 IterArgs)
    SmallVector<Type, 4> newResultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    // 2. 创建新的 ForOp
    // 注意：induction variable (iv) 通常保持 index 类型，不需要转化
    auto newOp = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    
    Block* oldBody = op.getBody();
    Block* newBody = newOp.getBody();
    rewriter.mergeBlocks(oldBody, newBody, newBody->getArguments());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// 同样需要处理 scf.yield，因为它的操作数必须与新的 scf.for 返回类型匹配
struct SCFYieldTypeConversion : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, 
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<scf::YieldOp>(op->getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct SCFForOpReplaceIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op, 
    PatternRewriter &rewriter) const override {
    auto oldIterArgs = op.getRegionIterArgs();
    auto oldInitArgs = op.getInitArgs();
    std::vector<Value> newInitArgs;
    DenseMap<int, int> map_argId_blockArgId;
    // if we found shm ptr in iterargs, it needs to be modified to corresponding global ptr
    bool needTransform = false;
    for(auto arg : oldInitArgs){
      auto ty = mlir::dyn_cast<threadimp::PointerType>(arg.getType());
      if(ty != nullptr && ty.getMemorySpace() != threadimp::MemSpace::GM){
        needTransform = true;
        break;
      }
    }
    if(!needTransform){
      return failure();
    }
    for(int i=0 ; i<oldIterArgs.size() ; ++i){
      auto oldInitArg = op.getInitArgs()[i];
      if(mlir::isa<threadimp::PointerType>(oldInitArg.getType())){
        // need to be repalced with globalPtr
        auto defop = oldInitArg.getDefiningOp<threadimp::PointerToOp>();
        if(defop == nullptr){
          assert(false);
        }
        else{
          int id = defop->getAttrOfType<IntegerAttr>("argId").getInt();
          auto globalPtr = s_map_argId_newGlobalPtr[id];
          newInitArgs.push_back(globalPtr);
          map_argId_blockArgId[id] = i;
        }
      }
      else{
        newInitArgs.push_back(oldInitArg);
      }
    }
    auto newOp = rewriter.create<scf::ForOp>(op->getLoc(), 
          op.getLowerBound(),
          op.getUpperBound(), 
          op.getStep(), newInitArgs);
    IRMapping mapping;
    mapping.map(op.getInductionVar(), newOp.getInductionVar());
    for(auto i=0;i < op.getNumRegionIterArgs() ; ++i){
      mapping.map(op.getRegionIterArg(i) , newOp.getRegionIterArg(i));
    }
    rewriter.setInsertionPointToStart(newOp.getBody());
    for(auto & subOp : op.getBody()->without_terminator()){
      auto loc = subOp.getLoc();
      if(auto old = mlir::dyn_cast<threadimp::CopyGlobalToShm>(subOp)){
        auto argId = old->getAttrOfType<IntegerAttr>("argId").getInt();
        auto shmPtr = s_map_argId_newShmPtr[argId];
        auto gid = map_argId_blockArgId[argId];
        auto iterGlobalPtr = newOp.getRegionIterArg(gid);
        // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultData, ::mlir::Value src_pointer, ::mlir::Value dst_pointer, ::mlir::DenseI64ArrayAttr offset, ::mlir::DenseI64ArrayAttr block_shape, ::mlir::DenseI64ArrayAttr src_stride, ::mlir::DenseI64ArrayAttr src_order);
        auto newSubOp = rewriter.create<threadimp::CopyGlobalToShm>(loc,  iterGlobalPtr, shmPtr, 
          old.getOffsetAttr(), old.getBlockShapeAttr(), old.getSrcStrideAttr(), old.getSrcOrderAttr()
        );
        // 手动处理 mapping。将oldresult 映射到newresult 以便后续op使用
        mapping.map(old->getResults(), newSubOp->getResults());
      }
      else if(auto old = mlir::dyn_cast<threadimp::ReadShmOp>(subOp)){
        auto argId = old->getAttrOfType<IntegerAttr>("argId").getInt();
        auto shmPtr = s_map_argId_newShmPtr[argId];
        auto newReadShm = rewriter.create<ReadShmOp>(subOp.getLoc(), old.getType(), shmPtr);
        mapping.map(old.getResultData(), newReadShm.getResultData());
      }
      else if(auto old = mlir::dyn_cast<threadimp::WriteShmOp>(subOp)){
        auto argId = old->getAttrOfType<IntegerAttr>("argId").getInt();
        auto shmPtr = s_map_argId_newShmPtr[argId];
        auto newWrite = rewriter.create<WriteShmOp>(subOp.getLoc(), old.getType(), shmPtr, old.getData());
        mapping.map(old->getResults(), newWrite->getResults());
      }
      else if(auto old = mlir::dyn_cast<threadimp::CopyShmToGlobal>(subOp)){
        auto argId = old->getAttrOfType<IntegerAttr>("argId").getInt();
        auto shmPtr = s_map_argId_newShmPtr[argId];
        auto gid = map_argId_blockArgId[argId];
        auto iterGlobalPtr = newOp.getRegionIterArg(gid);
        // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultData, ::mlir::Value src_pointer, ::mlir::Value dst_pointer, ::mlir::DenseI64ArrayAttr offset, ::mlir::DenseI64ArrayAttr block_shape, ::mlir::DenseI64ArrayAttr src_stride, ::mlir::DenseI64ArrayAttr src_order);
        auto newSubOp = rewriter.create<threadimp::CopyGlobalToShm>(loc,  iterGlobalPtr, shmPtr, 
          old.getOffsetAttr(), old.getBlockShapeAttr(), old.getDstStrideAttr(), old.getDstOrderAttr()
        );
        mapping.map(old->getResults(), newSubOp->getResults());
      }
      else if(auto old = mlir::dyn_cast<threadimp::PointerAdvanceOp>(subOp)){
        auto argId = old->getAttrOfType<IntegerAttr>("argId").getInt();
        auto globalPtr = newOp.getRegionIterArg(map_argId_blockArgId[argId]);
        // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type next_pointer, ::mlir::Value src_pointer, ::mlir::DenseI64ArrayAttr offsets);
        auto newPtrAdv = rewriter.create<threadimp::PointerAdvanceOp>(loc, globalPtr.getType(), globalPtr, old.getOffsetsAttr());
        mapping.map(old->getResults(), newPtrAdv->getResults());
      }
      else{
        rewriter.clone(subOp, mapping);
      }
    }
    // 5. 处理 scf.yield
    auto oldYield = cast<scf::YieldOp>(op.getBody()->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (Value operand : oldYield.getOperands()) {
        newYieldOperands.push_back(mapping.lookupOrDefault(operand));
    }
    auto newYield = rewriter.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);

    // 6. 替换旧循环的返回值
    rewriter.replaceOp(oldYield, newYield);
    // replace forop
    rewriter.replaceOp(op, newOp);
    return success();
  }
};




// --- Pass 注册部分 ---

class MemOpConversionPass : public impl::DeepgengraphToThreadImpBase< MemOpConversionPass >
{
public:
  void runOnOperation() {
    /**
     * @brief 
     * threadimp dialect下  pointer用于抽象global, tensor用于抽象shm 和 reg。
     */
    using namespace deepgengraph::triton;
    MLIRContext *context = &getContext();
    RewritePatternSet ps1(context);
    RewritePatternSet ps2(context);
    RewritePatternSet ps3(context);
    auto converter = GetThreadImpTypeConverter();


    // 为 block_load 和 block_store 添加信息。以免block_ptr_of被转化后丢失
    ps1.add<
      BlockLoadOpAddInfoPattern,BlockStoreOpAddInfoPattern,BlockAdvanceOpAddInfoPattern
    >(  context);
    // op转换
    ps2.add<
      PointerOfOpConverionPattern,
      BlockPtrOfOpConverionPattern,
      SCFForTypeConversion,
      SCFYieldTypeConversion,
      BlockAdvanceOpConverionPattern,
      BlockLoadOpConversionPatern,
      BlockStoreOpConversionPatern,
      ZeroOpConverionPattern
    >(*converter, context);
    ps3.add<SCFForOpReplaceIterArgs>(context);

    // 必须告知 Target 哪些 Dialect 是合法的
    ConversionTarget target(*context);
    target.addLegalDialect<
      threadimp::ThreadImpDialect, 
      tensor::TensorDialect,
      deepgengraph::triton::DeepgengraphTritonDialect,
      deepgengraph::DeepgengraphDialect,
      scf::SCFDialect,
      arith::ArithDialect
      >();
    
    target.addIllegalOp<
      deepgengraph::triton::BlockPointerOfOp,
      deepgengraph::triton::EmptyPointerOp,
      deepgengraph::triton::PointerOfOp,
      deepgengraph::triton::BlockLoadOp,
      deepgengraph::triton::BlockStoreOp,
      deepgengraph::triton::BlockAdvanceOp,
      deepgengraph::triton::TensorFromOp,
      deepgengraph::ZeroOp
    >();

    target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op){
      for(auto arg : op.getInitArgs()){
        if(mlir::isa<deepgengraph::triton::BlockPointerType, deepgengraph::triton::PointerType>(arg.getType())){
          return false;
        }
      }
      return true;
    });
    target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp op){
      for(auto operand : op.getOperands()){
        if(mlir::isa<deepgengraph::triton::BlockPointerType, deepgengraph::triton::PointerType>(operand.getType())){
          return false;
        }
      }
      return true;
    });

    if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(ps1)))){
      // 这一步不检查合法性。只要匹配就做转化。
      // 如果用 applyPartialConversion 却不声明非法Op，则此Pattern不会执行（因为op是合法的，只是加了属性）
      signalPassFailure();
    }
    if (failed(applyPartialConversion(getOperation(), target, std::move(ps2)))) {
      signalPassFailure();
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(ps3)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createConvertMemOpPass(){
  return std::make_unique<MemOpConversionPass>(); 
}

}
