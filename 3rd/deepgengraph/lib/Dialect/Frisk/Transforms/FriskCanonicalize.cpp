#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKCANONICALIZE

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

// 进行数据流分析。化简kernel中的部分结构（如不必要的 alloc_buffer.）
/*
blockOp ：分析其 src 和 dst buffer。如果遇到 blockOp的dst作为 convertOp的输入，则直接在blockOp内进行数据类型转换 
*/

struct BlockOpBorderInfo{
  BlockOp blockOp  {};
  SmallVector<affine::AffineLoadOp> ins {};
  affine::AffineStoreOp out {};
};

// 获取 blockOp内的边界RW buffer（外界通信buffer）
static BlockOpBorderInfo* GetInterfaceRWBufferOfBlockOps(BlockOp blockOp){
  auto info = new BlockOpBorderInfo{};
  blockOp->walk([&](affine::AffineLoadOp loadOp){
    auto loadDefOp = loadOp.getMemref().getDefiningOp();
    if(loadDefOp== nullptr || loadDefOp->getParentOp() != blockOp){
      info->ins.push_back(loadOp);
    }
  });
  blockOp->walk([&](affine::AffineStoreOp storeOp){
    auto storeDefOp = storeOp.getMemref().getDefiningOp();
    if(storeDefOp==nullptr || storeDefOp->getParentOp() != blockOp){
      info->out = storeOp;
    }
  });
  if(info->ins.empty() && info->out == nullptr){
    blockOp->erase();
    delete info;
    info = nullptr;
  }
  info->blockOp = blockOp;
  return info;
}

static affine::AffineLoadOp checkStoreDstMemIsLoadSrcMem(SmallVector<affine::AffineLoadOp>& loadOps, Value storeDstMem){
  for(auto load : loadOps){
    if(load.getMemref() == storeDstMem){
      return load;
    }
  }
  return nullptr;
}

static frisk::CopyOp checkStoreDstMemIsCopySrcMem(SmallVector<frisk::CopyOp>& copyOps, Value storeDstMem){
  for(auto copyOp : copyOps){
    if(copyOp.getSrc() == storeDstMem){
      return copyOp;
    }
  }
  return nullptr;
}

static bool opUsesValue(Operation *op, Value value) {
  bool found = false;
  op->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    if (found) {
      return;
    }
    for (Value operand : nested->getOperands()) {
      if (operand == value) {
        found = true;
        return;
      }
    }
  });
  return found;
}

static bool isFullShapeSameTypeCopy(frisk::CopyOp copyOp) {
  auto srcTy = dyn_cast<MemRefType>(copyOp.getSrc().getType());
  auto dstTy = dyn_cast<MemRefType>(copyOp.getDst().getType());
  if (!srcTy || !dstTy) {
    return false;
  }
  if (!copyOp.getMapOperands().empty()) {
    return false;
  }
  return srcTy.getShape() == dstTy.getShape() &&
         srcTy.getElementType() == dstTy.getElementType();
}

static frisk::CopyOp findOnlyCopyUseAfterBlock(BlockOp blockOp, Value data) {
  frisk::CopyOp candidate = nullptr;
  for (Operation *op = blockOp->getNextNode(); op != nullptr;
       op = op->getNextNode()) {
    if (auto copyOp = dyn_cast<frisk::CopyOp>(op)) {
      if (copyOp.getSrc() == data) {
        if (candidate || !isFullShapeSameTypeCopy(copyOp)) {
          return nullptr;
        }
        candidate = copyOp;
        continue;
      }
    }
    if (opUsesValue(op, data)) {
      return nullptr;
    }
  }
  return candidate;
}

static bool sinkBlockStoreToCopyDst(BlockOp blockOp, OpBuilder &builder) {
  SmallVector<affine::AffineStoreOp, 2> stores;
  blockOp->walk([&](affine::AffineStoreOp storeOp) {
    auto storeDefOp = storeOp.getMemref().getDefiningOp();
    if (storeDefOp == nullptr || storeDefOp->getParentOp() != blockOp) {
      stores.push_back(storeOp);
    }
  });
  if (stores.size() != 1) {
    return false;
  }

  auto storeOp = stores.front();
  Value data = storeOp.getMemref();
  auto copyOp = findOnlyCopyUseAfterBlock(blockOp, data);
  if (!copyOp) {
    return false;
  }

  auto dstTy = dyn_cast<MemRefType>(copyOp.getDst().getType());
  if (!dstTy || dstTy.getRank() != storeOp.getAffineMap().getNumResults()) {
    return false;
  }

  builder.setInsertionPoint(storeOp);
  builder.create<affine::AffineStoreOp>(
      storeOp.getLoc(), storeOp.getValue(), copyOp.getDst(),
      storeOp.getAffineMap(), storeOp.getMapOperands());

  Operation *dataDefOp = data.getDefiningOp();
  storeOp.erase();
  copyOp.erase();
  if (dataDefOp && data.use_empty()) {
    dataDefOp->erase();
  }
  return true;
}


static void MergeBlockOps(BlockOp from, BlockOp to) {
  if (from.getBodyRegion().empty() || to.getBodyRegion().empty())
    return;

  mlir::Block &fromBlock = from.getBodyRegion().front();
  mlir::Block &toBlock = to.getBodyRegion().front();

  // 1. 处理 Block Arguments 的映射
  // 假设两个 frisk.block 的参数数量和含义是一致的（都是对应的迭代坐标）
  if (fromBlock.getNumArguments() == toBlock.getNumArguments()) {
    for (unsigned i = 0; i < fromBlock.getNumArguments(); ++i) {
      fromBlock.getArgument(i).replaceAllUsesWith(toBlock.getArgument(i));
    }
  } else {
    // 如果参数不匹配，说明融合逻辑有安全隐患，直接返回或报错
    return;
  }

  // 2. 精确移动操作，跳过或删除 Terminator
  for (auto it = fromBlock.begin(); it != fromBlock.end(); ) {
    mlir::Operation &op = *it++;
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      op.erase(); // 删掉原 block 的结束符
    } else {
      // 移动到 toBlock 的末尾（在 toBlock 的 terminator 之前，如果没有 terminator 就放最后）
      if (!toBlock.empty() && toBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        op.moveBefore(&toBlock.back());
      } else {
        op.moveBefore(&toBlock, toBlock.end());
      }
    }
  }

  // 3. 此时 fromBlock 已经完全干净了（没有任何 Op，参数也没有任何 use），可以安全擦除
  from.erase();
}

// 融合in out相连的 blockOp
struct FuseBlockOps : public PassWrapper<FuseBlockOps, OperationPass<frisk::KernelOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseBlockOps)

  StringRef getArgument() const final { return "add-tensor-memspace"; }
  StringRef getDescription() const final { return "Add memspace encoding to tensors based on their position."; }

  void runOnOperation() override {
    auto kernelOp = getOperation();
    
    std::vector<BlockOpBorderInfo*> infoArr;
    kernelOp->walk<WalkOrder::PreOrder>([&](frisk::BlockOp blockOp){
      auto info = GetInterfaceRWBufferOfBlockOps( blockOp);
      infoArr.push_back(info);
    });
    
    int i = 0;
    while(i+1 < infoArr.size()){
      // 以 infoArr[baseIndex] 为基准，尽可能将后续 blockOp融合进该Op内
      while(infoArr[i] == nullptr && i < infoArr.size()){
        i++;
      }
      if(i+1 >= infoArr.size()){
        break;
      }
      BlockOp baseOp = infoArr[i]->blockOp;
      affine::AffineStoreOp baseStore = infoArr[i]->out;
      // 检查base之后的info。如果info[j].ins 含有 base.outs, 则将 info[j].blockOp 融合进 base.blockOp
      for(int j=i+1;j<infoArr.size();++j){
        if(infoArr[j] == nullptr){
          continue;
        }
        if(infoArr[j]->blockOp->getBlock() != baseOp->getBlock()){
          continue;
        }
        auto ld = checkStoreDstMemIsLoadSrcMem(infoArr[j]->ins, baseStore.getMemref());
        if(ld != nullptr){
          // load的结果直接替换为 baseStore 的value
          // 删除冗余的 store/load 对
          ld.replaceAllUsesWith(baseStore.getValue());
          ld->erase();
          baseStore->erase();
          MergeBlockOps(infoArr[j]->blockOp, baseOp);  // 融合
          baseStore = infoArr[j]->out;  // 更新baseStore
          delete infoArr[j]; 
          infoArr[j] = nullptr;  // 释放无用信息
        }
      }
      i++;
    }
    
    SmallVector<frisk::AllocBufferOp> dumpBufferAllocOps ;
    kernelOp->walk([&](frisk::AllocBufferOp allocOp){
      if(allocOp->getUsers().empty()){
        dumpBufferAllocOps.push_back(allocOp);
      }
    });
    for(auto op : dumpBufferAllocOps){
      op->erase();
    }

    SmallVector<BlockOp, 8> blockOps;
    kernelOp->walk<WalkOrder::PreOrder>([&](BlockOp blockOp) {
      blockOps.push_back(blockOp);
    });

    OpBuilder builder(kernelOp->getContext());
    bool changed = true;
    while (changed) {
      changed = false;
      for (BlockOp blockOp : blockOps) {
        if (blockOp->getParentOp() == nullptr) {
          continue;
        }
        changed |= sinkBlockStoreToCopyDst(blockOp, builder);
      }
    }
  }
};


// blockOp与 相连的 frisk.copy 融合（当copy具有 dtype类型转换的语义时）
struct FuseBlockOpWithTypeConversion : public PassWrapper<FuseBlockOpWithTypeConversion, OperationPass<frisk::KernelOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseBlockOpWithTypeConversion)

  StringRef getArgument() const final { return "add-tensor-memspace"; }
  StringRef getDescription() const final { return "Add memspace encoding to tensors based on their position."; }

  void runOnOperation() override {
    auto kernelOp = getOperation();
    
    std::vector<BlockOpBorderInfo*> infoArr;
    kernelOp->walk<WalkOrder::PreOrder>([&](frisk::BlockOp blockOp){
      auto info = GetInterfaceRWBufferOfBlockOps( blockOp);
      infoArr.push_back(info);
    });
    SmallVector<CopyOp> typeConversionOps {};
    kernelOp.walk<WalkOrder::PreOrder>([&](frisk::CopyOp copyOp){
      bool isTypeSame = copyOp.getDst().getType().getElementType() == copyOp.getSrc().getType().getElementType();
      bool isShapeSame = copyOp.getDst().getType().getShape() == copyOp.getSrc().getType().getShape();
      if(isShapeSame && !isTypeSame){
        // 具有类型转化的语义
        typeConversionOps.push_back(copyOp);
      }
    });


    OpBuilder b{kernelOp->getContext()};
    for(auto info : infoArr){
      auto cp = checkStoreDstMemIsCopySrcMem(typeConversionOps, info->out.getMemref());
      if(cp != nullptr){
        /*
        {
          ...
          affine.store %val, %mem[%arg0, %arg1] : f32
        }
        frisk.copy %mem, %2 : memref<5xf32>, memref<5xf16>

        */
        b.setInsertionPoint(info->out);
        auto srcBitWidth = info->out.getMemref().getType().getElementTypeBitWidth();
        auto dstBitWidth = cp.getDst().getType().getElementTypeBitWidth();
        auto eleType = cp.getDst().getType().getElementType();
        // 插入extf或 truncf 直接转化value，存入 frisk.copy 的 dstMem 
        mlir::Operation* convertOp {};
        if(srcBitWidth < dstBitWidth){
          if(eleType.isFloat()){
            convertOp = b.create<arith::ExtFOp>(b.getUnknownLoc(), eleType, info->out.getValue());
          }
          else if(eleType.isIntOrIndex()){
            convertOp = b.create<arith::ExtSIOp>(b.getUnknownLoc(), eleType, info->out.getValue());
          }
        }
        else{
          if(eleType.isFloat()){
            convertOp = b.create<arith::TruncFOp>(b.getUnknownLoc(), eleType, info->out.getValue());
          }
          else if(eleType.isIntOrIndex()){
            convertOp = b.create<arith::TruncIOp>(b.getUnknownLoc(), eleType, info->out.getValue());
          }
        }
        assert(convertOp != nullptr);
        auto copyDstDefineOp = cp.getDst().getDefiningOp();
        // 如果 frisk.copy 的 dstMem 的定义Op位于 blockOp后，则将其移到前面
        if(copyDstDefineOp != nullptr && copyDstDefineOp->getBlock() == info->blockOp->getBlock()){
          if(!copyDstDefineOp->isBeforeInBlock(info->blockOp)){
            copyDstDefineOp->moveBefore(info->blockOp);
          }
          // 转换后的element 直接存入 frisk.copy 的dstMem
          auto newStore = b.create<affine::AffineStoreOp>(b.getUnknownLoc(), convertOp->getResult(0), cp.getDst(), info->out.getIndices());
          // 删除冗余的 store/copy 对
          auto affineStoreMemrefDefOp = info->out.getMemref().getDefiningOp();
          info->out->erase();
          cp->erase();
          if(affineStoreMemrefDefOp){
            // 删除原本用于进行 类型转换创建的 中介memref allocOp
            affineStoreMemrefDefOp->erase();
          }
        }
      }
    }
    
  }
};
    

class FriskCanonicalizePass : public impl::FriskCanonicalizeBase<FriskCanonicalizePass> {
  void runOnOperation() override {
    auto kernelOp = getOperation();
    DenseMap<BlockOp, BlockOpBorderInfo*> map_blockop_bufferInfo{};
    kernelOp->walk<WalkOrder::PreOrder>([&](frisk::BlockOp blockOp){
      auto info = GetInterfaceRWBufferOfBlockOps( blockOp);

    });


    OpBuilder b{kernelOp->getContext()};
    // find frisk.copy op作为 数值类型转换的场景
    SmallVector<CopyOp, 4> convertDtypeOps;
    kernelOp->walk([&](CopyOp copy){
      auto srcTy = copy.getSrc().getType();
      auto dstTy = copy.getDst().getType();
      if(srcTy.getElementType() != dstTy.getElementType()){
        // 元素类型不同。为数值类型转换语义
        convertDtypeOps.push_back(copy);
      }
    });
    
    // 尝试融合 convertDTypeOp 到blockOp内
    for(auto copyOp : convertDtypeOps){
      affine::AffineStoreOp targetStore{};
      for(auto[blockOp , info] : map_blockop_bufferInfo){
        // for(auto storeOp : info->outs){
        //   if(copyOp.getSrcMemRef() == storeOp.getMemref()){
        //     targetStore = storeOp;

        //   }
        // }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFriskCanonicalizePass() {
  return std::make_unique<FriskCanonicalizePass>();
}
std::unique_ptr<Pass> createFriskFuseBlockOpsPass() {
  return std::make_unique<FuseBlockOps>();
}
std::unique_ptr<Pass> createFuseBlockOpWithDTypeConvertOpPass() {
  return std::make_unique<FuseBlockOpWithTypeConversion>();
}

} // namespace mlir::frisk
