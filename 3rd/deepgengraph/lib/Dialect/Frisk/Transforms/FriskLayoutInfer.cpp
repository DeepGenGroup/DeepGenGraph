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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "deepgengraph/Analysis/HardwareSpecification.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKLAYOUTINFER

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

// 还需要考虑 warp_layout, block_layout, warpgroup_layout
// inst -> 可同时确定 warp_layout / wg_layout, 以及 thread_layout (thread计算的reg如何排列)
// 从宏观 tile 可定 外部循环次数 
enum class LayoutKind : int {
  swizzle_32B = 0,
  swizzle_64B = 1,
  swizzle_128B = 2,
  gemm_acc = 3,
  gemm_operator = 4,
  undef = 9
};

enum class ElementType : int { f16 = 1, f32 = 2, i16 = 3, i32 = 4 };

// 本质：通过 warpId, laneId, regId(当前线程需要算几个数) 算出一个buffer的逻辑坐标 [i,j]
// 编程时，对于单个线程，warp lane 确定。假设有两个buffer, layout分别为 f1 f2,
// 对于任意 regId ∈ 需要计算的数据个数
// f1(regId, warp,lane) = f2(regId, warp,lane) = [i,j]
// 则可认为 buffer的布局一致

struct Layout {
  LayoutKind kind = LayoutKind::undef;
  std::vector<int> inst_mnk;
  SmallVector<int,2> warp_layout;
  SmallVector<int,2> wg_layout;

};


// exist incoming 根据二者关系，将 exist 的属性做调整或不变

struct LayoutConversionInsertPoint {
    mlir::Operation* before;  // 在哪个op前插入
    mlir::Value buffer;  // 是哪个buffer需要改layout
    Layout from;  // 原有layout
    Layout to;  // 目的layout
};

class LayoutResolver {
public:
  std::vector<LayoutConversionInsertPoint> convertsToAdd;
  bool IsLayoutEqual(const Layout& a, const Layout& b){
    if(a.kind == b.kind){
      if(a.kind == LayoutKind::gemm_operator || a.kind == LayoutKind::gemm_acc){
        if(a.inst_mnk == b.inst_mnk){
          return true;
        }
      }
    }
    return false;
  }
  // 逻辑有些弱。后续完善下
  void ResolveConflictOrAddLayoutConvert(mlir::Operation* incomingOp,mlir::Value buffer, Layout* exist, const Layout& incoming){
    if(exist->kind == LayoutKind::gemm_acc || exist->kind == LayoutKind::gemm_operator){
      if(incoming.kind == LayoutKind::gemm_acc || incoming.kind == LayoutKind::gemm_operator){
        if(IsLayoutEqual(*exist, incoming)){
          return;
        }
        LayoutConversionInsertPoint ip;
        ip.buffer = buffer;
        ip.before = incomingOp;
        ip.from = *exist;
        ip.to = incoming;
        convertsToAdd.push_back(ip);
        return;
      }
      else{
        // keep exist layout
        return; 
      }
    }
    else{
      if(incoming.kind == LayoutKind::gemm_acc || incoming.kind == LayoutKind::gemm_operator){
        exist->kind = incoming.kind;
        exist->inst_mnk = incoming.inst_mnk;
        // assert(exist->reg_count == incoming.reg_count) ;  // 对同一buffer，其 reg_count 必然不变
      }
      else{
        // keep exist
        return; 
      }
    }

  };
};

static LayoutResolver LR;

// 选择 wgmma 这类 tensorcore指令的MNK. 逻辑后期完善
static std::vector<int32_t> SelectTCInstMNK(int32_t tileM, int32_t tileN, int32_t tileK, VendorKind vendor) {
  std::vector<int32_t> mnk;
  switch (vendor) { 
    case mlir::frisk::VendorKind::DCU:
    // 16x16x4 f32
    // 16x16x8 f32
    // 16x16x16 ab_f16_acc_f32
    // 16x16x32 ab_i8_acc_i32
      return {16,16,16};
    case mlir::frisk::VendorKind::NVIDIA:
      return {};
    case mlir::frisk::VendorKind::AMD:
      return {};

  }
}


// GemmOp的 ABC判断是否满足硬件要求。如果 memspaceA B 不满足，则加上对应buffer alloc
struct GemmOperatorInsertBuffer : public mlir::OpRewritePattern<frisk::GemmOp> {
  using mlir::OpRewritePattern<frisk::GemmOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(frisk::GemmOp op, mlir::PatternRewriter &rewriter) const override {
    // auto hw = GetHWSpecification("dcu","bw1000", op->getContext());
    // auto memTypeA = mlir::cast<MemRefType>(op.getMatrixA().getType());
    // auto memspaceA = memTypeA.getMemorySpaceAsInt();
    // auto memTypeB = mlir::cast<MemRefType>(op.getMatrixB().getType());
    // auto memspaceB = memTypeB.getMemorySpaceAsInt();
    // mlir::Value bufferA = nullptr;
    // mlir::Value bufferB = nullptr;
    // llvm::outs() <<"hw:ms " << hw->gemmInfo.memspace_a <<", " << hw->gemmInfo.memspace_b << " | " << memspaceA << "," << memspaceB << "\n";llvm::outs().flush();
    // if(memspaceA != hw->gemmInfo.memspace_a){
    //   bufferA = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),  memTypeA.getShape(), memTypeA.getElementType(), 16, hw->gemmInfo.memspace_a);
    //   auto copyToA = rewriter.create<frisk::CopyOp>(op->getLoc(), op.getMatrixA(), bufferA);
    // }
    // if(memspaceB != hw->gemmInfo.memspace_b){
    //   bufferB = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),  memTypeB.getShape(), memTypeB.getElementType(), 16, hw->gemmInfo.memspace_b);
    //   auto copyToB = rewriter.create<frisk::CopyOp>(op->getLoc(), op.getMatrixB(), bufferB);
    // }
    // if(bufferA != nullptr || bufferB != nullptr){
    //   auto bufA = bufferA == nullptr ? op.getMatrixA() : bufferA;
    //   auto bufB = bufferB == nullptr ? op.getMatrixB() : bufferB;
    //   auto newGemm = rewriter.create<frisk::GemmOp>(op->getLoc(),  bufA, bufB, op.getMatrixC(), op.getTransA(), op.getTransB());
    //   rewriter.replaceOp(op, newGemm);
    //   return success();
    // }
    // else{
    //   return failure();
    // }
    return failure();
  }
};


class FriskLayoutInferPass : public impl::FriskLayoutInferBase<FriskLayoutInferPass> {
  void runOnOperation() override {
    auto kernelOp = getOperation();
    if(!kernelOp->hasAttr("thread_num")){
      return;
    }
    auto thread_num = mlir::cast<IntegerAttr>(kernelOp->getAttr("thread_num")).getInt();
    auto context = kernelOp->getContext();
    
    // -------- step1 : 结合HW特性，对GEMM 判定 abc memspace是否合规。不符合则添加alloc_buffer 
    mlir::RewritePatternSet ps{context};
    ps.add<GemmOperatorInsertBuffer>(context);
    if(failed(applyPatternsGreedily(kernelOp, std::move(ps)))){
      llvm::errs() << "GemmOperatorInsertBuffer error\n";
    }
    
    // -------- step2 ： 收集 alloc_buffer, 添加到 bufferLayouts。
    llvm::outs() << "collect  bufferLayouts" << "\n"; llvm::outs().flush();
    llvm::DenseMap<mlir::Value, Layout*> bufferLayouts;
    kernelOp->walk([&]( frisk::AllocBufferOp allocOp){
      auto layout = new Layout();
      bufferLayouts.insert(std::make_pair(allocOp.getResult(), layout));
    });
    // -------- step3 遍历 GEMMOP，更新map中 ab acc的 Layout
    auto inferRegCount = [&](mlir::Value buffer) -> int {
      auto memRefTy = mlir::cast<MemRefType>(buffer.getType());
      int64_t elementCount = 1;
      for (int64_t dim : memRefTy.getShape()){
        elementCount *= dim;
      }
      return static_cast<int>((elementCount + thread_num - 1) / thread_num);
    };
    auto setOperatorLayout = [&]( mlir::Operation* op ,mlir::Value buffer, int tileM, int tileN, int tileK) {
      auto it = bufferLayouts.find(buffer);
      if(it != bufferLayouts.end()){
        auto existLayout = it->getSecond();
        if(existLayout->kind == LayoutKind::undef){
          existLayout->kind = LayoutKind::gemm_operator;
          // existLayout->reg_count = inferRegCount(buffer);
          existLayout->inst_mnk = SelectTCInstMNK(tileM, tileN, tileK, VendorKind::DCU);
        }
        else{
          // 已有其他Layout。需消解冲突
          Layout incoming;
          incoming.kind = LayoutKind::gemm_operator;
          // incoming.reg_count = inferRegCount(buffer);
          incoming.inst_mnk = SelectTCInstMNK(tileM, tileN, tileK, VendorKind::DCU);
          LR.ResolveConflictOrAddLayoutConvert(op, buffer ,existLayout, incoming);
        }
      }
    };
    auto setAccLayout = [&](mlir::Operation* op, mlir::Value buffer, int tileM, int tileN, int tileK) {
      auto it = bufferLayouts.find(buffer);
      if(it != bufferLayouts.end()){
        auto existLayout = it->getSecond();
        if(existLayout->kind == LayoutKind::undef){
          existLayout->kind = LayoutKind::gemm_acc;
          // existLayout->reg_count = inferRegCount(buffer);
          existLayout->inst_mnk = SelectTCInstMNK(tileM, tileN, tileK, VendorKind::DCU);
        }
        else{
          // 已有其他Layout。需消解冲突
          Layout incoming;
          incoming.kind = LayoutKind::gemm_acc;
          // incoming.reg_count = inferRegCount(buffer);
          incoming.inst_mnk = SelectTCInstMNK(tileM, tileN, tileK, VendorKind::DCU);
          LR.ResolveConflictOrAddLayoutConvert(op, buffer,existLayout, incoming);
        }
      }
    };

    kernelOp->walk([&]( frisk::GemmOp gemmOp){
      auto memTyA = mlir::cast<MemRefType>(gemmOp.getMatrixA().getType());
      auto memTyB = mlir::cast<MemRefType>(gemmOp.getMatrixB().getType());
      int m,n,k = 0;
      if(!gemmOp.getTransA()){
        m = memTyA.getShape()[0];
        k = memTyA.getShape()[1];
      }
      else{
        m = memTyA.getShape()[1];
        k = memTyA.getShape()[0];
      }
      if(!gemmOp.getTransB()){
        n = memTyA.getShape()[1];
      }
      else{
        n = memTyA.getShape()[0];
      }
      setOperatorLayout(gemmOp, gemmOp.getMatrixA() , m,n,k);
      setOperatorLayout(gemmOp, gemmOp.getMatrixB() , m,n,k);
      setAccLayout(gemmOp, gemmOp.getMatrixC() , m,n,k);
    });
    // -------- step4 ： 插入 convertLayoutOp
    llvm::outs() << "LR.convertsToAdd size = " << LR.convertsToAdd.size() << "\n"; llvm::outs().flush();
    for(auto ip : LR.convertsToAdd){
      mlir::OpBuilder builder{ip.before};
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::Attribute value);
      auto from = std::to_string(int(ip.from.kind)); 
      auto to = std::to_string(int(ip.to.kind)); 
      std::string attr = from + ","+to;
      builder.create<frisk::ConvertLayoutOp>(ip.before->getLoc(), ip.buffer, builder.getStringAttr(attr.data()));
    }

    // -------- step4 Layout 传播：向User和向 Definer 两个方向. 但需要以 LayoutConversionInsertPoint 为分界。分界前后的 buffer Layout 不同
    // 步骤：
    // bool hasChanged = true;
    // while(hasChanged){
    //   kernelOp->walk([&](mlir::Operation* op){
    //     for(auto operand : op->getOperands()){
    //       auto it = bufferLayouts.find(operand);
    //       if(it->second->kind == LayoutKind::undef){
    //         // 需要修改
    //       }
    //     }
    //   });
    // }
    return;
  }
};

} // namespace

std::unique_ptr<Pass> createFriskLayoutInferPass() {
  return std::make_unique<FriskLayoutInferPass>();
}


} // namespace mlir::frisk
