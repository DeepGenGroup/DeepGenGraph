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





} // namespace




} // namespace mlir::frisk
