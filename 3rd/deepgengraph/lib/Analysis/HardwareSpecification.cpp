#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <sstream>
#include <vector>

namespace mlir::frisk {

std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs) {
  std::vector<mlir::affine::AffineForOp> loops;
  loops.reserve(upperBounds.size());

  // 确保标签数量与循环层数一致（可选的安全检查）
  size_t numLoops = upperBounds.size();
  
  for (size_t i = 0; i < numLoops; ++i) {
    // 1. 定义下界、上界和步长 (下界默认为 0，步长默认为 1)
    int64_t lowerBound = 0;
    int64_t step = 1;
    auto ub = upperBounds[i];
    // 2. 创建当前层的 AffineForOp
    auto forOp = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBounds[i], step);
    mlir::Value iv = forOp.getInductionVar();
    // 4. 收集当前循环的迭代变量 (Induction Variable) 和 Op 本身
    outIvs.push_back(iv);
    loops.push_back(forOp);
    // 5. 将 builder 的插入点移动到当前循环体的末尾（yield 之前），以便下一层循环嵌套在内部
    builder.setInsertionPointToStart(forOp.getBody());
  }

  return loops;
}

std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs,
    const std::vector<const char*> &labels) {
  std::vector<mlir::affine::AffineForOp> loops;
  loops.reserve(upperBounds.size());
  outIvs.reserve(upperBounds.size());

  // 确保标签数量与循环层数一致（可选的安全检查）
  size_t numLoops = upperBounds.size();
  
  for (size_t i = 0; i < numLoops; ++i) {
    // 1. 定义下界、上界和步长 (下界默认为 0，步长默认为 1)
    int64_t lowerBound = 0;
    int64_t step = 1;
    auto ub = upperBounds[i];
    // 2. 创建当前层的 AffineForOp
    auto forOp = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBounds[i], step);
    // 3. 如果提供了对应的 label，则为其添加 StringAttr 属性
    if (i < labels.size() && labels[i] != nullptr) {
      forOp->setAttr("iterLabel", builder.getStringAttr(labels[i]));
    }
    mlir::Value iv = forOp.getInductionVar();
    // 4. 收集当前循环的迭代变量 (Induction Variable) 和 Op 本身
    outIvs.push_back(iv);
    loops.push_back(forOp);
    // 5. 将 builder 的插入点移动到当前循环体的末尾（yield 之前），以便下一层循环嵌套在内部
    builder.setInsertionPointToStart(forOp.getBody());
  }

  return loops;
}

std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    mlir::DenseMap<const char*, std::pair<int, mlir::Value>> &loopInfoMap  // in out : 标签-{上界，迭代变量}
  ) {
  std::vector<mlir::affine::AffineForOp> loops;
  // 确保标签数量与循环层数一致（可选的安全检查）
  size_t numLoops = loopInfoMap.size();
  for(auto& entry : loopInfoMap){
    auto [label , _pair] = entry;
    auto& [ub, iterVar] = _pair;
    auto forOp = builder.create<mlir::affine::AffineForOp>(loc, 0, ub, 1);
    forOp->setAttr("iterLabel", builder.getStringAttr(label));
    iterVar = forOp.getInductionVar();
    loops.push_back(forOp);
    builder.setInsertionPointToStart(forOp.getBody());
  }
  return loops;
}

}

HWSpecification* GetHWSpecification(std::string hwKind, std::string version, mlir::MLIRContext* ctx){
    using mlir::frisk::friskMs;
    static HWSpecification* s = nullptr;
    if(s == nullptr){
        s = new HWSpecification{hwKind,version};
    }
    else{
        return s;
    }
    if(hwKind == HW_KIND_NVIDIA){
        s->dataCopyInfo.bankcount = 32;
        s->dataCopyInfo.singleBankBytes = 4;
        if(version == HW_VERSION_NV_H100){
            s->syncGranularity.innerBlockLevelSync = true;
            s->syncGranularity.innerWarpgroupSync = true;
            s->syncGranularity.innerWarpLevelSync = true;
        }
        else if(version == HW_VERSION_NV_A100){
            s->syncGranularity.innerBlockLevelSync = true;
            s->syncGranularity.innerWarpgroupSync = false;
            s->syncGranularity.innerWarpLevelSync = true;
        }
    }
    else if(hwKind == HW_KIND_DCU){
        s->dataCopyInfo.bankcount = 32;
        s->dataCopyInfo.singleBankBytes = 4;
        if(version == HW_VERSION_DCU_BW1000){
            s->dataCopyInfo.supportAsyncCopy = false;
            s->syncGranularity.innerBlockLevelSync = true;
            s->syncGranularity.innerWarpgroupSync = false;
            s->syncGranularity.innerWarpLevelSync = false;
            
            auto warpId = mlir::getAffineDimExpr(0, ctx);
            auto laneId = mlir::getAffineDimExpr(1, ctx);
            auto regId = mlir::getAffineDimExpr(2, ctx);
            
            MMAInstInfo info;
            LinearLayout2DDesc& _a = info.desc_a;
            LinearLayout2DDesc& _b = info.desc_b;
            LinearLayout2DDesc& _c = info.desc_c;
            // BW 1000
            auto get_inst = [](int m, int n, int k, mlir::frisk::FriskDType abTy, mlir::frisk::FriskDType cTy)-> MMAInstInfo {
              MMAInstInfo info;
              // info.name = "__builtin_amdgcn_mmac_f32_16x16x4f32"
              info.asm_str = "v_mmac_";
              info.constraints = "=v,v,v,0";  // 表示最后一个operand的寄存器 必须和输出D的寄存器分配相同
              // info.constraints = "=v,v,v,v";  // V4fV4hV4hV4f
              llvm::raw_string_ostream ss(info.asm_str);
              info.m = m;
              info.n = n;
              info.k = k;
              ss << mlir::frisk::FriskDTypeToString(cTy) << "_" << m << "x" << n << "x" << k << "_"<< mlir::frisk::FriskDTypeToString(abTy);
              ss << " $0, $2, $1, $3";  // $0 输出， 123 输入(b,a,c)
              /*
              <inline asm>:1:2: error: srcD is overlap with srcC
                      v_mmac_f32_16x16x16f16 v[2:5], v[0:1], v[2:3], v[4:7]
                      ^
              error: cannot compile inline asm
              */
              LinearLayout2DDesc& _a = info.desc_a;
              LinearLayout2DDesc& _b = info.desc_b;
              LinearLayout2DDesc& _c = info.desc_c;
              _a.memspace = friskMs::Local;_b.memspace = friskMs::Local;_b.memspace = friskMs::Local;
              _a.warp_layout = {16,4}; _a.warp_layout_order = {0,1}; _a.warp_repeat = {1,1}; _a.warp_repeat_order = {0,1};
              _b.warp_layout = {4,16}; _b.warp_layout_order = {1,0}; _b.warp_repeat = {1,1}; _b.warp_repeat_order = {0,1};
              _c.warp_layout = {16,4}; _c.warp_layout_order = {0,1}; _c.warp_repeat = {1, n / 4};_c.warp_repeat_order = {0,1};
              
              _a.elementType = abTy;
              _b.elementType = abTy;
              _c.elementType = cTy;

              _a.thread_creg = {1,k / _a.warp_layout[1]};
              _a.thread_creg_order = {1,0};
              _b.thread_creg = {k / _b.warp_layout[0],1};
              _b.thread_creg_order = {0,1};
              _c.thread_creg = {1,1};
              _c.thread_creg_order = {0,1};
              return info;
            };
            // v_mmac_16x16x8_f32
            s->gemmInfo.validInsts.push_back(get_inst(16, 16, 8, mlir::frisk::FriskDType::f32, mlir::frisk::FriskDType::f32));
            // v_mmac_f32_16x16x16_f16
            s->gemmInfo.validInsts.push_back(get_inst(16, 16, 16, mlir::frisk::FriskDType::f16, mlir::frisk::FriskDType::f32));
            // v_mmac_i32_16x16x32_i8
            // v_mmac_16x16x4_f32
            s->gemmInfo.validInsts.push_back(get_inst(16, 16, 4, mlir::frisk::FriskDType::f32, mlir::frisk::FriskDType::f32));
            // v_mmac_16x16x4_f32
            // v_mfma_f32_16x16x4f32
        }
    }
    return s;
}
