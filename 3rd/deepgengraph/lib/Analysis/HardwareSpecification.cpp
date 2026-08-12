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

coordXY_t PointwiseDot(coordXY_t a, coordXY_t b){
    return {a[0] * b[0], a[1] * b[1]};
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
                llvm::raw_string_ostream ss(info.name);
                info.m = m;
                info.n = n;
                info.k = k;
                ss << "m" << m << "n" << n << "k" << k << ":"<< int(abTy)<<":" << int(cTy);
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
