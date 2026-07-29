#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <vector>

HWSpecification* GetHWSpecification(std::string hwKind, std::string version, mlir::MLIRContext* ctx){
    static HWSpecification s;
    using mlir::frisk::friskMs;
    s.hwKind = hwKind;
    s.hwVersion = version;
    if(hwKind == HW_KIND_NVIDIA){
        s.warpSize = 32;
        s.dataCopyInfo.bankcount = 32;
        s.dataCopyInfo.singleBankBytes = 4;
        if(version == HW_VERSION_NV_H100){
            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = true;
            s.syncGranularity.innerWarpLevelSync = true;
        }
        else if(version == HW_VERSION_NV_A100){
            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = false;
            s.syncGranularity.innerWarpLevelSync = true;
        }
    }
    else if(hwKind == HW_KIND_DCU){
        s.warpSize = 64;
        s.dataCopyInfo.bankcount = 32;
        s.dataCopyInfo.singleBankBytes = 4;
        if(version == HW_VERSION_DCU_BW1000){
            s.dataCopyInfo.supportAsyncCopy = false;
            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = false;
            s.syncGranularity.innerWarpLevelSync = false;
            
            auto warpId = mlir::getAffineDimExpr(0, ctx);
            auto laneId = mlir::getAffineDimExpr(1, ctx);
            auto regId = mlir::getAffineDimExpr(2, ctx);
            
            MMAInstInfo info;
            info.memspaceA = friskMs::Local;
            info.memspaceB = friskMs::Local;
            info.memspaceAcc = friskMs::Local;
            info.warp_layout_a = {16,4};
            info.warp_layout_b = {4,16};
            info.warp_layout_acc = {16,4};
            
            // v_mmac_16x16x8_f32
            info.m = 16;info.n = 16;info.k = 8;
            info.accElementType = mlir::frisk::FriskDType::f32;
            info.fragmentElementType = mlir::frisk::FriskDType::f32;
            info.wlr_Aij = {laneId % 16, laneId.floorDiv(16) * 2 + regId} ;
            info.wlr_Bij = {laneId.floorDiv(16) * 2 + regId, laneId % 16 } ;
            info.wlr_Cij = {laneId % 16, laneId.floorDiv(16) + regId * 4 } ;
            s.gemmInfo.validInsts.push_back(info);
            // v_mmac_f32_16x16x16_f16
            info.m = 16;info.n = 16;info.k = 16;
            info.accElementType = mlir::frisk::FriskDType::f32;
            info.fragmentElementType = mlir::frisk::FriskDType::f16;
            info.wlr_Aij = {laneId % 16, laneId.floorDiv(16) * 4 + regId} ;
            info.wlr_Bij = {laneId.floorDiv(16) * 4 + regId, laneId % 16 } ;
            info.wlr_Cij = {laneId % 16, laneId.floorDiv(16) + regId * 4 } ;
            s.gemmInfo.validInsts.push_back(info);
            // v_mmac_i32_16x16x32_i8
            // v_mmac_16x16x4_f32
            // v_mmac_16x16x4_f32
            // v_mfma_f32_16x16x4f32
        }
    }
    return &s;
}
