#include "deepgengraph/Analysis/HardwareSpecification.h"

HWSpecification* GetHWSpecification(std::string hwKind, std::string version){
    static HWSpecification s;
    if(hwKind == "nvidia"){
        s.warpSize = 32;
        s.dataCopyInfo.bankcount = 32;
        s.dataCopyInfo.singleBankBytes = 4;
        if(version == "h100"){
            s.gemmInfo.needThreadCount = 128;
            s.gemmInfo.memspace_a = MS_REG | MS_SHM;
            s.gemmInfo.memspace_b = MS_REG;
            s.gemmInfo.memspace_acc = MS_REG;
            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = true;
            s.syncGranularity.innerWarpLevelSync = true;
        }
        else if(version == "a100"){
            s.gemmInfo.needThreadCount = s.warpSize;
            s.gemmInfo.memspace_a = MS_REG;
            s.gemmInfo.memspace_b = MS_REG;
            s.gemmInfo.memspace_acc = MS_REG;
            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = false;
            s.syncGranularity.innerWarpLevelSync = true;
        }
    }
    else if(hwKind == "dcu"){
        s.warpSize = 64;
        s.dataCopyInfo.bankcount = 32;
        s.dataCopyInfo.singleBankBytes = 4;
        if(version == "bw1000"){
            s.gemmInfo.needThreadCount = s.warpSize;
            s.gemmInfo.memspace_a = MS_REG;
            s.gemmInfo.memspace_b = MS_REG;
            s.gemmInfo.memspace_acc = MS_REG;

            s.dataCopyInfo.supportAsyncCopy = false;

            s.syncGranularity.innerBlockLevelSync = true;
            s.syncGranularity.innerWarpgroupSync = false;
            s.syncGranularity.innerWarpLevelSync = false;
        }
    }
    return &s;
}
