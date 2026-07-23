#ifndef _HARDWARESPECIFICATION_H_
#define _HARDWARESPECIFICATION_H_
#include <string>

#define  MS_REG 0x01
#define  MS_SHM 0x02

// 块级别 wmma 指令描述
struct TiledGEMMInfo {
    int needThreadCount;  // tile级别gemm由几个线程共同协作完成
    int m;  // 指令的mnk 
    int n;
    int k;
    int memspace_a;  // fragment ab 以及 accumulator 的内存位置（寄存器or SHM）
    int memspace_b;
    int memspace_acc;
};

// 数据传输特性的描述
struct DataCopyInfo {
    bool supportAsyncCopy;  // 是否支持 异步拷贝
    bool bankcount;  // shm中物理内存按照 bank=? 组织。（=32）
    int  singleBankBytes;  // 单个bank的字节数 (一般=4)
};

// 同步原语的描述. 支持什么级别的sync （block内的sync，或是 warp级别的sync）
struct SyncGranularityInfo {
    bool innerBlockLevelSync;  // 是否支持 block内 thread间的同步 __sync_threads()
    bool innerWarpLevelSync;  // 是否支持 warp内的同步 __syncwarp()
    bool innerWarpgroupSync;  // __sync_warpgroup()
};

// 描述硬件特性
struct HWSpecification {
    TiledGEMMInfo gemmInfo;  // 是否支持tile级别的gemm
    DataCopyInfo dataCopyInfo;  // 是否支持 GM<->shm 之间的异步拷贝
    int  warpSize;      // warp 大小（调度单元有几个线程）
    SyncGranularityInfo syncGranularity;   // 同步语句的控制粒度
};

static HWSpecification* GetHWSpecification(std::string hwKind, std::string version);

#endif