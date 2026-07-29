#ifndef _HARDWARESPECIFICATION_H_
#define _HARDWARESPECIFICATION_H_
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <string>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"


/**
 * @brief 描述块级别 mma指令
 * 
 */
struct MMAInstInfo {
    const char* name;  // inst的name
    int m;
    int n;
    int k;
    int coopThreadsCount;
    mlir::frisk::friskMs memspaceA; 
    mlir::frisk::friskMs memspaceB; 
    mlir::frisk::friskMs memspaceAcc; 
    mlir::frisk::FriskDType fragmentElementType;
    mlir::frisk::FriskDType accElementType;
    std::array<int64_t, 2> warp_layout_a;  // warp中的线程如何排布
    std::array<int64_t, 2> warp_layout_b;  // warp中的线程如何排布
    std::array<int64_t, 2> warp_layout_acc;  // warp中的线程如何排布
    std::array<int64_t, 2> wg_layout_acc;    // wg中的warp如何排列
    int dataCountPerThread_A;  // 描述单个mma指令中，一个线程持有多少data（按元素个数计）
    int dataCountPerThread_B;  // 描述单个mma指令中，一个线程持有多少data（按元素个数计）
    int dataCountPerThread_Acc;  // 描述单个mma指令中，一个线程持有多少data（按元素个数计）
    std::array<mlir::AffineExpr, 2> wlr_Aij;  // f(warp,lane,reg) -> buffer[i,j]
    std::array<mlir::AffineExpr, 2> wlr_Bij;  // f(warp,lane,reg) -> buffer[i,j]
    std::array<mlir::AffineExpr, 2> wlr_Cij;  // f(warp,lane,reg) -> buffer[i,j]
};

// 块级别 wmma 指令描述
struct TiledGEMMInfo {
    bool isAsync;      // 矩阵乘加指令是否异步
    std::vector<MMAInstInfo> validInsts;
};
// 
/*
总结其过程 ：C[m,n] += A[m,k] @ B[k,n]
由 th个线程协同。均摊数据： A[m*k/th] B[k*n/th] 结果C也均摊到 thread中： C[m*n/th]
AB取值需要符合一定规则。该规则为一个函数 fa(tid,va) -> A[i0,j0]  fb(tid,vb) -> B[i1,j1],  va∈[0, m*k/th ) vb∈[0, n*k/th )
va vb 表示输入的第几个数。  简单而言，tid确定了AB中该取哪几个位置的数。即 A B 各自的逻辑索引
同理，已知tid，也能知道该线程持有的全部 (m*n/th) 个 C元素的逻辑索引
 */ 

// 数据传输特性的描述
struct DataCopyInfo {
    bool supportAsyncCopy;  // 是否支持 异步拷贝
    bool bankcount;  // shm中物理内存按照 bank=? 组织。（=32）
    int  singleBankBytes;  // 单个bank的字节数 (一般=4)
    int  asyncCopyAlignBytesNonSwizzle;  // 异步拷贝时, 开启swizzle时的 起始地址对齐字节数要求
    int  asyncCopyAlignBytesWithSwizzle;  // 异步拷贝时 不开启swizzle时的 起始地址对齐字节数要求
};

// 同步原语的描述. 支持什么级别的sync （block内的sync，或是 warp级别的sync）
struct SyncGranularityInfo {
    bool innerBlockLevelSync;  // 是否支持 block内 thread间的同步 __sync_threads()
    bool innerWarpLevelSync;  // 是否支持 warp内的同步 __syncwarp()
    bool innerWarpgroupSync;  // __sync_warpgroup()
};

#define HW_KIND_DCU    "dcu"
#define HW_KIND_NVIDIA "nvidia"

#define HW_VERSION_NV_H100  "h100"
#define HW_VERSION_NV_A100  "a100"
#define HW_VERSION_DCU_BW1000  "bw1000"

// 描述硬件特性
struct HWSpecification {
    std::string hwKind;
    std::string hwVersion;
    TiledGEMMInfo gemmInfo;  // 是否支持tile级别的gemm
    DataCopyInfo dataCopyInfo;  // 是否支持 GM<->shm 之间的异步拷贝
    int  warpSize;      // warp 大小（调度单元有几个线程）
    SyncGranularityInfo syncGranularity;   // 同步语句的控制粒度
};

HWSpecification* GetHWSpecification(std::string hwKind, std::string version, mlir::MLIRContext* ctx);

#endif
