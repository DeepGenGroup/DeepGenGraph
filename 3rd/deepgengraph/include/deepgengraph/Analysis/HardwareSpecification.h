#ifndef _HARDWARESPECIFICATION_H_
#define _HARDWARESPECIFICATION_H_
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

using coordXY_t = std::array<int64_t, 2>;  // coordinate [x,y]

coordXY_t PointwiseDot(coordXY_t a, coordXY_t b);

// 一般的线性排布描述
struct LinearLayout2DDesc {
    mlir::frisk::friskMs memspace;  // 位于shm还是reg
    mlir::frisk::FriskDType  elementType;  // 数据类型
    coordXY_t warp_layout;   // warp中的线程排布形状
    coordXY_t warp_layout_order;  // [0,1]  表示自增顺序先x坐标再y坐标。即列优先
    coordXY_t thread_creg;  // thread持有的连续reg排布
    coordXY_t thread_creg_order;  // reg 排布为行/列优先
    coordXY_t warp_repeat;  // warp内所有thread持有的连续数据构成D. D如何重复
    coordXY_t warp_repeat_order;  // D的重复按行/列优先
    coordXY_t wg_layout;   // warpgroup 里 warp的排列形状
    coordXY_t wg_layout_order;  // [0,1]  表示自增顺序先x坐标再y坐标。即列优先

    inline coordXY_t get_warp_widths() {  // warp单次计算的连续区域（还没有replicate）
        return PointwiseDot(warp_layout, thread_creg);
    }
    inline coordXY_t get_warp_widths_total() {  // warp repeat后计算的区域大小
        return PointwiseDot(get_warp_widths(), warp_repeat);
    }
    inline coordXY_t get_wg_widths() {  // warpgroup计算的连续区域大小
        return PointwiseDot(get_warp_widths_total(), wg_layout);
    }
};

struct SwizzleLayoutDesc {
    int B,M,S;
};

/**
 * @brief 描述块级别 mma指令
 * 
 */
struct MMAInstInfo {
    const char* name;  // inst的name
    int m;
    int n;
    int k;
    LinearLayout2DDesc desc_a;  // operator
    LinearLayout2DDesc desc_b;  // operator
    LinearLayout2DDesc desc_c;  // accumulator
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
class HWSpecification {
public:
    TiledGEMMInfo gemmInfo;  // 是否支持tile级别的gemm
    DataCopyInfo dataCopyInfo;  // 是否支持 GM<->shm 之间的异步拷贝
    SyncGranularityInfo syncGranularity;   // 同步语句的控制粒度

    HWSpecification(std::string kind, std::string version) : hwKind(kind), hwVersion(version) {
        if(kind == HW_KIND_NVIDIA){
            warpSize = 32;
        }
        else if(kind == HW_KIND_DCU){
            warpSize = 64;
        }
    }
    inline const std::string getKind() {return hwKind;}
    inline const std::string getVersion() {return hwVersion;}
    inline int getWarpsize() {return warpSize;}
private:
    std::string hwKind;
    std::string hwVersion;
    int  warpSize;      // warp 大小（调度单元有几个线程）
};

HWSpecification* GetHWSpecification(std::string hwKind, std::string version, mlir::MLIRContext* ctx);

#endif
