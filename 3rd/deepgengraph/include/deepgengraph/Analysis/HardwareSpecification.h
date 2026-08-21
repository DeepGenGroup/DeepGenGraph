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

namespace mlir::frisk {

inline coordXY_t operator*(const coordXY_t& lhs, const coordXY_t& rhs) {
    return {
        lhs[0] * rhs[0],
        lhs[1] * rhs[1]
    };
}
inline coordXY_t operator/(const coordXY_t& lhs, const coordXY_t& rhs) {
    return {
        lhs[0] / rhs[0],
        lhs[1] / rhs[1]
    };
}
inline coordXY_t operator+(const coordXY_t& lhs, const coordXY_t& rhs) {
    return {
        lhs[0] + rhs[0],
        lhs[1] + rhs[1]
    };
}
inline coordXY_t operator-(const coordXY_t& lhs, const coordXY_t& rhs) {
    return {
        lhs[0] - rhs[0],
        lhs[1] - rhs[1]
    };
}

inline int64_t flat_size(const coordXY_t& x){
    return x[0] * x[1];
}

// 根据展平的idx, order, layout，还原出xy分量的计算式
static std::array<mlir::AffineExpr, 2> UnflattenIndexToXY(mlir::AffineExpr flattenIdx, 
    const coordXY_t& order, const coordXY_t& layout)
{
    // 寻找连续维
    auto consistentDim = order[0] == 0 ? 0 : 1;
    auto consistentLen = layout[consistentDim];
    std::array<mlir::AffineExpr, 2> ret = {0,0};
    ret[order[0]] = flattenIdx % (consistentLen);  // 列优先时，warp_layout_order[0] = 0， 行优先为1
    ret[order[1]] = flattenIdx.floorDiv(consistentLen) ;
    return ret;
}


std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs);

std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    const std::vector<int> &upperBounds,
    std::vector<mlir::Value> &outIvs,
    const std::vector<const char*> &labels);

std::vector<mlir::affine::AffineForOp> createNestedAffineFor(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    mlir::DenseMap<const char*, std::pair<int, mlir::Value>> &loopInfoMap  // in out : 标签-{上界，迭代变量}
  ) ;


}
using mlir::frisk::operator*;




// 一般的线性排布描述
/*
 * DCU WMMA Matrix Layout Mapping (16 x 16 Matrices for CUDA / MMA Tensor Core Operations)
 *
 * 1. MATRIX B (16 x 16):
 *    Cols:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
 *         +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
 * Row 0:  |T0  |T1  |T2  |T3  |T4  |T5  |T6  |T7  |T8  |T9  |T10 |T11 |T12 |T13 |T14 |T15 | (V0)
 * Row 1:  |T0  |T1  |T2  |T3  |T4  |T5  |T6  |T7  |T8  |T9  |T10 |T11 |T12 |T13 |T14 |T15 | (V1)
 * Row 2:  |T0  |T1  |T2  |T3  |T4  |T5  |T6  |T7  |T8  |T9  |T10 |T11 |T12 |T13 |T14 |T15 | (V2)
 * Row 3:  |T0  |T1  |T2  |T3  |T4  |T5  |T6  |T7  |T8  |T9  |T10 |T11 |T12 |T13 |T14 |T15 | (V3)
 * Row 4:  |T16 |T17 |T18 |T19 |T20 |T21 |T22 |T23 |T24 |T25 |T26 |T27 |T28 |T29 |T30 |T31 | (V0)
 * Row 5:  |T16 |T17 |T18 |T19 |T20 |T21 |T22 |T23 |T24 |T25 |T26 |T27 |T28 |T29 |T30 |T31 | (V1)
 * Row 6:  |T16 |T17 |T18 |T19 |T20 |T21 |T22 |T23 |T24 |T25 |T26 |T27 |T28 |T29 |T30 |T31 | (V2)
 * Row 7:  |T16 |T17 |T18 |T19 |T20 |T21 |T22 |T23 |T24 |T25 |T26 |T27 |T28 |T29 |T30 |T31 | (V3)
 * Row 8:  |T32 |T33 |T34 |T35 |T36 |T37 |T38 |T39 |T40 |T41 |T42 |T43 |T44 |T45 |T46 |T47 | (V0)
 * Row 9:  |T32 |T33 |T34 |T35 |T36 |T37 |T38 |T39 |T40 |T41 |T42 |T43 |T44 |T45 |T46 |T47 | (V1)
 * Row10:  |T32 |T33 |T34 |T35 |T36 |T37 |T38 |T39 |T40 |T41 |T42 |T43 |T44 |T45 |T46 |T47 | (V2)
 * Row11:  |T32 |T33 |T34 |T35 |T36 |T37 |T38 |T39 |T40 |T41 |T42 |T43 |T44 |T45 |T46 |T47 | (V3)
 * Row12:  |T48 |T49 |T50 |T51 |T52 |T53 |T54 |T55 |T56 |T57 |T58 |T59 |T60 |T61 |T62 |T63 | (V0)
 * Row13:  |T48 |T49 |T50 |T51 |T52 |T53 |T54 |T55 |T56 |T57 |T58 |T59 |T60 |T61 |T62 |T63 | (V1)
 * Row14:  |T48 |T49 |T50 |T51 |T52 |T53 |T54 |T55 |T56 |T57 |T58 |T59 |T60 |T61 |T62 |T63 | (V2)
 * Row15:  |T48 |T49 |T50 |T51 |T52 |T53 |T54 |T55 |T56 |T57 |T58 |T59 |T60 |T61 |T62 |T63 | (V3)
 *         +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
 *
 * 2. MATRIX A (16 x 16):
 *    Cols:  0..3 (V0..V3)  |  4..7 (V0..V3)  | 8..11 (V0..V3)  | 12..15 (V0..V3)
 *          +---------------+---------------+---------------+-----------------+
 * Row 0:   | T0  V0..V3    | T16 V0..V3    | T32 V0..V3    | T48 V0..V3      |
 * Row 1:   | T1  V0..V3    | T17 V0..V3    | T33 V0..V3    | T49 V0..V3      |
 * Row 2:   | T2  V0..V3    | T18 V0..V3    | T34 V0..V3    | T50 V0..V3      |
 * Row 3:   | T3  V0..V3    | T19 V0..V3    | T35 V0..V3    | T51 V0..V3      |
 * Row 4:   | T4  V0..V3    | T20 V0..V3    | T36 V0..V3    | T52 V0..V3      |
 * Row 5:   | T5  V0..V3    | T21 V0..V3    | T37 V0..V3    | T53 V0..V3      |
 * Row 6:   | T6  V0..V3    | T22 V0..V3    | T38 V0..V3    | T54 V0..V3      |
 * Row 7:   | T7  V0..V3    | T23 V0..V3    | T39 V0..V3    | T55 V0..V3      |
 * Row 8:   | T8  V0..V3    | T24 V0..V3    | T40 V0..V3    | T56 V0..V3      |
 * Row 9:   | T9  V0..V3    | T41 V0..V3    | T41 V0..V3    | T57 V0..V3      |
 * Row10:   | T10 V0..V3    | T26 V0..V3    | T42 V0..V3    | T58 V0..V3      |
 * Row11:   | T11 V0..V3    | T27 V0..V3    | T43 V0..V3    | T59 V0..V3      |
 * Row12:   | T12 V0..V3    | T28 V0..V3    | T44 V0..V3    | T60 V0..V3      |
 * Row13:   | T13 V0..V3    | T29 V0..V3    | T45 V0..V3    | T61 V0..V3      |
 * Row14:   | T14 V0..V3    | T30 V0..V3    | T46 V0..V3    | T62 V0..V3      |
 * Row15:   | T15 V0..V3    | T31 V0..V3    | T47 V0..V3    | T63 V0..V3      |
 *          +---------------+---------------+---------------+-----------------+
 *
 * 3. MATRIX ACC (16 x 16):
 *    Sub-block repeating pattern across 4x4 quad-groups (Columns 0..3, 4..7, 8..11, 12..15):
 *    Pattern per row group across the 16 columns:
 *    [T0..T3|T16..T19|T32..T35|T48..T51] -> [T4..T7|T20..T23...] -> [T8..T11...] -> [T12..T15...]
 *
 *    Row 0  (V0): | T0  T16 T32 T48 | T0  T16 T32 T48 | T0  T16 T32 T48 | T0  T16 T32 T48 |
 *    Row 1  (V0): | T1  T17 T33 T49 | T1  T17 T33 T49 | T1  T17 T33 T49 | T1  T17 T33 T49 |
 *    Row 2  (V0): | T2  T18 T34 T50 | T2  T18 T34 T50 | T2  T18 T34 T50 | T2  T18 T34 T50 |
 *    Row 3  (V0): | T3  T19 T35 T51 | T3  T19 T35 T51 | T3  T19 T35 T51 | T3  T19 T35 T51 |
 *    Row 4  (V0): | T4  T20 T36 T52 | T4  T20 T36 T52 | T4  T20 T36 T52 | T4  T20 T36 T52 |
 *    Row 5  (V0): | T5  T21 T37 T53 | T5  T21 T37 T53 | T5  T21 T37 T53 | T5  T21 T37 T53 |
 *    Row 6  (V0): | T6  T22 T38 T54 | T6  T22 T38 T54 | T6  T22 T38 T54 | T6  T22 T38 T54 |
 *    Row 7  (V0): | T7  T23 T39 T55 | T7  T23 T39 T55 | T7  T23 T39 T55 | T7  T23 T39 T55 |
 *    Row 8  (V0): | T8  T24 T40 T56 | T8  T24 T40 T56 | T8  T24 T40 T56 | T8  T24 T40 T56 |
 *    Row 9  (V0): | T9  T25 T41 T57 | T9  T25 T41 T57 | T9  T25 T41 T57 | T9  T25 T41 T57 |
 *    Row 10 (V0): | T10 T26 T42 T58 | T10 T26 T42 T58 | T10 T26 T42 T58 | T10 T26 T42 T58 |
 *    Row 11 (V0): | T11 T27 T43 T59 | T11 T27 T43 T59 | T11 T27 T43 T59 | T11 T27 T43 T59 |
 *    Row 12 (V0): | T12 T28 T44 T60 | T12 T28 T44 T60 | T12 T28 T44 T60 | T12 T28 T44 T60 |
 *    Row 13 (V0): | T13 T29 T45 T61 | T13 T29 T45 T61 | T13 T29 T45 T61 | T13 T29 T45 T61 |
 *    Row 14 (V0): | T14 T30 T46 T62 | T14 T30 T46 T62 | T14 T30 T46 T62 | T14 T30 T46 T62 |
 *    Row 15 (V0): | T15 T31 T47 T63 | T15 T31 T47 T63 | T15 T31 T47 T63 | T15 T31 T47 T63 |
 *                 +---------------+---------------+---------------+-----------------+
 *                 | Vector V0     | Vector V1     | Vector V2     | Vector V3       |
 以上述的排布举例, 说明 LinearLayout2DDesc 中各个字段含义
 C:
    warp_layout = [16,4]
    warp_layout_order = [0,1]   // warp中的tid按列优先排布

    wg_layout, wg_layout_order 同理

    thread_creg = [1,1]  // 一个线程持有[1,1] 个连续元素
    thread_creg_order = [1，0]  // 线程的连续元素按行优先排布( 存在dim=1，行列优先没区别 )
    warp_repeat = [1，4]  // 按照 {thread_creg+order, warp_layout+order} 给定的微观模式，以warp为单位重复 [1,4] 得到指令级别 layout
    warp_repeat_order = [1, 0]  // warp_repeat 为行优先顺序(即先迭代0序号的dim 再迭代1序号的dim)

    block_repeat = [br0 , br1]  // 对于此buffer，为了完成GEMM运算，需要以 wmmaInst 为单位在MN 上各做 [br0, br1] 次循环.
 */



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

    inline coordXY_t get_warp_widths() const {  // warp单次计算的连续区域（还没有 repeat）
        return warp_layout* thread_creg;
    }
    inline coordXY_t get_warp_widths_total() const {  // warp repeat后计算的区域大小
        return get_warp_widths() * warp_repeat;
    }
    inline coordXY_t get_wg_widths() const {  // warpgroup计算的连续区域大小
        return get_warp_widths_total() * wg_layout;
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
    std::string name;  // inst的name
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
