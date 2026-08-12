#ifndef FRISK_ANALYSIS_INFERLOWERINFO_H
#define FRISK_ANALYSIS_INFERLOWERINFO_H

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Support/LLVM.h"

struct HWSpecification;

namespace mlir::frisk {

#define TID  "threadIdx"

static const char* WARP_LABELS[] = {"iv_warpX", "iv_warpY"};
static const char* THREAD_LABELS[] = {"iv_threadX", "iv_threadY"};
static const char* BLOCK_LABELS[] = {"iv_blockX", "iv_blockY"};

class LowerInfoAnalysis ;
/**
 * @brief LowerInfo
 * 其表示了 在block层面的一块buffer，在降低到线程层面后，线程如何从 block-level的buffer里 根据自己的tid 去RW 该buffer里的数据（即索引[x,y]）
[x,y] 可通过不同级别的 loop_iv （block_repeat, warp_repeat, thread_width） 配合 wid laneid 算出来
 */
class LowerInfo {
  friend LowerInfoAnalysis;
public:
  Value buffer;
  int warp_threads;
public:
  explicit LowerInfo(int _warp_threads);
  MMAInstInfo*  mmaInst = nullptr;
  LinearLayout2DDesc base_layout;  // 某个指令决定的，基础访问模式。
  coordXY_t        base_layout_repeat;  // 以基础访问模式的subBuffer为粒度，(i,j)在YX方向上repeat，最终平铺整个buffer。本质:(i,j,warp,lane,reg) -> buffer[x,y]
  coordXY_t        base_layout_repeat_order;  // 以基础访问模式的subBuffer为粒度，(i,j)在YX方向上repeat，最终平铺整个buffer。本质:(i,j,warp,lane,reg) -> buffer[x,y]

  int get_dimcount() const {
    return dimCount;
  }
  const std::array<int64_t, 2>& get_warp_layout() const {
    return warp_layout;
  }
  const std::array<int64_t, 2>& get_block_layout() const {
    return block_layout;
  }
  const std::array<int64_t, 2>& get_warp_repeat() const {
    return warp_repeat;
  }
  const std::array<int64_t, 2>& get_block_repeat() const {
    return block_repeat;
  }
  const std::array<int64_t, 2>& get_thread_widths() const {
    return thread_widths;
  } 
  std::array<int64_t, 2> get_thread_total_widths() const {
    std::array<int64_t, 2> ret;
    for(int i=0;i<2;++i){
      ret[i] = (thread_widths[i] * warp_repeat[i] * block_repeat[i]);
    }
    return ret;
  }

  const std::array<int64_t, 2>& get_warp_widths() const {
    return warp_widths;
  }
  const std::array<int64_t, 2>& get_block_widths() const {
    return block_widths;
  }
  const auto& getOperandLabels() const {
    return mapOperandsLabel;
  }
  const auto& getIterVarLabels() const {
    return iterVarLabels;
  }
  const auto& getItervarUbs() const {
    return ivUpperBounds;
  }

  void show(const char* label = nullptr) {
    auto printI64Vec = [&](const char *name, const std::array<int64_t, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << vec[i];
        if (i + 1 < vec.size()) llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
    };
    auto printExprVec = [&](const char *name, const std::array<AffineExpr, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        vec[i].print(llvm::outs());
        auto& raw = vec[i];
        // llvm::outs() << " simplified:[ " << mlir::simplifyAffineExpr(raw, dimCount, 0) << "] ";
        if (i + 1 < vec.size()) llvm::outs() << ", ";

      }
      llvm::outs() << "]\n";
    };
    const char* _label = " ";
    if(label != nullptr){
      _label = label;
    }
    llvm::outs() << "=== LowerInfo "<< _label <<" ===\n";
    llvm::outs() << "buffer: ";
    if (buffer) {
      buffer.print(llvm::outs());
    } else {
      llvm::outs() << "<null>";
    }
    llvm::outs() << "\n";
    llvm::outs() << "buffer_memory: ";
    if (!buffer || !isa<MemRefType>(buffer.getType())) {
      llvm::outs() << "<non-memref>\n";
    } else {
      int64_t memorySpace = cast<MemRefType>(buffer.getType()).getMemorySpaceAsInt();
      if (memorySpace == 3) {
        llvm::outs() << "shared(memory_space=3)\n";
      } else if (memorySpace == 0 || memorySpace == 5) {
        llvm::outs() << "register/local(memory_space=" << memorySpace << ")\n";
      } else {
        llvm::outs() << "unknown(memory_space=" << memorySpace << ")\n";
      }
      auto affineMapIndices = getAffineMap();
      printExprVec("getAffineMap()", affineMapIndices);
    }
    llvm::outs() << "thread_bound: " << thread_bound << "\n";

    printExprVec("warp_indices", warp_indices);
    printExprVec("thread_indices", lane_indices);
    printI64Vec("warp_layout", warp_layout);
    printI64Vec("block_layout", block_layout);
    printI64Vec("warp_repeat", warp_repeat);
    printI64Vec("block_repeat", block_repeat);
    printI64Vec("thread_widths", thread_widths);
    printI64Vec("warp_widths", warp_widths);
    printI64Vec("block_widths", block_widths);
    llvm::outs() << "=================\n";
  }

  std::array<AffineExpr, 2> getAffineMap() {
    // 根据上述信息，生成不同层面的索引
    // 强制重新计算
    mapOperandsLabel.clear();
    iterVarLabels.clear();
    ivUpperBounds.clear();
    dimCount = 1;
    mapOperandsLabel.push_back(TID);
    OpBuilder b{buffer.getContext()};
    MemRefType type = dyn_cast<MemRefType>(buffer.getType());

    if (type.getMemorySpaceAsInt() == 0 || type.getMemorySpaceAsInt() == 5) { // local
      for (size_t i = 0; i < thread_widths.size(); ++i) {
        auto ib = b.getAffineDimExpr(
            i * 3 + 1); // block_repeat: [bm_ / (block_layout[0] * warp_layout[0] * thread_widths[0]), ...]
        auto iw = b.getAffineDimExpr(i * 3 + 2); // warp_repeat：[2, mma_k/(warp_layout[1] * thread_widths[1])]
        auto it = b.getAffineDimExpr(i * 3 + 3); // thread_widths: [1, 2]
        AffineExpr expr = ib * (warp_repeat[i] * thread_widths[i]) + iw * thread_widths[i] + it;
        indices[i]= expr;  // buffer-> thread 级别元素的映射
        // add labels
        mapOperandsLabel.push_back(BLOCK_LABELS[i]);
        mapOperandsLabel.push_back(WARP_LABELS[i]);
        mapOperandsLabel.push_back(THREAD_LABELS[i]);
        iterVarLabels.push_back(BLOCK_LABELS[i]);
        iterVarLabels.push_back(WARP_LABELS[i]);
        iterVarLabels.push_back(THREAD_LABELS[i]);
        
        dimCount+=3;
        // add upperBounds
        ivUpperBounds.push_back(block_repeat[i]);
        ivUpperBounds.push_back(warp_repeat[i]);
        ivUpperBounds.push_back(thread_widths[i]);
      }
    } else if (type.getMemorySpaceAsInt() == int(friskMs::Shared)) { // shared
      for (size_t i = 0; i < thread_widths.size(); ++i) { // 0:tidx, 1:iv_bx, iv_wx , iv_tx ,iv_by, iv_wy, iv_ty
        auto ib = b.getAffineDimExpr(i * 3 + 1);          // iv_bx
        auto iw = b.getAffineDimExpr(i * 3 + 2);          // iv_wx
        auto it = b.getAffineDimExpr(i * 3 + 3);          // iv_tx
        AffineExpr expr = ib * block_widths[i] + warp_indices[i] * (warp_repeat[i] * warp_widths[i]) +
                          iw * warp_widths[i] + lane_indices[i] * thread_widths[i] + it;
        indices[i] = expr;
        mapOperandsLabel.push_back(BLOCK_LABELS[i]);
        mapOperandsLabel.push_back(WARP_LABELS[i]);
        mapOperandsLabel.push_back(THREAD_LABELS[i]);
        iterVarLabels.push_back(BLOCK_LABELS[i]);
        iterVarLabels.push_back(WARP_LABELS[i]);
        iterVarLabels.push_back(THREAD_LABELS[i]);
        dimCount+=3;
        // add upperBounds
        ivUpperBounds.push_back(block_repeat[i]);
        ivUpperBounds.push_back(warp_repeat[i]);
        ivUpperBounds.push_back(thread_widths[i]);
      }
    }
    affine_map = mlir::AffineMap::get(dimCount, 0, indices, buffer.getContext());
  
    return indices;
  }

  std::array<AffineExpr, 2> getThreadIndices(
    OpBuilder b, std::array<int64_t, 2> warp_layout) {
      // tid -> lane_id
    auto tid = _theadIdx(b);
    auto ly = (tid % warp_threads).floorDiv(warp_layout[1]);
    auto lx = (tid % warp_threads) % warp_layout[1];
    return {ly, lx};
  }

  std::array<AffineExpr, 2> getWarpIndices(
    OpBuilder b, std::array<int64_t, 2> block_layout) {
      // tid -> warp_id
    auto tid = _theadIdx(b);
    auto wy = tid.floorDiv(warp_threads).floorDiv(block_layout[1]);
    auto wx = tid.floorDiv(warp_threads) % block_layout[1];
    return {wy, wx};
  }

  std::array<int64_t, 2> getWarpWidths(
      std::array<int64_t, 2> thread_widths,
      std::array<int64_t, 2> warp_layout) {
        // 一个warp计算的tile
    std::array<int64_t, 2> warp_widths;
    for (size_t i=0; i<thread_widths.size(); ++i) {
      int64_t ws = warp_layout[i] * thread_widths[i];
      warp_widths[i] = ws;
    }
    return warp_widths;
  }

  std::array<int64_t, 2> getBlockWidths(
      std::array<int64_t, 2> warp_widths,
      std::array<int64_t, 2> warp_repeat,
      std::array<int64_t, 2> block_layout) {
        // 一个block计算的tile（重复后才等于bm/bn）
    std::array<int64_t, 2> block_widths;
    for (size_t i=0; i<warp_repeat.size(); ++i) {
      int64_t wrs = warp_repeat[i] * warp_widths[i];
      int64_t bs = block_layout[i] * wrs;  // block中的warp排布 * warp_repeat计算的区域尺寸
      block_widths[i] = bs;
    }
    return block_widths;
  }

private:

  int64_t thread_bound;
  AffineMap affine_map;
  std::vector<const char*> mapOperandsLabel;  // mapOperands 的标签
  std::vector<const char*> iterVarLabels;  // for 循环的标签
  std::vector<int> ivUpperBounds;  // 迭代变量的上界
  std::array<AffineExpr, 2> indices;
  uint32_t dimCount = 0;

  std::array<AffineExpr, 2> warp_indices;  // warp_id: [(tid / 32) / block_layout[1], (tid / 32) % block_layout[1]]  warp_id[x,y]
  std::array<AffineExpr, 2> lane_indices;  // lane_id: [(tid % 32) / warp_layout[1], ...]   lane[x,y]
  std::array<int64_t, 2> warp_layout;  // lane
  std::array<int64_t, 2> block_layout;  // warp

  std::array<int64_t, 2> warp_repeat;
  std::array<int64_t, 2> block_repeat;

  std::array<int64_t, 2> thread_widths;  // 一个thread计算的元素个数 [tx,ty]
  std::array<int64_t, 2> warp_widths;
  std::array<int64_t, 2> block_widths;

protected:
  AffineExpr _theadIdx(OpBuilder& b){
    if(dimCount == 0){
      dimCount++;
      mapOperandsLabel.push_back(TID);
    }
    return b.getAffineDimExpr(0);
  }
};

class LowerInfoMap {
public:
  using LowerInfoMapTy = DenseMap<std::pair<Value, Operation*> , LowerInfo>;
  // 进行op顺序分析
  void getOpsOrder(mlir::Operation* rootNode);
  // 查询 <buffer，op> 对应的LowerInfo
  LowerInfo* getLowerInfo(const mlir::Value& buffer, mlir::Operation* op);
  // 添加 lowerinfo（info中已经含有buffer）
  LowerInfo* addLowerInfo(mlir::Operation* op, LowerInfo info);
  // 根据buffer查找infoMap，找到其中距离currOp最近的之前/之后的Op的 LowerInfo
  LowerInfo* getNearestInferedInfo(const mlir::Value& buffer, mlir::Operation* currOp, bool isBefore = true);
  auto begin() { return infoMap.begin(); }
  auto end() { return infoMap.end(); }
private:
  LowerInfoMapTy infoMap;
  DenseMap<Operation*, unsigned> opOrder;
};

class LowerInfoAnalysis {
public:
  static LowerInfoMap* run(mlir::Operation* kernelOp,
                                        const std::string& hwKind = HW_KIND_DCU,
                                        const std::string& version = HW_VERSION_DCU_BW1000);                                        
  struct GemmProblem {
    Value A;
    Value B;
    Value C;
    MemRefType aType;
    MemRefType bType;
    MemRefType cType;
    unsigned inElemBitWidth;
    int64_t bm;
    int64_t bn;
    int64_t bk;
  };
  static GemmProblem getGemmProblem(GemmOp gemmOp);
  static MMAInstInfo* selectGemmInst(GemmProblem problem, HWSpecification* hw);

private:
  static LowerInfoMap buf_info_maps;
  static llvm::SmallVector<Operation*, 5> collectNeedInferOps(mlir::Operation *kernelOp);
  static std::pair<int, int> squareFactor(int n);
  static uint64_t getRegionThreadNum(Operation *op);
  static bool checkGemmProblem(GemmProblem p, HWSpecification* hw);
  static bool getDirectGemmBlockLayout(uint64_t thread_num,
                                       std::array<int64_t, 2> &block_layout, HWSpecification* hw);
  static LowerInfo makeDirectGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                       MMAInstInfo *mma, uint64_t thread_num,
                                       HWSpecification *hw,
                                       std::array<int64_t, 2> block_layout);
  static LowerInfo makeRelyGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                     MMAInstInfo *mma, uint64_t thread_num,
                                     HWSpecification *hw, const LowerInfo &source_info,
                                     bool source_is_a);
  static void applyDirectGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                   MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw);
  static void applyRelyGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                 MMAInstInfo *mma, AffineExpr zero );
  static void applyGemmBInfo(LowerInfo &info, const GemmProblem &problem,
                             MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw);
  static bool inferDirectOp(Operation *op, LowerInfoMap& infoMap ,HWSpecification *hw);
  static bool inferRelyOp(Operation *op, LowerInfoMap& infoMap, HWSpecification *hw);
  static bool inferCopyOp(Operation *op, LowerInfoMap &buf_info_maps);
  static bool inferBlockOp(Operation *op, LowerInfoMap &buf_info_maps);
  static bool inferGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
                          HWSpecification *hw);
  static bool inferRelyGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
                              HWSpecification *hw);
  static bool inferReduceOp(Operation *op, LowerInfoMap &buf_info_maps);

  // void getTest() {
  //   llvm::outs() << "[D]need_infer_ops size: " << need_infer_ops.size() << "\n";
  //   llvm::outs() << "[D]buf_info_maps size: " << buf_info_maps.size() << "\n";
  // }
  // void showAllInfo() {
  //   llvm::outs() << "[D]show all lower info, count: " << buf_info_maps.size() << "\n";
  //   for ( auto &it : buf_info_maps) {
  //     llvm::outs() << "[D]buffer key: ";
  //     it.first.print(llvm::outs());
  //     llvm::outs() << "\n";
  //     it.second.show();
  //   }
  // }

  // LowerInfo getInfo(const Value& buffer){
  //   // return buf_info_maps.at(buffer);
  // }

};

}

#endif
