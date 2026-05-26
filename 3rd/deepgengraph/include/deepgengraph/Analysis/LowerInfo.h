#ifndef FRISK_ANALYSIS_INFERLOWERINFO_H
#define FRISK_ANALYSIS_INFERLOWERINFO_H

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
#include <cstdint>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

namespace mlir::frisk {

#define TID  "threadIdx"

static const char* WARP_LABELS[] = {"iv_warpX", "iv_warpY"};
static const char* THREAD_LABELS[] = {"iv_threadX", "iv_threadY"};
static const char* BLOCK_LABELS[] = {"iv_blockX", "iv_blockY"};

class LowerInfoAnalysis ;
class LowerInfo {
  friend LowerInfoAnalysis;
public:
  Value buffer;
public:
  int get_dimcount() const {
    return dimCount;
  }
  const llvm::SmallVector<int64_t, 2>& get_warp_repeat() const {
    return warp_repeat;
  }
  const llvm::SmallVector<int64_t, 2>& get_block_repeat() const {
    return block_repeat;
  }
  const llvm::SmallVector<int64_t, 2>& get_thread_widths() const {
    return thread_widths;
  } 
  llvm::SmallVector<int64_t, 2> get_thread_total_widths() const {
    llvm::SmallVector<int64_t, 2> ret;
    for(int i=0;i<2;++i){
      ret.push_back(thread_widths[i] * warp_repeat[i] * block_repeat[i]);
    }
    return ret;
  }

  const llvm::SmallVector<int64_t, 2>& get_warp_widths() const {
    return warp_widths;
  }
  const llvm::SmallVector<int64_t, 2>& get_block_widths() const {
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

  void show() {
    auto printI64Vec = [&](const char *name, const llvm::SmallVector<int64_t, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << vec[i];
        if (i + 1 < vec.size()) llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
    };
    auto printExprVec = [&](const char *name, const llvm::SmallVector<AffineExpr, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        vec[i].print(llvm::outs());
        auto& raw = vec[i];
        llvm::outs() << " simplified:[ " << mlir::simplifyAffineExpr(raw, dimCount, 0) << "] ";
        if (i + 1 < vec.size()) llvm::outs() << ", ";

      }
      llvm::outs() << "]\n";
    };

    llvm::outs() << "=== LowerInfo ===\n";
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

  llvm::SmallVector<AffineExpr, 2> getAffineMap() {
    // 根据上述信息，生成不同层面的索引
    // 强制重新计算
    indices.clear();
    dimCount = 1;
    OpBuilder b{buffer.getContext()};
    MemRefType type = dyn_cast<MemRefType>(buffer.getType());

    if (type.getMemorySpaceAsInt() == 0 || type.getMemorySpaceAsInt() == 5) { // local
      for (size_t i = 0; i < thread_widths.size(); ++i) {
        auto ib = b.getAffineDimExpr(
            i * 3 + 1); // block_repeat: [bm_ / (block_layout[0] * warp_layout[0] * thread_widths[0]), ...]
        auto iw = b.getAffineDimExpr(i * 3 + 2); // warp_repeat：[2, mma_k/(warp_layout[1] * thread_widths[1])]
        auto it = b.getAffineDimExpr(i * 3 + 3); // thread_widths: [1, 2]
        AffineExpr expr = ib * (warp_repeat[i] * thread_widths[i]) + iw * thread_widths[i] + it;
        indices.push_back(expr);
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
    } else if (type.getMemorySpaceAsInt() == int(MemorySpace::Shared)) { // shared
      for (size_t i = 0; i < thread_widths.size(); ++i) { // 0:tidx, 1:iv_bx, iv_wx , iv_tx ,iv_by, iv_wy, iv_ty
        auto ib = b.getAffineDimExpr(i * 3 + 1);          // iv_bx
        auto iw = b.getAffineDimExpr(i * 3 + 2);          // iv_wx
        auto it = b.getAffineDimExpr(i * 3 + 3);          // iv_tx
        AffineExpr expr = ib * block_widths[i] + warp_indices[i] * (warp_repeat[i] * warp_widths[i]) +
                          iw * warp_widths[i] + lane_indices[i] * thread_widths[i] + it;
        indices.push_back(expr);
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

  llvm::SmallVector<AffineExpr, 2> getThreadIndices(
    OpBuilder b, llvm::SmallVector<int64_t, 2> warp_layout) {
      // tid -> lane_id
    auto tid = _theadIdx(b);
    auto ly = (tid % 32).floorDiv(warp_layout[1]);
    auto lx = (tid % 32) % warp_layout[1];
    return llvm::SmallVector<AffineExpr, 2>{ly, lx};
  }

  llvm::SmallVector<AffineExpr, 2> getWarpIndices(
    OpBuilder b, llvm::SmallVector<int64_t, 2> block_layout) {
      // tid -> warp_id
    auto tid = _theadIdx(b);
    auto wy = tid.floorDiv(32).floorDiv(block_layout[1]);
    auto wx = tid.floorDiv(32) % block_layout[1];
    return llvm::SmallVector<AffineExpr, 2>{wy, wx};
  }

  llvm::SmallVector<int64_t, 2> getWarpWidths(
      llvm::SmallVector<int64_t, 2> thread_widths, 
      llvm::SmallVector<int64_t, 2> warp_layout) {
        // 一个warp计算的tile
    llvm::SmallVector<int64_t, 2> warp_widths;
    for (size_t i=0; i<thread_widths.size(); ++i) {
      int64_t ws = warp_layout[i] * thread_widths[i];
      warp_widths.push_back(ws);
    }
    return warp_widths;
  }

  llvm::SmallVector<int64_t, 2> getBlockWidths(
      llvm::SmallVector<int64_t, 2> warp_widths, 
      llvm::SmallVector<int64_t, 2> warp_repeat,
      llvm::SmallVector<int64_t, 2> block_layout) {
        // 一个block计算的tile（重复后才等于bm/bn）
    llvm::SmallVector<int64_t, 2> block_widths;
    for (size_t i=0; i<warp_repeat.size(); ++i) {
      int64_t wrs = warp_repeat[i] * warp_widths[i];
      int64_t bs = block_layout[i] * wrs;
      block_widths.push_back(bs);
    }
    return block_widths;
  }

private:

  int64_t thread_bound;
  AffineMap affine_map;

  std::vector<const char*> mapOperandsLabel;  // mapOperands 的标签
  std::vector<const char*> iterVarLabels;  // for 循环的标签
  std::vector<int> ivUpperBounds;  // 迭代变量的上界
  llvm::SmallVector<AffineExpr, 2> indices;
  uint32_t dimCount = 0;

  llvm::SmallVector<AffineExpr, 2> warp_indices;  // warp_id: [(tid / 32) / block_layout[1], (tid / 32) % block_layout[1]]
  llvm::SmallVector<AffineExpr, 2> lane_indices;  // lane_id: [(tid % 32) / warp_layout[1], ...]
  llvm::SmallVector<int64_t, 2> warp_layout;  // lane
  llvm::SmallVector<int64_t, 2> block_layout;  // warp

  llvm::SmallVector<int64_t, 2> warp_repeat;
  llvm::SmallVector<int64_t, 2> block_repeat;

  llvm::SmallVector<int64_t, 2> thread_widths;  // 一个thread计算的元素个数 [tx,ty]
  llvm::SmallVector<int64_t, 2> warp_widths;
  llvm::SmallVector<int64_t, 2> block_widths;

protected:
  AffineExpr _theadIdx(OpBuilder& b){
    if(dimCount == 0){
      dimCount++;
      mapOperandsLabel.push_back(TID);
    }
    return b.getAffineDimExpr(0);
  }
};

class LowerInfoAnalysis {
public:

  static DenseMap<Value, LowerInfo> run(mlir::Operation* kernelOp);

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
