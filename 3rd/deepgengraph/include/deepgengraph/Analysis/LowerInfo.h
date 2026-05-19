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

namespace mlir::frisk {

struct LowerInfo {

  Value buffer;
  int64_t thread_bound;

  llvm::SmallVector<AffineExpr, 2> warp_indices;  // tid / warp_size
  llvm::SmallVector<AffineExpr, 2> thread_indices;  // tid % warp_size
  llvm::SmallVector<int64_t, 2> warp_layout;
  llvm::SmallVector<int64_t, 2> block_layout;

  llvm::SmallVector<int64_t, 2> warp_repeat;
  llvm::SmallVector<int64_t, 2> block_repeat;

  llvm::SmallVector<int64_t, 2> thread_widths;
  llvm::SmallVector<int64_t, 2> warp_widths;
  llvm::SmallVector<int64_t, 2> block_widths;

  void show() const {
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
    printExprVec("thread_indices", thread_indices);
    printI64Vec("warp_layout", warp_layout);
    printI64Vec("block_layout", block_layout);
    printI64Vec("warp_repeat", warp_repeat);
    printI64Vec("block_repeat", block_repeat);
    printI64Vec("thread_widths", thread_widths);
    printI64Vec("warp_widths", warp_widths);
    printI64Vec("block_widths", block_widths);
    llvm::outs() << "=================\n";
  }


  llvm::SmallVector<AffineExpr, 2> getAffineMap() const {
    // 根据上述信息，生成不同层面的索引
    OpBuilder b(buffer.getDefiningOp());
    MemRefType type = dyn_cast<MemRefType>(buffer.getType());
    llvm::SmallVector<AffineExpr, 2> indices;
    if (type.getMemorySpaceAsInt() == 0 || type.getMemorySpaceAsInt() == 5) {  // local
      for(size_t i=0; i<thread_widths.size(); ++i) {
        auto ib = b.getAffineDimExpr(i*3+1);
        auto iw = b.getAffineDimExpr(i*3+2);
        auto it = b.getAffineDimExpr(i*3+3);
        AffineExpr expr = ib * (warp_repeat[i] * thread_widths[i]) + iw * thread_widths[i] + it;
        indices.push_back(expr);
      }
    } else if (type.getMemorySpaceAsInt() == 5) {  // shared
      for(size_t i=0; i<thread_widths.size(); ++i) {
        auto ib = b.getAffineDimExpr(i*3+1);
        auto iw = b.getAffineDimExpr(i*3+2);
        auto it = b.getAffineDimExpr(i*3+3);
        AffineExpr expr = ib * block_widths[i] + 
                          warp_indices[i] * (warp_repeat[i] * warp_widths[i]) + iw * warp_widths[i] + 
                          thread_indices[i] * thread_widths[i] + it;
        indices.push_back(expr);
      }
    }
    return indices;
  }

  static llvm::SmallVector<AffineExpr, 2> getThreadIndices(
    OpBuilder b, llvm::SmallVector<int64_t, 2> warp_layout) {
      // tid -> lane_id
    auto tid = b.getAffineDimExpr(0);
    auto ly = (tid % 32).floorDiv(warp_layout[1]);
    auto lx = (tid % 32) % warp_layout[1];
    return llvm::SmallVector<AffineExpr, 2>{ly, lx};
  }

  static llvm::SmallVector<AffineExpr, 2> getWarpIndices(
    OpBuilder b, llvm::SmallVector<int64_t, 2> block_layout) {
      // tid -> warp_id
    auto tid = b.getAffineDimExpr(0);
    auto wy = tid.floorDiv(32).floorDiv(block_layout[1]);
    auto wx = tid.floorDiv(32) % block_layout[1];
    return llvm::SmallVector<AffineExpr, 2>{wy, wx};
  }

  static llvm::SmallVector<int64_t, 2> getWarpWidths(
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

  static llvm::SmallVector<int64_t, 2> getBlockWidths(
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
};

class LowerInfoAnalysis {
public:
  LowerInfoAnalysis(func::FuncOp kernelOp): _kernelOp(kernelOp) {}
  void run();
  void getTest() {
    llvm::outs() << "[D]need_infer_ops size: " << need_infer_ops.size() << "\n";
    llvm::outs() << "[D]buf_info_maps size: " << buf_info_maps.size() << "\n";
  }
  void showAllInfo() {
    llvm::outs() << "[D]show all lower info, count: " << buf_info_maps.size() << "\n";
    for (const auto &it : buf_info_maps) {
      llvm::outs() << "[D]buffer key: ";
      it.first.print(llvm::outs());
      llvm::outs() << "\n";
      it.second.show();
    }
  }

private:
  func::FuncOp _kernelOp;
  SmallVector<Operation*, 5> need_infer_ops;
  DenseMap<Value, LowerInfo> buf_info_maps;
};

}

#endif
