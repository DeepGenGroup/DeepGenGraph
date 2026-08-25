#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

namespace mlir::frisk {

class LivelinessAnalyzer {
private:
  // 1. 估算 MemRefType 对应的寄存器需求（以 32-bit 寄存器为单位）
  int64_t calculateRegisters(MemRefType type);
  // 估算 MemRefType 对应的共享内存需求（以 byte 为单位）
  int64_t calculateBytes(MemRefType type);
  // 2. 追溯 View 算子至底层 Root Alloc (如 memref.alloca)
  Value getRootAllocation(Value value) ;
public:
  // 记录每个 Root Allocation 的全局存活区间 [start, end]
  llvm::DenseMap<Value, std::pair<unsigned, unsigned>> liveRanges;
  llvm::DenseMap<Value, int64_t> rootRegCounts;
  llvm::DenseMap<Value, int64_t> rootShmBytes;
  llvm::DenseMap<Value, llvm::DenseSet<Value>> shmInterferenceMap;
  llvm::DenseMap<Value, unsigned> rootShmColors;
  llvm::DenseMap<Value, int64_t> rootShmOffsets;
  llvm::DenseMap<unsigned, int64_t> shmColorBytes;
  // 进行分析
  void run(func::FuncOp funcOp) ;
  // 基于分析结果，进行 buffer -shm复用性分析（着色）
  // 方法是：建立 buffer 干扰图 interference map. 有干扰，则buffer间连线。最后着色，将不相邻Node染成相同颜色。过程中还要区分buffer基础数据类型
  void getColoredShmNodes();
};

} // namespace mlir:frisk
