#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace mlir::frisk{

struct RegPressurePoint {
  Operation *op = nullptr;

  /// 整个 block tile 上的 logical reg32 数量。
  uint64_t regUnits = 0;

  /// regUnits / threadNum。
  uint64_t regsPerThread = 0;

  /// 此处 live 的 register values。
  SmallVector<std::pair<Value, uint64_t>> liveValues;
};

struct RegPressureResult {
  uint64_t peakRegUnits = 0;
  uint64_t peakRegsPerThread = 0;

  Operation *peakOp = nullptr;
  SmallVector<std::pair<Value, uint64_t>> peakValues;

  /// 按 IR lexical order 保存 pressure curve。
  SmallVector<RegPressurePoint> curve;
};

static uint64_t ceilDiv(uint64_t x, uint64_t y) {
  return (x + y - 1) / y;
}

static uint64_t getElementBitWidth(Type type,
                                   unsigned indexBitWidth = 64) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth();

  if (auto floatTy = dyn_cast<FloatType>(type))
    return floatTy.getWidth();

  if (isa<IndexType>(type))
    return indexBitWidth;

  return 0;
}

static int64_t getMemorySpace(MemRefType type) {
  Attribute space = type.getMemorySpace();

  if (!space)
    return 0;

  if (auto intAttr = dyn_cast<IntegerAttr>(space))
    return intAttr.getInt();

  return -1;
}

/// 返回 Value 对应的 logical 32-bit register unit。
static uint64_t getRegUnits(Value value,
                            unsigned regBitWidth = 32,
                            unsigned indexBitWidth = 64) {
  Type type = value.getType();

  // vector -> register tile
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    if (!vecTy.hasStaticShape())
      return 0;

    uint64_t numElements = vecTy.getNumElements();
    uint64_t elemBits =
        getElementBitWidth(vecTy.getElementType(), indexBitWidth);

    if (!elemBits)
      return 0;

    return ceilDiv(numElements * elemBits, regBitWidth);
  }

  // memref
  if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    int64_t space = getMemorySpace(memrefTy);

    //
    // 按你的 Frisk 语义：
    //
    // memory space 3 : shared memory
    // memory space 1 : global memory / global view
    //
    // payload 都不计入 register tile。
    //
    if (space == 3 || space == 1)
      return 0;

    if (!memrefTy.hasStaticShape())
      return 0;

    uint64_t numElements = memrefTy.getNumElements();
    uint64_t elemBits =
        getElementBitWidth(memrefTy.getElementType(), indexBitWidth);

    if (!elemBits)
      return 0;

    return ceilDiv(numElements * elemBits, regBitWidth);
  }

  // scalar
  if (isa<IntegerType, FloatType, IndexType>(type)) {
    uint64_t bits = getElementBitWidth(type, indexBitWidth);

    if (!bits)
      return 0;

    // 一个独立 scalar SSA 至少占一个 logical reg。
    return std::max<uint64_t>(
        1, ceilDiv(bits, regBitWidth));
  }

  return 0;
}

/// 分析一个具体 Operation 所在位置的 pressure。
static RegPressurePoint
analyzePressureAtOp(Operation *op,
                    const Liveness &liveness,
                    unsigned threadNum) {
  RegPressurePoint point;
  point.op = op;

  Block *block = op->getBlock();
  if (!block)
    return point;

  const LivenessBlockInfo *blockInfo =
      liveness.getLiveness(block);

  if (!blockInfo)
    return point;

  auto liveValues =
      blockInfo->currentlyLiveValues(op);

  for (Value value : liveValues) {
    uint64_t units = getRegUnits(value);

    if (!units)
      continue;

    point.regUnits += units;
    point.liveValues.emplace_back(value, units);
  }

  point.regsPerThread =
      ceilDiv(point.regUnits, threadNum);

  return point;
}

/// 按 lexical IR order 递归遍历。
///
/// 注意这里不依赖 Operation::walk() 的具体 traversal order。
static void collectRegionPressure(
    Region &region,
    const Liveness &liveness,
    unsigned threadNum,
    SmallVectorImpl<RegPressurePoint> &curve) {

  for (Block &block : region) {
    for (Operation &op : block) {

      //
      // 对 region-owning op 本身不做 pressure 统计。
      //
      // 例如：
      //   affine.for
      //   scf.if
      //   frisk.mask
      //
      // currentlyLiveValues() 对包含 region 的 op 是 expansive 的，
      // 容易把 region 内的值都合并到父 op 上，使结果虚高。
      //
      if (op.getNumRegions() == 0) {
        curve.push_back(
            analyzePressureAtOp(
                &op, liveness, threadNum));
      }

      //
      // 然后继续分析 nested region。
      //
      for (Region &nestedRegion : op.getRegions()) {
        collectRegionPressure(
            nestedRegion,
            liveness,
            threadNum,
            curve);
      }
    }
  }
}

static RegPressureResult
analyzeRegPressure(func::FuncOp funcOp,
                   unsigned threadNum = 64) {
  RegPressureResult result;

  Liveness liveness(funcOp.getOperation());

  //
  // 收集所有 op 的 pressure。
  //
  for (Region &region : funcOp->getRegions()) {
    collectRegionPressure(
        region,
        liveness,
        threadNum,
        result.curve);
  }

  //
  // 找峰值。
  //
  for (const RegPressurePoint &point : result.curve) {
    if (point.regUnits <= result.peakRegUnits)
      continue;

    result.peakRegUnits = point.regUnits;
    result.peakRegsPerThread =
        point.regsPerThread;
    result.peakOp = point.op;
    result.peakValues = point.liveValues;
  }

  return result;
}

static void dumpPressureCurve(
    func::FuncOp funcOp,
    const RegPressureResult &result,
    unsigned threadNum = 64,
    bool printFullOp = false) {

  llvm::errs()
      << "\n"
      << "================ Register Pressure Curve ================\n";

  llvm::errs()
      << "Function: "
      << funcOp.getName()
      << "\n";

  llvm::errs()
      << "Threads/block: "
      << threadNum
      << "\n\n";

  uint64_t previousRegsPerThread = 0;

  for (size_t i = 0; i < result.curve.size(); ++i) {
    const RegPressurePoint &point =
        result.curve[i];

    int64_t delta =
        static_cast<int64_t>(point.regsPerThread) -
        static_cast<int64_t>(previousRegsPerThread);

    //
    // index
    //
    llvm::errs()
        << "["
        << llvm::format("%4zu", i)
        << "] ";

    //
    // block 总 logical reg32
    //
    llvm::errs()
        << "reg32/block="
        << llvm::format("%7llu",
             static_cast<unsigned long long>(
                 point.regUnits))
        << "  ";

    //
    // per-thread
    //
    llvm::errs()
        << "reg/thread="
        << llvm::format("%4llu",
             static_cast<unsigned long long>(
                 point.regsPerThread))
        << "  ";

    //
    // 与上一条 op 的变化
    //
    llvm::errs()
        << "delta="
        << llvm::format("%+5lld",
             static_cast<long long>(delta))
        << "  ";

    //
    // op name
    //
    llvm::errs()
        << point.op->getName().getStringRef();

    if (point.op == result.peakOp)
      llvm::errs() << "    <--- PEAK";

    llvm::errs() << "\n";

    //
    // 可选：把完整 operation 打印出来。
    //
    if (printFullOp) {
      llvm::errs() << "       ";
      point.op->print(llvm::errs());
      llvm::errs() << "\n";
    }

    previousRegsPerThread =
        point.regsPerThread;
  }

  llvm::errs()
      << "\nPeak reg32/block : "
      << result.peakRegUnits
      << "\n";

  llvm::errs()
      << "Peak regs/thread : "
      << result.peakRegsPerThread
      << "\n";

  if (result.peakOp) {
    llvm::errs()
        << "Peak op          : ";

    result.peakOp->print(llvm::errs());
    llvm::errs() << "\n";
  }

  llvm::errs()
      << "=========================================================\n";
}

static void dumpPeakLiveValues(
    func::FuncOp funcOp,
    const RegPressureResult &result,
    unsigned threadNum = 64) {

  if (!result.peakOp)
    return;

  SmallVector<std::pair<Value, uint64_t>>
      values = result.peakValues;

  llvm::sort(
      values,
      [](const auto &lhs, const auto &rhs) {
        return lhs.second > rhs.second;
      });

  AsmState asmState(funcOp);

  llvm::errs()
      << "\n"
      << "================ Peak Live Values =======================\n";

  for (auto &[value, units] : values) {
    llvm::errs() << "  ";

    value.printAsOperand(
        llvm::errs(), asmState);

    llvm::errs()
        << " : "
        << value.getType()
        << "\n"
        << "      reg32/block = "
        << units
        << ", approx/thread = "
        << ceilDiv(units, threadNum)
        << "\n";
  }

  llvm::errs()
      << "=========================================================\n";
}

/// 最外层调用接口。
void dumpRegPressure(
    func::FuncOp funcOp,
    unsigned threadNum = 64,
    bool printFullOp = false) {

  RegPressureResult result =
      analyzeRegPressure(
          funcOp,
          threadNum);

  //
  // 1. 输出逐 op pressure 曲线
  //
  dumpPressureCurve(
      funcOp,
      result,
      threadNum,
      printFullOp);

  //
  // 2. 输出峰值处具体哪些 Value 活跃
  //
  dumpPeakLiveValues(
      funcOp,
      result,
      threadNum);
}

} // namespace