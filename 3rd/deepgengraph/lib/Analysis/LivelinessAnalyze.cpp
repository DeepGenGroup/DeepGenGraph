#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include <tuple>

namespace mlir::frisk {

// 1. 估算 MemRefType 对应的寄存器需求（以 32-bit 寄存器为单位）
int64_t LivelinessAnalyzer::calculateRegisters(MemRefType type) {
  if (!type.hasStaticShape()) return 0; // 动态 Shape 暂不纳入静态峰值统计
  
  int64_t totalElements = type.getNumElements();
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  constexpr unsigned RegBitWidth = 32; // 假设架构寄存器宽度为 32-bit
  
  return (totalElements * bitWidth + RegBitWidth - 1) / RegBitWidth;
}

int64_t LivelinessAnalyzer::calculateBytes(MemRefType type) {
  if (!type.hasStaticShape()) return 0; // 动态 Shape 暂不纳入静态峰值统计

  int64_t totalElements = type.getNumElements();
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  constexpr unsigned ByteBitWidth = 8;

  return (totalElements * bitWidth + ByteBitWidth - 1) / ByteBitWidth;
}

// 2. 追溯 View 算子至底层 Root Alloc (如 memref.alloca)
Value LivelinessAnalyzer::getRootAllocation(Value value) {
  while (true) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp) break;

    if (auto subView = dyn_cast<memref::SubViewOp>(defOp)) {
      value = subView.getSource();
    } else if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      value = cast.getSource();
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(defOp)) {
      value = collapse.getSrc();
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(defOp)) {
      value = expand.getSrc();
    } 
    else {
      break;
    }
  }
  return value;
}

void LivelinessAnalyzer::run(func::FuncOp funcOp) {
  if(!funcOp->hasAttr("thread_num")){
    return;
  }
  liveRanges.clear();
  rootRegCounts.clear();
  rootShmBytes.clear();
  shmInterferenceMap.clear();
  rootShmColors.clear();
  rootShmOffsets.clear();
  shmColorBytes.clear();

  // // 初始化 MLIR 标准 Liveness 分析
  // Liveness liveness(funcOp);

  // 给 Block 内所有 Operation 建立线性拓扑索引
  llvm::DenseMap<Operation *, unsigned> opIndexMap;
  unsigned opIdx = 0;
  funcOp.walk([&](Operation *op) {
    opIndexMap[op] = opIdx++;
  });

  enum class MemoryKind { Register, Shm, Ignore };
  auto classifyRoot = [](Value root) {
    if (root.getDefiningOp<memref::AllocaOp>())
      return MemoryKind::Register;

    auto allocBuffer = root.getDefiningOp<frisk::AllocBufferOp>();
    if (!allocBuffer)
      return MemoryKind::Ignore;

    switch (allocBuffer.getMemorySpace()) {
    case 0:
    case 5:
      return MemoryKind::Register;
    case 3:
      return MemoryKind::Shm;
    default:
      return MemoryKind::Ignore;
    }
  };

  // 3. 遍历算子提取 Root MemRef 的活跃区间，并按 reg/shm 分类统计大小
  funcOp.walk([&](Operation *op) {
    unsigned currentIdx = opIndexMap[op];

    auto processValue = [&](Value val) {
      auto memrefType = dyn_cast<MemRefType>(val.getType());
      if (!memrefType) return;

      // 仅关注片上分配：local/register 与 shared memory。
      Value root = getRootAllocation(val);
      auto rootType = dyn_cast<MemRefType>(root.getType());
      if (!rootType) return;

      MemoryKind kind = classifyRoot(root);
      if (kind == MemoryKind::Ignore) return;

      if (kind == MemoryKind::Register && !rootRegCounts.count(root)) {
        rootRegCounts[root] = calculateRegisters(rootType);
      } else if (kind == MemoryKind::Shm && !rootShmBytes.count(root)) {
        rootShmBytes[root] = calculateBytes(rootType);
      }

      // 更新存活区间起点与终点
      if (!liveRanges.count(root)) {
        liveRanges[root] = {currentIdx, currentIdx};
      } else {
        liveRanges[root].first = std::min(liveRanges[root].first, currentIdx);
        liveRanges[root].second = std::max(liveRanges[root].second, currentIdx);
      }
    };

    for (Value operand : op->getOperands()) processValue(operand);
    for (Value result : op->getResults()) processValue(result);
  });

  // 4. 构建扫描线事件 (Sweep-Line Events)。区间不重叠的 shm buffer
  // 可复用同一段共享内存，峰值即复用后的最小并发需求。
  struct Event {
    unsigned time;
    int64_t delta; // +Resource (Alloc/Start), -Resource (Dealloc/End)
  };

  auto calculatePeak = [&](const llvm::DenseMap<Value, int64_t> &rootSizes) {
    std::vector<Event> events;

    for (auto &[root, size] : rootSizes) {
      auto rangeIt = liveRanges.find(root);
      if (rangeIt == liveRanges.end()) continue;

      auto range = rangeIt->second;
      events.push_back({range.first, size});       // Start: 增加资源占用
      events.push_back({range.second + 1, -size}); // End+1: 释放资源占用
    }

    // 优先按时间升序排序；同一时间点先释放(-Resource)再分配(+Resource)
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
      if (a.time != b.time) return a.time < b.time;
      return a.delta < b.delta;
    });

    int64_t current = 0;
    int64_t peak = 0;
    unsigned peakOpIdx = 0;

    for (const auto &event : events) {
      current += event.delta;
      if (current > peak) {
        peak = current;
        peakOpIdx = event.time;
      }
    }

    return std::pair<int64_t, unsigned>{peak, peakOpIdx};
  };

  // 5. 模拟扫描过程，分别计算 reg 与 shm 的复用后峰值
  auto [peakRegs, peakRegOpIdx] = calculatePeak(rootRegCounts);
  auto [peakShmBytes, peakShmOpIdx] = calculatePeak(rootShmBytes);

  // 打印分析结果
  llvm::outs() << "[MemRefLivelinessAnalyzePass] Function: " << funcOp.getName() << "\n";
  llvm::outs() << "  -> Peak Register Count: " << peakRegs << " (32-bit units)\n";
  llvm::outs() << "  -> Peak Register Occurred Near Operation Index: " << peakRegOpIdx << "\n";
  llvm::outs() << "  -> Peak Shared Memory: " << peakShmBytes << " bytes\n";
  llvm::outs() << "  -> Peak Shared Memory Occurred Near Operation Index: " << peakShmOpIdx << "\n";
  getColoredShmNodes();
  llvm::outs() << "  -> Reused Shared Memory Slots: " << shmColorBytes.size() << "\n";
}

void LivelinessAnalyzer::getColoredShmNodes() {
  rootShmColors.clear();
  rootShmOffsets.clear();
  shmColorBytes.clear();
  shmInterferenceMap.clear();

  struct ShmNode {
    Value root;
    std::pair<unsigned, unsigned> range;
    int64_t bytes;
    MemRefType type;
  };

  SmallVector<ShmNode> nodes;
  nodes.reserve(rootShmBytes.size());
  for (auto &[root, bytes] : rootShmBytes) {
    auto rangeIt = liveRanges.find(root);
    if (rangeIt == liveRanges.end()) continue;

    auto type = dyn_cast<MemRefType>(root.getType());
    if (!type) continue;

    nodes.push_back({root, rangeIt->second, bytes, type});
  }

  auto rangesOverlap = [](std::pair<unsigned, unsigned> lhs,
                          std::pair<unsigned, unsigned> rhs) {
    return lhs.first <= rhs.second && rhs.first <= lhs.second;
  };

  auto canShareSlot = [&](const ShmNode &lhs, const ShmNode &rhs) {
    return lhs.type.getElementType() == rhs.type.getElementType() &&
           !rangesOverlap(lhs.range, rhs.range);
  };

  for (const ShmNode &node : nodes)
    shmInterferenceMap[node.root];

  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      if (canShareSlot(nodes[i], nodes[j])) continue;
      shmInterferenceMap[nodes[i].root].insert(nodes[j].root);
      shmInterferenceMap[nodes[j].root].insert(nodes[i].root);
    }
  }

  SmallVector<ShmNode> coloringOrder(nodes.begin(), nodes.end());
  std::sort(coloringOrder.begin(), coloringOrder.end(),
            [&](const ShmNode &lhs, const ShmNode &rhs) {
              size_t lhsDegree = shmInterferenceMap[lhs.root].size();
              size_t rhsDegree = shmInterferenceMap[rhs.root].size();
              if (lhsDegree != rhsDegree) return lhsDegree > rhsDegree;
              if (lhs.bytes != rhs.bytes) return lhs.bytes > rhs.bytes;
              return std::tie(lhs.range.first, lhs.range.second) <
                     std::tie(rhs.range.first, rhs.range.second);
            });

  for (const ShmNode &node : coloringOrder) {
    llvm::DenseSet<unsigned> forbiddenColors;
    auto neighborIt = shmInterferenceMap.find(node.root);
    if (neighborIt != shmInterferenceMap.end()) {
      for (Value neighbor : neighborIt->second) {
        auto colorIt = rootShmColors.find(neighbor);
        if (colorIt != rootShmColors.end())
          forbiddenColors.insert(colorIt->second);
      }
    }

    unsigned color = 0;
    while (forbiddenColors.contains(color))
      ++color;

    rootShmColors[node.root] = color;
    auto colorBytesIt = shmColorBytes.find(color);
    if (colorBytesIt == shmColorBytes.end()) {
      shmColorBytes[color] = node.bytes;
    } else {
      colorBytesIt->second = std::max(colorBytesIt->second, node.bytes);
    }
  }

  SmallVector<unsigned> colors;
  colors.reserve(shmColorBytes.size());
  for (auto &[color, bytes] : shmColorBytes)
    colors.push_back(color);
  std::sort(colors.begin(), colors.end());

  llvm::DenseMap<unsigned, int64_t> colorOffsets;
  int64_t nextOffset = 0;
  for (unsigned color : colors) {
    colorOffsets[color] = nextOffset;
    nextOffset += shmColorBytes[color];
  }

  for (const ShmNode &node : nodes) {
    auto colorIt = rootShmColors.find(node.root);
    if (colorIt == rootShmColors.end()) continue;

    unsigned color = colorIt->second;
    int64_t offset = colorOffsets[color];
    rootShmOffsets[node.root] = offset;

    if (auto allocBuffer = node.root.getDefiningOp<frisk::AllocBufferOp>()) {
      MLIRContext *ctx = allocBuffer->getContext();
      Builder builder(ctx);
      allocBuffer->setAttr("shm_reuse_color", builder.getI64IntegerAttr(color));
      allocBuffer->setAttr("shm_reuse_offset", builder.getI64IntegerAttr(offset));
      allocBuffer->setAttr("shm_reuse_bytes",
                           builder.getI64IntegerAttr(node.bytes));
    }
  }

  llvm::outs() << "  -> Shared Memory Reuse Total: " << nextOffset
               << " bytes\n";
  for (unsigned color : colors) {
    llvm::outs() << "     color " << color << ": " << shmColorBytes[color]
                 << " bytes";
    auto offsetIt = colorOffsets.find(color);
    if (offsetIt != colorOffsets.end())
      llvm::outs() << ", offset " << offsetIt->second;
    llvm::outs() << "\n";
  }
}

} // namespace mlir:frisk
