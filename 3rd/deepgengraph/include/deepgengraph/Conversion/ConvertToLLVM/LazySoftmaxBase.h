#ifndef DEEPGENGRAPH_CONVERSION_LAZYSOFTMAXBASE_H
#define DEEPGENGRAPH_CONVERSION_LAZYSOFTMAXBASE_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
namespace mlir::frisk {
/// 解析 --lazy-softmax-base=off|<阈值>，阈值是正的有限浮点数（如 8）。
bool parseLazySoftmaxBaseOption(llvm::StringRef arg, bool &enabled, float &threshold, bool &consumed);

/// safe online softmax 的“滞后基准”：把 running max 状态 M 改成 N = -B，用 N 直接作 QK
/// 累加器的初值，于是 S' = S - B 不再需要逐元素减法；只有当本 wave 某一行的 S' 峰值超过
/// 阈值 T 时，才在一个 wave uniform 的慢路径里重缩放 S'、O、L 并推进基准。
/// 慢路径上的算术与原来的 running max 递推逐条对应（B 取代 M），所以对跨 lane 行最大值的
/// 前提和原来完全一致；快路径只是跳过这次重缩放。详见 .cpp 的说明。
/// 只处理 non-causal：causal 的 -inf mask 没有建模。
class LazySoftmaxBasePass : public llvm::OptionalPassInfoMixin<LazySoftmaxBasePass> {
  float threshold;

public:
  explicit LazySoftmaxBasePass(float threshold = 8.0f) : threshold(threshold) {}
  llvm::PreservedAnalyses run(llvm::Function &, llvm::FunctionAnalysisManager &);
};

/// 必须在 prefetchNextK 和 elideBackedgeBarrier 之后运行，因为它引入的
/// 分支只是 wave uniform，不是 workgroup uniform，会让依赖 analyzeWave64Kernel 的 Pass 放弃优化。
/// 不依赖该分析的 SoftmaxCvtPack 可在它之后运行。
void lazySoftmaxBase(llvm::Module &, bool enabled = false, float threshold = 8.0f);
} // namespace mlir::frisk
#endif
