通用 DeferLaneSumPass 的 GPU A/B 输出
=================================

kernel.ll：Pass 自动生成的 BN128 测试 IR，已转换为 DTK/LLVM22 浮点常量格式。
kernel.llvm24.ll：独立 Pass 的原始 LLVM24 输出。
baseline.ll：与前次手工 deferred-sum 实验相同的优化前基线。
launch.json：对应启动参数。默认未开启此 Pass；需显式选择 approx。

执行：
  bash generated_ir/deferred_lane_sum_pass/run_gpu.sh
  # 或仅测试新版本：
  bash generated_ir/deferred_lane_sum_pass/run_gpu.sh deferred

复现 Pass：
  3rd/deepgengraph/build/test/LegacyLLVMExportTest \
    --defer-lane-sum=approx --defer-lane-sum-wave-size=64 \
    generated_ir/online_softmax_deferred_sum/baseline.ll > /tmp/deferred.llvm24.ll
  3rd/deepgengraph/build/test/LegacyLLVMExportTest \
    --llvm22-text /tmp/deferred.llvm24.ll > /tmp/deferred.ll

检查：
  generic-validation.json：46 个通用 IR 用例，20 个变换，100 组实际 LLVM A/B 执行。
  validation.json：8 个 attention 用例，16 次实际 LLVM 执行；包括延迟 DMA 检查、
    全 Q 单次装载、组内缩放一致性、动态 shuffle 数与数值误差。
  tune-tests.log：22 项 TuneAttention 入口、参数和过滤测试。
  full-bn128-validation.json、full-bn64-validation.json：完整流水线各 6 组数值回归。
  compiler_final/：BN64/BN128 的 off/legal/approx 完整 MyTest 导出。
  compiler/：开发期间的旧导出，仅用于排查；测试最终版本请使用 compiler_final/。

本轮未运行 GPU。CPU lane 模型通过不代表硬件性能一定提高。
该优化减少循环内通信，并不改变剩余 MMAC/exp2/布局转换的工作量。
原 BN128 基线循环内 8 次 shuffle -> 4 次；出口 64 次 -> 68 次。
32 次迭代的分母部分由 128 次 -> 4 次，最大值归约仍在循环内。

源代码和通用性约束：3rd/deepgengraph/test/defer_lane_sum_README.txt。
