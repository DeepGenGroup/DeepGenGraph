DeferLaneSumPass：通用循环跨 lane 求和延迟
=========================================

入口
  include/deepgengraph/Conversion/ConvertToLLVM/DeferLaneSum.h
  lib/Conversion/ConvertToLLVM/DeferLaneSum.cpp
  C++：deferLaneSums(Module &, DeferLaneSumOptions)
  New PM：DeferLaneSumPass(options)，使用 LoopAnalysis 和 DominatorTreeAnalysis。

数学变换（R 为固定 lane 子组的求和 all-reduce）
  原始：S(0)=0；S(t+1)=a(t)*S(t)+R(x(t))。
  变换：p_lane(0)=0；p_lane(t+1)=a(t)*p_lane(t)+x_lane(t)。
        退出循环后 S=R(p)。
  a 可以省略（普通循环求和）。带缩放时，a 必须在 R 的整个子组内一致；
  可以随迭代变化，也可以来自可证明组内一致的循环状态。
  整数 i32 在模 2^32 运算下保持结果；新的局部递推不沿用 nsw/nuw。
  浮点 f32 会改变加法和乘法顺序，只在显式 approx 模式下启用。
  approx 不承诺位级一致、异常行为或所有 NaN/Inf/下溢边界的保持；需验证数值容差。

与 attention 无关的识别条件
  * 不匹配函数名、SSA 编号、张量形状、BN、softmax/exp2 的固定表达式。
  * 从自然循环 header PHI、更新值和 def-use 识别零初值加法/缩放加法递推。
  * 支持标量以及 array/struct/fixed-vector 字段中的标量 f32/i32 状态；
    支持 <1 x T> 包装与单入边标量 LCSSA 转发。
  * 初版通信适配器支持 AMDGPU ds.bpermute 的固定 XOR butterfly，
    2/4/8/16/32/64 个 lane 子组，组内掩码可以不连续，例如 XOR 16、32。
  * 对完整工作组的每个 lane 有限枚举 shuffle 地址表达式，证明是固定 XOR，
    而不是依赖地址 IR 的文本格式。支持 mbcnt、线性 workitem.id.x、整数计算和 select。
  * 缩放一致性包括常数/kernel 参数、组内一致整数地址表达式、确定性算术、
    完整子组归约结果，以及可归纳证明一致的循环 PHI。
  * 检查整个函数的控制流保持完整 wave；拒绝不透明调用、未知 inline asm、
    EXEC 修改。DMA、barrier 和已知不修改 EXEC 的后端模板有显式适配。
  * 累计状态在循环中只能读一次用于匹配的更新；不允许中间 store、归一化、
    比较、非线性自乘，或归约/更新结果逃逸到其他用途。

保守边界（遇到时不变换，不是编译错误）
  当前要求 amdgpu_kernel、reqd_work_group_size、完整 wave、明确 wave 大小；
  无子循环的候选循环、preheader、单 latch、header 单出口、专用 exit。
  非零初值、不规则/重复 XOR、动态 shuffle 地址、无法证明一致的 load、
  lane 发散控制、部分 wave、严格浮点、其他通信原语、整个 aggregate 的
  LCSSA 转发/外部使用，以及不支持的类型，均保留原 IR。
  这是通用递推分析 + AMDGPU 通信适配器，不是任意目标的全能归约 Pass。
  缩短动态 shuffle 数不保证性能提升；短循环会增加出口开销。

CLI，默认关闭
  MyTest 和 TuneAttention.py：
    --defer-lane-sum=off|legal|approx
    TuneAttention：--set defer-lane-sum=approx
                   --sweep 'defer-lane-sum=off;legal;approx'
  approx 要求 math=approx。legal 目前只变换 i32。
  MyTest 的现有 lowering 使用 wave64 ABI；Pass 在 LocalReduction 之后、
  prefetch/register-lifetime 调整之前执行。结果计入 defer_lane_sum_reports。

  独立处理已导出的 IR（不运行其他优化）：
    3rd/deepgengraph/build/test/LegacyLLVMExportTest \
      --defer-lane-sum=approx --defer-lane-sum-wave-size=64 input.ll > output.ll
  若下游使用 DTK/LLVM22，再运行 --llvm22-text output.ll 转换浮点常量打印格式；
  MyTest 已自动做此转换。
  wave-size=0（独立入口默认）要求函数 +wavefrontsize32/+wavefrontsize64 特性。
  显式 32/64 是调用者的运行 ABI 承诺；与 target-features 冲突时跳过。
  C++ options 默认 approximate=false、waveSize=0；没有隐式 wave64 假设。

构建、回归
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 2
  python3 3rd/deepgengraph/test/check_defer_lane_sum.py \
    3rd/deepgengraph/build/test/LegacyLLVMExportTest --output /tmp/defer-tests.json
  python3 -m unittest script.test_tune_attention

  check_defer_lane_sum.py 包含非 attention 普通/加权/动态最大值累计、
  wave32/64、标量/aggregate、各 XOR 子组、零次/1/2/7/31 次迭代、
  i32 溢出、拒绝条件、CLI、重复执行幂等性、LLVM verifier 和 O3。
  对实际变换前后 LLVM 做 CPU lane 模型执行比较，不仅比较代数公式。
  CPU 执行模型不能验证 GPU 指令调度、性能或所有硬件行为。

供 GPU 测试的输出
  generated_ir/deferred_lane_sum_pass/kernel.ll
  generated_ir/deferred_lane_sum_pass/baseline.ll
  generated_ir/deferred_lane_sum_pass/run_gpu.sh
  generated_ir/deferred_lane_sum_pass/validation.json
  以上 baseline 与先前手工实验相同；kernel.ll 由新 Pass 自动生成。
  该目录保留独立 GPU A/B 入口：
    bash generated_ir/deferred_lane_sum_pass/run_gpu.sh
  新 Pass 生成的 IR 仍需用户实测 GPU 性能，不将历史手工实验计为本轮 GPU 测试。
