# early-wave-baseline 多 shape 性能汇总（2026-11-25）

合并 shape_eval、shape_eval_continue 和 early_wave_baseline；同一 shape / 模式采用最近一次成功运行结果。
8 个 shape，共 16 组；全部通过 atol=rtol=0.01 的正确性检查。
设备 BW1000 64G；warmup=20，repeat=100，seed=0；BM/BN/BK=128/128/32。
时间为中位数，单位 ms；加速比 = SDPA / Kernel，大于 1 表示 Kernel 更快。

| BHSD | causal Kernel | causal SDPA | causal 加速比 | non-causal Kernel | non-causal SDPA | non-causal 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| 1×32×512×128 | 0.066881 | 0.116480 | 1.742× | 0.062400 | 0.105280 | 1.687× |
| 1×32×1024×128 | 0.178401 | 0.181041 | 1.015× | 0.162881 | 0.209201 | 1.284× |
| 1×32×2048×128 | 0.417203 | 0.420962 | 1.009× | 0.448962 | 0.509283 | 1.134× |
| 1×32×4096×64 | 0.776005 | 0.798725 | 1.029× | 0.959845 | 1.135127 | 1.183× |
| 1×8×4096×128 | 0.365834 | 0.378554 | 1.035× | 0.497756 | 0.549195 | 1.103× |
| 1×16×4096×128 | 0.662325 | 0.682404 | 1.030× | 0.826645 | 0.888085 | 1.074× |
| 1×32×4096×128 | 1.181687 | 1.104887 | 0.935× | 1.480168 | 1.555529 | 1.051× |
| 2×8×4096×128 | 0.666484 | 0.679764 | 1.020× | 0.827205 | 0.887766 | 1.073× |

non-causal：8/8 组快于 SDPA；causal：7/8 组快于 SDPA。
1×32×4096×128 causal 比 SDPA 慢 6.95%。
约 1–3% 的加速幅度较小，需多次独立运行确认稳定性；本表覆盖整除 tile 的 S、D=64/128，尚不代表尾块和所有其他 shape。

## 编译优化命中情况

以下结论来自本轮保存的 lowering.log；开关启用不保证变换命中。

| 配置 | causal | non-causal |
|---|---|---|
| lazy-softmax-base=8 | 跳过：score 带 mask/bias | 命中 |
| cross-bn-k0=1 | 共享内存规划跳过：无法确认访存/读取关系 | 生成共享内存规划 |
| k-tail-prefetch=4 | regions=0、moved=0 | regions=1、moved=7 |
| v-double-buffer=1 | unsupported stage order，跳过 | unsupported stage order，跳过 |

相同参数在两种模式下的实际优化覆盖不同。各项变换的性能贡献仍需单因素对照，
共享内存规划或指令移动的报告本身不证明 GPU 性能收益。

## 结果分析

- 固定 B=1、H=32、D=128，non-causal 随 S=512/1024/2048/4096 的加速比分别为
  1.687x/1.284x/1.134x/1.051x。短序列优势更大；启动、调度开销及 SDPA 后端选择
  可能影响差异，未做 profiling，不能把加速比直接解释为计算吞吐比。
- 1x32x4096x128 causal 的两次加速比为 0.939x、0.935x，回退可复现。
  同一 shape 从 non-causal 切换到 causal，Kernel 耗时减少约 20%，SDPA 减少约 29%。
  输入 IR 已限制 causal 的 key 循环；mask、动态循环、block 负载不均衡和变换覆盖
  是待验证线索，尚未确定寄存器或带宽瓶颈。
- BxH=16 的 1x16x4096x128 与 2x8x4096x128，Kernel 耗时差异约为 causal 0.63%、
  non-causal 0.07%；在这些测试上 batch/head 拆分没有明显额外成本。
- 固定 B=1、H=32、S=4096，D 从 128 减为 64，Kernel 耗时减少约 34-35%。
- 1x32x2048x128 causal 的 P10-P90 跨度约为中位数的 8.3%，而中位数加速仅 0.9%。
  小幅优势与分布重叠需要多轮独立测量确认；原始计时包含 100 次交替顺序的 GPU event 测量。

## 版本与数据范围

测量来自 2026-11-24/25 使用当时现有 MyTest 二进制的运行。记录这些数据时本地
源码 HEAD 为 85bde2ab。随后同步的上游编译器变更需另外构建、测量；本表不代表
新编译器版本的性能。只有这些整除分块的 shape、seed=0 和上述容差得到验证。
原始 IR、编译日志及 HSACO 保存在下表的本地 generated_ir 目录中。

[完整精度 CSV](2026-11-25-early-wave-baseline.csv)

## 来源

| BHSD | 模式 | 运行目录 |
|---|---|---|
| 1×32×512×128 | causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×512×128 | non-causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×1024×128 | causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×1024×128 | non-causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×2048×128 | causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×2048×128 | non-causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×4096×64 | causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×32×4096×64 | non-causal | generated_ir/shape_eval/run-nquk6jyu/compiled/run-20261124-144255-2llo8vuo |
| 1×8×4096×128 | causal | generated_ir/shape_eval_continue/run-74357jm6/compiled/run-20261124-103532-h0jsek3f |
| 1×8×4096×128 | non-causal | generated_ir/shape_eval_continue/run-74357jm6/compiled/run-20261124-103532-h0jsek3f |
| 1×16×4096×128 | causal | generated_ir/shape_eval_continue/run-ctu0v6rv/compiled/run-20261125-101759-yjpcwqr9 |
| 1×16×4096×128 | non-causal | generated_ir/shape_eval_continue/run-ctu0v6rv/compiled/run-20261125-101759-yjpcwqr9 |
| 1×32×4096×128 | causal | generated_ir/shape_eval_continue/run-ctu0v6rv/compiled/run-20261125-101759-yjpcwqr9 |
| 1×32×4096×128 | non-causal | generated_ir/shape_eval_continue/run-ctu0v6rv/compiled/run-20261125-101759-yjpcwqr9 |
| 2×8×4096×128 | causal | generated_ir/shape_eval_continue/run-qo6ufx5x/compiled/run-20261125-094126-920fza3d |
| 2×8×4096×128 | non-causal | generated_ir/shape_eval_continue/run-qo6ufx5x/compiled/run-20261125-094126-920fza3d |
