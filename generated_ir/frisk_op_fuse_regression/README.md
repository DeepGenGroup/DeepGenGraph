# FriskOpFuse 性能回退检查

检查输入：

- 基准：`../best_noncausal_early_wave/kernel.o3.ll`。
- 中间版本：`../best_noncausal_gemm_accumulator/kernel.o3.ll`。
- 新版本：`../best_noncausal_frisk_op_fuse/kernel.o3.ll`。
- 最新目标汇编：仓库根目录 `k_test.gcn.log`，UTF-16；解码副本为 `k_test.gcn.log.txt`。
- 旧性能日志：仓库根目录 `pmc_test.log`；解码副本为 `pmc_test.log.txt`。没有将其中历史 hipprof 计数当作新版计数。

基准和中间版本的 kernel 函数体在去除 debug location 后完全一致。下面比较中间版本与新版本。

已确认：

| 静态 LLVM 指令 | 旧 | 新 |
| --- | ---: | ---: |
| MMAC | 256 | 256 |
| 128-bit DMA | 16 | 16 |
| 64-bit DMA | 20 | 20 |
| matrix LDS read | 32 | 32 |
| load / store | 80 / 64 | 80 / 64 |
| fadd / fmul | 68 / 42 | 68 / 42 |
| scheduler barrier | 306 | 306 |
| ds.bpermute | 72 | 8 |
| insertelement | 240 | 111 |

PV interleave4、V double buffer、K DMA64-swizzle、K LDS prefetch8、tail prefetch4、cross-BN K0、lazy softmax base8 和 RTZ packing 的命中记录一致。没有看到这些优化失效。

最新汇编的 Attn_p2：231 VGPR、44 SGPR（汇编报告）、ScratchSize=0、LDS=16384、Occupancy=1；256 条 MMAC、32 条 matrix read、8 条 ds_bpermute、64 条 global_store_short。旧 hipprof 报告 VGPR=232、scratch=0、LDS=16384。两种工具的资源计数口径不同，不能据此声称 VGPR 或 SGPR 显著上升。

## 两个明确变化

1. 输出布局改变。旧版在循环出口将硬件 C 布局转成 contiguous C；新版布局推断把硬件布局传播到最终 store。省去的 64 次 bpermute 全部是出口转换。旧版每个线程连续输出四个元素，新版相邻寄存器输出列间隔为 4（f16 字节间隔 8）。旧 LLVM 见 `kernel.o3.ll:2774`，新 LLVM 见 `kernel.o3.ll:2240`；新汇编 `k_test.gcn.log.txt:3631` 开始的 store 确实为 short，offset 为 0/8/16 等。旧版是否实际生成更宽的 store，须有对应 gfx936 汇编才能确认。

   对同一行的 lanes 0/16/32/48，旧版第一条 store 的列为 [0,4,8,12]，第二条为 [1,5,9,13]；新版第一条为 [0,1,2,3]，第二条为 [4,5,6,7]。不能只因单线程 stride 变大就断言整个 wave 的访存合并变差；新版同一条指令跨 lane 的地址反而更连续，但失去了原来的单线程相邻元素写入形式。需要目标 codegen/实测区分收益与损失。

2. PV seed 提取位置前移。每组四条累加链的 seed 原来分别在对应 MMAC 前提取，新版在该组 wait/barrier 前集中提取。最后一组还在 cross-BN K0 分支之前提取。这可能改变 tied-operand 寄存器复制、活跃区间和机器调度；LLVM 中的 extract/shuffle 不等价于必然执行一条硬件指令，不能从文本距离直接推导耗时。

## 动态语义核对

`python3 generated_ir/frisk_op_fuse_regression/compare_trace.py` 使用仓库已有 delayed-DMA CPU LLVM 模型，在 random seed42、head0/query0、完整 4096 keys 上比较：

- 8192 次 MMAC 的 A/B/acc 数值及调用顺序逐项相同。
- LDS 读取来源/时序记录、LDS 值摘要、global reads 和 K0 发起位置相同。
- 每个版本 260 次 VM wait、261 次 block barrier、512 次 64-bit DMA。
- 输出逐位一致；所有线程合计写入的输出地址集合相同。
- 按线程的最终 store 地址不同，符合上述输出布局改变。

此模型不模拟机器指令调度、寄存器分配或 GPU 时间。

## 结论与下一步验证

目前证据把差异集中到输出布局和后端调度，不能确认哪一个解释用户报告的速度下降。没有足够证据直接删除 FriskOpFuse 或认定“融合本身增加了 MMAC 依赖”。应优先获得旧版同一 DTK 的 gfx936 汇编，比较输出 store 宽度、PV seed 的 v_mov 和 wait 分布，再分别做只恢复输出布局、只恢复 seed 提取位置的 A/B。

旧日志的实际中位耗时是 HSACO=1.484159 ms、PyTorch=1.570719 ms、speedup=1.058322。用户更正新版 HSACO=1.5635 ms、PyTorch=1.570 ms，按这组耗时计算 speedup=1.004157。HSACO 耗时增加 0.079341 ms（5.35%），PyTorch 耗时变化 -0.046%。因此回退主要来自 kernel 本身，不能用对照基线波动解释。具体瓶颈仍待旧版汇编或新版 profiling/A/B 验证。

本次没有修改编译器源码，也没有将未实测的推断作为性能修复。

证据：`metrics.json`、`structural.diff`、`trace_comparison.json`、解码汇编/日志。`current.named.ll` 仅用于分析 SSA，不是新的调优产物。
