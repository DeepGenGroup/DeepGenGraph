最新稳定版 SHM 操作增加分析（2026-10-23 GPU capture）

输入确认：k_test.gcn.log 的 .file 指向 online_softmax_best_aligned 的
run-20261023-173120-_o7ne8xg/retained32-baseline/input.mlir，确为对齐后的稳定版。
GPU 实测 median=1.448637009 ms、SDPA=1.117996991 ms，speedup=0.771757855x。
unsafe group4 best median=1.168558955 ms；稳定版耗时增加23.97%。
当前 scratch=0，PMC VGPR=248；汇编 NumVgprs=246，资源按分配粒度取整。
当前 LDS=16384，没有新增显式 ds_write，没有本次 scratch spill 的证据。

SHM 对照使用 pmc_test.log 中有完整 PMC 的 retained_32words 原版：
4,890,624 -> 9,224,192，增加4,333,568（88.61%）。group4 best 在提供的
日志中只有计时，没有该项 PMC；不能将原版 group8 的 PMC 冒充 group4 测量。
对比两份 LLVM，group4 best 与稳定版的普通 LDS half4 load 静态站点都为144，
matrix read 都为32；分组改变执行次序，不改变这里固定形状的 causal 工作量。

根因并非 LDS 数据读取翻倍。ds_bpermute 使用 DS 跨 lane 通信路径，
本次 shared memory operation 的增加可以由它的执行次数精确闭合。
旧版：每轮 BN 内4条 sum shuffle，循环结束后64条 O 布局 shuffle。
稳定版：每轮 BN 内4条 max + 4条 sum + 64条 PV 布局 shuffle。
只比较静态68条与72条会漏掉布局 shuffle 从循环外移入循环内这一变化。

BHSD=(1,32,4096,128)、BM=BN=128、每block4个wave：
W = 32 heads * 32 query tiles * 4 waves = 4096
N = 32 heads * (1+...+32) * 4 waves = 67584 次 BN-wave 迭代

项目                           原版              稳定版
Q 一次性 LDS 读取：8*W           32768             32768
K/V 每轮读取：64*N              4325376           4325376
sum shuffle：4*N                270336            270336
max shuffle                     0                 270336
PV/O layout shuffle             64*W=262144       64*N=4325376
合计                            4890624           9224192

布局转换贡献64*(N-W)=4063232，解释增量的93.76%。MAX贡献4*N=270336，
占增量6.24%。统计包含wave通信操作，不能当成LDS数据传输字节数。
assembly.txt:2613 起对应MAX，3203起对应sum，3351起对应PV布局转换。
PV片段里反复出现 ds_bpermute + lgkmcnt 等待 + v_cndmask，均在BN循环中。

优先优化1：让缩放递推也能把布局转换移出循环。
  lib/Conversion/FriskToBase/FriskBaseToTheadlevelIR_plan2.cpp
  sinkLoopLayoutConversions（约1144行）只匹配 add 的另一输入直接为
  BlockArgument。稳定版另一输入为 mul(O_old, alpha)，约1159行匹配失败。
  扩展为 scaled recurrence：让 O carrier、scaledO、PV 都保持 native
  MMAC C layout，循环结束后再一次转换。alpha 的行广播必须按该 native
  layout 的逻辑行映射取值，不能直接复用原 contiguous layout 的lane编号。
  可保持“PV先归约、再与scaledO相加”的浮点求值顺序，仅移动布局边界。
  理想计数目标为5160960，较当前减少44.05%，仍保留MAX通信。
  这是未加入额外alpha重排开销的计数目标，不是已实现或实测性能提升。

优先优化2：在上述布局一致后，再融合 scaledO 到 MMAC C 输入。
  lib/Conversion/ConvertToLLVM/FuseMMACAccumulator.cpp
  目前seed前移只允许 extract/shuffle，不允许晚定义的fmul；现有布局
  shuffle还会隔开最终fadd与MMAC链，因此现有融合不能直接覆盖该模式。
  可把 scaledO 提前计算并作为合法seed，消除PV临时结果及额外向量加法。
  这会改变浮点累加顺序，应独立做正确性与极值测试，并核对VGPR资源。

MAX归约本身是较小的新增通信项，不应以移除数值稳定性为代价优化。
bank conflict比例从原版11.58%到当前10.09%，当前数字不支持“冲突比例
恶化导致操作量翻倍”的解释；总DS请求增加仍会带来等待和资源占用。
L2命中率、调度与VGPR可能影响剩余耗时，不能由SHM计数推算各自耗时占比。

复核：python3 generated_ir/online_softmax_best_aligned/shm_analysis/analyze.py
analysis.json 保存静态/动态分解、源位置与输入日志哈希。当前只完成分析，
未修改编译器、未生成上述两项优化的候选内核。
