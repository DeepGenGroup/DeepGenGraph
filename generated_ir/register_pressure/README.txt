共享寄存器压力分析：实际 attention 日志
=====================================

主要查看以下四份日志（每份包含各 block 峰值、热点、共享内容来源和 IR 行号）：
  noncausal.fragment.pressure.log
  noncausal.thread.pressure.log
  causal.fragment.pressure.log
  causal.thread.pressure.log

对应的 *.fragment.mlir / *.thread.mlir 是完整阶段快照。
日志中的 loc 指向这些快照的实际行号，方便定位热点与存活值的定义。
*.pressure.log（不含阶段名）是 MyTest 内部打印的原始报告，loc 指向输入程序。
*.lowering.log 是包含所有 IR dump 的完整编译日志。

kernel @Attn_p2 的摘要（单位：32-bit words/thread）：

case        stage       SSA data -> shared data   packed   ideal-packed   live   overlap
noncausal   fragment         352 -> 350              314            314    350       352
noncausal   thread           321 -> 294              231            231    304       306
causal      fragment         354 -> 352              316            316    352       354
causal      thread           351 -> 332              296            263    340       344

packed 是 peak-packed-data：各来源的存活位数分别向上取整再相加。
ideal-packed 是新增的 peak-ideal-packed-data：先汇总去重后的存活 data bits，
再统一除以 32 向上取整，允许不同 SSA 的子字值乐观共享一个 word。
例如两个独立 f16 SSA：data=2、packed-data=2、ideal-packed-data=1。
三个独立 f16 SSA 的 ideal-packed-data=2；不能对各 SSA 先取整再相加。
理想打包忽略硬件指令/寄存器类别/复制约束，不能当成实际 VGPR 分配。

主指标 live（peak-live）现为 data + 显式 pointer SSA，排除 memref 及其完整描述符。
global/shm 的存储内容始终不计为寄存器；它们的 load 结果会计为 data。
显式指针单列 addresses，动态 index/offset 算术仍计为 data。
本次口径修正主要影响 peak-live，之前的 peak-data 本来就不包含描述符。
完整 memref 描述符仍保留为 descriptor-diagnostic，仅供诊断，不参与热点排名。
旧合计口径保存在 peak-with-descriptors：noncausal fragment/thread 为 444/402，
causal fragment/thread 为 446/426；它们不再是主寄存器压力指标。

SSA data 是旧的整值计数（ssa-peak-data）；shared data 是按来源内容并集计数
（peak-data）。静态 extract/insert 精确追踪切片；pack/unpack/shape_cast/bitcast
保留来源；from_elements 和标量 broadcast 复用输入内容。算术和加载仍产生新内容。
当旧 vector 与更新结果共存时，未改写部分只计一次，被改写的旧内容仍需保留。
日志保留 ssa-before/ssa-after/ssa-overlap，方便逐点检查改变量。

更直观的小例子见 sharing_examples.mlir 和 sharing_examples.pressure.log：
  static_fragment_sharing: 旧 tile(8)、更新结果(8)、旧切片(4) 同时存活，
      旧口径 peak-live=20 / peak-overlap=24，内容并集均为 12（8+新 patch 4）。
  static_fragment_replace: insert 前为 12，旧 tile 死亡后为 8。
  dynamic_fragment_sharing: 返回点为 12；全程峰值为 14，额外 2 来自动态 index。
  memory_only: global 和 shm memref 同时存活，peak-live=0；描述符诊断为 28。
  two_f16_from_global_and_shm: 两个 load 结果的 peak-data=2、peak-packed-data=2、
      peak-ideal-packed-data=1；源缓冲区与描述符不进入上述三项。
  explicit_addresses: 指针和动态 offset 是实际 SSA 值，仍计入主压力；
      peak-live=4、peak-addresses=4、peak-data=2（各项峰值不一定同时发生）。

peak-live 是同一程序点的数据与显式指针合计峰值；peak-data 是单独的数据峰值。
不同类别的峰值不一定出现在同一位置，不能将类别峰值相加。
peak-overlap 对普通指令额外考虑最后使用的输入与新结果同时存在。
所有 kernel 报告的 max-unknown-live 均为 0。
causal 的 FragmentIR 有一个无 RegionBranch 接口的 mask_frag region，按保守规则
保留外部存活值并记录 unknown-region-ops=1；ThreadIR 展开后该项为 0。
宿主包装函数 @Attn 中的 tensor 类型没有寄存器成本模型，会单独标出 unmodeled-type；
宿主函数的已知成本为 0 不代表其实际寄存器使用为 0。

这些数值是内容需求诊断估计，不是硬件 VGPR/SGPR 数，也不是 occupancy。
不能把 peak-live 或 peak-data 与 255 直接比较，也不能把任一 packed 指标当作实际 VGPR。
peak-live 不再包含完整 memref 描述符，但尚未区分 SGPR/VGPR；早期 memref lowering
尚未显式化的地址计算开销也未预测。内存空间与指针寄存器类别并不是同一个概念。
当前仍使用整值 SSA 生存期，尚未实现每个元素的最后使用/循环迭代间版本分析。
FragmentIR 的非因果峰值附近（noncausal.fragment.mlir:167）同时保留源 %63 和
两个 vector<2x32xf32> carrier，三者共 192 个 f32。不能仅凭 insert 就把整组
carrier 与源合并：既要追踪每轮更新的切片，也要知道哪些旧元素以后不会再读。
因此来源共享对 FragmentIR 的全局 peak-data 只减少 2；以排除描述符后的相同口径
比较，ssa-peak-overlap=406 降至 peak-overlap=352（causal: 408 -> 354），
消除了不少插入操作输入/输出共存时的重复计费。

两组 FragmentIR 都有 unresolved-extracts=82、conservative-inserts=77。
动态 extract 在来源内容已经被计入时不另收费；否则按独立切片计费。
动态 insert 保守保留 destination+input 并集，可能多计被覆盖部分。
ThreadIR 无 conservative-inserts，但仍有 128 个动态 extract。
后续 lowering 内部临时值、常量物化、硬件操作数约束引入的复制和分配粒度尚未计算，
因此这些估计也不是硬件占用上界。用于早期 pass 比较压力变化时必须保留上述限制；
真正的硬件上限判定需要匹配目标后端分配结果，并分别看 VGPR/SGPR 和 scratch/spill。
当前只增加分析和日志，没有更改归约链数、调度或其他优化策略。

实现复用 MLIR Liveness、RegionBranchOpInterface、LoopLikeOpInterface、DataLayout
及 AnalysisManager。可复用接口位于：
  3rd/deepgengraph/include/deepgengraph/Analysis/RegisterPressure.h
实现与完整成本说明：
  3rd/deepgengraph/lib/Analysis/RegisterPressure.cpp
  3rd/deepgengraph/test/register_pressure_README.txt

验证结果见 tests.log、validation.json 和 build.log。
包含静态/动态切片、插入新旧版本共存、fragment 坐标缩放、bitcast 子字范围、
from_elements、跨 SSA 乐观打包、global/shm 排除、加载结果/显式地址计费、描述符独立诊断、
常量、CFG 回边、嵌套循环、循环捕获、跨循环存活、分支、DLTI、
未知类型、IR 不变性、日志确定性和分析缓存失效测试。
两组实际 attention 均比较了开启/关闭分析的 LLVM 输出，并验证阶段快照独立重放的
分析摘要与 MyTest 内部结果一致。GPU 性能和物理寄存器分配未在此次任务中测试。

重现（仓库根目录）：
  python3 generated_ir/register_pressure/reproduce.py --baseline

在原有 MyTest 命令后增加：
  --analyze-register-pressure=1

对独立阶段快照分析：
  3rd/deepgengraph/build/test/ThreadTilingTest \
    generated_ir/register_pressure/noncausal.thread.mlir \
    '--frisk-analyze-register-pressure=stage=thread top=8 all-points=true' \
    -o /dev/null 2> /tmp/pressure-all-points.log

all-points=true 输出每条指令前后的完整压力曲线；默认只打印摘要、热点和 reduce_frag。
