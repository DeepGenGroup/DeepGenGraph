# Frisk Base CP-SAT 调度框架

在 `PromoteGemmMatrixLoads` 之后、`PipelineSchedule` / fragment lowering 之前交换 JSON：

```text
MyTest：识别 mainloop、分析依赖和 RW 规模 → problem.json
Python OR-Tools CP-SAT：联合求 op 顺序、stage、buffer 槽数 → schedule.json
MyTest：重新验证方案 → 同迭代移动，或 PipelineSchedule 展开与 buffer 轮转
```

使用当前 conda base 即可；开发验证版本为 Python 3.13、OR-Tools 9.15.6755。
没有使用 SMT 的 Z3 接口；这里直接使用 Google OR-Tools 的 CP-SAT 整数约束接口。

## 文件与入口

| 文件 | 作用 |
|---|---|
| `cpsat_scheduler.py` | CLI、`solve(problem, ...)`、整数规划与目标函数 |
| `cost_model.py` | 可替换的相对耗时估计 |
| `shm_liveness.py` | 按候选顺序计算 SHM 生命周期/峰值，生成 cumulative 约束 |
| `schedule_schema.py` | JSON 读写、结构及调度合法性检查 |
| `capture_baselines.py` | 保存最佳 causal/noncausal 配置的 Base 原始顺序，验证回放 |
| `test_scheduler.py` | 导出/求解/导入集成测试、延迟 DMA 的 CPU 数值测试 |
| `../../3rd/deepgengraph/lib/Pipeline/ScheduleJSON.cpp` | C++ mainloop 分析、JSON 导出/导入 |

构建与测试（仓库根目录）：

```bash
cmake --build 3rd/deepgengraph/build --target MyTest FriskScheduleJSONTest -j2
python -m unittest discover -s python/smt -v
```

CPU 数值测试默认用 `/data2/xsl/install/bin` 的 MLIR/LLVM 工具，可通过 `LLVM_BIN` 修改。
`CPSAT_TEST_ATTENTION=1 python -m unittest discover -s python/smt -v` 额外对两个真实 attention
case 求解、导入并完整 lowering，随后用 `llvm-as` 验证，约需数分钟。

## 使用

保留原来的 MyTest 输入与调优参数，第二个位置参数必须是 `0`。新选项不改变历史位置参数。
以已保存的 noncausal 输入演示最小命令；重现最佳配置时须使用其 `provenance.json` 内完整参数。
MyTest 会产生诊断文件，建议在独立工作目录运行。

```bash
# 在仓库根目录取得绝对路径，然后切入独立工作目录。
repo="$PWD"
mytest="$repo/3rd/deepgengraph/build/test/MyTest"
input="$repo/python/smt/baselines/noncausal/input.mlir"
mkdir -p /tmp/frisk-cpsat-demo
cd /tmp/frisk-cpsat-demo

# 1. 导出到 problem.json，保存原始顺序，停在 Base 阶段。
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  "$mytest" "$input" 0 unused.ll \
  --cpsat-export=problem.json --cpsat-baseline=baseline.json \
  --cpsat-dump-ir=base.mlir --cpsat-stop-after=1

# 2. 禁止跨迭代；仍可重排同一迭代的独立 op。
python "$repo/python/smt/cpsat_scheduler.py" problem.json -o schedule.json \
  --mode intra --time-limit 15

# 3. 用相同输入、环境和调优参数重跑前半段，导入后继续 lowering。
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  "$mytest" "$input" 0 kernel.ll --cpsat-import=schedule.json
```

支持跨迭代的区域可以将求解参数改成：

```bash
python "$repo/python/smt/cpsat_scheduler.py" problem.json -o pipeline.json \
  --mode pipeline --max-stages 2 --lds-budget 65536 --dma-capacity 4
```

`pipeline` 模式允许求出全部 stage=0 的方案，不强制双缓冲。
`--baseline` 可由 problem JSON 直接生成原始顺序；不经过求解器。
`--costs costs.example.json` 覆盖估计参数；`--scale-work` 开启按 RW/计算规模的粗粒度缩放。
资源默认值集中在 `cpsat_scheduler.py` 开头：`SHM_LIMIT_BYTES`、`DMA_MAX_IN_FLIGHT`、
`MAX_PIPELINE_STAGES`、`MAX_BUFFER_SLOTS`；CLI 可分别用 `--lds-budget`、`--dma-capacity`、
`--max-stages`、`--max-slots` 覆盖。SHM 默认 `None`（不限），其余默认 4 / 2 / 2。
VGPR（每 thread）和 SGPR（每 wave）上限已预留，但当前缺少寄存器活跃区间信息，
设为非 `None` 会明确报错。求解输出的 `model.resource_limits` 记录实际采用的限额。
无可行解或超时尚无解时返回非零退出码，不写新方案；目标路径若已有旧文件仍保留，调用方须检查退出码。

`examples/prefetch.mlir` 提供一个可直接演示双缓冲的 Base-IR 小循环，在仓库根目录运行：

```bash
bridge=3rd/deepgengraph/build/lib/Pipeline/FriskScheduleJSONTest
"$bridge" python/smt/examples/prefetch.mlir --export=/tmp/prefetch.problem.json
python python/smt/cpsat_scheduler.py /tmp/prefetch.problem.json \
  -o /tmp/prefetch.schedule.json --mode pipeline
"$bridge" python/smt/examples/prefetch.mlir --import=/tmp/prefetch.schedule.json \
  --ir=/tmp/prefetch.scheduled.mlir
```

mainloop 自动选择唯一最外层、包含 copy/GEMM 的 affine loop。可在 loop 标记 `cpsat.mainloop`，
或显式传 `--cpsat-loop=Attn_p2:0`；序号是函数内 affine loop 的前序编号。
歧义时会列出候选并报错，不随意选循环。一次导入只处理一个 loop。
启用 JSON 路径会跳过前面的 AutoPipeline，防止先展开再调度；输入及配置必须保持一致。

导出点已经在 `PromoteGlobalToSharedAsync` 和 `PromoteGemmMatrixLoads` 之后，
`ConvertFriskBaseToFragmentIR` 之前，仍是 block 级 IR。启用 CP-SAT 且未设置
`FRISK_GM_TO_SHM_MODE` 时默认使用 `async`，将可转换的 GM→SHM copy 变为
`frisk.async_copy_gm_to_shm` + `frisk.wait_buffer_ready`。
显式选择 `legacy` 会保留同步 copy；选择 `sync` 时在导出前降回同步 copy，
求解器按同步占用处理，不假设它能异步重叠。无法提升的 copy 仍按同步 op 建模。
`dma_summary` 区分直接调度的 DMA/wait 与嵌套 region 内的 DMA/wait；后者仍为原子 fence 的内部实现。

## JSON 与调度含义

`problem.json` 的 `schema` 是 `deepgengraph.cpsat.problem`，`version=1`：

- `fingerprint`：整个所属函数的规范 generic IR SHA-256。变化了函数、布局、tile 或相关参数，必须重新导出。
- `loop` / `candidates`：循环 ID、step、可确定的上下界、跨迭代能力及不支持原因。
- `operations`：按原始顺序的稳定整数 ID、op 名称、类别、`read_bytes/write_bytes/compute_ops`、buffer 读写集合、stage 锚定要求。
- `dependencies`：SSA、RAW/WAR/WAW、DMA completion/ready、opaque fence，以及仅同迭代模式使用的同步顺序约束。
- `buffers`：类型、规模、memory space、读写者、全部使用者、是否允许版本化、`liveness` 所有权/迭代局部性证据。
- `dma_summary`：当前区域直接/嵌套的异步 issue、wait 数，以及仍为同步的 GM→SHM copy 数。
- `limits`：当前 materializer 的 `max_stages=2`、`max_slots=2`。

规模来自静态类型，未知用 `-1`；GEMM 计算量估计为 `2*M*N*K`。
嵌套 region 保留为原子 op，其规模只是内部 body 聚合估计，不等于动态执行总量。
别名无法证明独立时保守保留依赖；不会仅凭 SSA 名字不同就判定内存不冲突。

`schedule.json` 的 `schema` 是 `deepgengraph.cpsat.schedule`，`version=1`：

```json
{
  "schema": "deepgengraph.cpsat.schedule",
  "version": 1,
  "fingerprint": "<problem 中的 SHA-256>",
  "loop_id": "Attn_p2:0",
  "mode": "pipeline",
  "status": "OPTIMAL",
  "operations": [
    {"id": 0, "stage": 0, "order": 1},
    {"id": 1, "stage": 1, "order": 0}
  ],
  "buffers": [{"id": 0, "slots": 2}]
}
```

此片段仅说明字段，完整方案必须列出每个 op/buffer。
`order` 是稳态每一拍中的总顺序（0..N-1 的排列）；`stage` 是逻辑迭代偏移层级。
例如 stage0 的 DMA 和 stage1 的消费放入同一拍后，执行的是下一 tile 的 DMA 与当前 tile 的消费。
同 stage 依赖必须满足 order；普通 SSA 跨 stage 不搬运，只有已支持的 token/版本化存储可以跨 stage。
Buffer 槽数由写入到最晚使用的 stage 距离推导，并被 Python/C++ 双重检查。

`model` 下的 period、phase、latency 是求解诊断信息，MLIR 不将它们解释成实际等待时间。
MLIR 重新分析当前 IR，验证 fingerprint、全部 ID、依赖、锚点及槽数，在克隆模块验证成功后才提交修改。
错误方案不会部分修改原模块。

## 目标与耗时模型

默认 `DMA=10, SHM→REG=5, ALU=1, WMMA=1`，DMA issue=1、wait 指令开销=1。
同步 op 占用串行 issue 流，DMA 在 issue 后异步持续；wait 的 stall 来自 completion 约束。
模型包含周期边界上的 DMA 并发容量和后端 wait-all 行为。

依次最小化：估计稳态 period → 新增 SHM 字节数 → 相对原始顺序的逆序对数。
每个目标证实最优后才固定它、继续下一目标；超时但已有可行解时返回 `FEASIBLE`。
`model.lexicographic_solves` 保存各层目标的状态和 bound。
这些相对权重足以探索方案，但 `OPTIMAL` 只代表该成本模型内的最优，不代表实际 GPU 最快。

`--lds-budget` 现在约束生命周期峰值。MLIR 导出“自有、单次完整写入、不逃逸”的证据，
Python 在每个候选方案上重新计算 `begin = stage(writer)*N + order(writer)`，
`end = max(stage(user)*N + order(user) + 1)`；`N` 是 loop 内 op 数。
同一 op 的输入和输出同时占用存储；DMA 从 issue 开始占用 buffer，至少到真实 wait 之后才结束。
生命周期使用离散 op 顺序，不会因为估计耗时或虚构 idle gap 而提前释放内存。
仓库已有的 C++ `analyzeShmLiveness` 主要分析 `memref.alloc` 及其逻辑/物理访问；
本阶段仍有 `frisk.alloc_buffer` 和计算产生的 tile，因此这里导出 Base 层所有权事实，
由 Python 随候选顺序重算，后端再对实际 allocation 做自己的 liveness/reuse 分析。

动态大小在指定预算时拒绝求解；无法证明迭代局部、别名/逃逸未知的 buffer 全程保留。
跨迭代 FIFO 的所有物化槽也全程计入预算：当前后端不会对这些槽做生命周期复用，
即使其逻辑内容并非同时存活，也不能把双槽的物理容量算成单槽。
旧 v1 JSON 没有 `liveness` 字段时，只信任原先 `rotatable` 的完整 DMA 所有权证明，其余保守保留。

```bash
# 小模型：直接将活跃区间和字节需求加入 CP-SAT cumulative。
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --lds-budget 65536 --shm-check constraint

# 大模型：先最小化 II，再检查内存；拒绝超限的 stage/order 组合后继续求解。
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --lds-budget 65536 --shm-check postcheck
```

默认 `--shm-check auto`：有预算且周期区间数不超过 `SHM_MAX_CUMULATIVE_INTERVALS` 时加约束，
否则使用 post-check。未设置预算仍输出生命周期报告，但不限制容量。
`sum` 保留旧版 `sum(bytes * slots)` 模型，方便对照。
post-check 使用 stage/order 的 no-good cut，避免只修改 phase 后重复检查同一方案；
只有通过内存检查的候选才会固定 II 并进入次级目标。
耗尽总时间或 `SHM_MAX_POSTCHECK_REJECTIONS` 时，返回已有合法方案；若一个也没有则非零退出，
不会把超限方案作为可用解输出。达到重试限额不等于证明不存在合法解。

输出 `model.shm_liveness` 包含峰值、峰值位置、各 buffer 区间、保守保留原因和预算检查结果。
这是 Base 层的逻辑容量规划，尚不包含物理对齐/布局/packing、后续 scratch 和 loop 外无关存储；
也没有替后端分配共享地址。后端仍需重新做物理存储分配及跨 warp 复用交接 barrier 检查。
现阶段未建模寄存器占用、具体 WMMA 发射吞吐、bank conflict 或 fragment 指令调度。

## 当前支持边界

同迭代模式支持独立 op 重排，不生成额外 buffer；未知副作用和嵌套循环形成 fence。
真实 wait/barrier 不能被估计耗时替代，逻辑阶段的 DMA/wait 相对同步约束保守保留。
非 identity 方案标记 `pipeline.scheduled`，使尊重该标记的后续 DMA 调度器跳过此作用域。
identity baseline 回读是真正的 no-op，不增加标记，因此原有后续优化保持原样。

跨迭代 v1 使用现有 `PipelineSchedule` 的 prologue/steady/epilogue 与槽轮转实现，限两级流水、双槽：

- 非空、step 整除区间的常量 affine loop；所属函数有 `thread_num`，外层为常量 affine loop。
- 没有 opaque/nested region；常量边界保证集体 barrier 一致执行。
- 版本化对象为自有 SHM `frisk.alloc_buffer`，单个完整 tile 的异步 DMA 写者、唯一局部 wait、直接消费者，无视图或逃逸，源内存不可变。
- 非版本化可变数据、loop carrier 和计算保持在最后 stage。
- 支持普通 SSA yield 的多 carrier；旧 destination-style `frisk.copy ... to iter_arg` 限单 carrier。
- 空循环、动态边界或复杂别名仍可用 intra；跨迭代会说明原因并拒绝。

因此当前 attention 的最外层 mainloop 含嵌套 QK/PV 循环时，先用 intra，不能把嵌套 region 当成已经支持跨迭代的任务。
后续扩展可在此 JSON 契约下增加递归依赖、更多版本和更通用的展开器，不必改变 Python/MLIR 分工。

## 已保存的最佳 case

```bash
python python/smt/capture_baselines.py --verify-lowering
```

保存位置是 `baselines/{causal,noncausal}/`，每项包含输入、`base.mlir`、`problem.json`、
`schedule.json` 和 `provenance.json`。当前选用：

| case | 历史已测产物 | 记录的 median |
|---|---|---|
| causal | `causal_v_double_buffer_reverse/group4.ll` | 1.176870 ms |
| noncausal | `noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll` | 1.505759 ms |

Base 原始方案均为 stage0 的原始顺序；历史最快 kernel 的后续 V 双缓冲、K 预取、LLVM 指令排序不在这份 Base 方案中。
provenance 另存历史 artifact SHA-256、完整配置和已有的 late-schedule 元数据。
重新从当前编译器导出的 Base 快照不等于历史 LLVM 产物的逐指令复现。
`--verify-lowering` 比较当前编译器不导入/identity 导入后的最终 LLVM 是否字节一致；不重新做 GPU 性能测试。
