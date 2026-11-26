# Frisk Fragment CP-SAT 调度框架

MyTest 在 `ConvertFriskBaseToFragmentIR` 和累加器融合之后、`ConvertFragmentToThreadIR` 之前交换 JSON。
此时 LowerInfo/shared layout 已确定，寄存器 fragment 与真实 SHM 访问已经分离：

```text
MyTest：完整展开 mainloop 内静态循环、分析 Fragment 依赖和真实存储 → problem.json
Python OR-Tools CP-SAT：联合求 op 顺序、stage、buffer 槽数 → schedule.json
MyTest：重新验证方案 → 同迭代移动，或 PipelineSchedule 展开与 buffer 轮转
```

使用当前 conda base 即可；开发验证版本为 Python 3.13、OR-Tools 9.15.6755。
没有使用 SMT 的 Z3 接口；这里直接使用 Google OR-Tools 的 CP-SAT 整数约束接口。

## 文件与入口

| 文件 | 作用 |
|---|---|
| `run_pipeline.py` | 一键导出 → CP-SAT 求解 → MyTest 验证导入并完整 lowering |
| `cpsat_scheduler.py` | CLI、`solve(problem, ...)`、整数规划与目标函数 |
| `cost_model.py` | 可替换的相对耗时估计 |
| `makespan.py` | 有限 N 的周期调度完成时间、常量次数推导与动态次数上界 |
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

### 一键运行

在仓库根目录执行（使用安装了 OR-Tools 的 Python 和已构建的 MyTest）：

```bash
python python/smt/run_pipeline.py python/smt/baselines/noncausal/input.mlir \
  --provenance python/smt/baselines/noncausal/provenance.json \
  --mode intra --time-limit 15 -o /tmp/cpsat-kernel.ll
```

自定义输入时，把原 MyTest 第三个位置参数（输出文件）之后的调优参数放在 `--` 后：

```bash
python python/smt/run_pipeline.py /path/to/input.mlir \
  --work-dir /tmp/my-cpsat-run --mode intra --time-limit 30 \
  -- --k-tail-prefetch=4
```

脚本自动执行两次 MyTest（第二个位置参数固定为 `0`），共用输入、调优参数和环境。
第一次停在 Fragment 导出点，求解成功后第二次导入方案，由 MyTest 验证并继续完整 lowering。
支持 `--loop`、`--mytest`，以及求解器的 `--costs`、`--scale-work`、`--max-stages`、
`--max-slots`、`--lds-budget`、`--dma-capacity`、`--shm-check`、`--workers`、`--seed`。
`--provenance` 读取 baseline 的完整 MyTest 参数和环境；`--` 后的额外参数追加到其后。
未指定 provenance 时继承当前环境；`FRISK_GM_TO_SHM_MODE` 未设置时使用 `async`。
`--cpsat-*` 流程控制参数由脚本管理，不接受透传。

默认每次创建 `generated_ir/cpsat_runs/run-*`；显式 `--work-dir` 必须是新目录，避免误用旧产物。
其中保留 `problem.json`、`baseline.json`、`schedule.json`、`base.mlir`、`scheduled.mlir`、
最终 `kernel.ll` 和 `export.log` / `solve.log` / `lower.log`，日志首行包含实际命令。
两次 MyTest 各自在独立子目录中写诊断文件。任一步失败即非零退出并指明日志，
不会继续执行后续步骤；`-o` 指定的目标只在所有阶段成功且最终 LDS 检查通过后替换。
LDS 硬上限默认 **16 KiB（16384 字节）**，用 `--lds-budget 32768` 等字节数覆盖。
同一预算同时约束求解器的逻辑 SHM 峰值和最终 LLVM 的物理 LDS 分配。
最终检查结果写入 `lds_check.json`；超限或无法确定容量时非零退出，候选 LLVM 留在
`lower/kernel.ll` 供诊断，不发布顶层 `kernel.ll`，也不覆盖 `-o` 的已有文件。
物理检查支持 MyTest 当前输出的静态 `addrspace(3)` 字节数组，计入对齐，保守累加
模块全部 LDS 全局对象；未知形式或动态 LDS 明确拒绝。逻辑模型和物理分配可能不同，
因此仅运行 `cpsat_scheduler.py` 得到的方案仍需经过此完整流程才能确认物理容量合规。
这里的验证是调度合法性和编译验证，不包含 GPU 数值或性能测试。

### Fragment 展开与调度所有权

CP-SAT 的第一优化目标默认是有限 N 的周期调度 makespan，随后依次优化 LDS 占用和对源顺序
的改动；只有前一个目标已证明最优才进入下一个目标。`FEASIBLE` 不表示最优 makespan。
`--objective ii` 保留旧稳态 II 目标用于对照。makespan 覆盖所选循环已导出的执行节点和结果完成延迟，
包含 stage/phase 导致的填充与排空跨度；不计 lowering 额外插入但未导出的边界工作、循环外
setup/finish、实际 occupancy 或精确的目标 GPU 时延，不能作为整个 kernel 的实测运行时间。

Fragment LDS→MMAC 链默认不再施加固定两批 read-ahead 顺序约束：读取可以在依赖、寄存器、
LDS 和 DMA 资源约束允许的范围内提前。`--lds-read-ahead` 可显式启用旧策略做对照：
下一批 LDS 读在当前批消费者前发出，当前批 MMAC 完成发射后才能打开第三批读取。
小型 `load_frag` 两个一批，矩阵读一个一批；这是发射顺序约束，不会用虚拟 idle 或
零成本 view 假装隐藏延迟。规则遇到数据/地址依赖环或不支持的消费者时跳过该组。
所有方案仍须满足寄存器与 LDS 容量硬约束，预算不足时不会静默放宽上限。
规则保存在 `schedule.json` 的 `model.lds_read_ahead` 中，并独立检查最终导出顺序。
`--no-lds-read-ahead` 仍兼容，等同于当前默认值；求解器和 `run_pipeline.py` 均支持两个选项。
这些约束是可调的调度策略，会缩小搜索空间，不能保证在所有 GPU/形状上更快。
关闭规则后，大图仍会比较无约束构造和旧预读顺序两种初始提示，按当前主目标选择较好者。
后者只提供 `add_hint` 初值，不向优化模型添加两批策略边；最终顺序可以超出两批。
`model.initial_hint_policy` 记录提示来源，`model.lds_read_ahead.rules.enabled/edges` 才表示实际策略约束。

MyTest 的 CP-SAT 路径自动完整展开所选 mainloop 内所有静态 affine 循环，保留外层
mainloop。不再使用 16 次或 2048 新增操作阈值；动态/不支持的内层循环明确报错。
展开保留 SSA iter_args/yield 依赖，在克隆模块上完成，导出/导入重复同样准备并校验 fingerprint。
`--unroll-max-trip-count` / `--unroll-max-expanded-ops` 仅保留旧 Base bridge 的兼容语义，
不会限制 MyTest Fragment 路径。独立 `FriskScheduleJSONTest --fragment` 可测试 Fragment 输入。

CP-SAT 路径屏蔽 AutoPipeline、FriskFragIRReorder、
ScheduleAsyncCopies，以及后续 V 双缓冲、cross-BN、QK/PV interleave、LDS/fragment/packet
prefetch、Softmax fragment schedule、LazySoftmaxBase、局部归约/延迟 lane sum 重组、
寄存器 math-consumer 重排、IRDeepOptimize 的 reorder、LLVM cleanup/O3 和全局 barrier 优化。
即使 provenance 中开启了这些参数，此路径也会屏蔽；MyTest 日志会明确提示调度所有权。
保留数值 lowering、必要的 layout/packing、存储生命周期重算、复用与同步插入。
LDS 访问合并限制在计算/同步/写入边界内；`--register-lifetimes=1` 在此路径仅重建纯地址
表达式并删除死代码，不移动 DMA/MMA。地址复用窗口最多覆盖 8 次访存，避免展开后的长块
把大量地址值保留到循环末尾。局部重复 wait/barrier 清理在 CP-SAT 路径始终启用。
非 CP-SAT 路径保持原行为。
循环简化和 invariant fragment retention 在导出/指纹生成前执行；它们不会覆盖求解结果。
常驻值计入寄存器预算，已经结束的前导 LDS 暂存经用途/异步完成证明后可复用 periodic
pool，并在交接处插入 barrier；不能证明生命周期互斥的存储仍保持独立。

最终物理 VGPR/spill 必须检查目标后端汇编；Fragment/Thread 活跃值分析不能代替寄存器分配。
以下检查默认要求 VGPR 数量不超过 250、LDS 不超过 16 KiB、无 spill、无 private/scratch：

```bash
python python/smt/check_assembly_resources.py kernel.amdgcn.log --vgpr-budget 250 --lds-budget 16384
```

支持 UTF-8 和带 BOM 的 UTF-16。失败退出码为 1，缺失元数据退出码为 2；预算可由用户调整。
`v0..v255` 表示 256 个寄存器，后端分配粒度也可能使报告数量向上取整。
这是后端验收检查，当前 CP-SAT 仍未联合约束物理 VGPR，不能将 IR 生成成功视为无 spill。

Fragment 新导出包含 `register_liveness`：按底层 origin 的 32-bit 字去重计算
pack/insert/extract 的切片活跃期，并把循环输入映射到前一次迭代的 yield 内容。
CP-SAT 对周期活跃区间加入 cumulative 硬约束，默认 `--vgpr-budget=250`
和 `--register-reserve=32`（数据峰值加预留不超过 250）。预留用于地址和后端临时值，
不等于精确的物理分配；SGPR 不与 VGPR 相加。求解后再次按实际 order 独立检查，
报告位于 `schedule.json/model/register_liveness`。未知载荷拒绝建模；动态或未更新的
循环切片保守地保留整周期，
显式请求预算时缺少活跃期数据也会失败。旧 Base 导出未显式请求预算时仍兼容运行。
合并访存由固定 layout 和边界内 packing 实现；求解器目前不搜索 lane layout、
全局访存 transaction 或银行冲突，因此建模周期不能当作 GPU 性能预测。
静态 pack/insert/extract 等表示操作导出为 `zero_issue`，仅用亚时间单位 tick 保持
唯一 order，全部 tick 的总成本小于一个模型时间单位；动态 extract 仍计费。
`shm_issue`（默认 1）与 `shm_reg` 结果延迟（默认 5）分离，SSA 消费者必须等结果，
中间允许插入独立工作。物理 pool 的 writer handoff barrier 也计入发射成本。
这些默认值仍是相对代价，尚未用 gfx936 测量校准，也没有细粒度 MMAC/VALU 多发射模型。
大图初始解优先把下一迭代的 DMA 放在尾部 LDS 读完成之后，并把本轮下一 tile 的
DMA 提前到上一 tile 的 ready 之后；依赖或周期 packing 不合法时退回保守初始解。
这些只是 CP-SAT hints，不固定最终决策。过早占住下一迭代 buffer 会阻止本轮双缓冲。
可用 `compare_llvm_schedule.py best.o3.ll candidate.o3.ll -o comparison.json`
对比自然循环内的 DMA/MMAC/LDS/wait/barrier；统计会明确列出条件分支块，
不会把 peeled epilogue 的静态副本误算成每轮两倍工作量。
若安装了目标兼容的后端，可在 `run_pipeline.py` 上追加
`--llc /path/to/llc --mcpu gfx936`，自动编译并验收汇编；超限或出现 spill 时不会发布
`kernel.ll` 或覆盖 `-o` 的已有输出。不提供后端时 `assembly_check.json` 明确记录物理
寄存器尚未验证。

`fragment-origin-words-v2` 将窄类型 ALU/转换结果按每个标量一个寄存器计费，
不提前假定独立的 f16 转换结果已经打包；LDS/矩阵读取仍按已打包的 32-bit 字计费。
v1 导出仍可读，但它的理想打包估算可能偏低。250 是 VGPR 估算硬预算，
不是精确物理寄存器分配证明；地址、布局降低和后端临时量仍需预留及汇编验收。

CP-SAT preparation 在分配 op ID 之前化简静态 extract/insert 切片、pack/unpack
和同类型 cast，解除整块向量拼装产生的伪依赖。仅折叠可证明完全包含或不相交的
静态切片；动态选择、部分重叠保留原依赖。这一步不移动计算 op。新的 source
fingerprint 会使旧 schedule 失效，应重新导出和求解。
`register_hint.py` 在求解器内生成优先消费活跃值的拓扑初始解，保持预取模板的
内存事件顺序，且只将通过寄存器扫描及周期 LDS 检查的 hint 交给 CP-SAT。
最终 order/stage 仍由 CP-SAT 决定并验证；不启用后续 Fragment/Thread 调度 Pass。
`issue_units` 估算向量 ALU/reduction 展开后的每线程指令数量，MMAC 的
`wmma_latency`（默认 4）与发射成本分离，所有 SSA 边保留结果延迟。
这些是可调的相对成本，并非 gfx936 测量值。

```bash
python python/smt/run_pipeline.py /path/to/input.mlir \
  --provenance /path/to/provenance.json \
  --mode intra --lds-budget 16384 --time-limit 120 --workers 8 -o /tmp/fragment.ll
```

`base.mlir` 是兼容旧文件名的 **Fragment 调度前快照**，`scheduled.mlir` 是调度后快照。
Fragment 图不再限制 256 个节点；大图使用 NoOverlap/phase 顺序编码，避免成对排序变量的
平方增长。仍逐项检查 SSA、读写别名、DMA 完成和 LDS 生命周期；不会用估计时间代替真实 wait。
大图的第三目标是相对源顺序时间位置的偏移，小图仍使用逆序对数；两者均在主目标/存储大小之后。

`pipeline` 仍允许全 stage 0。非零 stage 必须对应实际缓冲版本重叠，不再仅因分配操作被提前而展开流水。
`--require-cross-iteration` 进一步排除全 stage 0，要求至少一个 buffer 使用多个槽；它只适用于
`pipeline`，无可行解或超时无解会失败，不会静默退回 intra。也可直接传给 `cpsat_scheduler.py`。
展开增加求解规模，可能需要更长时间或多个 worker；跨迭代还会增加 LDS 占用，实际性能仍需测量。

### 手动分步运行

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

# 1. 导出到 problem.json，保存原始顺序，停在 Fragment 阶段。
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
`--max-stages`、`--max-slots` 覆盖。SHM 默认 16384 字节（与 driver 共用
`lds_budget.py` 的默认值），其余默认 4 / 2 / 2。不支持用 `None` 取消 LDS 约束；
`--baseline` 导出的原顺序也必须满足给定逻辑预算，MyTest 导出的 `baseline.json` 仅为诊断快照。
旧 `VGPR_LIMIT_PER_THREAD` / `SGPR_LIMIT_PER_WAVE` 常量仍不约束物理寄存器，
设为非 `None` 会明确报错；新的 Fragment 数据活跃期约束使用 `--vgpr-budget` /
`--register-reserve`。求解输出的 `model.resource_limits` 记录实际采用的限额。
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

导出点位于 `ConvertFriskBaseToFragmentIR` 之后。启用 CP-SAT 且未设置
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

默认依次最小化：有限 N 的 makespan → 物理池大小（旧保守路径为新增 SHM 字节数）→ 相对原始顺序的变化。
每个目标证实最优后才固定它、继续下一目标；超时但已有可行解时返回 `FEASIBLE`。
`model.lexicographic_solves` 保存各层目标的 name、状态和 bound（时间目标使用整数 ticks）。
这些相对权重足以探索方案，但 `OPTIMAL` 只代表该成本模型内的最优，不代表实际 GPU 最快。

设真实执行节点的逻辑起点 `a[v] = stage[v] * II + phase[v]`，结果完成延迟为 `L[v]`：

```text
D = max(a[v] + L[v]) - min(a[v])
T(N) = (N - 1) * II + D
```

max/min 只覆盖有真实 issue 成本的节点，不让零成本 alloc/view 等表示节点决定计时边界；
异步任务的末尾结果完成延迟仍计入。N=0 或全为零成本节点时 T=0。
这是有限次周期时刻表的完成时间，不是简单的 `N * II`，允许较大 II 换取更短的跨度 D。
现有周期资源约束保持不变；尚未独立压缩物化后 prologue/epilogue 的执行时刻表。

常量循环从 `loop.lower/upper/step` 自动推导 `N=max(0,ceil((upper-lower)/step))`。
动态次数使用 `loop.trip_count_upper_bound` 或 `--trip-count-upper-bound N`，该值须覆盖
所有合法 bid/tid；不能把非单调表达式直接代入最大的 ID。当前导出器只自动提供常量边界，
动态 case 需调用者提供已知上界；缺少上界会报错，不会静默退回 II。若常量 N 已知则优先使用精确 N，
并拒绝比它更小的上界。次数上界仅用于成本目标，不会放宽动态循环的跨迭代物化限制。

输出记录 `model.objective/trip_count/trip_count_source/makespan/iteration_span`，
`model.finite_horizon` 保存公式、适用范围及整数 ticks。`period` 仍保留为诊断值。
已知 N 时选择 `--objective ii` 也会记录 makespan，便于在相同成本口径下对照。

```bash
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --mode pipeline --objective makespan --time-limit 120
# 仅当循环次数动态、且已知合法 bid/tid 范围内的最大次数为 32 时：
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --objective makespan --trip-count-upper-bound 32
```

`--lds-budget` 现在约束生命周期峰值。MLIR 导出“自有、单次完整写入、不逃逸”的证据，
Python 在每个候选方案上重新计算 `begin = stage(writer)*N + order(writer)`，
`end = max(stage(user)*N + order(user) + 1)`；`N` 是 loop 内 op 数。
同一 op 的输入和输出同时占用存储；DMA 从 issue 开始占用 buffer，至少到真实 wait 之后才结束。
生命周期使用离散 op 顺序，不会因为估计耗时或虚构 idle gap 而提前释放内存。
仓库已有的 C++ `analyzeShmLiveness` 主要分析 `memref.alloc` 及其逻辑/物理访问；
MyTest Fragment 路径按 `memref.alloc` 的物理类型和访问导出所有权事实；旧 Base bridge 仍支持
`frisk.alloc_buffer` 和计算 tile，
由 Python 随候选顺序重算，后端再对实际 allocation 做自己的 liveness/reuse 分析。

动态大小在指定预算时拒绝求解；无法证明迭代局部、别名/逃逸未知的 buffer 全程保留。
Fragment 导出的 `storage_contract=periodic-offsets-v1` 支持联合物理规划：当所选循环的
SHM 均为已证明所有权的完整 DMA buffer 时，pipeline 求解同时决定 stage/order 和
每个 buffer 在偶数、奇数逻辑迭代中的字节偏移。CP-SAT 的二维不重叠约束保证
“生命周期 × 物理地址范围”不能同时相交，包含跨周期和两轮后再次使用同一版本的冲突。
同一 buffer 的两版本、以及不同 buffer 的版本，均可在生命周期不重叠时使用同一地址。
槽位偏移满足 allocation 对齐，整个物理池不超过用户 LDS 预算。
旧 Base/缺少所有权证明的路径仍保守地将跨迭代 FIFO 的所有物化槽全程计入预算。
旧 v1 JSON 没有 `liveness` 字段时，只信任原先 `rotatable` 的完整 DMA 所有权证明，其余保守保留。

`schedule.json` 的 `storage` 字段包含 `pool_bytes`、`alignment`、`budget_bytes` 和
`buffers: [{id, offsets: [even, odd]}]`。Python 和 C++ 导入端分别按照最终离散顺序校验
六个相邻迭代的版本冲突。物化器为每次访问选择对应偏移，处理 prologue、steady loop 和
epilogue；layout packing 后严格转换为同一个 byte pool 上的 views，不再重新分配这些 FIFO。
每次覆盖写前插入 block barrier，原有 DMA token、wait/readiness 同步保留。
对称偏移相同也合法，例如先读完当前 K，再覆盖成下一轮 K，可只用一个物理 K 槽。

```bash
python python/smt/run_pipeline.py input.mlir --provenance baseline/provenance.json \
  --mode pipeline --require-cross-iteration --lds-budget 16384 --time-limit 120 --workers 8
```

`--require-cross-iteration` 禁止退回全 stage=0 方案。大图先构造单块 lookahead 的搜索提示，
使用独立 CP 求解补齐提示中的资源和地址变量；最终优化仍允许改变所有调度/地址决策。
`FEASIBLE` 表示找到合法方案，不保证性能最优，仍需 GPU 正确性和性能测试。

```bash
# 小模型：直接将活跃区间和字节需求加入 CP-SAT cumulative。
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --lds-budget 65536 --shm-check constraint

# 大模型：先优化主目标，再检查内存；拒绝超限的 stage/order 组合后继续求解。
python python/smt/cpsat_scheduler.py problem.json -o schedule.json \
  --lds-budget 65536 --shm-check postcheck
```

默认 `--shm-check auto`：有预算且周期区间数不超过 `SHM_MAX_CUMULATIVE_INTERVALS` 时加约束，
否则使用 post-check。未指定预算使用 16 KiB 默认硬限制。
`sum` 保留旧版 `sum(bytes * slots)` 模型，方便对照。
小图 post-check 使用 stage/order 的 no-good cut；大图使用 stage/phase，可能重试同一顺序的
不同时间安排，因此大图优先使用 `constraint`；
只有通过内存检查的候选才会固定主目标并进入次级目标。
耗尽总时间或 `SHM_MAX_POSTCHECK_REJECTIONS` 时，返回已有合法方案；若一个也没有则非零退出，
不会把超限方案作为可用解输出。达到重试限额不等于证明不存在合法解。

输出 `model.shm_liveness` 包含峰值、峰值位置、各 buffer 区间、保守保留原因和预算检查结果。
这是布局已确定后的所选循环容量规划，仍需最终检查覆盖对齐、后续物化和循环外存储；
有 `storage` 的方案已指定共享地址，后端验证 packing 没有改变容量并严格使用这些偏移；
没有 `storage` 的方案仍由后端分配。最终 LLVM 检查继续覆盖后续 scratch 和循环外存储。
现阶段未建模物理寄存器占用、具体 WMMA 发射吞吐或 bank conflict。
未显式物化 LDS scratch 的非 shuffle `convert_layout_frag` 会拒绝导出，不能按零字节放行。

## 当前支持边界

同迭代模式支持独立 op 重排，不生成额外 buffer；未知副作用和嵌套循环形成 fence。
真实 wait/barrier 不能被估计耗时替代，逻辑阶段的 DMA/wait 相对同步约束保守保留。
非 identity 方案标记 `pipeline.scheduled`，使尊重该标记的后续 DMA 调度器跳过此作用域。
identity baseline 回读不移动节点；MyTest 的整个 CP-SAT 路径仍屏蔽旧调度 Pass。

跨迭代 v1 使用现有 `PipelineSchedule` 的 prologue/steady/epilogue 与槽轮转实现，限两级流水、双槽：

- 非空、step 整除区间的常量 affine loop；所属函数有 `thread_num`，外层为常量 affine loop。
- 没有 opaque/nested region；常量边界保证集体 barrier 一致执行。
- 版本化对象为自有静态 SHM `frisk.alloc_buffer` 或 `memref.alloc`，单个完整 tile 的异步 DMA 写者、唯一局部 wait、直接消费者，无视图或逃逸，源内存不可变。
- 非版本化可变数据、loop carrier 和计算保持在最后 stage。
- 支持普通 SSA yield 的多 carrier；旧 destination-style `frisk.copy ... to iter_arg` 限单 carrier。
- 空循环、动态边界或复杂别名仍可用 intra；跨迭代会说明原因并拒绝。

Fragment 全展开后仍需通过跨迭代合法性、buffer 版本和预算检查；能重排不代表必然能
在同一 LDS 预算下实现跨迭代重叠。`--require-cross-iteration` 无解时不会静默降级。

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
这些已保存的 Base JSON 属于历史接入点，不能直接导入新的 Fragment 接入点；需要使用原输入
和 provenance 重新导出。当前重新捕获的是 Fragment 快照，不等于历史 LLVM 产物的逐指令复现。
`--verify-lowering` 比较相同 CP-SAT 路径上仅导出/identity 导入后的最终 LLVM 是否字节一致；不重新做 GPU 性能测试。
