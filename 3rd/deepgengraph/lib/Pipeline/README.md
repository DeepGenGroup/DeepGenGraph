# Pipeline 模块（FA3 风格软件流水）

## 流水线清理 Pass 与共享内存复用

两个清理 Pass 独立于 attention 的尺寸、函数名和 Q/K/V 命名：

* `frisk-simplify-loop-control`：对 `affine.for` / `scf.for` 的静态小循环
  完整展开，然后折叠 IV、slot 取模、条件分支、selector 和地址表达式。
  默认只处理带 `pipeline.scheduled` 或 `{frisk.simplify_loop}` 的循环。
  `only-marked=false` 可用于其他小循环；默认 `max-trip-count=8`、
  `max-expanded-ops=2048`（每个函数的展开操作总预算）。动态/unsigned
  bounds、末次增量溢出及超预算循环不展开。保留迭代顺序与 iter_args。
  在 MyTest 中运行于 **FinalizeThreadTiling 之后、LowerFriskFragments
  的 physical packing 和共享内存生命周期规划之前**。过早展开会破坏旧
  layout inference 对循环 carrier 的假设，过晚则会错过存储复用。
* `frisk-coalesce-waits-and-barriers`：在 named readiness 已降为
  `rocdl.wait.asyncmark 0` 和 `gpu.barrier` 后，合并同一 block 中重复的
  wait-all/publication。可以跨越无副作用、可推测执行的普通索引/算术操作，
  不跨越任何内存读写、调用、不透明 asm、新 async group、未知 target op、
  partial wait、region 或 CFG 边界。不同 buffer 的 `frisk.wait_buffer_ready`
  **不会**在 Base 层被视为等价。单独的 `barrier; wait-all; barrier` 也不会
  错删后一个 publication barrier。它与已有基于依赖分析的
  `barrier-optimize` 是不同 Pass，先合并冗余序列，再做原有依赖分析。

MyTest 默认开启，命名参数不改变原有位置参数：

```bash
--simplify-loop-control=1 --coalesce-waits=1
# 单独测同步合并：
--simplify-loop-control=0 --coalesce-waits=1
# 旧的流水线输出：
--simplify-loop-control=0 --coalesce-waits=0
```

独立执行：

```bash
build/test/ThreadTilingTest input.mlir \
  --frisk-simplify-loop-control='only-marked=true max-trip-count=8 max-expanded-ops=2048'
build/test/ThreadTilingTest lowered.mlir --frisk-coalesce-waits-and-barriers
python3 test/check_pipeline_cleanup.py build/test/ThreadTilingTest
python3 test/check_pipeline_cleanup_attention.py --mytest build/test/MyTest
```

原有共享内存复用的限制在 `lib/Analysis/LivelinessAnalyze.cpp`：
`hasLocalCompletion` 只接受同一 block 内、同一 descriptor 的 issue→wait；
`findIterationLocalLoops` 对多个 alias 保留整个循环的生命周期。环形流水线的
prologue/guard/下一轮 wait 和 selector 不满足这些证明条件，所以 Q/K 槽被
标记为 `unsupported or escaping use: frisk.async_copy_gm_to_shm`，V 单独池化。
这不是 `PackSharedMemory`（物理布局变换）没做地址合并，也不是只需要调整
buffer 大小。循环展开后，固定槽位、发射、等待和最后读者可以落入同一
顺序 block，让现有 `reuseSharedMemory` 重新证明异步完成和不重叠的生命周期。
没有放宽动态流水线的安全条件，也没有依赖 `pipeline.scheduled` 属性直接
假定存储可以复用。动态或超预算循环仍可能保守保留独立缓冲。

`test_input_spidD.mlir`、QK32/VO32、2 stages 的导出验证结果（每个 key tile、
每个 workgroup 的 CPU 模型动态指令次数；不是汇编静态条数或 GPU 性能）：

| 配置 | 静态 LDS | wait-all | block barrier |
|---|---:|---:|---:|
| 原始 pipeline | 40 KiB | 12 | 20 |
| 仅合并等待 | 40 KiB | 8 | 16 |
| 仅简化循环 | 32 KiB | 12 | 16 |
| 两项同时开启 | 32 KiB | 8 | 12 |

CPU 延迟 DMA 模型检查所有异步写在 wait 前不可读取/覆盖，并覆盖首个、
第二个和最后一个 query block 的数值。它不能证明独立 wave 时序或实际 GPU
性能；用相同编译选项和 launch 参数对四份导出做上机 A/B。

## 自动 tile 预取：`frisk-auto-pipeline`

输入循环可写：

```mlir
%result = scf.for %dk = %c0 to %cD step %cBK
    iter_args(%acc = %zero) -> tensor<128x128xf32> {
  // block_load Q/K、计算并更新 acc
  ...
  scf.yield %next : tensor<128x128xf32>
} {frisk.pipeline_hint = 2 : i64}
```

`frisk.pipeline_hint` 是循环上的整数属性，值是 `num_stages`。
裸标记 `{frisk.pipeline_hint}` 兼容为 2；1 表示禁用；0、负数和非整数报错。
没有标记时不尝试优化。当前代码生成预算最多 64 stages，超过时保留原循环并输出
remark。SCF→affine 和 tensor→Frisk 的转换会保留该标记。

新 Pass 在 **Frisk Base、thread tiling 和物理 packing 之前**运行。它从循环中
识别完整 tile 的 global→shared `frisk.copy` 或 `frisk.async_copy_gm_to_shm`，
分析地址依赖和内存副作用；不依赖函数名、attention 模式、矩阵尺寸或 GEMM 数量。
`MyTest` 已自动接入，hint 本身就是开关，不受旧的命令行 `pipeline:0|1`
控制；那个参数仍只控制手工 `pipeline.stage/order` 调度。

对于 N stages，每个加载目标分配 N 份 shared buffer，预取距离为 N−1：

```text
prologue: 有效时加载 tile 0 ... N-2
for tile i（原来的 bounds、step、iter_args）:
    wait 当前槽 i % N
    block barrier，确保旧读者已完成
    若 tile i+N-1 存在:
        async load tile i+N-1 -> (i+N-1) % N
    用当前槽计算 tile i，按原顺序更新累加器
```

因此 N=2 是“加载 i+1、计算 i”；N=3 是“加载 i+2、计算 i”。
所有序言/未来加载都有 guard，零次循环不加载，短循环不越界。保留原消费循环，
尾部通过 guard 自然排空，不再单独克隆计算 epilogue。边界检查先比较剩余距离，
再构造未来 IV，避免排空时加法溢出。等待放在未来加载**之前**，因为当前后端
可以将 readiness 降为 wait-all；实际性能仍取决于拷贝布局和目标后端。

成功后移除 hint，在消费循环上设置 `pipeline.scheduled`、`pipeline.num_stages`
及 `pipeline.slots`，重复运行不重复变换。无法证明安全则输出
`automatic pipeline skipped: ...`，保留原循环。当前支持范围：

* Base 层的 `affine.for`，常量正步长，静态或动态 affine 上下界；
  bounds 和外围控制流必须能证明 block-uniform。
* 完整、同形状/元素类型、静态大小的 global tile，目标是循环内独占的
  `frisk.alloc_buffer`；支持同步 copy 的 destination-alias 返回句柄。
* scalar/vector 累加器和多个 iter_args，多个独立带标记的循环。
* 预取地址不能依赖本循环累加器或内存读取。存在未知副作用、外部写入、
  未支持的同步、目标 buffer 别名/逃逸/额外写入、memref iter_args 等情况会跳过。
* 已有手工 pipeline scheme 不叠加自动调度；standalone Pass 不直接转换
  tensor/SCF IR，应先运行既有前端 lowering。

所选槽位沿用完整 shared alias group 的布局约束；最大宽度 DMA 对整组规划，
统一修改所有槽位，防止 prologue/steady 两次规划改变 consumer width 或 kpack。

使用和回归（仓库根目录）：

```bash
cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j2
# 单独处理 Frisk Base IR：
3rd/deepgengraph/build/test/ThreadTilingTest base.mlir \
  --frisk-auto-pipeline --verify-each

python 3rd/deepgengraph/test/check_auto_pipeline.py
OPENBLAS_NUM_THREADS=1 python \
  3rd/deepgengraph/test/check_auto_pipeline_attention.py
```

第一项执行 CPU 递推/轮换验证和拒绝路径验证，覆盖 2/3/4 stages、动态/负下界、
非单位步长、整数边界、零次和短循环、多个累加器、无标记与幂等性。
第二项对 split-D attention 的无标记、双/三缓冲和 BK=32/64 运行完整 LLVM 导出；
执行模型将 DMA 延迟到 wait 才完成，检查槽位地址冲突、读前等待、MMA 执行时的
未完成预取和数值结果。这些是编译与执行模型检查，不是 GPU 计时或独立 wave
调度的硬件验证。

## 手工 stage/order 调度

独立模块，参考 tilelang 的 `T.Pipelined(..., stage = [...], order = [...])`，
把带标注的 `affine.for` 重写成 FA3 Algorithm 2 那种 prologue / steady /
epilogue 结构。不依赖其它 pass，也不改 IR 里的 op，只在循环上读方案、在跨级
buffer 上做多槽化。

## 文件

| 文件 | 作用 |
| --- | --- |
| `include/deepgengraph/Pipeline/Passes.td` | `frisk-pipeline-schedule` pass 声明 |
| `include/deepgengraph/Pipeline/PipelineSchedule.h` | `PipelineScheduleOptions` + 入口 |
| `lib/Pipeline/PipelineSchedule.cpp` | 分析与重写实现 |
| `lib/Pipeline/test/pipeline_schedule_test.cpp` | 结构自检（`FriskPipelineTest`） |
| `lib/Pipeline/test/attn_p2_pipeline_scheme.mlir` | 输入：`test/test_friskBaseDebug.mlir` + 2-stage/2-buffer 方案 |

输入文件是 `test/test_friskBaseDebug.mlir`（`createConvertFriskToBasePass` 在
`@Attn_p2` 上的输出）**逐字复制**，op 一个都没改，只在 `affine.for` 上加了
方案标记。循环体有 22 个顶层 op。

为此顺手修了 `lib/Dialect/Frisk/IR/FriskOps.cpp` 里 `frisk.buffer_view` 的
parser：它的 printer 把下标打印成 SSA 名字代入后的 affine map
（`%arg2[0, (%i + 4) floordiv 8, %j]`），原来的 parser 却按裸操作数列表解析，
导致这种写法根本无法回读。现在改成 `parseAffineMapOfSSAIds`，同时把
attr-dict 挪到 `:` 之前（与 printer 及其它 op 的约定一致）。

## 标注语义

方案挂在 `affine.for` 上，`pipeline.stage` / `pipeline.order` 都是
`array<i64: ...>`：

```mlir
%9 = affine.for ... attributes {
  // 每个顶层 op 一项；同一条语句的 op 重复同一个 (stage, order) 对
  pipeline.stage = array<i64: 0,0, 1,1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2, 1>,
  pipeline.order = array<i64: 0,0, 1,1, 2, 4,4,4,4,4,4,4,4,4,4,4,4,4, 3,3,3, 5>
}   //                   K    V    QK  softmax(mask..copy到P槽)          PV   rowsum
```

* `pipeline.stage`：语句所属流水级。0 最超前，数字越大越靠近当前迭代。
* `pipeline.order`：稳态循环体内的发射顺序，小的先发射。同一循环体内必须构成全序。
* 两个列表长度必须都等于循环体**顶层 op 数**（不含 terminator），且按顶层 op 在
  循环体里的位置下标；相邻且 `(stage, order)` 相同的 op 自动合成一条语句，所以
  同一条语句的 op 重复写同一个 pair。长度对不上会直接报错。
* 注意列表不是按发射顺序排的；上例里 `order` 的 4（softmax）出现在 3（PV）前面，
  就是因为 PV 在循环体里靠后。

即 tilelang 的说法：`stage` 是逻辑流水级（**数字小的先跑**，
`num_stages = max(stage) + 1`），`order` 只决定发射先后，不改变每个语句处理哪个
tile。

### 槽位规则

应用标注后还会检查数据依赖：生产者语句的 `stage` 不能大于消费者的 `stage`
（生产者不能排在消费者之后）；两者同 `stage` 时，生产者的 `order` 必须更小。
违反会直接报错。

* `numStages = max(stage) + 1`，语句 `s` 在稳态第 `i` 步处理 tile
  `i + offset(s)`，其中 `offset(s) = numStages - 1 - s`；槽位
  `slot = tile % slots`，`slots = 1 + (写者 offset - 读者 offset)`。
  `slots == 1` 的 buffer 留在原位（例如 QK 的结果 S 不必物化）。

## FA3 2-stage / 2-buffer 方案

| 语句 | stage | order | 稳态第 i 步处理 |
| --- | --- | --- | --- |
| K view + load | 0 | 0 | tile i+2（提前 2 级，K 双缓冲） |
| V view + load | 1 | 1 | tile i+1（V 双缓冲） |
| QK gemm | 1 | 2 | tile i+1 |
| PV gemm + `acc += PV` | 2 | 3 | tile i（P 双缓冲） |
| mask + softmax + rowsum | 1 | 4 | tile i+1 |
| rowsum -> 寄存器 | 1 | 5 | tile i+1 |

生成的骨架：

```
prologue:   K(0)                     -- 第 1 步
            K(1) + V(0) + QK(0) + P(0)  -- 第 2 步（K(1) 带越界保护）
steady  :   affine.for 0 .. Ub-step   （T-1 次）
              一个 barrier（每一步唯一的同步点）
              1 个 scf.if(isEven)      -- 只按奇偶选出这一步要用的槽
              K load(tile i+2) -> V load(tile i+1) -> QK(tile i+1)
                  -> PV(tile i) 叠加到 acc -> softmax(tile i+1)
epilogue:   barrier + 选槽 + 最后一个 tile 的 PV + 原 tail（div / truncf / store）
```

稳态循环里每步只有顶部一个 barrier：它正好把“本步写”和“上一步读”隔开，
满足双缓冲的距离 2 依赖。发射顺序把 PV 排在 softmax 之前，是为了让
`P·V` 的 MMA 和下一块的 softmax 在硬件上重叠（IR 层面是结构性重叠，
真实重叠取决于下游是否把它降成异步 copy / 双发射队列）。

### 奇偶只用来选槽，不用来复制语句

稳态（和 epilogue）里 tile 的奇偶只有运行期才知道，但没必要时把整段语句复制成
`scf.if` 的两个大分支——那样循环体里每个 op 都会出现两次。这里只对 **buffer**
做选择，而且整个 step 只发**一个**多结果的 `scf.if`：它一次产出这一步所有要用
的槽，两个 `yield` 就是整张槽表

```mlir
// 稳态循环体：一个 if 选出这一步要用的 6 个槽
%44:6 = scf.if %isEven -> (memref<128x32xf16, 3>, memref<32x128xf16, 3>,
                           memref<128x32xf16, 3>, memref<32x128xf16, 3>,
                           memref<64x32xf16, 3>,  memref<64x32xf16, 3>) {
  // 偶块：K 写槽 0 / 读槽 1，V 写槽 1 / 读槽 0，P 写槽 1 / 读槽 0
  scf.yield %kSlot0, %vSlot1, %kSlot1, %vSlot0, %pSlot0, %pSlot1
} else {
  // 奇块：写者和读者互换
  scf.yield %kSlot1, %vSlot0, %kSlot0, %vSlot1, %pSlot1, %pSlot0
}
scf.if %guard { frisk.copy ... to %44#0 }   // K load -> 槽
frisk.copy ... to %44#1                     // V load -> 槽
%qk = frisk.gemm(%qtileScale, %44#2)        // QK 读 K 槽
%pv = frisk.gemm(%44#4, %44#3)              // PV 读 P 槽 / V 槽
frisk.copy %qkme to %44#5                   // softmax 写 P 槽
```

每个槽只归约到 `(buffer, offset % slots)` 这个键上：槽位是
`(parity + offset) % slots`，只跟 `offset` 对槽数取模有关，所以 offset 相差整数
倍槽数的语句共用同一个句柄。后序语句照常只发射一遍、直接用这些槽句柄。
epilogue 同理，只是那边只剩 `offset == 0` 的语句有活干，所以只选 V/P 两个槽。

### `frisk.copy` 的 SSA 句柄就是它写入的 buffer

这一版 Frisk IR 不给 `frisk.copy` 单独写一行了，而是用它的返回值当 buffer 句柄
往下传（`%ktile = frisk.copy %view to %ktile_slot`，gemm 读 `%ktile`）。这种句柄
和它的 dst 指向同一份存储，所以 pass 里凡是处理 buffer 的地方都跟着走：

* 依赖分析把句柄上的 use 算成对 buffer 的 use——否则 K/V/P 的 `maxLead` 会算成 0，
  一个 buffer 都不会双缓冲（句柄 `%ktile` 有 use，buffer 本身没有）；
* `statementTouches` 认句柄，`(buffer, offset)` 的槽才会被选中；
* 发射语句时句柄和 buffer 一起绑到选出来的槽，读句柄的 gemm 才读到正确的槽。

因此「`frisk.copy` 的结果 = 它的 dst」是本模块的一条基本假设；这也是回写
iter_arg 的写法（`%oAcc_next = frisk.copy %22 to %oAcc`），prologue 里同一套逻辑
把它折成稳态循环的初值。

## 使用

```bash
cmake -S . -B build && cmake --build build --target FriskPipelineTest -j8
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --idempotence
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --bad-annotation
```

默认模式打印变换后的 IR 并做结构自检；`--idempotence` 验证跑第二遍无变化；
`--bad-annotation` 验证非法标注（重复 order）被拒绝。

默认模式还会做一次**同步检查**：解析输入和 `test/test_friskBaseDebug.mlir`（从输入
文件往上找），把输入上的 `pipeline.stage`/`pipeline.order` 去掉后逐行比对打印出
来的 module，所以注释和排版不影响结果，只有真正的 IR 改动会被报出来（会指出第
一行差异）。做法是「输入 = 冻结 IR 逐字复制 + 循环标记」，这条检查保证两边不会
悄悄漂移。也可以单独跑：

```bash
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --sync-check
# 或指定基准文件
./build/lib/Pipeline/FriskPipelineTest <input.mlir> --sync-check test/test_friskBaseDebug.mlir
```

在代码里作为普通 pass 用：

```cpp
pm.addNestedPass<func::FuncOp>(mlir::pipeline::createPipelineSchedulePass());
// 或 mlir::pipeline::applyPipelineSchedule(func, {.peeledTiles = 1, .maxSlots = 2});
```

## 限制

* 只支持 2 槽（`maxSlots = 2`），即一个流水级的距离；
* 跨流水级的值必须是 `frisk.alloc_buffer` 的结果，否则报错；
* 语句之间的 SSA 依赖只认「buffer + 它的 `frisk.copy` 句柄」；跨语句直接传一个
  中间 memref（例如某条语句的 gemm 结果被另一条语句当输入）时，只能靠同 `stage`
  的 slot 折叠，跨级则要求它是 `frisk.alloc_buffer`；
* 稳态/epilogue 的槽句柄是由 `scf.if` 选出来的 **memref** 值，语句通过它引用
  buffer；下游要能把 memref 类型的 `scf.if` 结果一路处理下去（这种值的
  defining op 已经不是 `frisk.alloc_buffer` 了）；
* 循环必须带 `iter_args`；
* 每次运行只重写 func 里第一个被标注的循环（想处理多个就多跑几次）。

## 下游 shared 布局与寄存器直传

`FriskToBase/SharedBufferAliases.h` 将完整、同类型的 owned shared allocation
与其 `scf.if`、`arith.select`、SCF/affine 循环携带值和 copy 返回句柄组成别名组。
这表示可能引用哪些存储，不表示槽位里的数据相同。分析不依赖 attention 名称、
tile 尺寸或 pipeline 属性，也用于普通 GEMM 和控制流。

* 布局规划检查整组的 producer/consumer，然后给所有槽位同一 packing 方案。
* 物理 packing 同时改写所有 allocation、选槽/循环描述符及实际访存地址。
  未标记或布局冲突的备选槽位、未知来源、逃逸、未支持的 view/访存会让整组回退，
  不会出现写入布局和读取布局不一致的半转换。
* 计算结果可按 producer 的寄存器布局直接写入选中的存储；GEMM 按自己的布局读取。
* 提升到循环外的归约临时量可在各个代码块内分别直传。必须证明读者观察的是本块的
  归约结果、没有中间写入且逐 lane 地址映射相同；跨块读写和存储逃逸保留原路径。
* 桥接折叠仍比较地址布局和实际线程形状；`block_repeat` 仅描述全 tile 预算，
  不会再仅因 rowsum 广播改变预算而插入一次 LDS 中转。

从仓库根目录运行回归：

```bash
python 3rd/deepgengraph/test/check_shared_buffer_aliases.py
python 3rd/deepgengraph/test/check_attention_pipeline.py \
  3rd/deepgengraph/build/test/MyTest /data2/xsl/install/bin/llvm-as
python 3rd/deepgengraph/test/check_attention_codegen.py
```

第一项包含 CPU 逐坐标读写、两种选槽结果、零次循环和保守回退验证。第二项验证
完整导出的 LLVM IR、相邻迭代 K/V/P 写入到 MMA 读取的地址关系，以及 rowsum 的
FP32 shuffle/add 递推。第三项保护非 pipeline 的原有优化。这些检查不替代 gfx936
上的完整数值验证和计时。

当前 `test_input.mlir`（BM=64、BN=32、128 threads）的修复前后静态对照：
LDS 49,408 → 40,960 B，稳态 LLVM barrier 调用点 8 → 4，K 读取
128 条标量 half load → 16 条 `<8 x half>` load，稳态 P/rowsum 的 FP32 LDS
读写均消除。调度层的单个同步点不代表最终代码只有一个 barrier；copy lowering
仍保留发布同步，不能按 pipeline 标记直接删除。
