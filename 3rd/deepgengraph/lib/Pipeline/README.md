# Pipeline 模块（FA3 风格软件流水）

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
| `lib/Pipeline/test/attn_p2_pipeline_scheme.mlir` | 输入：`test/test_friskBase.mlir` + 2-stage/2-buffer 方案 |

输入文件是 `test/test_friskBase.mlir`（`createConvertFriskToBasePass` 在
`@Attn_p2` 上的输出）**逐字复制**，op 一个都没改，只在 `affine.for` 上加了
方案标记。循环体有 17 个顶层 op。

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
  pipeline.stage = array<i64: 0,0, 1,1, 1, 1,1,1,1,1,1,1,1, 2,2,2, 1>,
  pipeline.order = array<i64: 0,0, 1,1, 2, 4,4,4,4,4,4,4,4, 3,3,3, 5>
}   //                   K    V    QK  softmax(mask..copy)  PV   rowsum
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

## 使用

```bash
cmake -S . -B build && cmake --build build --target FriskPipelineTest -j8
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --idempotence
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --bad-annotation
```

默认模式打印变换后的 IR 并做结构自检；`--idempotence` 验证跑第二遍无变化；
`--bad-annotation` 验证非法标注（重复 order）被拒绝。

默认模式还会做一次**同步检查**：解析输入和 `test/test_friskBase.mlir`（从输入
文件往上找），把输入上的 `pipeline.stage`/`pipeline.order` 去掉后逐行比对打印出
来的 module，所以注释和排版不影响结果，只有真正的 IR 改动会被报出来（会指出第
一行差异）。做法是「输入 = 冻结 IR 逐字复制 + 循环标记」，这条检查保证两边不会
悄悄漂移。也可以单独跑：

```bash
./build/lib/Pipeline/FriskPipelineTest lib/Pipeline/test/attn_p2_pipeline_scheme.mlir --sync-check
# 或指定基准文件
./build/lib/Pipeline/FriskPipelineTest <input.mlir> --sync-check test/test_friskBase.mlir
```

在代码里作为普通 pass 用：

```cpp
pm.addNestedPass<func::FuncOp>(mlir::pipeline::createPipelineSchedulePass());
// 或 mlir::pipeline::applyPipelineSchedule(func, {.peeledTiles = 1, .maxSlots = 2});
```

## 限制

* 只支持 2 槽（`maxSlots = 2`），即一个流水级的距离；
* 跨流水级的值必须是 `frisk.alloc_buffer` 的结果，否则报错；
* 稳态/epilogue 的槽句柄是由 `scf.if` 选出来的 **memref** 值，语句通过它引用
  buffer；下游要能把 memref 类型的 `scf.if` 结果一路处理下去（这种值的
  defining op 已经不是 `frisk.alloc_buffer` 了）；
* 循环必须带 `iter_args`；
* 每次运行只重写 func 里第一个被标注的循环（想处理多个就多跑几次）。
