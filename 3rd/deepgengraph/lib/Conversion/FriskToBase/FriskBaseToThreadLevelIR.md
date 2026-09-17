# FriskBaseToThreadLevelIR 维护说明

实现位于 `FriskBaseToTheadlevelIR_selfTry.cpp`。建议从文件末尾的
`ConvertFriskBaseToThreadLevelIR::runOnOperation()` 开始阅读，再查看对应阶段注册的 Pattern。
文件名、Pass 工厂函数和 CMake 注册保持原样。

## 输入和转换顺序

输入已经是 Frisk 的 block 级表达，函数具有 `thread_num`，且存在 `gpu.thread_id x`。
`LowerInfoAnalysis` 推导每个 `(buffer, 使用点)` 的布局；存在布局冲突时先插入
`ConvertLayoutOp`。缺少 `thread_num` 的函数会跳过本 Pass。

| 阶段入口 | Pattern | 主要行为 |
| --- | --- | --- |
| `lowerStructuredOps` | `BlockOpConversion` | 收集 load/store，分配 local thread tile，建立线程循环，按每个 buffer 的布局重建索引并克隆 region |
| 同上 | `GemmOpConversion` | DCU 路径：MN 循环选择输出片段，K 循环生成 `WarpMmaRR`，最后合并线程持有的 C vector |
| 同上 | `MaskOpConversion` | 将 region 参数映射为 `starts + block 坐标`，逐元素构造线程 mask vector |
| 同上 | `CopyConvertOpRewrite` | 同 shape 的浮点 dtype 转换：读寄存器片段、`extf/truncf`、写回；本阶段强制处理 shared/shared 的此类 copy |
| `lowerValueFills` | `FillOpRewrite` | 优先将不需要地址语义的 SSA fill 转成线程 vector 常量 |
| `lowerElementwiseOps` | `FriskBinaryOpConversion`、`FriskExp2OpConversion` | 统一输入 tile 类型，生成 `arith` / `math` vector 运算 |
| `lowerReductions` | `ReduceOpConversion` | 线程内归约、warp XOR shuffle、leader 写回 |
| `lowerCopiesAndLayouts` | `CopyOpRewrite`、`FillOpRewrite`、`ConvertLayoutOpConversion` | 展开剩余搬运、填充和布局重排；随后调整循环携带值并清理 `copy_to_reg` |
| `lowerAllocations` | `AllocBufferOpConversion` | Shared 分配为 `memref.alloc`，Local 分配为 `memref.alloca`，最终禁止残留高层操作和临时 cast |

顺序体现生产者与消费者的依赖。例如，逐元素运算必须能看到 GEMM 的 thread vector；
循环携带值的收缩必须发生在提取循环被折叠之前。每个阶段的合法性配置与 Pattern 注册
放在一起。转换失败会立即向 PassManager 报告，避免在未完成的中间 IR 上继续后续阶段。

## 布局与坐标

- **block 坐标**用于访问原来的 shared/global buffer，包含线程 ID 的影响。
- **thread 坐标**用于访问单个线程持有的 vector/local memref。
- `LowerInfo` 是某个使用点的访问布局；`thread_own_data_size` 是线程持有的形状。
  不应仅凭 shape 推断两个值拥有相同布局。

`buildMappedAccessIndices` 对每一维进行以下分解：

```text
iv = (((br * instUnroll + iu) * warpRepeat + wr) * threadWidth + reg)
```

然后向 `LowerInfo::getAffineMap()` 传入：

```text
[tid, br0, br1, iu0, iu1, flattened_warp_repeat, flattened_register]
```

`wr` / `reg` 的展平顺序来自各自的 layout order。
`buildThreadTileOffsetMap` 做线程内部的逆映射，不包含 tid。
`ignoreDim` 对应的访问维度固定为 0；需要保留该维度的 tile 形状时使用长度 1。

## 主要辅助对象

`VectorTileLoopNest` 统一创建携带 vector 的 affine 循环，处理单位维度、
`iter_args`、`affine.yield` 和插入点。各 `*Element::emit` 只描述一个元素的操作：

| 元素生成器 | 单元素行为 |
| --- | --- |
| `GemmAccumulatorElement` | 将一个 MMA 片段元素插入完整 C thread vector |
| `MaskTileElement` | 映射并执行 mask region，插入标量结果 |
| `ExtractThreadTileElement` | 按布局从 block vector/memref 读取线程元素 |
| `InsertThreadTileElement` | 将线程元素放回 block vector |
| `BroadcastTileElement` | 将源长度为 1 的维度广播到目标 tile |
| `LayoutReadBackElement` | 按新布局读 shared scratch |
| `CopyToRegElement` | 基址加连续偏移读取标量，可选浮点转换 |

扩展新的元素操作时，显式声明所需状态并实现 `emit(indices, currentVector)`；
不要在元素生成器中重复创建整个循环或手动生成外层 yield。

`BlockLowering` 和 `CopyLowering` 都只存在于一次 Pattern 调用期间。
它们将原来由大型 lambda 隐式捕获的上下文变成具名成员，Pattern 本身仅负责入口分派。

## Copy 的四条路径

`CopyLowering::run()` 保留以下尝试顺序：

1. `lowerVectorCopy`：已有 vector 表示时，local 寄存器目标更新 SSA 替换关系；
   shared/global 目标按线程布局生成实际 store，shared 写后同步，供后续内存读取。
   同 shape copy 不将旧式 `() -> (2)` 标记解释为地址偏移。
   目标是完整 block vector 时，按布局插入 thread tile。
2. `lowerGlobalSharedCopy`：连续划分整个 copy tile，每线程处理
   `ceil(totalElements / thread_num)` 个元素，并检查尾部越界。
   shared → global 在读前同步；global → shared 在写后同步。
3. `lowerBufferViewCopyWithThreadMap`：仅一侧为 view 且存在 LowerInfo 时，
   用线程布局生成 tile 坐标，再叠加 view 的物理基址。
4. `lowerScalarCopy`：普通 memref 搬运，使用 copy tile 循环和可选 offset map。

`computeCopyPlan` 只计算计划：同 shape 直接复制；单侧 view 使用 view 的 map；
不同 shape 时，以元素数较小的一侧作为 tile，offset map 作用于较大的一侧。
不同 shape 但元素数相同、或两侧均为 view 时，保留原有不支持的行为。

扩展匹配范围时，先明确新路径属于哪个阶段、使用哪种坐标，以及是否需要同步。
不要将“当前路径不匹配”与“已生成部分 IR 后失败”混为一谈；后者需要注意
ConversionPatternRewriter 的回滚边界。

## SSA、物化与清理

`findLowerInfoForValue` 依次查询当前 consumer、其他 user、分析表中的 buffer；
必要时再解开一层单输入 `unrealized_conversion_cast`。
Reduce 还需要 nearest-inference 和反向替换查询，因此保留独立的查找函数。

`s_buffer_replace` 保存原值到线程表示的映射。替换 vector 必须支配使用点；
不能将仅在循环体内定义的值拿到循环外使用。新生产的 vector 要通过
`registerConvertedValueLowerInfo` 登记布局。

TypeConverter 的临时 cast 标注 `b->t` / `t->b` 和 `thread_tile_shape`，并传播布局信息。
早期 `CopyToRegCastRewrite` 只折叠可直接复用 vector 的情况；后期
`CopyToRegOpRewrite` 才展开真实读取。

`normalizeLoopCarriedBlockVectors` 同时重建循环的初值、region 参数、结果和 yield。
它保留 carrier 顺序，既处理 splat 常量，也处理非 splat 初值。

## 当前范围与验证

本次重构保留现有 DCU 路径。NVIDIA GEMM 分支仍未实现；Reduce 仅支持浮点
add/mul/min/max、静态正长度和单 warp 内的二次幂 lane 归约。
分析层与本文件仍保留共享静态状态，未在本次改动中改造为并发执行模型。

函数级 `warp_layout` / `block_layout` / `block_layout_order` 仍从 LowerInfoMap 的首项
读取。该表为 DenseMap，首项可能随进程变化；比较 IR 时要区分这一既有属性差异和
实际生成的访问/计算指令差异。本文件生成索引时使用每个值对应的 LowerInfo。

在仓库根目录运行：

```bash
cmake --build 3rd/deepgengraph/build --target MyTest -j 2
python3 3rd/deepgengraph/test/check_loop_thread_tiles.py \
  3rd/deepgengraph/build/test/MyTest
```

现有回归覆盖常量 fill、活跃累加器、非零初值、非 splat 初值以及交换循环 carrier 顺序。
这些是编译/IR 回归，不等同于 GPU 数值测试，也没有穷尽所有 Pattern 分支。

### 本次重构验证记录

- `MyTest` 构建通过；上述 5 个回归用例通过。
- 额外检查了 Reduce 的 min/max/mul、逐元素 sub、shared Block 和 local Block
  往返搬运，共 6 个变体，均完成到 LLVM IR 的转换。
- 对以上 11 个用例比较重构前后的线程级 IR：仅归一化临时文件路径和前述不稳定的
  函数布局属性后，文本一致。原版本独立重复运行也复现了布局属性的变化。
- 两个仓库旧样例存在既有失败，重构前后均复现：`test_friskBase.mlir` 残留
  `unrealized_conversion_cast`，`attn_p2_pipeline_scheme.mlir` 在 LowerInfo 推导中断言失败。
  本次未修改这两个样例或分析算法。
