# Fragment IR 与 affine 循环融合

当前实现：`FriskBaseToTheadlevelIR_plan2.cpp`、`FriskFragIRReorder.cpp`。

## Fragment 组织

每线程的计算 fragment 形状为 `thread_creg * warp_repeat`，与 GEMM
操作数 fragment 的定义一致。完整 thread tile 中每维的 fragment 数为
`block_repeat * warpInstUnroll`。长度为 1 的广播/归约维保持为 1；rank-1
结果使用 `ignoreDim` 选择对应的布局轴。

`FragmentTileLoopNest` 对 fragment 网格按行优先展平，生成一个
`affine.for`。循环体内的寄存器坐标静态展开：

```
fragment = thread_creg * warp_repeat
count[d] = tile[d] / fragment[d]
origin[d] = (fragment_iv floordiv product(count[d+1:]) mod count[d]) * fragment[d]
thread_index[d] = origin[d] + register_offset[d]
```

例如 thread tile 为 `vector<2x32xf32>`、fragment 为 `1x4`，外层循环
执行 16 次，每次操作一个 `vector<1x4xf32>`。动态片段起点通过
`vector.extract/insert` 实现；片段内算术仍使用 fragment vector。

Add/Sub/Mul/Div、Exp2、Cast、Zero、广播、Mask、Fill、Copy、Block、
ConvertLayout 及 bridge 最终物化均使用该组织方式。Reduce 的输出轴
与归约轴分别按 fragment 遍历，片段内仍按原顺序逐元素累加，warp shuffle
发生在完整线程内归约之后。GEMM 使用显式的 MN 片段循环与内层 K 累加循环。A/B 在 K 片段使用处
装载，不先物化整个 thread tile。

Global→Shared Copy 使用连续传输 packet 作为搬运 fragment，其宽度为
`gcd(perThread, rowWidth)`，继续保证不跨行及尾部 guard；这个物理搬运布局
不套用 MMA 计算布局。没有 LowerInfo 的显式线程切片视为一个 fragment。
Alloc、BufferView、同步及表示转换本身没有计算迭代空间，继续保留其原语义。

循环属性：`frisk.fragment_loop`、`frisk.fragment_shape`、`frisk.tile_shape`、
`iterLabel`。`frisk.fragment` 是一个无条件、单 block 的 region，捕获外部
SSA 值，通过 `frisk.fragment_yield` 返回寄存器 fragment；`kind` 分为
`load`、`store`、`compute`、`gather`。多个标量访存和组装寄存器的操作仍
处于一个 fragment 边界内。它具有递归内存副作用，不能在 canonicalize
阶段提前内联。完整 tile 的 scatter 在 region 外以保留融合的依赖证明。

单片段循环在本 Pass 的单次迭代提升阶段保留；后续 canonicalize
可以按通常规则消除。属性是调度提示，不作为安全融合的依据。

## 融合 Pass

工厂函数：`mlir::frisk::createFriskFragIRReorderPass()`。
命令行：`--frisk-frag-ir-reorder`。已注册到 Passes.td/CMake/ThreadTilingTest，
MyTest 在 `createFinalizeThreadTilingPass()` 后执行。

推荐顺序：

```
convert-friskbase-to-thread
finalize-thread-tiling
canonicalize  # 可选：简化索引，增加可证明的融合机会
frisk-frag-ir-reorder     # 融合 + 下一片段预取
lower-frisk-fragments     # 内联 region、shared memory 复用、延后展开 K
canonicalize
cse
# ThreadLevelIRLegalize -> IRDeepOptimize -> LLVM lowering
```

Pass 合并边界、边界操作数和步长完全相同的相邻 affine.for，也允许跨过
不依赖前循环结果的纯、可推测执行的无 region 初始化操作。它拼接所有
iter_args/yield/results，支持 vector、memref 和独立 scalar carrier，保留
循环外使用者。嵌套循环按相同规则处理，直到无法继续融合。

安全范围：

- 独立 SSA 循环，包括动态边界，可融合。
- 生产者 vector 必须是逐元素 insert 链，各迭代写集合不重叠。消费者只能
  读取对应迭代已经生成的位置。整个 vector 的运算、归约及反向/跨片段访问
  不满足条件时保留原循环。
- 当生产者结果是消费者初值时，消费者必须更新同样的位置。融合后先将
  生产者当前片段覆盖到消费者 carrier，再执行消费者，保留已计算的其他片段。
- memref carrier 必须保持 descriptor 不变。通过别名分析证明独立，或者对
  同一 identity-layout memref 的静态标量地址检查所有因融合而交换顺序的
  `B(i)` / `A(j), j > i` 读写对。无法证明的别名/动态地址保留原循环。
- barrier、未知副作用、分配/释放、调用以及不支持的内存操作阻止融合。

依赖证明最多枚举 256 次迭代、65536 个 vector 写位置及 1048576 个内存
访问对，超出预算时保持原 IR。该 Pass 尚不实现跨 barrier 调度、通用符号
内存依赖证明或整个 vector 的任意跨迭代重排。

## 预取与延后展开

融合后，对只读且有 compute fragment 的静态片段循环生成一级 lookahead：

```
carried = load(first)
for i = first to last exclusive step step:
    next = load(i + step)
    acc = compute(i, carried, acc)
    carried = next
acc = compute(last, carried, acc)
```

没有最后一次越界预取，累加顺序保持原样。load 的地址依赖切片必须是纯且
可推测执行的操作，不能依赖 loop-carried 值。整个循环内出现写、barrier、
未知副作用、shuffle 或嵌套控制流时跳过；零次/一次迭代也跳过。使用
`frisk.pipelined` 和 `frisk.pipeline_stage` 标记，重复运行不再次流水化。
这只是把下一个 fragment 的真实 load 放到当前 MMA/计算之前，为后端隐藏
访存延迟提供指令顺序，不保证目标 GPU 的性能收益。

`lower-frisk-fragments` 才去掉 region。当前 MLIR 的 affine LICM 不检查
未知 region 的隐式 captures，因此 affine LICM、buffer hoisting 和 loop
normalize 也在此 Pass 之后运行。共享内存复用在此阶段按调度后的
生命周期执行。标记 `frisk.loopUnrollFull` 的 K 循环此时才展开，单循环上限
64 次，额外操作总预算 65536；更大的循环保留。后续 IRDeepOptimize 继续
展开能静态化 vector 索引的循环。fragment 内固定的寄存器坐标仍允许静态
展开，这不会消除用于融合/调度的 fragment 网格循环。

验证：

```bash
cmake --build 3rd/deepgengraph/build --target ThreadTilingTest MyTest -j 2
python3 3rd/deepgengraph/test/check_fragment_loops.py
python3 3rd/deepgengraph/test/check_plan2_thread_tiles.py \
  3rd/deepgengraph/build/test/ThreadTilingTest
python3 3rd/deepgengraph/test/check_fragment_schedule.py
```

第一项测试包含融合正反例、CPU baseline/fused 数值执行及 attention
fragment IR/finalization 检查。调度测试涵盖步长、符号地址、阻止移动的副作用、
幂等性及 baseline/scheduled CPU 数值对照，并检查完整 attention 的 LLVM 导出、
vector 索引常量化、65536 个 QK/PV 操作数地址及 64 个 MMA 累加输入；GPU 数值与性能需在对应设备上另行验证。

## 展开后的调度与 LLVM 导出

`IROptimize.cpp` 的 `--ir-deep-optimize` 默认在展开后执行依赖调度，
可用 `reorder-after-unroll=false` 单独关闭调度。调度仅在直线代码窗口内进行，
保留 SSA、可能别名的 RAW/WAR/WAW 和 MMA 相对顺序；barrier、控制流、
未知 asm、调用和原子操作是边界。

同一缓冲区、同一类型的至多 4 个标量 load 作为一组调度，组内只允许
必要依赖穿插；vector load 自成一组。该组织保留 SLP 向量化机会，避免
逐条 load/compute 交替拆散 MMA 操作数读取。没有合并地址或增加读取。
窗口内新定义数据的 SSA 活跃量达到 512 bit 时优先计算，以限制超前读取；
这是启发式优先级，不是硬上限，也不是实际 VGPR/occupancy 估计。

`prepareRegisterMMAForLLVM` 只识别已知的完整寄存器 MMA asm、类型和
约束，补上 `memory(none)` 与 `convergent`。允许前后各一条或多条
`s_nop`，十进制等待立即数为 0..15；不要求两侧数量或等待立即数相同。
识别的是内存副作用，不负责证明硬件等待时间。`sideeffect`、MMA 顺序及
用户调好的 NOP 文本原样保留，包括当前前后各一条 `s_nop 1`。
不能把识别条件写死为旧的前后各 8 条 `s_nop 7`，否则调参后会同时失去
展开后调度和 LLVM 重复 load 消除。额外指令、memory clobber、重复 MMA、
缺少任一侧 padding 或签名不匹配仍拒绝识别。
兼容旧版 LLVM 的输出将 `memory(none)` 写成
`readnone`，避免原先直接删除它而阻止 O3 复用重复读取。其他未知 asm
继续保守处理。

同输入 A/B 导出（在仓库根目录执行）：

```bash
mkdir -p 3rd/deepgengraph/build/perf-fix
3rd/deepgengraph/build/test/MyTest 3rd/deepgengraph/test/test_input.mlir 0 \
  3rd/deepgengraph/build/perf-fix/reorder.ll 1 > /tmp/reorder.log 2>&1
3rd/deepgengraph/build/test/MyTest 3rd/deepgengraph/test/test_input.mlir 0 \
  3rd/deepgengraph/build/perf-fix/no-reorder.ll 0 > /tmp/no-reorder.log 2>&1
python3 3rd/deepgengraph/test/check_post_unroll_reorder.py
python3 3rd/deepgengraph/test/check_register_mma_export.py
python3 3rd/deepgengraph/test/check_attention_codegen.py
```

MyTest 参数依次为输入、block pipeline 开关、输出路径、展开后 reorder 开关。
后两个参数可省略，默认输出 `finalLLVMText.ll` 并开启 reorder。O3 回归
检查同时覆盖两版导出、LLVM 15 解析、操作数地址、读取复用与分组调度的
访存指令数；LLVM 指令数不等于 gfx936 ISA 指令数或实际性能。
