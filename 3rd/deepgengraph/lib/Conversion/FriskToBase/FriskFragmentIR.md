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

MyTest 参数依次为输入、block pipeline 开关、输出路径、展开后 reorder 开关、
shared operand packing 开关（最后一项为 `0|1`，默认开启）。
后两个参数可省略，默认输出 `finalLLVMText.ll` 并开启 reorder。O3 回归
检查同时覆盖两版导出、LLVM 15 解析、操作数地址、读取复用与分组调度的
访存指令数；LLVM 指令数不等于 gfx936 ISA 指令数或实际性能。

## 通用 shared operand packing

`PackSharedMemory.cpp` 将逻辑二维共享矩阵的连续行片段映射成连续物理地址。
对于逻辑 `[K,N]`、pack 宽度 `p`，物理布局是 `[K/p,N,p]`，以一维 memref 存储：

```
physical(k,n) = ((k floordiv p) * N + n) * p + k mod p
```

这是元素数不变的一一映射；不改变全局张量布局、算子接口或线程的计算数据归属。
没有额外分配一个重排缓冲区。`K` 必须能被 `p` 整除。

自动规划在 `convert-friskbase-to-thread` 中进行，从 GEMM B 的
`LowerInfo.thread_widths[0]` 推导 `p`，不匹配 kernel 名、attention pattern 或
固定的 M/N/K。要求本地拥有的静态 identity-layout shared allocation，
所有消费者都是相容的 GEMM B，写入是完整的 global→shared copy，且每线程
搬运量能被 `p` 整除。多个相容 GEMM 消费者可以共享同一个物理布局。
未知用户、作为 A 同时使用、外部参数、切片/逃逸别名和轮转 selector 保持原布局。
开关：`--convert-friskbase-to-thread=pack-shared-operands=false`。

规划只附加 `frisk.shared_pack = p : i64`，此时缓冲区仍具有原逻辑语义。
协作搬运按未来物理连续区间分工，反解为逻辑坐标读取 global、写入 shared。
每个 packet 是逻辑矩阵的 `p` 行与若干连续列；global 使用逐行显式
vector load，再在寄存器中交错为物理 packet，避免依赖 SLP 从交错标量
读取中重新发现连续访问。自动规划要求 global 源的末维具有 unit stride；
即使暂不执行后续 packing pass，这些 scalar stores 的语义也成立。

普通 global→shared copy 同样显式生成连续源的 vector load，不依赖 shared
packing 或 LLVM SLP。每次读取取不超过 128 bit、且不超过当前行剩余元素数的
最大二次幂长度；FP16/FP32/FP64 分别最多读取 8/4/2 个元素，尾部缩小读取包。
线程搬运总量、读取指令宽度和 shared 写入向量的长度是独立的，不能把一个
线程搬运的全部元素都视为一条硬件读取。view 偏移仍合成到原始 buffer，
只有原始 buffer 的末维已知为 unit stride 时才生成向量读取；非单位或未知
stride 保留标量读取。不会给外部指针或切片增加未经证明的对齐属性。
`test/check_affine_copy.py` 覆盖偏移、尾部、不同位宽和 CPU 数值；
`test/check_attention_codegen.py` 在存在 LLVM 15 时还检查 gfx90a 的实际
global load 指令，防止出现“IR 已向量化、机器指令仍是标量”的回归。

`--pack-shared-memory` 在桥接/view 已消除、共享内存池化之前执行；
`lower-frisk-fragments` 自动调用它。它也可独立处理带上述属性的普通
affine/memref IR，不要求含 GEMM、Frisk fragment 或 GPU thread id。
当前只接收可完整枚举的 scalar affine/memref load/store 和 dealloc；
未知用途（包括 subview、cast、原有 vector access、atomic、call）整块回退。
先验证完整使用集合，再统一改写地址，因此不会只改变生产者或部分消费者。

地址合并通过组合 affine map 证明相邻地址差为 1，最多生成 128 bit 的
power-of-two vector load/store。任何中间访存、控制流、同步或不可推测计算
都会终止合并；读取移动到组首、写入移动到组尾，不多读尾部或 padding。
搬运生成的标量写因此重新合并成宽写，避免以增加 scatter 写换取宽读。
这个 pass 不调整 MMAC/NOP，也不保证目标硬件的 bank conflict 或性能收益；
实际 LDS 指令、VGPR、occupancy 和时间仍须用目标 DTK/设备验证。

验证入口：`test/check_shared_packing.py`（普通 IR 的 CPU 数值对照、回退及
不同尺寸独立 GEMM）、`test/check_packed_attention_copy.py`（K/V 完整搬运地址）、
`test/check_attention_codegen.py`（全部 MMAC 操作数、P、rowsum、LLVM 15/O3）。
