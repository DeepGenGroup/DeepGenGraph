# LowerInfo 推导逻辑

本文梳理 `mlir::frisk::LowerInfoAnalysis::run` 的实际执行逻辑。相关代码主要在：

- `3rd/deepgengraph/include/deepgengraph/Analysis/LowerInfo.h`
- `3rd/deepgengraph/lib/Analysis/LowerInfo.cpp`
- 消费端：`3rd/deepgengraph/lib/Conversion/FriskToBase/FriskBaseToTheadlevelIR.cpp`

## 1. LowerInfo 表示什么

`LowerInfo` 描述一个 block-level buffer 在降低到 thread-level 后，每个线程应该持有、访问哪些元素。它不是单纯的 shape 信息，而是把 buffer 的二维访问拆成几层：

- `block_layout`：一个 block 内 warp / warpgroup 在二维方向上的排布。
- `warp_layout`：一个 warp 内 lane 在二维方向上的排布。
- `block_repeat`：block 级 tile 在每个维度需要重复多少次。
- `warp_repeat`：warp 级 tile 在每个维度需要重复多少次。
- `thread_widths`：单个 thread 在每个维度连续持有多少元素。
- `warp_widths = warp_layout * thread_widths`。
- `block_widths = block_layout * warp_repeat * warp_widths`。
- `thread_bound`：当前 region 使用的线程数。
- `warp_indices` / `lane_indices`：由 `threadIdx.x` 推出当前线程属于哪个 warp、哪个 lane 二维坐标。

最终消费端常用的是：

- `get_thread_total_widths()`：生成 thread-local buffer 的形状，等于 `thread_widths * warp_repeat * block_repeat`。
- `get_block_repeat()` / `get_block_widths()`：生成 GEMM 外层 M/N/K 循环。
- `getAffineMap()`：生成从 thread-level loop iv / `threadIdx` 到原 buffer 坐标的 affine map。

## 2. run 的总体流程

`LowerInfoAnalysis::run(kernelOp, hwKind, version)` 的核心策略是：

1. 根据 `hwKind` 和 `version` 调 `GetHWSpecification` 得到硬件描述，默认是 `dcu` / `bw1000`。
2. 在 `kernelOp` 里按 preorder 收集需要推导的 op：`frisk.copy`、`frisk.block`、`frisk.gemm`、`frisk.reduce`。
3. 找到第一个可以直接推导的 op。当前 `inferDirectOp` 只认识 `GemmOp`，所以本质上是用第一个 GEMM 作为锚点。
4. 直接推导该 GEMM 的 A/B/C 三个 buffer 的 `LowerInfo`，写入 `DenseMap<Value, LowerInfo> buf_info_maps`，并把该 op 从待推导列表移除。
5. 对剩余 op 反复做依赖推导：只要某个 op 的部分 buffer 已经有 `LowerInfo`，就尝试把信息传播到同 op 的其它 buffer。
6. 每轮如果没有任何 op 推导成功，打印 unresolved op 并 assert。
7. 全部 op 推导完后返回 `buf_info_maps`。

伪代码：

```cpp
hw = GetHWSpecification(hwKind, version)
buf_info_maps = {}
need_infer_ops = collectNeedInferOps(kernelOp)

for op in need_infer_ops:
  if inferDirectOp(op):        // 当前只有 GEMM 可作为锚点
    remove op
    break

while need_infer_ops is not empty:
  progress = false
  pending = []
  for op in need_infer_ops:
    if inferRelyOp(op):
      progress = true
    else:
      pending.push_back(op)
  if !progress:
    error + assert
  need_infer_ops = pending

return buf_info_maps
```

这个算法是一个 worklist/fixpoint 推导：先用 GEMM 定出一组 buffer 的布局，然后沿 copy、block、reduce、后续 GEMM 的数据依赖向前后传播。

## 3. 初始锚点：直接推导 GEMM

`inferDirectOp` 只处理 `GemmOp`，实际调用 `inferGemmOp`。

### 3.1 提取 GEMM problem

`getGemmProblem` 从 `GemmOp` 中拿到：

- `A`、`B`、`C` 三个 memref value。
- `aType`、`bType`、`cType`。
- `bm = C.shape[0]`。
- `bn = C.shape[1]`。
- `bk = transA ? A.shape[1] : A.shape[0]`。
- `inElemBitWidth = A.elementType.bitWidth`。

### 3.2 选择 MMA 指令

`selectGemmInst` 遍历硬件描述里的 `hw->gemmInfo.validInsts`：

- A/B fragment dtype 要与指令 `fragmentElementType` 匹配。
- C accumulator dtype 要与指令 `accElementType` 匹配。
- 指令的 `m/n/k` 不能超过 problem 的 `bm/bn/bk`。
- 在满足条件的指令里选择 `m * n` 最大者。

目前 `checkGemmProblem` 只是占位，直接返回 `true`。

### 3.3 计算 block_layout

`getDirectGemmBlockLayout(thread_num, block_layout, hw)` 根据硬件类型决定 block 内 warp 如何排布：

- NVIDIA：
  - `warpgroup_num = thread_num / 128`。
  - `squareFactor(warpgroup_num)` 得到接近平方的 `{y, x}`。
  - `block_layout = {y * 4, x}`，因为一个 warpgroup 有 4 个 warp。
- DCU：
  - `warp_num = thread_num / hw->warpSize`。
  - `squareFactor(warp_num)` 得到 `{y, x}`。
  - `block_layout = {y, x}`。

`squareFactor(n)` 从 `sqrt(n)` 往下找最大因子 `a`，返回 `{n / a, a}`。

### 3.4 生成 C 的 LowerInfo

`makeDirectGemmCInfo` 先以 C 为中心生成完整二维 tiling。

NVIDIA 分支：

- `thread_widths = {1, 32 / inElemBitWidth}`。
- `warp_layout = {8, 4}`。
- `block_layout` 使用上一节的结果。
- `warp_widths = warp_layout * thread_widths`。
- `warp_repeat = {2, mma.n / warp_widths[1]}`。
- `block_widths = block_layout * warp_repeat * warp_widths`。
- `block_repeat = {bm / block_widths[0], bn / block_widths[1]}`。
- `warp_indices` / `lane_indices` 由 `threadIdx.x` 按 block/warp layout 算出。

DCU 分支：

- `thread_widths = {1, 1}`。
- `warp_layout = mma->warp_layout_acc`，BW1000 里当前是 `{16, 4}`。
- `warp_widths = {16, 4}`。
- `warp_repeat = {1, 1}`。
- `block_widths = block_layout * warp_repeat * warp_widths`。
- `block_repeat = {bm / block_widths[0], bn / block_widths[1]}`。

生成 C 后，`inferGemmOp` 把它写入 `buf_info_maps[C]`。

### 3.5 从 C 派生 A/B 的 LowerInfo

直接 GEMM 会复制 C 的 `LowerInfo` 作为基础，然后分别改写 A、B 与 K 轴相关的维度。

A 的规则在 `applyDirectGemmAInfo`：

- `buffer = A`。
- 第 0 维基本沿用 C 的 M 轴布局。
- 第 1 维改成 K 轴：
  - NVIDIA：`thread_widths[1] = 32 / inElemBitWidth`。
  - DCU：`thread_widths[1] = mma.k / 4`。
  - `warp_widths[1] = 0`。
  - `block_widths[1] = mma.k`。
  - `block_repeat[1] = bk / mma.k`。
  - `warp_indices[1] = 0`，`lane_indices[1] = 0`。

B 的规则在 `applyGemmBInfo`：

- `buffer = B`。
- 第 1 维基本沿用 C 的 N 轴布局。
- 第 0 维改成 K 轴：
  - `thread_widths[0] = 0`。
  - `warp_widths[0] = 0`。
  - `warp_repeat[0] = 0`。
  - `block_widths[0] = mma.k`。
  - `block_repeat[0] = bk / mma.k`。
  - `warp_indices[0] = 0`，`lane_indices[0] = 0`。

至此，锚点 GEMM 的 A/B/C 都有了 LowerInfo。

## 4. 依赖推导：inferRelyOp

`inferRelyOp` 是剩余 op 的统一入口。

它会先收集该 op 关联的 memref：

- 普通 op：遍历 operands，取类型是 `MemRefType` 的 value。
- `BlockOp`：额外 walk block body，收集内部 `affine.load` 的 memref 和 `affine.store` 的 memref。

然后判断这些 memref 在 `buf_info_maps` 里的覆盖情况：

- 全部已知：直接返回 `true`，不再改写。
- 全部未知：返回 `false`，等待后续轮次；日志语义是“需要 GEMM 做锚点”。
- 部分已知：按顺序尝试 `inferCopyOp`、`inferBlockOp`、`inferRelyGemmOp`、`inferReduceOp`。

## 5. 各类 op 的传播规则

### 5.1 CopyOp

`inferCopyOp` 支持双向传播：

- 如果 dst 已知，则把 dst 的 `LowerInfo` 复制给 src，并把 `buffer` 字段改成 src。
- 否则如果 src 已知，则把 src 的 `LowerInfo` 复制给 dst，并把 `buffer` 字段改成 dst。
- 两边都未知则失败。

这让 `global/shared/local` 之间的 copy 可以把同一份逻辑 tiling 传过去。

### 5.2 BlockOp

`inferBlockOp` 用 block 内的 load/store buffer 做传播：

1. 取 `blockOp.getBlockRanges()` 作为 block 的逻辑迭代范围。
2. walk body，收集所有 `affine.load` 的 memref，并记录 `affine.store` 的 memref。
3. 只有 rank 和 shape 都等于 block ranges 的 buffer 才能作为“主 buffer”。
4. 优先从已知的 load buffer 中找主 buffer；找不到再看 store buffer。
5. 找到主 buffer 后，把它的 `LowerInfo` 复制给所有尚未知的 load/store buffer。

注意：已有 `LowerInfo` 的 buffer 不会在这里被覆盖。

### 5.3 依赖 GEMM

`inferRelyGemmOp` 用已经推导出的 A 或 B 或 C 反推另一个 GEMM 的布局。

如果 A 或 B 已知：

- 若 A 已知，认为 source 是 A，保留 source 的 M 轴布局，用 `makeRelyGemmCInfo(..., source_is_a=true)` 推出 C 的 N 轴布局。
- 若 A 未知但 B 已知，认为 source 是 B，保留 source 的 N 轴布局，用 `makeRelyGemmCInfo(..., source_is_a=false)` 推出 C 的 M 轴布局。
- 会检查 `thread_num == source_info.thread_bound`。
- 推出 C 后写入 `buf_info_maps[C]`。

随后根据已知输入补齐另一侧：

- 如果 A 已知，则以 C 的 info 为基础推 B，调用 `applyGemmBInfo`。
- 如果 B 已知，则以 C 的 info 为基础推 A，调用 `applyRelyGemmAInfo`。

`applyRelyGemmAInfo` 和 direct A 的区别是：K 轴上的 `thread_widths[1]`、`warp_widths[1]`、`warp_repeat[1]` 都设为 0，只保留 `block_widths[1] = mma.k` 和 `block_repeat[1] = bk / mma.k`。

如果 A/B 都未知但 C 已知，当前实现会读出 C 的 info 后直接返回 `true`，但不会补 A/B。这一点像是未完成逻辑，见最后的注意点。

### 5.4 ReduceOp

`inferReduceOp` 要求 src 已知：

1. 把 src 的 `LowerInfo` 复制给 dst。
2. 把 `buffer` 改成 dst。
3. 根据 `reduceOp.getDim()` 删除对应维度：
   - 对 `warp_indices` / `lane_indices`：从 reduce dim 开始左移，最后一维置为 affine 常量 0。
   - 对 `warp_layout`、`block_layout`、`warp_repeat`、`block_repeat`、`thread_widths`、`warp_widths`、`block_widths`：从 reduce dim 开始左移，最后一维置为 1。

这套实现目前按二维数组写死，适合 2D buffer 的 reduce 传播。

## 6. getAffineMap 的生成方式

`LowerInfo::getAffineMap()` 会根据 buffer memory space 生成二维索引表达式。

### 6.1 local/register buffer

memory space 为 `0` 或 `5` 时走 local/register 分支：

```text
idx[i] = iv_block_i * (warp_repeat[i] * thread_widths[i])
       + iv_warp_i  * thread_widths[i]
       + iv_thread_i
```

这里不显式乘上 `warp_indices` / `lane_indices`，因为 local buffer 已经是每个 thread 自己持有的 thread-local 视图。它主要描述当前 thread-local buffer 内部的循环访问。

### 6.2 shared buffer

memory space 为 `friskMs::Shared`，即 `3` 时走 shared 分支：

```text
idx[i] = iv_block_i * block_widths[i]
       + warp_indices[i] * (warp_repeat[i] * warp_widths[i])
       + iv_warp_i * warp_widths[i]
       + lane_indices[i] * thread_widths[i]
       + iv_thread_i
```

shared buffer 是 block 内所有线程共享的一整块 tile，因此索引需要包含当前线程的 warp/lane 二维位置。

`warp_indices` 和 `lane_indices` 都来自 `threadIdx.x`：

```text
warp_y = (tid / warp_threads) / block_layout[1]
warp_x = (tid / warp_threads) % block_layout[1]

lane_y = (tid % warp_threads) / warp_layout[1]
lane_x = (tid % warp_threads) % warp_layout[1]
```

## 7. run 的结果如何被使用

`ConvertFriskBaseToThreadLevelIR` 中会调用：

```cpp
s_info = LowerInfoAnalysis::run(kernel);
```

之后主要做几件事：

- 从 `s_info` 里取任意一个 info 的 `warp_layout` / `block_layout` 写回 kernel attr。
- GEMM lowering 时取 `s_info.at(A/B/C)`：
  - 用 `infoC.get_thread_total_widths()` 分配 local accumulator。
  - 用 `infoA.get_block_widths()[0]`、`infoB.get_block_widths()[1]`、K 方向 repeat 生成 GEMM loop。
- Block lowering 时对 block 内 local `alloc_buffer` 用 `get_thread_total_widths()` 改成 thread-local shape。

所以 `LowerInfoAnalysis::run` 的目标不是给 IR 加 attr，而是提前算出后续 lowering 所需的 thread-level buffer layout 和循环参数。

## 8. 实现注意点

当前实现里有几个值得留意的点：

- `selectGemmInst` 使用 `for (auto inst : hw->gemmInfo.validInsts)`，然后返回 `&inst`。这里 `inst` 是循环局部拷贝，返回指针有悬垂风险。更稳妥应改成引用遍历：`for (auto &inst : ...)`。
- `GetHWSpecification` 返回 static `HWSpecification s`，但 DCU 分支每次都会 `push_back` MMA 指令，没有清空 `validInsts`，多次调用可能重复积累。
- `inferRelyGemmOp` 在“只有 C 已知，A/B 都未知”的情况下会返回 `true`，但实际上没有补齐 A/B info，后续 `s_info.at(A/B)` 可能失败。
- `makeRelyGemmCInfo` 的 A 已知分支使用了 `32 / inElemBitWidth`，没有按 `hw->warpSize` 区分；对 DCU 默认 warpSize=64 的路径需要确认是否符合预期。
- `getAffineMap()` 会修改 `dimCount`、`mapOperandsLabel`、`iterVarLabels`、`ivUpperBounds`，但没有清空 label/vector。多次调用同一个 `LowerInfo` 的 `getAffineMap()` 时，label 信息可能重复追加。
- `checkGemmProblem` 目前永远返回 `true`，shape 整除、memory space、dtype 支持等检查主要还没有真正落地。

