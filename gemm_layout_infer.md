# GEMM / WGMMA Layout Infer

这份笔记只讲 NVIDIA 路径里，GEMM 何时会走 `wgmma`，`MNK` 是谁选的，以及 Gluon 的 `LayoutInfer` 是怎么把布局约束传下去的。

## 先说结论

- `LayoutInfer` **不负责搜索 `wgmma` 的 `MNK`**。
- `MNK` 的选择发生在 `AccelerateMatmul.cpp` 里，把 `DotOp` 变成 `NvidiaMmaEncodingAttr` 时完成。
- `LayoutInfer` 做的是：把已经确定的布局 seed，沿着 `load / store / reshape / transpose / join / split` 等算子做固定点传播；`dot / wgmma` 自己更像布局锚点，布局通常已经在 MMA 选择阶段定好了。

## `wgmma` 的 `MNK` 怎么选

相关代码：

- `triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`
- `triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp`
- `triton/lib/Dialect/TritonGPU/IR/Dialect.cpp`
- `triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp`

### 1. 先选 MMA 版本

`getMMAVersionSafe()` 会按硬件能力挑版本。对 Hopper/SM90+，如果 `supportMMA(op, 3)` 成功，就会走 version 3，也就是 WGMMA。

### 2. 再算 `instr_shape`

核心函数是 `mmaVersionToInstrShape(version, shape, eltType, numWarps)`：

- version 3 时：
  - `m = 16`
  - `k = 256 / bitwidth`
  - `n` 从一个“合法列表”里从大到小挑
- dtype 不同，合法 `n` 不同：
  - fp8 / f16 / bf16 / f32: `256, 248, 240, ..., 8`
  - int8: `224, 208, 192, ..., 8`
- 额外约束：
  - `shape[0] % 64 == 0`
  - `shape[1] % 8 == 0`
  - `n` 必须整除 `shape[1]`
  - `n <= maxN`

其中：

```text
mWarps = max(shape[0] / 16, 1)
nWarps = max(numWarps / mWarps, 1)
maxN   = max(shape[1] / nWarps, 8)
```

所以本质上是：

1. 先固定 `m = 16`。
2. `k` 由元素位宽决定。
3. `n` 在允许范围内尽量取大，但要能被当前 tile 切开。
4. 如果同一个 CTA 里 warps 更多，`maxN` 会变小，`n` 也可能被迫变小。

### 3. 再算 `warps_per_cta`

version 3 用 `warpsPerTileV3()`：

- 如果 forward slice 里还能看到后继 dot，直接偏向 `[numWarps, 1]`，方便链式 GEMM。
- 否则从 `[4, 1]` 开始，按需要把 warps 在 M/N 上二分扩展。
- 选择策略会偏向让 M 方向先吃掉更多 warp，避免寄存器压力和后续布局冲突。

### 4. 把结果写进 MMA layout

最终会构造：

```cpp
NvidiaMmaEncodingAttr::get(ctx, versionMajor, versionMinor,
                           warpsPerTile, cgaLayout, instrShape)
```

这个 layout 之后会成为：

- dot result 的布局 seed
- `DotOperandEncodingAttr` 的 parent
- lowering 到 WGMMA 时的唯一真值来源

### 5. lowering 时直接读这个 shape

`WGMMA.cpp` 里直接：

- `auto instrMNK = mmaEncoding.getInstrShape();`
- 用 `instrMNK[0] / [1] / [2]` 算重复次数、寄存器展开、shared memory 访问

也就是说，前面选出来的 `MNK` 会一路传到 PTX 生成阶段，不会再二次搜索。

## `LayoutInfer` 怎么推

相关代码：

- `triton/lib/Dialect/Gluon/Transforms/InferLayoutUtils.cpp`
- `triton/lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp`
- `triton/lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp`
- `triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp`

### 1. 入口

Gluon 里有两条典型入口：

- `gluon-infer-coalesced-encodings`
- `gluon-resolve-auto-encodings`

它们最终都调用同一个 `inferLayout(func, typeCheck, seedEncodings)`。

### 2. Seed 从哪来

- `ResolveAutoEncodings`：从 `set_auto_layout(x, layout)` 收集 seed。
- `InferCoalescedEncodings`：从 coalesced 的 load/store 推出 pointer 的布局 seed。

### 3. 先做边界检查

`inferLayout()` 会先拒绝两种情况：

- 函数参数带 auto encoding
- 函数返回值带 auto encoding

意思是：自动布局不能跨函数边界悬空，必须在函数内部完全解析掉。

### 4. Worklist 固定点传播

它维护一个 `value -> LayoutInfo` 表和 worklist：

1. 先把 seed 放进去。
2. 反复 pop 一个 value。
3. 把它的 encoding 推给使用者。
4. 也把它的 encoding 反推给定义它的操作输入。
5. 直到 worklist 为空或发生冲突。

### 5. 传播规则

#### 向使用者传播

调用 `inferDstEncoding(op, info.encoding)`。

常见规则：

- elementwise / same encoding trait：原样传递
- `reduce`：输出变成 `SliceEncodingAttr`
- `expand_dims`：把 slice 还原回 parent
- `join` / `split`：通过 dialect 接口推导
- `transpose`：按逆 permutation 推
- `reshape`：走 `inferReshapeOpEncoding`
- `gather`：结果布局跟 index 一样
- `fp4_to_fp`：按轴把 `kWidth` 翻倍或减半

#### 向定义者传播

调用 `inferSrcEncoding(definingOp, info.encoding)`。

规则和上面大体对称，目的是把同一个布局约束尽量推回源头。

### 6. loop / yield 的特殊处理

对 `scf.for / scf.while / scf.yield`，它不会把布局当成普通算子处理，而是把布局传到 tied args 上。

这一步很关键，不然循环 phi 节点会把布局链条断掉。

### 7. 冲突怎么解

如果同一个 value 收到多个 encoding：

- 一样就合并
- 如果某一侧来自 `join / split / reshape / transpose / cat` 这类“可能变化”的节点，就优先保留另一侧更稳定的布局
- 真冲突且都不能变，就直接报错

### 8. 最后写回 IR

传播结束后，`inferLayout()` 会：

- 把推出来的 encoding 写回 tensor type
- 对常量 splat 一并修正类型
- 再跑一次 `doubleCheckEncodings()`

## 如果看的是 `wgmma` GEMM，实际步骤可以这样记

1. 先由 pass 决定是否走 MMA/WGMMA 版本。
2. 用 `mmaVersionToInstrShape()` 选 `MNK`。
3. 用 `warpsPerTileV3()` 选 `warps_per_cta`。
4. 把这两个信息封进 `NvidiaMmaEncodingAttr`。
5. `LayoutInfer` 把这个结果布局往 A/B operand、load/store、reshape、transpose 等地方传播；`dot / wgmma` 这类点本身通常是 seed 的来源，而不是再被它反推出来。
6. 最后 `WGMMA.cpp` 直接读 `instr_shape` 生成 PTX。

## 一句话版

`MNK` 是在 MMA 选择阶段算出来的，`LayoutInfer` 只是把这个 layout seed 沿图传播；`wgmma` lowering 再从同一个 `NvidiaMmaEncodingAttr` 里取出 `instr_shape` 来发 PTX。
