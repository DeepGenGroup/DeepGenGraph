# Triton 中 data layout 的处理方式

本文基于 `/data2/xsl/DeepGenGraph/triton` 源码梳理。这里说的 data layout 主要指 TTGIR/TritonGPU tensor 或 memdesc type 上的 `encoding` attribute，不是 LLVM/MLIR 的 ABI `DataLayout`。

## 1. layout 如何表示

TritonGPU 的 tensor layout 存在 `RankedTensorType` 的 `encoding` 字段里，shared memory descriptor layout 存在 `ttg.memdesc` 类型的 `encoding` 字段里。相关定义：

- `RankedTensorType(..., encoding)`：普通 SSA tensor 的分布式 layout。
- `ttg.memdesc<shape x elem, encoding, memorySpace, ...>`：shared/tensor memory descriptor 的 layout。
- 所有 TritonGPU layout attr 定义在 `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`。
- `MemDescType` 的 encoding、memory space、shape/allocShape 校验在 `triton/lib/Dialect/TritonGPU/IR/Types.cpp`。

概念上，`TritonGPU_Attr` 的注释把 layout 定义为函数 `L(index) -> thread set`：逻辑 tensor 元素由哪些 GPU 线程拥有。实现上，很多 layout 又可以统一转换成 `LinearLayout`：

- `LinearLayout` 定义在 `triton/include/triton/Tools/LinearLayout.h`。
- 含义是“硬件位置 -> 逻辑 tensor index”的 GF(2) 线性映射。
- 分布式 register layout 常用输入维度：`register`, `lane`, `warp`, `block`。
- shared memory layout 常用输入维度：`offset`, `block`。
- 输出维度统一叫 `dim0`, `dim1`, ...。
- `LinearLayout` 用 basis vectors 表示映射，只需记录输入坐标为 2 的幂时对应的输出 basis，再用 xor 组合出完整映射。

转换入口在 `triton/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h` 和 `triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`，例如：

- `toLinearLayout(RankedTensorType type)`
- `toLinearLayout(MemDescType type)`
- `toLinearLayout(ArrayRef<int64_t> shape, Attribute layout)`
- `toLinearEncoding(...)`
- `inferEncodingFromLinearLayout(...)`

也就是说，源码里存在两套表达：历史上各类专用 `EncodingAttr`，以及越来越通用的 `LinearLayout`/`LinearEncodingAttr`。

## 2. layout 有哪些种类，含有什么信息

layout attribute 大致分两类：distributed encoding 和 shared encoding。共同 trait 在 `TritonGPUAttrInterfaces.td`：

- `LayoutEncodingTrait`：所有 TTGIR layout 的共同接口，核心信息是 `getCGALayout()` 和 rank。
- `DistributedEncodingTrait`：描述 register/线程层级分布，可转 `LinearLayout`，可查每线程元素数、rep order 等。
- `SharedEncodingTrait`：描述 shared memory layout，默认 alignment 为 16。
- `LinearEncodingTrait`：基于 `LinearLayout` 的 distributed encoding，提供 `getOrder()`、`getThreadsPerWarp()`、`getWarpsPerCTA()`、contiguity 等兼容接口。

主要 distributed encoding：

- `BlockedEncodingAttr`
  - 用于普通 tensor、load/store 合并访问、默认布局。
  - 字段：`sizePerThread`、`threadsPerWarp`、`warpsPerCTA`、`order`、`CGALayout`。
  - `order` 是最快变化维在前。
  - 默认 encoding 由 `getDefaultBlockedEncoding` 创建：每线程每维 1 个元素，order 为反向维度顺序。

- `LinearEncodingAttr`
  - 字段：`LinearLayout linearLayout`。
  - 输入维固定为 `register/lane/warp/block`，输出维为 `dim0..`。
  - 限制较强：移除 broadcast basis 后必须是 permutation matrix，register/lane/warp/block basis 不能随意 swizzle。

- `GenericLinearEncodingAttr`
  - 字段同样是 `LinearLayout`。
  - 比 `LinearEncodingAttr` 宽松：warp basis 可以 swizzle；移除 broadcast 后要求 surjective，不要求 bijective。

- `NvidiaMmaEncodingAttr`
  - 用于 NVIDIA tensor core dot 的 accumulator/result。
  - 字段：`versionMajor`、`versionMinor`、`warpsPerCTA`、`CGALayout`、`instrShape`。
  - version 表示 tensor core 代际，如 v2 Ampere/Turing、v3 Hopper、v5 Blackwell 相关路径另有 TMEM encoding。

- `AMDMfmaEncodingAttr`
  - 用于 AMD CDNA MFMA result。
  - 字段：`version`、`warpsPerCTA`、`instrShape`、`isTransposed`、`CGALayout`、`tilesPerWarp`、`elementBitWidth`。

- `AMDWmmaEncodingAttr`
  - 用于 AMD RDNA WMMA result。
  - 字段：`version`、`ctaLayout`、`isTransposed`、`CGALayout`、`instrShape`。
  - `ctaLayout` 本身是 `LinearLayout`，表达 warps 在 WMMA tile 上的排列，可支持更复杂的 swizzle。

- `DotOperandEncodingAttr`
  - 用于 `tt.dot` 的 A/B operands。
  - 字段：`opIdx`、`parent`、`kWidth`。
  - `opIdx=0` 表示 A，`opIdx=1` 表示 B。
  - `parent` 是 dot result/accumulator 的 MMA layout。
  - `kWidth` 表示每线程沿 K 维连续持有/加载的元素数，NVIDIA Ampere/Hopper builder 会按元素 bitwidth 推默认值。

- `SliceEncodingAttr`
  - 用于 reduce/expand_dims 等 rank 变化。
  - 字段：`dim`、`parent`。
  - 含义是从 parent layout squeeze 掉某一维。

主要 shared encoding：

- `SwizzledSharedEncodingAttr`
  - shared memory swizzle layout。
  - 字段：`vec`、`perPhase`、`maxPhase`、`order`、`CGALayout`。
  - 用 xor swizzle 降低 bank conflict；`vec` 表示成组 swizzle，`perPhase/maxPhase` 控制相位变化。

- `PaddedSharedEncodingAttr`
  - shared memory padding + 线性重排。
  - 字段：`intervals`、`paddings`、`linearComponent`。
  - `interval:padding` 表示每隔若干元素插入 padding；`linearComponent` 描述 offset 到逻辑维的线性映射。

- `SharedLinearEncodingAttr`
  - shared memory 版本的 linear layout。
  - 字段：`linearLayout`、`layoutAlignment`。

- `NVMMASharedEncodingAttr`
  - NVIDIA MMAv3/MMAv5 shared-memory operand layout。
  - 字段：`swizzlingByteWidth`、`transposed`、`elementBitWidth`、`fp4Padded`、`CGALayout`。
  - builder 会根据连续维字节数选择 32/64/128B swizzle。

- `AMDRotatingSharedEncodingAttr`
  - AMD rotating shared swizzle。
  - 字段：`vec`、`perPhase`、`maxPhase`、`order`、`CGALayout`。
  - 用于读写 order 不同、希望降低 LDS conflict 的场景。

- `PartitionedSharedEncodingAttr`
  - 把 tensor 沿某维切成多个 shared memory allocation。
  - 字段：`numPartitions`、`numGroups`、`partitionDim`、`partitionLayout`。
  - 用于减少 shared memory partition conflict。

辅助布局：

- `CGAEncodingAttr`
  - 定义在 `CGAEncodingAttr.td`。
  - 字段：`LinearLayout linearLayout`。
  - 表示 CTA/block 在 cooperative group array 里如何映射到逻辑 tensor 维度。
  - 常用派生信息：`CTAsPerCGA`、`CTASplitNum`、`CTAOrder`。

## 3. 算子里如何选锚点推定 layout

layout 的“锚点”可以理解为必须保留或优先保留的 layout 来源。代码里主要有以下几类。

### 默认锚点：类型转换阶段

`TritonGPUTypeConverter` 在 `TritonGPUConversion.cpp` 中给所有还没有 encoding 的 `RankedTensorType` 加默认 `BlockedEncodingAttr`：

- `sizePerThread = [1, ...]`
- `order = [rank-1, rank-2, ... , 0]`
- 按 `numWarps`、`threadsPerWarp`、`numCTAs` 构造 `threadsPerWarp/warpsPerCTA/CGA`。

如果 operand/result 类型不一致，target materialization 会插入 `ttg.convert_layout`。

### dot/MMA 锚点

`tt.dot` 是最典型的 layout anchor。

在旧的 Triton-to-TritonGPU conversion 中，`TritonDotPattern` 会：

- 给 dot result/accumulator 选一个 blocked result layout。
- 如果 A/B 不是 `DotOperandEncodingAttr`，插入 `ConvertLayoutOp` 转成 `DotOperandEncodingAttr(opIdx, dEncoding, eltTy)`。
- C accumulator 转成 result type 的 encoding。

后续 `AccelerateMatmul.cpp`/AMD `AccelerateAMDMatmul.cpp` 会进一步根据硬件选择 MMA layout：

- NVIDIA：`createMMAEncodingForDot` 根据 compute capability、result shape per CTA、元素类型、numWarps 选 `NvidiaMmaEncodingAttr`，并把 A/B 转成 `DotOperandEncodingAttr`。
- AMD MFMA：根据 MFMA 指令 shape、warpsPerTile、tilesPerWarp、isTransposed 等创建 `AMDMfmaEncodingAttr`，再给 A/B 创建 `DotOperandEncodingAttr(opIdx, mfmaEnc, kWidth)`。
- AMD WMMA：创建 `AMDWmmaEncodingAttr`，A/B 同样转为 `DotOperandEncodingAttr`。

`DotOp::inferReturnTypes` 和 `DotOp::verify` 会通过 `DialectInferLayoutInterface::inferDotOpEncoding` / `verifyDotOpEncodingCompatibility` 校验：

- A/B 必须都是合适的 `DotOperandEncodingAttr`，或 Hopper 场景接受 `NVMMASharedEncodingAttr`/`SharedLinearEncodingAttr` 等。
- `opIdx` 必须匹配。
- `parent` 必须和 result encoding 兼容。
- A/B `kWidth` 必须一致。
- NVIDIA MMA 版本必须一致；AMD 多 CTA 时还会检查 A/B/result 的 CGA layout 关系。

### load/store 合并访问锚点

`Coalesce.cpp` 对 load/store 的 pointer tensor 选择更适合 global memory coalescing 的 `BlockedEncodingAttr`：

- 用 `ModuleAxisInfoAnalysis` 得到 pointer 的 contiguity。
- `getOrderFromContiguity` 选择最快变化维。
- `getNumElementsPerThread` 选择每线程 vectorization。
- 构造 `BlockedEncodingAttr(shape, sizePerThread, order, numWarps, threadsPerWarp, cgaLayout)`。
- 然后 `convertDistributedOpEncoding` 把 memory op 的 operands/results 调整到该 layout，原有使用处再必要时转回。

descriptor load/store 也有单独策略：最多按 16B/128bit 方向选择 vector size，并偏向 row-major。

### RemoveLayoutConversions 的显式 anchor

`RemoveLayoutConversions.cpp` 的 `isLayoutAnchor` 明确定义哪些 op 的 layout 不希望改掉：

- `DescriptorOpInterface`
- expensive `LoadOp` / `StoreOp`
- `DotOpInterface`
- `AtomicRMWOp` / `AtomicCASOp`
- `triton::nvidia_gpu::TMEMLoadOp`
- 带 `efficientLayout` 的 `GatherOp`
- `allowReorder` 的 `ReshapeOp`
- 函数参数也作为 anchor，方便测试和保留外部签名 layout。

这些 anchor 的 result encoding 会进入传播工作表，作为候选 layout 向下游传播。

### shared/memdesc layout 锚点

shared memory layout 通常由 local alloc/store/load、descriptor memory layout pass、MMA shared operand 构造决定：

- `SwizzledSharedEncodingAttr::get(dotOpEnc, shape, order, CGALayout, typeWidth)` 会根据 dot operand 和硬件派生 shared layout。
- `NVMMASharedEncodingAttr` builder 根据 shape/order/bitwidth 选择 swizzle byte width。
- `AssignDescriptorMemoryLayouts` 会从 descriptor load/store users、`tt.desired_encoding`、已有 memdesc/tensor layout 收集希望的 shared encoding。

descriptor layout 冲突时有自己的 fallback 策略，见后文。

## 4. layout 如何传播

传播主要分三层。

### 类型推断中的局部传播

Triton dialect 通过 `DialectInferLayoutInterface` 给 rank-changing/view-like 算子定义 transfer function。接口在 `triton/include/triton/Dialect/Triton/IR/Dialect.h`：

- `inferTransOpEncoding`
- `inferReduceOpEncoding`
- `inferExpandDimsOpEncoding`
- `inferDotOpEncoding`
- `inferReshapeOpEncoding`
- `inferDefaultJoinOpEncoding`
- `inferSplitOpEncoding`
- `inferFp4ToFpOpEncoding`
- `verifyLayoutsAreEqual`
- `verifyDotOpEncodingCompatibility`
- `verifyCatOpEncodingCompatibility`

TritonGPU 的实现是 `TritonGPUInferLayoutInterface`，在 `triton/lib/Dialect/TritonGPU/IR/Dialect.cpp`。

典型规则：

- elementwise / SameOperandsAndResultEncoding：operand 和 result encoding 相同。
- `reduce(axis)`：result 是 `SliceEncodingAttr(axis, operandEncoding)`；降到 scalar 则没有 tensor encoding。
- `expand_dims(axis)`：operand 必须是对应 `SliceEncodingAttr`，result 回到 parent encoding。
- `transpose(order)`：目标是 no-op transpose，即同一线程仍持有同一物理值，只是逻辑维重命名；对 blocked/swizzled 等字段按维度 permutation 变换，对 `order` 用 `inverse(order) * inputOrder` 的逻辑处理；通用情况转成 `LinearLayout` 后 `transposeLinearLayout`。
- `reshape`：先尝试 legacy blocked encoding 的 no-op reshape；失败则用 `inferReshapeLinearLayout` 生成 linear layout，再包装成 `LinearEncodingAttr` 或 `GenericLinearEncodingAttr`。
- `join/split/fp4_to_fp`：优先走 legacy blocked/dot 规则，否则用 `LinearLayout` 加/去 register 维上的 basis。

### 优化 pass 中的一跳传播工具

`triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp` 提供：

- `inferDstEncoding(Operation *op, Attribute srcEncoding)`
- `inferSrcEncoding(Operation *op, Attribute dstEncoding)`

这些函数用于优化 pass 沿 def-use 或 use-def 推一跳：

- 对 elementwise、SameOperandsAndResultEncoding、SCF yield/condition/for/while 等，多数直接返回同 encoding。
- 对 reduce/expand/trans/reshape/join/split/gather/fp4_to_fp 调用对应 transfer function。
- `GatherOp` 特殊：indices 和 result layout 相同；source shared encoding 不通过这个规则传播。
- `UpcastFpOpInterface` 有自己的 `inferDstEncoding/inferSrcEncoding`，通用工具默认不强推。

### RemoveLayoutConversions 的全图前向传播

`RemoveLayoutConversions.cpp` 的 `LayoutPropagation` 是更完整的一次性分析+重写：

1. `initAnchorLayout`
   - 收集函数参数和 `isLayoutAnchor` op 的 results。

2. `propagateLayout`
   - 从 anchor 的 candidate encodings 出发，反复调用 `propagateToUsers`。
   - 对可传播 op 调用 `setEncoding`，内部用 `inferDstEncoding` 得到 user result 的候选 encoding。
   - 一个 value 可以临时拥有多个候选 encoding。

3. `resolveConflicts`
   - 如果某 value 有多个候选，选一个。
   - 当前策略比较简单：load/store/atomic 相关优先 `BlockedEncodingAttr`，其他场景优先 `MmaEncodingTrait`，否则取集合里的第一个。

4. `rewrite`
   - 按 region/dominance 顺序重写 IR。
   - `setEncodingInPlace` 直接改 result/block argument type 的 encoding。
   - operand 需要别的 encoding 时，`getValueAs` 插入 `ConvertLayoutOp`。
   - SCF for/while/if/yield/condition 有专门处理，保证 region arg/result/yield operand 类型一致。

之后 pass 会清理冗余 `ConvertLayoutOp`，并做 backward rematerialization / hoist convert，尽量把转换推到更便宜的位置或消除掉。

### 反向传播和 rematerialization

同一个文件里 `LayoutRematerialization` 用 `getConvertBackwardSlice` 从一个 `ConvertLayoutOp` 的 source use 反向追溯 producer slice：

- 遇到可重算、可传播的 op，就用 `inferSrcEncoding` 推出 operand 应该采用的 encoding。
- 能重算且收益合适时，clone 一段 producer slice 直接产出目标 layout，从而删除原 convert。
- 如果传播到已有兼容 remat value，就复用。
- 如果遇到不可传播/不可重算/冲突，则返回 failure，保留 convert。

## 5. layout 传播冲突如何处理

冲突处理不是一个单点，而是分场景。

### 前向候选冲突

`LayoutPropagation::resolveConflicts` 允许一个 value 在传播阶段有多个候选 encoding，最后强制选一个：

- load/store/atomic 类倾向 `BlockedEncodingAttr`，服务 memory coalescing。
- 非 memory 类倾向 `MmaEncodingTrait`，服务 tensor core/MMA。
- 其他情况取第一个候选。

选定后，rewrite 阶段如果某 operand 当前不是目标 encoding，就插入 `ttg.convert_layout`。因此冲突的主要解决方式是：每个 value 选一个主 layout，边界上用 convert 连接。

### 反向 slice 冲突

`getConvertBackwardSlice` 内部维护 `DenseMap<Value, Attribute> layout`。同一个 value 如果已记录 encoding，又被要求另一个不同 encoding：

```cpp
if (existing && existing != encoding)
  return failure();
```

这类 conflict 会让本次反向 rematerialization 失败。调用方通常只是放弃该优化，保留原 `ConvertLayoutOp`，不会硬改 IR。

### 算子语义冲突/非法 layout

一些算子的 verifier 或 infer interface 会直接报错：

- `tt.dot`：A/B encoding 不匹配、`opIdx` 错、`parent` 和 result 不兼容、MMA 版本不一致、`kWidth` 不一致等会 `emitError`。
- `LocalGatherOp`：source 必须 shared encoding，indices/result shape 和 layout 必须一致。
- `LocalScatterOp`：values/indices shape 和 layout 必须一致，destination 必须 shared encoding。
- `Fp4ToFpOp`：src/dst shape、axis、encoding 必须可由 infer rule 对上，否则报错。
- `MemDescType`：shared encoding 必须配 shared memory space；tensor memory encoding 必须配 tensor memory space；shape/allocShape 大多要求 power-of-two，rank 要匹配 encoding。

### descriptor memory layout 冲突

`AssignDescriptorMemoryLayouts::combineEncodings` 处理 descriptor users 给出的 desired shared encoding：

- 如果 CGA layout 冲突，fallback 到默认 CGA layout：把所有 CTA 放到最后一维。
- 如果 desired shared encoding 冲突，设置 `forcedToDefault = true`，后续用 backend 提供的 fallback shared encoding。
- shape 不同时取逐维较小值，适配 gather/scatter 等形状差异。

这类冲突不会用普通 distributed `ConvertLayoutOp` 解决，而是重新选择 descriptor shared memory encoding。

### 结构等价而非字面相等

`verifyLayoutsAreEqual` 会先看 attr 是否相等；不相等时转成 `LinearLayout` 比较。必要时可忽略 register broadcast basis。这样可以接受“名字/具体 attr 不同，但线性映射等价”的 layout，避免把可等价的情况误判为冲突。

## 6. 一条典型链路

1. Triton IR tensor 进入 TritonGPU conversion，如果没有 encoding，先获得默认 `BlockedEncodingAttr`。
2. Coalesce pass 会给昂贵 load/store 选更适合 memory coalescing 的 blocked layout，并插入必要 convert。
3. Matmul acceleration pass 会把 dot result/accumulator 转成 NVIDIA/AMD MMA encoding，把 A/B 转成 `DotOperandEncodingAttr` 或 shared MMA operand layout。
4. View/rank-changing 算子通过 `DialectInferLayoutInterface` 保持 no-op layout 变换，尽量不移动数据。
5. `RemoveLayoutConversions` 从 anchor 出发前向传播候选，解决冲突，重写 value type，并在边界插入 `ConvertLayoutOp`。
6. 对剩余 convert，尝试反向 rematerialization 和 hoist，能消就消，不能消就保留为显式 layout 转换。

## 7. 关键源码索引

- layout attr 定义：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`
- layout trait 定义：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td`
- CGA layout：`triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td`
- `LinearLayout` 定义：`triton/include/triton/Tools/LinearLayout.h`
- encoding 到 linear layout：`triton/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h`、`triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`
- TritonGPU infer layout 实现：`triton/lib/Dialect/TritonGPU/IR/Dialect.cpp`
- dialect infer layout 接口：`triton/include/triton/Dialect/Triton/IR/Dialect.h`
- 一跳传播工具：`triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp`
- remove/propagate layout conversions：`triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`
- 默认 type conversion：`triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp`
- dot 初始转换：`triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`
- NVIDIA matmul layout：`triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`
- AMD matmul layout：`triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- coalesced load/store layout：`triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`、`triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp`
- descriptor shared memory layout：`triton/lib/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.cpp`
