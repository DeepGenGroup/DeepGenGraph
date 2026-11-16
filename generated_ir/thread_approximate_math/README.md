# Thread IR approximate math

`optimizeApproximateRegisterMath` 的倒数复用已前移到 MLIR pass
`frisk-approximate-thread-math`，位于 thread IR 循环展开、寄存器向量规范化之后，
`firstLowering` 之前。`MyTest` 在 `approx` / `mmac-recip` 下调用它。

本配置中 16 个 `arith.divf` 改为 16 个乘法，共用 2 个
`arith.divf 1, denominator fastmath<arcp>`。只在同一基本块内复用，处理 FP32
标量和向量；常量分母、FP16/FP64、strictfp/optnone 函数不改写。

native exp2 继续走已有的 `math.exp2 -> rocdl.exp2` lowering。
GEMM 累加器已有 block/fragment 优化；独立的 LLVM MMAC 融合兜底仍保留。
旧 LLVM 近似数学函数仅供 `LegacyLLVMExportTest` 处理历史 `.ll`，主流程不再调用。

## 测试 IR

- `kernel.ll`：当前完整调优配置的 LLVM 导出。
- `kernel.o3.ll`：`default<O3>,verify` 后的 LLVM IR。
- `thread.before.mlir` / `thread.after.mlir`：新 pass 前后的 thread IR。
- `kernel.launch.json` / `kernel.o3.launch.json`：对应 launch 参数。
- `input.mlir` / `command.sh` / `lowering.log`：复现输入、命令及完整阶段日志。
- `baseline/kernel.ll`：修改前编译器用相同输入和调优参数生成的对照。
- `generation.json` / `tests.log` / `validation.log`：验证结果。

沿用 `direct_native_exp2` 的 non-causal 配置：BHSD=1x32x4096x128，
BM/BN/BK=128/128/32，256 threads，WMMA 3.3，math=approx，
K swizzle=8:6，LazySoftmaxBase=8，SoftmaxCvtPack=rtz。
完整参数见 `command.sh`。

在仓库根目录执行 GPU 正确性和性能测试：

```bash
bash generated_ir/thread_approximate_math/run_gpu.sh
```

脚本使用 `kernel.ll` 和对应 launch 配置，结果写入本目录的 `runs/`。
本次只生成 IR 并做编译器/CPU 验证，未进行 GPU 实测。

验证覆盖五种 math 模式、thread IR 序列化重放、LLVM 原始/O3 verifier，
以及 1005 组 CPU 数值用例和作用域/属性边界。原始 IR 的倒数指令位置已提前，
所以直接 `llvm-diff` 有差异；在比较副本中把两个倒数移至各自首次使用处后，
原始和 O3 两组都通过 `llvm-diff`，见 `baseline/comparison.json`。
比较副本没有用于 GPU 测试。

重新生成和验证：

```bash
cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j 6
bash generated_ir/thread_approximate_math/command.sh > generated_ir/thread_approximate_math/lowering.log 2>&1
python3 3rd/deepgengraph/test/check_thread_math.py
python3 generated_ir/thread_approximate_math/validate.py
```

单独重放 thread IR pass：

```bash
3rd/deepgengraph/build/test/ThreadTilingTest \
  generated_ir/thread_approximate_math/thread.before.mlir \
  --frisk-approximate-thread-math --canonicalize --cse \
  -o /tmp/thread-math.mlir
```
