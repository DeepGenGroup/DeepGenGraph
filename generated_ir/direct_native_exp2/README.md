# Direct native exp2 lowering

本目录主产物沿用 `../best_noncausal_gemm_accumulator` 的 non-causal attention 输入、launch 配置和全部调优参数，用本次修改后的编译器重新生成。

- BHSD = 1 / 32 / 4096 / 128，FP16 输入，FP32 exp2。
- BM / BN / BK = 128 / 128 / 32，256 threads，block-warps = 4,1。
- math = approx，WMMA 3.3，K swizzle 8:6，LazySoftmaxBase = 8，SoftmaxCvtPack = rtz。
- 直接转换路径：`frisk.exp2 -> math.exp2 -> rocdl.exp2 -> llvm.amdgcn.exp2.f32`。
- 主编译流程已移除 LLVM 阶段的 OCML exp2 调用替换；历史 `.ll` 实验兼容代码仅留在 LegacyLLVMExportTest。

## 测试入口

在仓库根目录执行（需要已配置 DTK 的 GPU 环境）：

```bash
bash generated_ir/direct_native_exp2/run_gpu.sh
```

脚本编译 `kernel.ll` 并使用 `kernel.launch.json` 做 non-causal 正确性和性能测试，结果位于本目录 `runs/`。尚未进行 GPU 实测。

## 文件

- `kernel.ll`：完整调优配置的 LLVM 导出，可交给 GenHsaco.sh。
- `kernel.o3.ll`：经 `default<O3>,verify` 的 LLVM IR。
- `kernel.launch.json` / `kernel.o3.launch.json`：launch 参数。
- `input.mlir` / `command.sh` / `lowering.log`：完整复现输入、命令和中间 IR。
- `checks/{strict,mmac,mmac-recip,mmac-exp2,approx}/kernel.ll`：同一输入、统一基础配置的五种模式对照。它们不启用主产物中的完整调优选项，不能据此把性能差异全归因于 exp2。
- `checks/modes.json`：LLVM 翻译刚完成时以及最终导出时的 exp2 调用统计。
- `generation.json`：产物校验与 SHA256。

重新生成主产物：

```bash
bash generated_ir/direct_native_exp2/command.sh > generated_ir/direct_native_exp2/lowering.log 2>&1
/data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/direct_native_exp2/kernel.ll -o generated_ir/direct_native_exp2/kernel.o3.ll
```

重新验证：

```bash
python3 3rd/deepgengraph/test/check_native_exp2.py
python3 generated_ir/direct_native_exp2/check_modes.py
```

`strict` / `mmac` / `mmac-recip` 保留 OCML；`approx` / `mmac-exp2` 在 MLIR lowering 时直接生成原生 FP32 exp2。FP16 / FP64 继续使用 OCML，带 strictfp / optnone 的 LLVM 函数也保留 OCML。
