# Non-causal GEMM accumulator 回归产物

基准 case：`../best_noncausal_early_wave/kernel.o3.ll`。
输入、launch 参数、MyTest 命令均沿用该目录，用当前编译器重新生成。

- BHSD：1 / 32 / 4096 / 128；non-causal。
- BM / BN / BK：128 / 128 / 32；256 threads；block-warps=4,1。
- WMMA 3.3；group32；K swizzle 8:6；LazySoftmaxBase=8；SoftmaxCvtPack=rtz。
- 其余参数见 `command.sh`；新产物尚未做 GPU 正确性或性能实测。

文件：

- `kernel.o3.ll`：当前编译器导出后执行 `default<O3>,verify` 的 LLVM IR，供对比测试。
- `kernel.ll`：MyTest 原始 LLVM 导出。
- `kernel[.o3].launch.json`：对应 launch 参数。
- `input.mlir` / `command.sh` / `lowering.log`：输入、复现命令和各阶段 IR。
- `frisk-base.mlir` / `frisk-fragment.mlir`：三操作数 GEMM 及 fragment lowering 的中间 IR。
- `generation.json`：输入、工具、产物哈希及检查结果。

在仓库根目录重新生成：

```bash
bash generated_ir/best_noncausal_gemm_accumulator/command.sh > generated_ir/best_noncausal_gemm_accumulator/lowering.log 2>&1
/data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/best_noncausal_gemm_accumulator/kernel.ll -o generated_ir/best_noncausal_gemm_accumulator/kernel.o3.ll
```

在已配置 DTK 的 GPU 环境编译并测试（`GenHsaco.sh` 自带 O3 流程）：

```bash
bash generated_ir/best_noncausal_gemm_accumulator/run_gpu.sh
```

脚本使用当前目录的 `kernel.ll` 和 launch 配置，在独立的 `runs/` 子目录保存 HSACO、构建日志及测试结果。
