# Non-causal FriskOpFuse 回归产物

基准为 `../best_noncausal_early_wave/kernel.o3.ll` 对应的 case。
`input.mlir`、MyTest 命令和 launch 配置沿用该目录，使用加入 `FriskOpFusePass` 的编译器重新生成。

- BHSD：1 / 32 / 4096 / 128；non-causal。
- BM / BN / BK：128 / 128 / 32；256 threads；block-warps=4,1。
- WMMA 3.3；group32；K swizzle 8:6；LazySoftmaxBase=8；SoftmaxCvtPack=rtz。
- Frisk block IR 的 5 处 GEMM + Add 融合成显式累加器 GEMM，Add 数量从 7 降至 2。
- strict 模式不启用该重结合；本 case 沿用 approx 模式。
- 尚未做目标 GPU 正确性或性能实测。CPU 模型不代替 GPU 实测。

产物与复现：

- `kernel.o3.ll`：经过 `default<O3>,verify` 的 LLVM IR，供测试。
- `kernel.ll`：MyTest 原始 LLVM 导出。
- `kernel[.o3].launch.json`：对应 launch 参数。
- `frisk-base-before-fuse.mlir` / `frisk-base.mlir`：融合前后的 block IR。
- `frisk-fragment.mlir`：fragment lowering 中间 IR。
- `lowering.log`：完整 lowering 和 LLVM 优化阶段 dump。
- `generation.json`：输入、工具和产物哈希及验证结果。
- `verify.py` / `validation.json`：现有 delayed-DMA CPU LLVM 模型的随机输入与未来 key 主导测试，覆盖 raw/O3。

在仓库根目录重新生成：

```bash
bash generated_ir/best_noncausal_frisk_op_fuse/command.sh > generated_ir/best_noncausal_frisk_op_fuse/lowering.log 2>&1
/data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/best_noncausal_frisk_op_fuse/kernel.ll -o generated_ir/best_noncausal_frisk_op_fuse/kernel.o3.ll
python3 generated_ir/best_noncausal_frisk_op_fuse/verify.py
```

在已配置 DTK 的 GPU 环境编译并测试（`GenHsaco.sh` 自带 O3）：

```bash
bash generated_ir/best_noncausal_frisk_op_fuse/run_gpu.sh
```

脚本读取该目录的 `kernel.ll` 和 launch 配置，在独立的 `runs/` 子目录保存 HSACO、构建日志及测试结果。
