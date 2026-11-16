# Non-causal GEMM computation/result layout 分离

此目录使用 `../best_noncausal_early_wave/kernel.o3.ll` 对应的同一 case：输入 MLIR、
MyTest 命令和 launch 参数保持一致。BHSD=1/32/4096/128，BM/BN/BK=128/128/32，
256 threads，block-warps=4,1，WMMA 3.3，non-causal，approx 模式。

`LowerInfoMap::GemmLayouts` 现在分别记录指令的 `computation` C 布局与对外的
`result` 布局。acc 推断传播 result；tiling 显式建立输入、输出转换。
已融合的 GEMM 递推在循环内部保持 computation，并在循环出口恢复 result。
这不依赖独立 Add；FriskOpFuse 的 5 处融合仍保留，block Add 数量仍由 7 降至 2。

原版与本次 O3 IR 的 MMAC=256、DMA128=16、DMA64=20、matrix read=32、
scheduler barrier=306、fadd=68、fmul=42、store=64 均一致。
`ds.bpermute` 恢复为 72（上一次融合产物为 8）；恢复的是出口转换，循环内部
没有额外的累加器 shuffle/LDS scratch。K DMA、PV interleave、V double buffer、
cross-BN K0 等仍按原配置启用。

验证记录：

- `regression.log`：LowerInfo 两套布局、两种 B-permute、非零和循环 acc、
  typed/vector lowering、正反向 shuffle、融合/未融合循环及 strict attention 回归。
- raw/O3 均通过 `llvm-as` 与 LLVM verifier。
- `validation.json`：delayed-DMA CPU LLVM 模型，raw/O3 各测试 random 与
  future_key_dominates。最大绝对误差分别约 0.00012854、4.62e-9。
- `trace_comparison.json`：在 random seed42、head0/query0、完整 4096 keys 上，
  原版与本次 8192 次 MMAC 输入及顺序、LDS 读值/轨迹、global 访问轨迹、K0 issue、
  每次 store 的逐线程地址和最终输出均一致，输出逐位相等。
  lane0/16/32/48 首次写回列为 [0,4,8,12]，下一次为 [1,5,9,13]。
- 未运行目标 GPU；以上不代表 GPU 性能已恢复。

测试入口：

```bash
bash generated_ir/best_noncausal_gemm_layout_contract/run_gpu.sh
```

该脚本在已配置 DTK 的 GPU 环境下，从 `kernel.ll` 经仓库 `GenHsaco.sh` 的 O3
流程编译 HSACO，并在独立 `runs/` 子目录保留日志、结果与输入哈希。
也可直接测试本目录已优化好的 `kernel.o3.ll`，launch 参数见 `kernel.o3.launch.json`。

重新生成与验证：

```bash
bash generated_ir/best_noncausal_gemm_layout_contract/command.sh > generated_ir/best_noncausal_gemm_layout_contract/lowering.log 2>&1
/data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/best_noncausal_gemm_layout_contract/kernel.ll -o generated_ir/best_noncausal_gemm_layout_contract/kernel.o3.ll
python3 generated_ir/best_noncausal_gemm_layout_contract/verify.py
python3 generated_ir/best_noncausal_gemm_layout_contract/compare_trace.py
```

`frisk-base-before-fuse.mlir`、`frisk-base.mlir`、`frisk-fragment.mlir` 保存融合前后
和 fragment IR；`lowering.log` 保存完整过程，包含四个 PV 循环出口的 `loop_result`
转换；`generation.json` 保存工具、输入、产物和源码哈希。
