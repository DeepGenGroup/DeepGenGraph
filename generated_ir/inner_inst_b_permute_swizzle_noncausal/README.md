# Layout 3.4 的 non-causal K swizzle

当前形状：BHSD=`1,32,4096,128`，BM=BN=128，QK/VO tile=32，256 threads，waves=`4,1`。

建议配置为 `layout=3.4 --swizzle-k=7:9 --k-dma=64-swizzle`，Q/V swizzle 保持 off。
这里“最优”指下述地址模型和搜索空间中的 K bank 冲突最小值；没有 gfx936 GPU 实测，不能据此断言总耗时最优。

## 为什么是 7:9

3.4 仅在单条 MMAC 的 16 列内交换 B 的列位。一个 K b64 读取的前 16 个线程，未 swizzle 的相对地址为：

```text
0,64,128,192,16,80,144,208,32,96,160,224,48,112,176,240
```

旧 `8:6` 加 `9:3` 没有把这组线程变化的关键 bit 7 引入 bank 索引，仍有 2 路冲突。
新配置按顺序执行（a 为 allocation 相对字节地址）：

```text
a1 = a  XOR (((a  >> 7) & 1) << 9)   # --swizzle-k=7:9
a2 = a1 XOR (((a1 >> 9) & 1) << 3)   # --k-dma=64-swizzle 的固定 9:3
```

因此最终 bit 3 包含原 bit 7；上述地址变成：

```text
0,64,648,712,16,80,664,728,32,96,680,744,48,112,696,760
```

在 32 banks、4 B/bank、b64 每 16 连续线程一个 phase、相同 word 可广播的模型中，这组读取覆盖全部 32 banks，没有冲突。两个 swizzle 必须配合使用；只改 `--swizzle-k=7:9` 而使用 128-bit K DMA 不能消除冲突。

`analyze.py` 从实际 IR 中提取全部 64 个带 K ownership/预取标记的 b64 读取，逆变换旧布局，再枚举：

- off 或 source/target 在 byte bits 4..12 且不同的单 XOR：73 种。
- K DMA 为 128、64、64-swizzle：3 种，共 219 组地址模型配置。

只在 `7:9 + 64-swizzle` 时全部 K 读取达到 multiplicity=1。无额外 9:3 的 packet-preserving 单 XOR 最好仍为 2。只编译了基线与胜出配置，不把模型扫描描述为 GPU benchmark。完整记录见 `swizzle_analysis.json`。

## 产物与复现

- `kernel.ll`：当前编译器直接导出的推荐 IR。
- `kernel.o3.ll`：本地 LLVM O3 + verifier 产物。
- `kernel.launch.json`：non-causal 启动配置。
- `baseline.ll`：同样的 3.4 配置，K swizzle 为旧 `8:6`。
- `command.sh`：完整生成命令；保持现有 PV interleave=4、V 双缓冲、K LDS prefetch=8、K tail prefetch=4、cross-BN K0 和 group32 forward。
- `numerical_validation.json`：与 baseline 的逐位比较、FP64 reference、延迟 DMA/LDS ownership 检查及文件哈希。

验证结果：7 个输入/位置场景 × 原始和 O3 两种 IR，共 14 项全部通过，均与旧 3.4 baseline 逐位一致；最大 FP64 reference 绝对误差 `0.000145755813135362`。每个 workgroup 的 wait/barrier/MMAC/DMA64 次数保持 `260/261/8192/512`，V 与 K0 调度检查通过，LDS 为 16384 B。

从仓库根目录运行：

```bash
cmake --build 3rd/deepgengraph/build --target MyTest LegacyLLVMExportTest -j 2
bash generated_ir/inner_inst_b_permute_swizzle_noncausal/command.sh > generated_ir/inner_inst_b_permute_swizzle_noncausal/lowering.log 2>&1
/data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/inner_inst_b_permute_swizzle_noncausal/kernel.ll -o generated_ir/inner_inst_b_permute_swizzle_noncausal/kernel.o3.ll
OPENBLAS_NUM_THREADS=1 python3 generated_ir/inner_inst_b_permute_swizzle_noncausal/analyze.py
OPENBLAS_NUM_THREADS=1 python3 generated_ir/inner_inst_b_permute_swizzle_noncausal/validate.py
```

## 编译兼容性修复

新布局的地址经 LLVM 化简后包含 `icmp + select`。原 `LDSAddressAnalysis.h` 未支持这两类表达式，导致 V double buffer 报 `unknown LDS read address`、K DMA 报 `unknown shared read footprint`。
本次补充逐线程整数比较和 select 的精确求值，未知谓词仍拒绝，只求值被选中的分支。`check_k_dma.py` 增加 signed/unsigned 比较、未选中 poison、上半 wave 越界和未知指针比较的回归检查。

回归检查：K DMA 7 项、V double buffer 28 cases、cross-BN K0 8 项通过。CPU 数值模型不模拟独立 wave 时序、寄存器分配或 GPU 性能；硬件正确性和耗时仍需在目标机测量。
