本目录保存 mask-add-bypass 前移的编译和验证产物。

主产物为 [mask_finite_pk_rtz.o3.ll](mask_finite_pk_rtz.o3.ll)，配套
[launch 配置](mask_finite_pk_rtz.o3.launch.json)、[原始 LLVM IR](mask_finite_pk_rtz.ll)、
[完整生成命令](mask_finite_pk_rtz.command.sh) 和 [manifest](manifest.json)。
参数来自 `pmc_test.log` 第 31 行指向的历史产物及其同名 command.sh，所有调优参数保持原样：
causal、shape=(1,32,4096,128)、256 threads、group4 reverse、useKPack、PKRTZ、
causal-mask=finite、mask-add-bypass=approx、uniform-masks=false、fuse-mask-add=false。

本次实现：Base -> Fragment 保存并验证 `bypass_mode`；Fragment -> Thread 与直接
Base -> Thread 共用 `lowerFragmentMask`。approx 在已有 uniform 分支、融合 add、
clear=+0 且 FP 环境允许时直接 yield score；masked 分支保留原加法和操作数顺序。
legal 继续交给 LLVM 证明，未命中高层融合的情况也保留 LLVM fallback。
TuneAttention 分开记录 early/LLVM 命中数，early elements 表示每线程标量元素数。

指定参数的生成结果：

- LLVM assembler 和 `default<O3>,verify` 均通过。
- early bypass：0 groups / 0 elements，原因是 no-uniform-branch。
- LLVM fallback：0 groups / 0 additions。
- finite 模式已有直接 select；O3 中 score/mask 的 vec4 加法为 0，PKRTZ packed pairs 为 32。
- 新原始 LLVM IR 与修改前重新生成的 baseline.ll SHA256 完全一致。
- 新 O3 IR 与 baseline.o3.ll 除 ModuleID 路径外完全一致。
- 当前编译器的修改前基线已与历史 pmc IR 不同，因此不声称复现历史 IR 字节或 GPU 耗时；
  两项比较及输入、编译器、产物哈希均记录于 manifest.json。
- 本次未执行 GPU benchmark 或目标 GPU 代码生成。

独立命中样例：[fragment IR](bypass.fragment.mlir)、[thread IR](bypass.thread.mlir)、
[生成命令](bypass_commands.json)。样例保留完整的 overflow guards；clear 分支只有 yield score，
masked 分支保留 fused add。该样例使用测试 fixture，独立于上面的 pmc 参数产物。

验证全部通过：

- `check_early_mask_add_bypass.py`：46 个 direct/typed 正向及回退用例、非法模式验证、
  序列化默认值与幂等性；327,680 次实际 lowered mask 的原生 CPU 对照。
  覆盖两种操作数顺序、全部 128 lanes 和每线程 16 个寄存器元素、clear/diagonal/masked/
  wrapping tiles、signed zero、NaN、正负无穷、subnormal 和有限极值。
- `check_mask_add_bypass.py`：LLVM off/legal/approx、回退和幂等性、600 次原生 CPU FP 对照。
- 原有 `check_mask_add_fusion.py`、`check_uniform_masks.py`、`check_causal_mask_lowering.py`、
  `check_semantic_fragments.py`、`check_typed_fragments.py` 全部通过。
- `python3 -m unittest script.test_tune_attention`：27 项通过。

日志分别保存在 early_tests.log、llvm_fallback_tests.log、check_*.log、tune_tests.log。

重生成主产物（仓库根目录）：

```bash
python3 generated_ir/early_mask_add_bypass/generate.py
```

重跑新增回归：

```bash
python3 3rd/deepgengraph/test/check_early_mask_add_bypass.py \
  3rd/deepgengraph/build/test/ThreadTilingTest
```

另保留 baseline_uniform.log：修改前编译器只将指定参数的 uniform-masks 改为 true 后，
已有 K DMA pass 报 `requires a single compute/latch block`。因此没有修改主产物的参数来
强行制造 bypass 命中，也没有更改 K DMA 的控制流支持范围。
