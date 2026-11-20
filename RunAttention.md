# Attention 编译与启动

`RunAttention.py` 根据 BHSD、BM/BN/BK 和 causal 模式生成输入，再调用现有
`TuneAttention.py` 串起 `MyTest → GenHsaco.sh → launchKernel.py`。
默认执行完整流程，包括随机输入、正确性校验和计时。

## 单组配置

在仓库目录执行：

```bash
python3 RunAttention.py --b 1 --h 32 --s 4096 --d 128 --causal true
python3 RunAttention.py --shape 2,8,1024,64 --non-causal --bm 64 --bn 32 --bk 32
```

BHSD 分别是 batch、head 数、序列长度、head 维度。默认 BHSD 为
`1,32,4096,128`，BM/BN/BK 为 `128/128/32`，使用 online softmax。
`--causal` 接受 `true/false/1/0/both`，省略值等同于 `true`；兼容 `--casual` 拼写。

## 批量配置

```bash
# 各维度取笛卡尔积，同时测试 causal 和 non-causal
python3 RunAttention.py --b 1 2 --h 8 --s 1024 4096 --d 64 128 --causal both

# 只跑指定的两组 BHSD
python3 RunAttention.py --shape 1,32,4096,128 --shape 2x8x1024x64 --causal both

# 分块也可扫描
python3 RunAttention.py --bm 64 128 --bn 32 64 --bk 32 --causal both
```

`--shape` 不能与 `--b/--h/--s/--d` 混用。重复配置会去重。
S 支持任意正整数，BM/BN 不必整除 S；BK 仍须为 16 的正整数倍且整除 D。
尾块使用固定 BM/BN：Q/K/V 越界读取补零、O 越界行不写回，softmax 屏蔽无效 key。
non-causal 也保留尾块 mask。当前非整除配置使用受保护的同步访存，整除配置保持原路径。
生成合法输入后，分块仍需满足 MyTest 后端的硬件指令和共享内存限制。

```bash
python3 RunAttention.py --s 1 65 129 1000 --bm 64 --bn 32 --bk 32 --causal both --stage ir
```

这是静态 shape 支持：每个 S 生成独立 IR，尚未增加运行时 seqLen 参数。
生成器为非整除配置设置 `frisk.checked_global_io` kernel 属性，并向下传递到 `func.func`。
`FinalizeThreadTiling` 在 view 索引合并后、地址展平前，按原始 tensor 的每个维度检查
global load/store；向量包完全有效时保留向量访存，跨边界时逐元素保护。
该属性的语义是越界读零、越界写忽略，不依赖调用方分配 padding。
此路径不提升为 DMA；手写的未降为同步 copy 的异步访存会报错。

回归测试（CPU LLVM 执行模型，不模拟 GPU 时序）：

```bash
python3 3rd/deepgengraph/test/unittest/check_attention_tails.py
```

## 工具与编译参数

文件可以直接平铺到目标机器的同一目录：

```text
RunAttention.py
TuneAttention.py
generate_attention_ir.py
GenHsaco.sh              # 也支持 genHsaco.sh
launchKernel.py
MyTest
test_input.mlir
```

模板优先读取脚本同目录的 `test_input.mlir`，缺失时回退到仓库的
`3rd/deepgengraph/test/test_input.mlir`。MyTest 同样优先使用脚本同目录的二进制。
平铺目录内已有 MyTest 时直接运行，不加 `--build`。

只生成 MLIR 或 LLVM IR 不需要 GPU。默认 MyTest 路径是
`3rd/deepgengraph/build/test/MyTest`，可以用 `--tool` 指定或用 `--build` 先构建。

生成 HSACO 需要目标 DTK 的 LLVM 工具及 device bitcode。`GenHsaco.sh` 继承
`LLVM_DIR`、`ROCM_BC`、`GPU_ARCH`、`GPU_FEATURES` 和 `DWARF_VERSION`。
例如，将路径换成目标机器上的安装位置：

```bash
export LLVM_DIR=/path/to/dtk/aillvm/bin
export ROCM_BC=/opt/dtk/amdgcn/bitcode
export GPU_ARCH=gfx936
python3 RunAttention.py --shape 1,32,4096,128 --causal both \
  --launch-python /path/to/hip-python/bin/python --opt-level 3
```

启动需要对应 GPU、HIP runtime 和 HIP 版 PyTorch。
`--launch-python` 默认使用运行脚本的 Python；也可指定 `--gen-hsaco` 和 `--launcher`。

默认采用 `TuneAttention.py` 的 `baseline` 编译预设，可以选择其它预设并覆盖参数：

```bash
python3 RunAttention.py --causal false --case online-vdb-k64
python3 RunAttention.py --case baseline --set qk-tile=64 --set vo-tile=32
python3 TuneAttention.py --help
```

预设仅控制编译优化；是否 causal 由生成的输入 IR 决定。
`--k-layout bhds` 仅用于未启用 K normalization 的旧版 MyTest；当前默认是 `bhsd`。

## 执行阶段和结果

```bash
python3 RunAttention.py --causal both --stage ir     # 仅生成输入
python3 RunAttention.py --causal both --stage llvm   # 编译到 LLVM IR
python3 RunAttention.py --causal both --stage hsaco  # 生成 HSACO 和启动配置
python3 RunAttention.py --causal both --stage launch --warmup 10 --repeat 50 --device 0
```

`--stage` 默认 `launch`。`--timeout` 控制每个编译或启动阶段的超时，默认 300 秒。
`--seed` 控制输入随机种子。

每次在 `generated_ir/attention_runs/run-*` 下新建运行目录，也可用 `--output-dir`
指定父目录。启动时打印实际路径。目录内包含：

- `inputs/*.mlir`、`inputs.json`：各配置的原始输入和参数。
- `pipeline.json`：本次交给 TuneAttention 的完整参数。
- `compiled/run-*/.../`：各候选的 `input.mlir`、`kernel.ll`、`lowering.log`、
  `config.json`、`result.json`、可重放的 `command.sh`。
- HSACO 阶段另有 `kernel.hsaco`、汇编文件、`hsaco.log` 和 `launch.json`；
  启动阶段另有 `benchmark.log` 和 `benchmark.json`。
- `summary/summary.json`、`summary/best_by_shape.json`：实时进度和最佳结果。
  最佳结果按 `BxHxSxD-causal` / `BxHxSxD-noncausal` 分开记录。

`launch.json` 从生成的 MLIR 和 LLVM IR 提取启动参数。当前模板的
`gridDim = (S / BM, H, B)`，`blockDim` 从 LLVM 的 `!reqd_work_group_size`
读取，并同步传递 BHSD 和 causal 状态。例如第二条单组命令的 grid 是 `(16,8,2)`。

任一配置编译失败、缺少产物、启动失败或正确性校验失败，最终命令都会返回非零状态；
已完成的结果和失败日志会保留，后续配置仍会继续执行。

## 本地回归检查

```bash
python3 3rd/deepgengraph/test/check_attention_runner.py
python3 3rd/deepgengraph/test/check_attention_shapes.py
```

第一组测试使用后端替身验证完整脚本的参数、产物和失败处理，不替代目标 GPU 验证。
