# Block 访问顺序对照 IR

基于 `../pipeline_cleanup/nohint_unrolled.ll`，只改 `Attn_p2` 入口的 workgroup ID 映射。
固定 BHSD=`1,32,4096,128`、BM/BN/BK=`128/128/32`、FP16、causal，Q/K/V/O 连续 BHSD。
所有版本均使用 **grid=(32,32,1)、block=(256,1,1)、dynamic shared=0**。
这是该固定基线的实验，不是任意 shape 的通用重写。

另外提供 [baseline_swap_bx_by.ll](baseline_swap_bx_by.ll)：直接从 `baseline.ll` 只修改两行，
将 `%7` 的 `workgroup.id.y` 改成 `workgroup.id.x`，将 `%9` 的 `workgroup.id.x` 改成
`workgroup.id.y`，即 **bx=query tile、by=head**。它与 `query_fast.ll` 的映射完全相同，
省去了入口的两个 `add 0`，便于做最小 diff 对照；不在八版本批量脚本的默认列表中。
验证记录见 [baseline_swap_bx_by.validation.json](baseline_swap_bx_by.validation.json)。
当前两个 grid 维度都为 32，launch 参数仍为 `(32,32,1)`。

## 版本和映射

令物理 block 为 `(x,y)`，线性编号 `t=x+32*y`，逻辑 `(h,q)` 为 head 和 query tile。
本实验将相邻线性编号分配给共享 K/V 的逻辑 block；并不强制硬件按 t 顺序执行，缓存和时间收益需实测。

| 文件 | G | 按 t 递增时的逻辑顺序示例 |
|---|---:|---|
| `baseline.ll` | 1 | `(h0,q0),(h1,q0),…,(h31,q0),(h0,q1),…` |
| `group2.ll` | 2 | `(h0,q0),(h0,q1),(h1,q0),(h1,q1),…` |
| `group4.ll` | 4 | 同一 head 的 q0..3，再下一 head；所有 head 结束后再 q4..7 |
| `group8.ll` | 8 | 同一 head 的 q0..7，再下一 head |
| `group16.ll` | 16 | 同一 head 的 q0..15，再下一 head |
| `query_fast.ll` | 32 | `(h0,q0),(h0,q1),…,(h0,q31),(h1,q0),…` |
| `group8_reverse.ll` | 8 | group8 的每个逻辑 q 替换为 `31-q`：先 q31..24 |
| `query_fast_reverse.ll` | 32 | 同一 head 先 q31..0，再下一 head |

统一公式：

```text
h = (t // G) % 32
q = (t // (32*G)) * G + (t % G)
逆序版本：q = 31-q
```

`baseline` 是源文件的逐字节副本。`query_fast` 简化为 `h=y,q=x`。
分组版本只增加 uniform 整数运算，公式中的除法/取模已用移位/掩码实现。
保留原数值 SSA 编号 `%7`（逻辑 q）、`%9`（逻辑 h），因此所有 Q/K/V/O 地址、
causal 循环终点和 mask 都沿用原代码。不能只改 Q/O 地址却留下旧的循环终点。
原始入口之后的计算、LDS 地址、DMA、MMAC、wait/barrier、归约和写回文本完全不变。
新增入口计算仍可能影响后端寄存器分配，因此还要比较目标汇编资源。

同一 head 的相邻 query tile 会读取重叠的 K/V 前缀，分组希望增加其缓存复用机会。
G 太大也可能让长短任务集中、加重尾部不平衡，所以不能假定 query_fast 一定最好。
reverse 只改变哪个 block 做哪份工作，不改变因果语义、每个输出值或总计算量。

## 生成与本地验证

从仓库根目录运行：

```bash
python3 generated_ir/block_order/generate.py
OPENBLAS_NUM_THREADS=1 python3 generated_ir/block_order/validate.py --numerics
```

生成器锁定源 SHA-256；源版本变化时会拒绝，防止 SSA/ABI 静默错配。
`manifest.json` 保存各 IR 哈希及配置；`validation.json` 记录实际完成的检查：

- 执行每份 IR 的入口，枚举全部 1024 个物理 block，核对所有 `(head,query)` 恰好覆盖一次。
- 还原入口后与源文件逐字节一致。
- LLVM assemble、O3 和 verifier。
- `--numerics` 对每个版本执行逻辑 `(h,q)=(0,0),(1,7),(1,31)` 的完整延迟 DMA CPU 模型，
  结果与基线逐位一致、对稳定 attention 参考满足 `atol=rtol=1e-2`，只写目标输出 tile。

CPU 模型不模拟独立 wave 时序、缓存、GPU 调度和目标 DTK 机器码。
所有 head 的映射均已枚举；完整数学模型仅抽样两个 head。性能与硬件正确性需要上机验证。

## 在 gfx936 机器编译并测量

先按机器安装位置设置 `LLVM_DIR`（目标 DTK 的 llvm-link/opt/llc/ld.lld 所在目录）
和 `ROCM_BC`（DTK amdgcn/bitcode 目录）；沿用原来能编译基线的环境。
`GPU_ARCH` 默认 gfx936。运行 Python 环境需有 HIP PyTorch，可用 `--launch-python` 指定。

先测四个版本：

```bash
python3 generated_ir/block_order/run.py \
  --variants baseline group4 group8 query_fast \
  --rounds 5 --warmup 50 --repeat 200
```

全部八个版本：

```bash
python3 generated_ir/block_order/run.py --rounds 5
```

只查看命令、不编译运行：

```bash
python3 generated_ir/block_order/run.py \
  --variants baseline group8 query_fast --rounds 1 --dry-run
```

脚本先独立编译全部候选，再轮换测试顺序。每轮每候选内部由 `launchKernel.py`
交替测 kernel 与 SDPA，并校验输出。全部保持同一个输入 seed；换 seed 可另起一次测试。
如果编译或数值校验失败，脚本停止并给出日志路径；不把失败版本计入排名。
每次创建新的 `run-时间/`，不覆盖旧实验：

```text
run-时间/
  run.json                 # 参数、编译环境变量、脚本哈希
  manifest.json            # 各 IR 配置和哈希
  summary.json             # 每轮完成后更新；比较时确认 completed_rounds 相等
  group8/
    kernel.ll / kernel.hsaco / kernel.hsaco.amdgcn
    binary.json            # IR/HSACO/汇编哈希、核对 16 KiB LDS
    build.log / passes.log
    round-1.json / round-1.log  # 正确性、两种实现原始样本、median 等
```

`summary.json` 按多轮 kernel median 的中位数比较：`speedup_vs_baseline > 1` 表示比原顺序快；
`median_sdpa_speedup` 是相对 SDPA 的加速比，两者不要混淆。参考 SDPA backend 仍由 PyTorch 自动选择，
该脚本不强制 backend；需从 profiling 确认实际参考 kernel。

若手工运行单份已经编译的 HSACO：

```bash
python3 launchKernel.py --hsaco <run目录>/group8/kernel.hsaco --kernel Attn_p2 \
  --shape 1,32,4096,128 --grid 32,32,1 --block 256,1,1 \
  --shared-mem 0 --k-layout bhsd --warmup 50 --repeat 200
```

## 怎么解释结果

先看绝对 median 和跨轮波动，0.2% 的差别通常不足以下结论。
若 group4/group8 稳定快，再对 baseline 和优胜者单独采集 PMC，比较 L2 hit、
L2/HBM 请求字节数、VGPR/SGPR/scratch、实际 occupancy 和执行尾部。
计时与 PMC 分开，使用同一 GPU/DTK/输入及参考 backend；L2 hit 提高本身不算成功。
若资源数基本相同、HBM 流量减少且时间下降，才支持缓存复用解释。
若 reverse 更好但流量基本不变，应进一步检查尾部调度。
若所有重排都无稳定收益，应降低这个方向的优先级，转向分块/寄存器和 Q/K 等待。
