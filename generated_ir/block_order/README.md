# Block 访问顺序对照 IR

本组实验已统一以 [baseline_swap_bx_by.ll](baseline_swap_bx_by.ll) 为源文件，
其映射为 **bx=query tile、by=head**，只改 `Attn_p2` 入口的 workgroup ID 映射。
源 SHA-256：`24bdcbd39aa0761fee22119b1ff6c80bda2d637eaa8e691b51deda885ce2214c`。
固定 BHSD=`1,32,4096,128`、BM/BN/BK=`128/128/32`、FP16、causal，Q/K/V/O 连续 BHSD。
所有版本均使用 **grid=(32,32,1)、block=(256,1,1)、dynamic shared=0**。
这是该固定基线的实验，不是任意 shape 的通用重写。

`baseline.ll` 和 `query_fast.ll` 现在都是该源文件的逐字节副本。
旧基线的 head 优先顺序改名为 `head_fast.ll`，仍用于对照。
`query_fast` 只作为兼容名称保留，默认不重复测量；默认共八个不同映射。
之前的 IR、脚本和记录已归档到 `history/head_fast_baseline-时间/`，
原有 `run-*` 结果不改动；历史的 `speedup_vs_baseline` 不能与本轮直接混用。
源文件验证记录见 [baseline_swap_bx_by.validation.json](baseline_swap_bx_by.validation.json)。

## 版本和映射

令物理 block 为 `(x,y)`，线性编号 `t=x+32*y`，逻辑 `(h,q)` 为 head 和 query tile。
本实验将相邻线性编号分配给共享 K/V 的逻辑 block；并不强制硬件按 t 顺序执行，缓存和时间收益需实测。

| 文件 | G | 按 t 递增时的逻辑顺序示例 |
|---|---:|---|
| `baseline.ll` | 32 | 新基线：`(h0,q0),(h0,q1),…,(h0,q31),(h1,q0),…` |
| `head_fast.ll` | 1 | 旧顺序对照：`(h0,q0),(h1,q0),…,(h31,q0),(h0,q1),…` |
| `group2.ll` | 2 | `(h0,q0),(h0,q1),(h1,q0),(h1,q1),…` |
| `group4.ll` | 4 | 同一 head 的 q0..3，再下一 head；所有 head 结束后再 q4..7 |
| `group8.ll` | 8 | 同一 head 的 q0..7，再下一 head |
| `group16.ll` | 16 | 同一 head 的 q0..15，再下一 head |
| `query_fast.ll` | 32 | 与新 `baseline.ll` 逐字节相同；默认不重复运行 |
| `group8_reverse.ll` | 8 | group8 的每个逻辑 q 替换为 `31-q`：先 q31..24 |
| `query_fast_reverse.ll` | 32 | 同一 head 先 q31..0，再下一 head |

统一公式：

```text
h = (t // G) % 32
q = (t // (32*G)) * G + (t % G)
逆序版本：q = 31-q
```

`baseline`、`query_fast` 是源文件的逐字节副本，都是 `h=y,q=x`。
G 仍表示“同一 head 连续分配多少个 query tile”，不因换基线而交换它的含义。
因此 group2/4/8/16 及两个 reverse 的映射和计算主体保持原定义，重新从新源生成后
部分文件哈希会与旧版相同。这是预期行为，不是遗漏了重生成，也不会再次交换新源的 bx/by。
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

生成器锁定源 SHA-256，且不会覆写 `baseline_swap_bx_by.ll`；源版本变化时会拒绝，防止 SSA/ABI 静默错配。
`manifest.json` 保存各 IR 哈希及配置；`validation.json` 记录实际完成的检查：

- 执行每份 IR 的入口，枚举全部 1024 个物理 block，核对所有 `(head,query)` 恰好覆盖一次。
- 还原入口后与源文件逐字节一致。
- LLVM assemble、O3 和 verifier。
- `--numerics` 对每个版本执行逻辑 `(h,q)=(0,0),(1,7),(1,31)` 的完整延迟 DMA CPU 模型，
  结果与交换 bx/by 后的新基线逐位一致、对稳定 attention 参考满足 `atol=rtol=1e-2`，只写目标输出 tile。

CPU 模型不模拟独立 wave 时序、缓存、GPU 调度和目标 DTK 机器码。
所有 head 的映射均已枚举；完整数学模型仅抽样两个 head。性能与硬件正确性需要上机验证。

## 在 gfx936 机器编译并测量

先按机器安装位置设置 `LLVM_DIR`（目标 DTK 的 llvm-link/opt/llc/ld.lld 所在目录）
和 `ROCM_BC`（DTK amdgcn/bitcode 目录）；沿用原来能编译基线的环境。
`GPU_ARCH` 默认 gfx936。运行 Python 环境需有 HIP PyTorch，可用 `--launch-python` 指定。

先测四个版本：

```bash
python3 generated_ir/block_order/run.py \
  --variants baseline head_fast group4 group8 \
  --rounds 5 --warmup 50 --repeat 200
```

全部八个不同映射（不重复运行 `query_fast` 别名）：

```bash
python3 generated_ir/block_order/run.py --rounds 5
```

只查看命令、不编译运行：

```bash
python3 generated_ir/block_order/run.py \
  --variants baseline head_fast group8 --rounds 1 --dry-run
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

`summary.json` 按多轮 kernel median 的中位数比较：`speedup_vs_baseline > 1` 表示比
**新基线 `baseline_swap_bx_by.ll`（bx=query、by=head）** 快，每项包含 `baseline_sha256`；
`median_sdpa_speedup` 是相对 SDPA 的加速比，两者不要混淆。参考 SDPA backend 仍由 PyTorch 自动选择，
该脚本不强制 backend；需从 profiling 确认实际参考 kernel。

若手工运行单份已经编译的 HSACO：

```bash
python3 launchKernel.py --hsaco <run目录>/group8/kernel.hsaco --kernel Attn_p2 \
  --shape 1,32,4096,128 --grid 32,32,1 --block 256,1,1 \
  --shared-mem 0 --k-layout bhsd --warmup 50 --repeat 200
```

## 怎么解释结果

**用户确认 hipprof 时间不准；性能排序和收益计算统一采用程序的 `time_our_mid`。**
`launchKernel.py` 写入 JSON 的 `kernel_median_ms` 就是该值，批量脚本已使用这个字段。
hipprof 只提供 counter 线索，不使用其 `kernel time` 或据其推导的吞吐给候选排序。
先看程序 median 和跨轮波动，0.2% 的差别通常不足以下结论。
对 baseline 和优胜者单独采集 PMC，比较 L2 hit、
L2/HBM 请求字节数、VGPR/SGPR/scratch、实际 occupancy 和执行尾部。
计时与 PMC 分开，使用同一 GPU/DTK/输入及参考 backend；L2 hit 提高本身不算成功。
若资源数基本相同、HBM 流量减少且时间下降，才支持缓存复用解释。
若 reverse 更好但流量基本不变，应进一步检查尾部调度。
若所有重排都无稳定收益，应降低这个方向的优先级，转向分块/寄存器和 Q/K 等待。

## 已提供的测量（2026-10-21，八种映射均已测）

来自仓库根目录 `pmc_test.log`，程序 `time_our_mid` 与同条 L2 counter 如下：

| 版本 | time_our_mid ms | L2 hit | 相对 head_fast 耗时减少 |
|---|---:|---:|---:|
| head_fast | 2.159039 | 11.77% | — |
| group2 | 2.064561 | 23.96% | 4.38% |
| group4 | 2.014640 | 34.05% | 6.69% |
| group8 | 2.087359 | 53.56% | 3.32% |
| group16 | 1.978400 | 61.25% | 8.37% |
| baseline_swap_bx_by（当前 baseline） | 1.973675 | 60.35% | 8.59% |
| group8_reverse（当前已测最快） | **1.848958** | 59.28% | **14.36%** |
| query_fast_reverse（L2 命中率最高） | 1.945200 | **70.16%** | 9.90% |

group8_reverse 的同条 SDPA 时间为 1.117680 ms，加速比 **0.604492**。
相对 group8 耗时减少 **11.42%**，相对 swap 减少 **6.32%**，相对 group16 减少 **6.54%**。
八组资源均为 VGPR=216、SGPR=112、LDS=16 KiB、scratch=0，LDS operations 也相同。
reverse 的 L2 hit 低于 swap/group16，bank conflict 反而为 25.37%，说明不能只用这些百分比解释收益。
逆序将长 query tile 提前、短任务留在后面，可能减少执行尾部；也改变 K/V 前缀进入缓存的时序。
这两个机制还需 block/CU 时间线及 L2/HBM bytes 区分，当前没有证明全部收益来自长尾改善。

query_fast_reverse 的同条 SDPA 加速比为 **0.576846**，命中率比 group8_reverse 高
**10.88 个百分点**，但程序耗时多 **5.21%**，因此继续以 group8_reverse 作为性能优化起点。
这里的 `L2 cache hit rate` 是命中率，不是带宽/硬件利用率，不能直接推出谁的 HBM 流量更小。
G=32 reverse 相对 swap 仅少用时 **1.44%**，G=8 reverse 相对 group8 却少 **11.42%**，
下一步应配对复测，确认分组大小和顺序的组合收益。

query_fast_reverse 每个 head 都排完整 q31..0；group8_reverse 则先给所有 head 排长 query 组，
最后给所有 head 排短 query 组。在线性编号的最后 256 个 block 中，前者仍有 4,224 轮 key-tile
迭代，后者仅 1,152 轮；这可能影响尾部负载，但编号顺序不是硬件执行顺序，仍需时间线验证。

八种映射已完成首轮测量，建议轮换复测以下四组：

```bash
python3 generated_ir/block_order/run.py \
  --variants baseline group8 group8_reverse query_fast_reverse \
  --rounds 5 --warmup 50 --repeat 200
```

下一阶段性能优化可从 `group8_reverse.ll` 出发，但本目录生成源和命名 `baseline.ll`
仍为 swap，未改写 IR 或历史数据；`summary.json` 的 `speedup_vs_baseline` 仍相对 swap，
group8_reverse 本次约为 **1.067452 倍**。更改 BM/BN 时应按新的 query tile 数重建映射，
不能直接沿用该固定 shape 的 32 个 query tile 公式。

原日志中另有名为 `baseline.ll` 的 2.230714 ms 旧记录，不能按现文件名归到新基线；
当前 `baseline.ll` 和 `baseline_swap_bx_by.ll` 已是同一文件内容，历史记录缺少二进制哈希。
本节是对用户上机日志的整理，`validation.json` 中的 `gpu_tested=false` 仍表示本地验证流程
只运行了 CPU/LLVM 检查，不将这些未绑定 HSACO 哈希的日志伪装成该流程的 GPU 验证。
