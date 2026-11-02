Non-causal attention：最新汇编归因与可测试优化候选

结论
  SHM operation 上涨主要来自 key tile 迭代数，不是新增 spill 或 LDS 容量。
  每个 query block 原来执行 1..32 轮、平均 16.5 轮，现在固定执行 32 轮。
  本内核的 tile 工作量增加 1.9394 倍，耗时增加 1.6293 倍；按处理 tile 数
  归一化，吞吐反而提高约 19.0%。相对 SDPA 比值降低，是因为本内核时间
  从 1.201280 增到 1.957201 ms，而 SDPA 仅从 1.117440 增到 1.571121 ms
  （1.4060 倍）。新任务仍比 SDPA 慢约 24.57%，要追平需减少自身时间 19.73%。
  两次测量来自日志中的不同实验，不能将差异全部归因于单项改动；没有 SDPA
  的对应 kernel/汇编，不能确定其内部 tiling 或 backend 是否变化。

数据来源
  ../../pmc_test.log 第 1–28 行：最新 non-causal；第 70–98 行：旧 causal。
  ../../k_test.gcn.log 中 Attn_p2 为第 9–3628 行，主循环为第 321–2921 行。
  后续设备库函数不计入主内核指令数。UTF-16 日志按编码读取。
  analysis.json 保存源文件哈希、指令分类、实测值和计算公式。

SHM 的精确核对
  grid=(32,32,1)，block=256，wave64，因此总计 4096 个 wave。
  每 wave 的循环外 DS 指令：Q 读取 8，出口归约和 O 布局转换 68，共 76。
  每轮 DS：K 的 ds_read2* 32，V 的 ds_read_m32x16_b16 32，max bpermute 4，
  合计 68。分母跨 lane 求和已经延迟到出口，循环内没有恢复这项工作。
    causal     = 4096 × (76 + 68 × 16.5) = 4,907,008
    non-causal = 4096 × (76 + 68 × 32)   = 9,224,192
  与两份 PMC 记录完全一致，增长 87.98%。这是该记录下 DS 指令执行计数的
  精确匹配，不能把它当作 LDS 字节数，也不能把 ds_bpermute 当作数组读写。

其他证据与限制
  LDS 容量均为 16 KiB，scratch 均为 0；VGPR 从 248 降至 216。
  最新汇编 NumVgprs=215，硬件分配计数为 216，二者口径不同。
  编译器仍给出 Occupancy:1；这是静态估计，不是实测 active waves。
  每轮还有 256 条 MMAC、66 条 exp、12 个 s_barrier、55 条 s_waitcnt
  （后者同时含 VM/LDS 等等待，不能都算作 DMA wait）。寄存器驻留与
  等待/计算重叠值得优化；现有 counter 不足以给出各项耗时百分比。
  bank conflict 为 12.07% → 14.34%，L2 hit 为 62.12% → 74.08%。
  bank conflict 的小幅变化不能解释已由循环次数完全匹配的 operation 增量；
  cache hit 百分比也不等价于 HBM bytes 或带宽利用率。

候选与用途
  baseline.ll：原始 non-causal LLVM 的逐字节副本，保留 1.957201 ms 对照。

  k0_prefetch.ll：优先测试的调度候选，算法和 FP 运算顺序不变。
    Q 装载完成后先发首轮 K0；每轮最后 V tile 的 LDS 读取结束后，将原有
    reader-completion barrier 提前，在该 barrier 后预取下一轮 K0，随后
    执行当前轮最后 32 条 PV MMAC。32 轮共提供 31×32=992 条 MMAC 的重叠
    窗口（基线为 0），并非节省了 992 条计算。保留下一轮 K0 的原有 wait
    与 publication barrier，最后一轮不再发起下一轮 DMA。
    不增加 LDS 容量、不增加 K/V 读取、不减少必要同步；目标是隐藏延迟。
    因此不应以 SHM operation 必须下降为验收条件；看正确性和 median。
    这是绑定输入 SHA 的 LLVM 实验，不是已经集成的通用编译器 Pass。

  k0_late_control.ll：与预取版有相同的 prologue、地址、guard 和 barrier
    移动，但在最后 32 条 PV 之后才发下一轮 K0，用于区分结构变化与重叠收益。

  threads512.ll：独立的寄存器压力候选，threads=512、waves=8,1。
    仍为 BM=BN=128、BK=32、group4 reverse。每线程逻辑 O/score/Q 状态缩小，
    每 wave 的 MMAC 由 256 变成 128，全网格 MMAC 总工作量不变。
    更多 wave 会重复读取 K/V，预期可能增加 DS operation；最终 VGPR、
    occupancy、scratch 和性能须看目标汇编。它不是“SHM 更少”的候选。

没有交付为优化候选的尝试
  waves=2,2：softmax 需要跨 wave 归约，当前 lowering 拒绝，日志在 waves22/。
  coalesce-waits=1：本例 O3 后 barrier/wait/内存和 MMAC 计数没有减少，未列入
  默认 GPU 测试。原始编译记录保存在 coalesce/。

已完成的验证
  k0_prefetch 和 threads512 各通过 7 组实际 LLVM CPU 数值执行；包含随机
  query=0/1/31、正溢出、负下溢、running max 改变、最后 key tile 支配首 query。
  最大绝对误差均约 1.45e-4，所有 Q 元素每 block 只装载一次。
  所有交付 IR 通过 llvm-as 和 opt default<O3>,verify。
  预取候选在 O3 后仍保留预期 K0 issue → 32 PV MMAC 的顺序。
  baseline/control/prefetch 另做逐位输出相等、全局/LDS 有序访问轨迹相等、
  跨 wave 的 LDS barrier-epoch 检查：均通过，wait=260、barrier=389/block。
  三个负例（删除 reader barrier、删除 K0 wait、删除末轮 guard）均被拒绝。
  查看 *.validation.json、prefetch-audit.json 和 verification.json。
  当前环境未运行目标 GPU 编译或测速，不能宣称候选已优于 baseline。

GPU 测试（仓库根目录，沿用 LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_optimization/run_gpu.sh
  默认顺序为 baseline → k0_prefetch → baseline，用前后基线检查漂移。
  单独测试：
    bash generated_ir/noncausal_optimization/run_gpu.sh k0_prefetch
    bash generated_ir/noncausal_optimization/run_gpu.sh threads512
  拆分预取收益：
    bash generated_ir/noncausal_optimization/run_gpu.sh k0_late_control k0_prefetch
  默认 warmup=20、repeat=100，可设置 ATTN_WARMUP / ATTN_REPEAT。
  全部使用 non-causal SDPA；各候选独立 launch.json 指定 block=256 或 512。
  每次运行创建 runs/<variant>-XXXXXX，不覆盖既有结果；保存 target 汇编、
  build.log、gpu.log、result.json 和输入哈希。重点返回 median、VGPR、scratch、
  LDS、occupancy；只有实际测速通过才应替换原最佳配置。

复现
  python3 generated_ir/noncausal_optimization/analyze.py
  python3 generated_ir/noncausal_optimization/generate_prefetch.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_optimization/verify_prefetch.py
  512 线程导出的原始 command.sh/config.json 保存在 threads512/run-*/online-baseline/。
