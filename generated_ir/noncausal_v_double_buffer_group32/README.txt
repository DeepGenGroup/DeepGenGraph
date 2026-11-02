V 双缓冲 LLVM IR：以用户确认的最新最佳 pv_interleave4 为基线

运行文件
  v_double_buffer.ll            V ping-pong 候选，交给 GenHsaco.sh
  baseline_pv_interleave4.ll     原始最佳基线的逐字节副本
  *.launch.json                配套启动配置
  *.o3.ll                      本地 LLVM O3 验证产物

基线来源
  ../noncausal_pv_interleave_pass_group32/pv_interleave4.ll
  SHA256: 2bef36b60ab7309928d68bea12a033ee0615d907f577333871fae41f74156c51
  最佳选择来自本次用户确认；未虚构 pv_interleave4 的本地测速数据。

固定参数
  BHSD=(1,32,4096,128)，non-causal，BM=BN=128，BK=32。
  256线程，waves=4,1，group32，reverse=0，Q retain-fragment-words=32。
  layout=3.3，math=approx，K swizzle=8:6，K整组预取与配对保护保留。
  PV四链交错 A0 B0 C0 D0 ... 保持完全相同的MMAC顺序和SSA操作数。
  其余继承参数和输入/输出哈希记录在 manifest.json。

LDS布局（总计仍为16384字节）
  slot0 = [0,8192)       V0、V2
  slot1 = [8192,16384)   V1、V3
  每片128×32×f16 = 8192字节，每片两条原有buffer-to-LDS DMA。
  V1/V3复用原K高半区的DMA目的基址；对应矩阵读取地址增加8192字节。
  V0发起时机不变：最后32条QK MMAC之前，K3仍在高半区。
  V1必须等最后的K3读取结束且所有wave通过V0-ready barrier后才发起。

新调度
  原V0 DMA → 剩余QK/softmax → wait V0 + release/barrier/acquire
  → DMA V1到slot1 → read V0(slot0) → PV0全部32条MMAC
  → wait V1 + release/barrier/acquire
  → DMA V2到slot0 → read V1(slot1) → PV1全部32条MMAC
  → wait V2 + release/barrier/acquire
  → DMA V3到slot1 → read V2(slot0) → PV2全部32条MMAC
  → wait V3 + release/barrier/acquire
  → read V3(slot1) → PV3全部32条MMAC → 原循环末尾barrier

同步依据
  V0-ready barrier同时结束K3对slot1的读取，之后才能写V1。
  V1-ready barrier同时结束V0对slot0的读取，之后才能写V2。
  V2-ready barrier同时结束V1对slot1的读取，之后才能写V3。
  所有ready等待及fence/barrier均保留，原循环末尾barrier保护下一轮K。
  删除当前片前4条PV后的三组reader release/barrier/acquire：
  下一片DMA已改写另一个slot，不再覆盖当前片，不再需要这三组同步。
  每个key tile少3个硬件barrier；32轮的动态计数从389降到293。
  vmcnt等待保持260次；所有256条静态MMAC和32条V读取保持不变。
  下一片DMA可覆盖当前片的V读取和全部32条PV MMAC（原为后28条）。
  仍是一次发出当前片8个V packet；本候选只改变双缓冲/DMA/sync，
  不另外引入V LDS读取分组或改变四链交错。

复现（仓库根目录）
  python3 generated_ir/noncausal_v_double_buffer_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_double_buffer_group32/verify.py

验证
  生成器进行LLVM/O3 verify，并检查O3后V调度、全部MMAC操作数及顺序、
  K读取配对与预取锚点、16 KiB LDS以及硬件同步计数。
  verify.py执行实际LLVM的CPU模型，DMA延迟至vmcnt等待时才写入LDS。
  检查256线程的LDS字节地址、交替slot、读写间的workgroup barrier、
  global访存轨迹、LDS读取值、与最佳基线逐bit一致及float64参考误差。
  覆盖随机首/末query、running-max变化、未来key主导的non-causal输入；
  另对O3产物执行随机case，并用错误slot/漏barrier负例确认检查有效。
  CPU模型不模拟GPU机器调度或独立wave推进；GPU数值与性能尚未验证。

GPU A/B测试（沿用LLVM_DIR、ROCM_BC环境配置）
  bash generated_ir/noncausal_v_double_buffer_group32/run_gpu.sh
  默认：baseline_pv_interleave4 → v_double_buffer → baseline_pv_interleave4。
  只测候选：
  bash generated_ir/noncausal_v_double_buffer_group32/run_gpu.sh v_double_buffer
  默认warmup=20、repeat=100，可通过ATTN_WARMUP/ATTN_REPEAT调整。
  每次产物、编译日志、GPU数值检查及result.json独立保存在runs/。
  验收目标汇编的LDS=16384、scratch/VGPR、交替DMA地址和真实延迟。
