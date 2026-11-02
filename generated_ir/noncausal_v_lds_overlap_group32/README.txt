Non-causal group32：V LDS读取与PV MMAC交错实验

基线为 ../noncausal_group_sweep_forward/group32.ll，SHA绑定在generate.py。
baseline.ll 是逐字节副本。BM=BN=128、BK=32、threads=256、waves=4,1、
retain=32、group32、reverse=0、LDS=16384、原有K整组预取和合并保护均保留。

候选
  v_tail_prefetch2.ll：只处理第四个V/PV阶段；不改变任何下一块V DMA的发起
    时机。用于先观察LDS/MMAC交错本身的效果，影响范围最小。
  v_prefetch2.ll：四个V/PV阶段均交错；读取提前至首次使用前最多2条MMAC。
  v_prefetch4.ll：同上，最多4条MMAC，测试更大的LDS延迟隐藏窗口。
  v_late_dma_control.ll：四阶段仍先读完全部V，再执行前8条PV MMAC，随后
    reader barrier与后续DMA；用来区分DMA窗口缩短和LDS交错的收益。

原结构（前三个V阶段）
  read V0..V7 → reader barrier → 下一块V DMA → PV MMAC0..31
v_prefetch2
  read V0,V1,V2 → MMAC0 → read V3 → MMAC1 → read V4 → MMAC2
  → read V5 → MMAC3 → read V6 → MMAC4 → read V7 → MMAC5,6,7
  → reader barrier → 下一块V DMA → MMAC8..31
v_prefetch4开始时读取V0..V4，此后每条MMAC之间发下一次读取。
v_late_dma_control
  read V0..V7 → MMAC0..7 → reader barrier → 下一块V DMA → MMAC8..31
第四阶段没有下一块V DMA；原末尾硬件barrier保持在全部32条PV MMAC之后。

重要权衡
  当前后续DMA覆盖同一LDS缓冲区，不能把读取直接挪到reader barrier/DMA后。
  本实验将纯寄存器PV计算移进读取段，所有内存操作与硬件同步的相对顺序
  完全不变。前三阶段的下一块V DMA可重叠窗口由32条PV MMAC缩为24条。
  这是要测试的权衡，不保证一定提速；tail候选没有这项DMA窗口代价。
  2/4是MMAC条数，不是VGPR预算；具体寄存器分配以目标汇编为准。

约束及本地校验
  不增加/删除原有指令，只重排纯计算并加入sched.barrier调度提示。
  所有256条MMAC的全局顺序、各自累加链、浮点运算和内存访问保持不变。
  保留全部硬件barrier、fence、vmcnt等待，不手写lgkmcnt，不增加LDS。
  sched.barrier不是硬件workgroup barrier。
  LLVM及O3校验，并检查O3后读取在预定MMAC之间、K的32个配对不被拆开。
  每轮仍为32条V矩阵读取；不会以降低LDS操作数作为本实验成功标准。
  verify.py运行实际LLVM CPU模型：随机head0/query0、head1/query31，另有
  running-max变化输入；逐bit对照、延迟DMA、内存访问地址轨迹和动态发起
  顺序均检查。它不模拟独立wave时序或目标GPU性能；GPU仍须验证。

复现（仓库根目录）
  python3 generated_ir/noncausal_v_lds_overlap_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_lds_overlap_group32/verify.py

GPU测试
  bash generated_ir/noncausal_v_lds_overlap_group32/run_gpu.sh
默认 baseline → v_tail_prefetch2 → baseline → v_prefetch2
     → v_late_dma_control → v_prefetch4 → baseline。
只测主要候选：
  bash generated_ir/noncausal_v_lds_overlap_group32/run_gpu.sh baseline v_prefetch2 v_prefetch4
单独指定上述任一名字也可以。沿用LLVM_DIR/ROCM_BC；warmup=20、repeat=100，
可通过ATTN_WARMUP/ATTN_REPEAT调整；每次结果独立保存在runs/中。

目标汇编重点
  V ds_read_m32x16_b16是否穿插在PV MMAC之间，而非全部堆在首条PV前。
  仍应保留32个K ds_read2配对和32条V矩阵读取；检查局部lgkmcnt、VGPR、
  scratch及绝对median。新候选未经目标GPU编译/测速，不预先宣称收益。
