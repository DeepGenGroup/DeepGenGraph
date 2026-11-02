Non-causal group32：V reader barrier/DMA位置扫描

输入：../noncausal_v_lds_overlap_group32/v_late_dma_control.ll。
dma_after8.ll与该输入逐字节一致；dma_after2/4只把前三个V阶段的reader
fence/barrier和下一块V DMA提前到对应PV MMAC后。第四阶段没有下一块V DMA，
保持此前after8的代码不变（末尾硬件barrier仍在32条PV MMAC之后）。

执行顺序
  全部8次V LDS读取尽早发出
  → 当前阶段前N条PV MMAC
  → 原reader release fence / s_barrier / acquire fence
  → 下一块V DMA
  → 当前阶段剩余32-N条PV MMAC

测试文件                 N      下一块V DMA重叠窗口
  dma_after2.ll          2      30条PV MMAC
  dma_after4.ll          4      28条PV MMAC
  dma_after8.ll          8      24条PV MMAC（已测试的原对照）

固定项
  group32，reverse=0，BM=BN=128、BK=32、256线程、waves=4,1。
  Q retain=32、原有K预取/32组读取保护、16 KiB LDS均保持不变。
  三个版本的全部100个sched.barrier调度提示数量和原有MMAC顺序一致。
  未删除任何硬件同步；所有内存操作与硬件同步的相对顺序保持一致。
  未手写lgkmcnt；实际等待由目标后端生成。
  2/4/8是barrier前已经执行的PV MMAC数量，不是预取距离或寄存器预算。

本地验证
  LLVM及O3校验；O3后逐一核对32条V读取、barrier/DMA的MMAC位置、K预取。
  CPU LLVM执行模型检查随机head0/query0、head1/query31及running-max变化；
  数值逐bit对照after8，内存访问轨迹、延迟DMA和动态发起位置均验证。
  不模拟目标GPU时序；性能及最终汇编仍须GPU实测。

复现（仓库根目录）
  python3 generated_ir/noncausal_v_handoff_sweep_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_handoff_sweep_group32/verify.py

GPU测试（沿用LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_v_handoff_sweep_group32/run_gpu.sh
默认 after8 → after2 → after8 → after4 → after8，用after8穿插对照。
每种只测一次：
  bash generated_ir/noncausal_v_handoff_sweep_group32/run_gpu.sh dma_after2 dma_after4 dma_after8
warmup=20、repeat=100，可设置ATTN_WARMUP、ATTN_REPEAT；结果分别保存在
runs/独立目录，包括汇编、数值检查及result.json。
