Non-causal group32：交错独立 PV 累加链

基线：../noncausal_v_handoff_sweep_group32/dma_after4.ll。
保持 group-size=32、reverse=0、BM=BN=128、BK=32、256线程、waves=4,1，
Q retain32、K预取/读取组合并保护、16 KiB LDS、FP32累加及原有softmax算法。

每个 V 阶段有 A/B/C/D 四条输出累加链，每条8条MMAC。
下标0..7表示该链的归约步骤，不是新的tile或block配置。

文件                     每阶段32条PV MMAC的顺序
baseline_after4.ll       A0..A7, B0..B7, C0..C7, D0..D7（原基线逐字节复制）
pv_serial_guarded.ll     同上，但所有PV MMAC都加上与实验相同的调度提示
pv_interleave2.ll        A0,B0,A1,B1,...,A7,B7, C0,D0,...,C7,D7
pv_interleave4.ll        A0,B0,C0,D0, A1,B1,C1,D1,...,A7,B7,C7,D7

优先测试pv_interleave4，同时测试serial_guarded以区分调度提示本身的影响。
所有四个V阶段都调整。每条链内部的SSA依赖、浮点归约顺序、MMAC输入与
输出保持不变，只移动独立MMAC及其纯寄存器操作（包含输出缩放）。

全部8次V LDS读取仍在每阶段首条PV MMAC前发出。
前三阶段的reader release fence / s_barrier / acquire fence及下一块V DMA，
仍在当前阶段总计第4条PV MMAC后。第四阶段没有下一块V DMA。
注意：总MMAC位置相同，但交错后这4条指令消费的V片段不同，实际lgkmcnt
等待及寄存器分配可能改变，需检查目标后端汇编。

为约束目标后端保持实验顺序，三个guarded版本在全部128条PV MMAC前后
放置sched.barrier(i32 0)。全kernel调度提示由100增至292；这些是编译期
调度约束，不是新增的工作组s_barrier。串行对照使用完全相同数量及位置
（按MMAC序号）的约束。没有删除硬件同步或手写waitcnt。

本地验证
  python3 generated_ir/noncausal_pv_interleave_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_pv_interleave_group32/verify.py
  bash -n generated_ir/noncausal_pv_interleave_group32/run_gpu.sh

generate.py固定输入SHA，检查原始和O3 IR中的MMAC顺序、读取及DMA位置、
内存/同步顺序、K读取组及预取锚点，并调用llvm-as和opt verify。
verify.py执行LLVM CPU模型，包含随机head0/query0、head1/query31和running-max
变化三种情况；对照原after4检查逐bit输出、访存轨迹、延迟DMA及动态位置。
该模型不模拟独立wave时序或目标GPU性能；GPU结果和实际VGPR数需实测。

GPU测试（仓库根目录，沿用GenHsaco.sh的LLVM_DIR、ROCM_BC等环境变量）
  bash generated_ir/noncausal_pv_interleave_group32/run_gpu.sh

默认：baseline_after4 → pv_serial_guarded → pv_interleave4 → baseline_after4
      → pv_interleave2 → baseline_after4。
只测指定版本：
  bash generated_ir/noncausal_pv_interleave_group32/run_gpu.sh pv_interleave4

默认warmup=20、repeat=100；可用ATTN_WARMUP、ATTN_REPEAT覆盖。
每次独立保存到runs/<variant>-XXXXXX，包含input.sha256、build.log、GPU数值
检查/gpu.log、result.json、kernel.hsaco和GenHsaco.sh产生的汇编。

汇编验收重点
  1. 同一V阶段前几条PV MMAC的输出累加器是否按A/B/C/D（或A/B）轮换。
  2. 下一批V DMA是否仍在该阶段第4条PV MMAC后。
  3. 8条V矩阵读取是否都提前发出，K的32组ds_read2是否保留。
  4. 是否增加VGPR/scratch、等待或其他额外指令。
