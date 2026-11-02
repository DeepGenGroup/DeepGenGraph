Non-causal softmax fragment 调度 Pass 测试包

测试入口（仓库根目录，目标 gfx936 / DTK 环境）
  bash generated_ir/noncausal_softmax_fragments/run_gpu.sh
默认 control -> softmax_fragment -> control -> softmax_pv -> control。
只测试 PV 交错候选：
  bash generated_ir/noncausal_softmax_fragments/run_gpu.sh softmax_pv
脚本沿用 LLVM_DIR / ROCM_BC，默认 warmup=20、repeat=100；
可通过 ATTN_WARMUP / ATTN_REPEAT 调整。每次测试保存独立 runs/ 目录。

供测试的 IR
  control.ll           当前 pmc_test.log 最佳 non-causal 实测 IR 的逐字节副本
  softmax_fragment.ll  --softmax-fragments=fragment
  softmax_pv.ll        --softmax-fragments=pv
对应 *.launch.json 固定 BHSD=(1,32,4096,128)，grid=(32,32,1)，block=(256,1,1)。
*.raw.ll、*.named.ll、*.o3.ll 是审计中间文件；测试优先使用上述三个 .ll。

基线来源
  generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/
    run-20261030-171621-98fko2hz/online-vdb-k64-lds2/kernel.ll
  pmc_test.log 首项，HSACO median=1.735664 ms。
  实际 IR 属性 frisk.k_lds_prefetch="8"，与日志 k-prefetch 8 一致。
  归档 config.json/目录名称仍写 2，已过时；本包不按旧配置重建测试基线。
  baseline.config.json 原样保留以便追溯；manifest.json 记录实际值和 SHA256。
  保留 group32、reverse=0、QK/VO tile32、retain32、pv-interleave4、
  V 双缓冲、K 64-bit swizzle DMA、8:6 K 布局及已有数值优化。

变换与约束
  真正的 C++ LLVM Function Pass：ScheduleSoftmaxFragmentsPass。
  从 P 的 <4 x float> -> <4 x half> 和 MMAC 消费关系识别 fragment，
  不依赖 SSA 名称或固定 attention tile 大小。
  fragment：按原 P 顺序执行减 max、exp、已就绪的 FP32 sum 更新、half 打包。
  pv：保持 MMAC 顺序，将上述转换推迟到该 P fragment 首次 PV 消费前。
  只重排指令；原浮点表达式、操作数、fast-math flags、舍入/精度均保留。
  原 max、alpha、FP32 denominator 结合顺序、FP32 PV 累加完全保留。
  第一片 exp 可作为原 sum 链初值，必须活到下一片，不另加 +0 或删掉它。
  原 FP32/half tied identity 标记都保留，不承诺消除所有 v_mov。
  所有 LDS 读取、DMA、硬件 wait/barrier、原调度提示、MMAC 顺序不变。
  每片打包后新增一个 sched.barrier(0)，仅约束编译调度，不是 s_barrier，
  不增加 LDS、不手写 lgkmcnt。保留每个 K 读取组原有合并边界。
  未知副作用、非匹配算子、跨 block packet、strictfp/optnone 保守跳过。
  Pass 默认关闭，重复运行不会再次改动已调度的函数。

IR 结构指标（O3 前后均核对）
                         control    fragment    pv
  P fragment 数             16          16       16
  首次打包前减 max 数        16           1        1
  首次 PV 前已打包片数       16          16        1
  最后一片打包前 PV MMAC 数   0           0       30
  新增 softmax 调度提示       0          16       16
P 会在后续 VO tile 中复用；这里只推迟首次准备，不重算/重复打包。
PV 模式可能延长原 score 和 V 的共同存活区间，真实 VGPR、scratch、
指令复制和速度需实测，不能由以上 IR 指标推断性能提升。

重新生成和验证
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 3
  python3 generated_ir/noncausal_softmax_fragments/generate.py
  python3 3rd/deepgengraph/test/check_softmax_fragments.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_softmax_fragments/verify.py
generate.py 直接对已测 IR 应用编译好的 Pass，保存输入/日志/编译器/产物 SHA。
执行 llvm-as 和 opt default<O3>,verify，并对相同 SSA 的变换前后检查指令
多重集、每条指令的操作数及 flags、访存/同步/MMAC 顺序完全一致。
verify.py 使用现有 LLVM CPU 执行模型，验证随机输入、正向溢出、负向下溢、
跨 key tile max 改变、future key 主导；候选与 control 逐 bit 比较，
FP64 参考阈值保持 atol=rtol=0.01，同时检查延迟 DMA 和 LDS epoch 所有权。
随机样例还检查 O3 后 IR。validation.json/log 记录实际结果。
执行模型不是 GPU 仿真，不验证独立 wave 时序；未在目标 GPU 测速。

复用 Pass
  3rd/deepgengraph/build/test/LegacyLLVMExportTest \
    --softmax-fragments=pv input.ll > output.raw.ll
  3rd/deepgengraph/build/test/LegacyLLVMExportTest \
    --llvm22-text output.raw.ll > output.ll
也可在原 MyTest 命令末尾追加 --softmax-fragments=off|fragment|pv。
运行在最终 MMA、PV、V double-buffer、K DMA/LDS 调度之后；
fresh-command.json 记录了按实际 k-lds-prefetch=8 验证 MyTest 的命令。

GPU 验收建议
  在相同编译参数下比较三组正确性和 median；检查 stable softmax 极端输入。
  比较主循环 exp/pack/MMAC 实际交错、v_mov、VGPR、scratch、waitcnt，
  同时核对 K ds_read2 组合与 LDS 操作数没有退化。
