V 双缓冲：编译器 Pass 产物

源文件：../noncausal_pv_interleave_pass_group32/pv_interleave4.ll
用户确认的当前最佳case，保留其K整组预取、PV四链交错和其他调优参数。
BM=BN=128、BK=32、256线程、waves=4,1、group32、reverse=0、Q retain32。

文件
  v_double_buffer.ll / .launch.json       推荐GPU A/B候选
  baseline_pv_interleave4.ll              原最佳基线逐字节副本
  v_double_buffer.pass.log               C++ Pass实际匹配结果
  manifest.json                          工具/输入/输出哈希与结构检查
  validation.json                        CPU延迟DMA及数值检查
  mytest_smoke/                          从MLIR完整编译的独立验证产物

生成器直接调用LegacyLLVMExportTest --v-double-buffer=1，Python不重排IR。
Pass自动推导两个8192字节slot，总LDS不变；V1/V3使用高半区。
三个后续DMA在当前片读取前发起；每key tile减少3个reader barrier。
所有ready等待/barrier、最终reader barrier和MMAC操作数/顺序均保留。

复现（仓库根目录）
  python3 generated_ir/noncausal_v_double_buffer_pass_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_double_buffer_pass_group32/verify.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_double_buffer_pass_group32/verify_mytest.py

GPU测试（沿用LLVM_DIR、ROCM_BC）
  bash generated_ir/noncausal_v_double_buffer_pass_group32/run_gpu.sh
默认 baseline_pv_interleave4 → v_double_buffer → baseline_pv_interleave4。
数值和性能结果保存到各自runs/目录；本地未进行目标GPU测速。

MyTest直接使用 --pv-interleave=4 --v-double-buffer=1。
TuneAttention.py支持 --set v-double-buffer=1、--sweep 'v-double-buffer=0;1'，
也支持pv-interleave组合扫描。result.json记录命中详情或跳过原因。
开关只负责V双缓冲；fresh MyTest不会额外生成先前独立的K整组预取实验。
mytest_smoke/command.json记录完整编译参数，baseline.ll为同编译路径关闭
V双缓冲的对照；不能把该smoke产物当作顶层最佳基线的逐指令复制。

通用性、同步证明、限制以及现有K双缓冲的关系详见：
  3rd/deepgengraph/lib/Conversion/ConvertToLLVM/VDoubleBuffer.md
