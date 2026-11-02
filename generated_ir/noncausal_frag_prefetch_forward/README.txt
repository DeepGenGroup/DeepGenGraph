Non-causal attention：group4、关闭 bx/query reverse，LDS fragment 预取对照

供测试的 IR
  control/kernel.ll    frag-prefetch=0, frag-prefetch-words=4（关闭时预算不生效）
  prefetch4/kernel.ll  frag-prefetch=1, frag-prefetch-words=4
  prefetch8/kernel.ll  frag-prefetch=1, frag-prefetch-words=8

三组统一参数
  BHSD=(1,32,4096,128)，stable online softmax，non-causal。
  BM=BN=128，QK/VO tile=32，block=256，waves=4,1。
  group-size=4，reverse=0，query-axis=x，head-axis=y。
  retain-fragment-words=32，K swizzle=8:6，Q/V swizzle=off。
  async copy，alignment=16，useKPack，asm128。
  其他参数继承归档 non-causal 最佳基线；详见各目录 config.json。

frag-prefetch-words 的单位为每线程额外活跃 SSA words（每 word 32 bit），
不是 fragment 个数、half 个数，也不是最终分配的 VGPR 增量。
当前 Pass 在单个 MMAC 前瞻窗口内提前下一组完整 A/B operand 读取；
是否实际移动、因预算或 read-group 跳过多少，查看 lowering.log 的 FragPrefetch。
这不是下一轮 BN 的 K0 global DMA 预取实验。

本次生成结果：prefetch4 和 prefetch8 均提前 4 组完整 A/B packet，
0 budget skips、48 read-group skips。去除源文件路径和预算属性差异后，
两份 LLVM IR 完全相同；当前 Pass 下增加预算并未带来更多预取。
保留两份供测试，但预计二者不会有稳定性能差异，主要比较候选与 control。

使用当前本地 MyTest 分别重新生成三组，避免将新编译器与旧 control 混比。
输入快照、完整命令、环境、launch 参数和编译日志均在各变体目录中。
manifest.json 记录输入/编译器/IR 哈希和 LLVM 校验结果。
本地产物执行 llvm-as 和 opt default<O3>,verify 校验；未运行目标 GPU。
正确性和最终性能以目标 GPU 实测为准。

重新生成（从仓库根目录）
  python3 generated_ir/noncausal_frag_prefetch_forward/generate.py

在 gfx936 DTK/GPU 环境测试（沿用 LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_frag_prefetch_forward/run_gpu.sh
默认顺序：control → prefetch4 → control → prefetch8 → control。
只跑两个候选：
  bash generated_ir/noncausal_frag_prefetch_forward/run_gpu.sh prefetch4 prefetch8
默认 warmup=20、repeat=100，可通过 ATTN_WARMUP / ATTN_REPEAT 修改。
每次测试创建独立 runs/ 子目录，保存汇编、正确性/计时输出和 result.json。
对比 kernel median、正确性、VGPR、scratch，以及 LDS wait/MMAC 的实际调度。
