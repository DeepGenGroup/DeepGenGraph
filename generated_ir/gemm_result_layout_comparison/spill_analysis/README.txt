3.1 spill 根因分析及隔离实验（2026-10-27）

结论
当前存在明确的 Pass 阶段衔接漏优化：QK 的分块累加没有融合进 MMAC，
四个 BK 分块的结果同时存活。BM=BN=128 / BK=32 的组合将后果放大。
共享中转本身的 LDS/同步开销仍存在，但不能据此把当前严重 spill 解释为
3.1 策略的必然代价。此轮仅分析并生成诊断 IR，未修改编译器源码。

硬件日志（pmc_test.log 按 UTF-16 解码）
  L2-29：exp2_fixed/layout_3_1/kernel.ll，median=8.209191 ms，VGPR=256，
         scratch=420，LDS=65536，shared operations=13787136。
  L38-66：历史 online_softmax_deferred_sum，median=1.201280 ms，VGPR=248，
          scratch=0，LDS=16384，shared operations=4907008。
  L34 仅有新 3.3 的标题，尚无对应计时/PMC。不能把历史数据写成同轮新 3.3 实测。
  当前 k_test.gcn.log 的 private_segment_fixed_size=0，不是上述 spilling 3.1
  的可匹配汇编；本地工具链也不支持目标 gfx936，因此未宣称定位具体 spill 指令。

直接证据：QK 分块结果活跃范围
  D=128、BK=32，共四个 DK 分块。
  每个部分 QK 128x128 / 256 threads = 64 FP32 元素/线程。
  原 3.1：64 条长度 2 的 MMAC 链；四组最终结果共 256 FP32 元素，全部在
  最后一条 QK MMAC（kernel.ll:988）后，才从 :995 起执行相加。
  同期外层 O 状态还需 4 x vector<2x8xf32> = 64 FP32 元素/线程。
  因而仅 QK 部分结果和 O 状态就有约 320 个逻辑 FP32 值要跨该位置存活，
  尚未计行状态、地址和输入。这里是 IR 数据活跃量，不是后端物理 VGPR 预测。
  原 3.3：16 条长度 8 的 QK MMAC 链，四个 DK 分块在同一组 C 累加，
  最终 QK 结果只需 64 FP32 元素/线程。

为什么融合没命中
  原 3.1 kernel.ll:992 有 br i1 true，将 MMAC 所在块与其 fadd 用户隔开。
  此分支来自共享写回的 leader guard；对这个无副本 tile，条件已化为真。
  FuseMMACAccumulator.cpp:72 明确要求 MMAC 与 fadd 在同一基本块。
  fuseMMACAccumulators() 仅先运行 InstCombine，没有先清掉这个控制流边界。
  后续普通 opt O3 能消除分支，却不会再次调用项目自己的融合 Pass；O3 输出
  仍保留四组部分 MMAC 结果以及尾部 fadd。

隔离实验（保持 3.1、分块、LDS、DMA/同步策略）
  A. 对原 IR 直接再运行 FuseMMACAccumulator：0 条融合。
  B. 先 simplifycfg，再运行同一 Pass：64 条融合（fusion.log）。
  B 的 QK 变成 16 条长度 8 的链；QK 最终部分结果活跃量从 256 降为 64。
  MMAC 总数仍为 256，LDS 仍为 65536 bytes，循环内 DMA/barrier/shuffle
  静态数量都不变。这一实验没有切换到 3.3，也没有缩小 tile。
  B 的 kernel.ll 已通过 LLVM O3 verification 和六组 CPU 实际 LLVM 数值检查，
  最大绝对误差 0.001061586，延迟 DMA/输出范围/Q 重载次数检查通过。
  尚未在 DTK/GPU 上验证 B 的最终 VGPR、scratch、ISA 和耗时，不能宣称已清除 spill。

另外存在的组合代价（static call sites，不是 GPU PMC 指令计数）
                                原 3.1      原 3.3
  QK MMAC chains                64 x 2      16 x 8
  循环内 bpermute               72          4
  循环外 bpermute               0           68
  循环内 block barrier          31          12
  循环外 block barrier          0           5
  循环内 DMA128                 24          16
  循环外 DMA128                 0           8
  循环内 f32 LDS store/load     76 / 76     0 / 0
  完整 Q fragment 保留          未命中      32 words/thread
  mask bypass / deferred sum    未命中      命中
  static LDS                    64 KiB      16 KiB
  3.1 的 PV 输出转换仍留在循环内，alpha*O 与 PV 也没有融合进 native C。
  这些说明优化开关一致并不等于实际优化效果一致；应在解决累加链问题后再比较
  布局策略的固有代价及 BM/BN/BK 选择。

不是预取过深或 Q 保留过量的证据
  两份的 pipeline=0、frag-prefetch=0，PrefetchGlobalPackets 实际均为 0 packets。
  3.1 RetainInvariantFragments 实际保留 0 tiles。
  carrier=34816 是前期循环 carrier 的预算，并不约束展开后的全部临时活跃值，
  因此不能当作后端不 spill 的保证。

建议修复顺序
  1. 在项目 MMAC 融合前规范化恒真/无条件 CFG，或让 Pass 安全识别这类边界。
     不能不加证明地把任意跨块 load/call/真实分支搬入融合。
  2. 在目标工具链对比本诊断候选的 scratch/VGPR/耗时，确认硬件影响。
  3. 再处理 3.1 下 native PV accumulator、alpha 布局和额外同步，然后调 tile。

文件与复现
  kernel.ll：简化 CFG 后补做 MMAC 融合的 DTK 文本格式测试候选。
  structure.json / provenance.json：结构统计、输入/输出哈希。
  validation.json / validation.log：六组数值检查。
  fusion.log / fuse_without_cfg.log：有/无 CFG 简化的融合结果。
  bash generated_ir/gemm_result_layout_comparison/spill_analysis/reproduce.sh
  bash generated_ir/gemm_result_layout_comparison/spill_analysis/run_gpu.sh
第二条在目标 DTK 环境使用已有 LLVM_DIR/ROCM_BC，输出保存到 gpu/。
