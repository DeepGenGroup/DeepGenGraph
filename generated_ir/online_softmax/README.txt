Attention online softmax / reduce MAX

generate_attention_ir.py 默认生成 online softmax；--softmax legacy 生成原算法。
新文件名增加 -online，避免与已测非稳定 IR 混淆。历史测试输入与已测 .ll
保留原样；TuneAttention 的 retained32-baseline 只选择编译参数，算法由
传入的 MLIR 决定，因此务必显式 --input 指向新的 online 文件。

生成当前 BM=BN=128、BK=32 的稳定输入：
  python3 generate_attention_ir.py --bm 128 --bn 128 --bk 32 \
    --softmax online --output-dir generated_ir/online_softmax

用当前完整 Q 保留配置生成 LLVM IR：
  python3 TuneAttention.py \
    --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
    --case retained32-baseline --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/online_softmax/export

在原 DTK/GPU 环境，追加 --benchmark --warmup 50 --repeat 200 可生成
HSACO 并检查正确性/性能。重新生成 C++ 工具：
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j 2

每个 key tile 的递推（score 已包含原有 log2(e)/sqrt(D) 预缩放）：
  m_new = max(m_old, reduce_max(masked_score, dim=1))
  alpha = exp2(m_old - m_new)
  p = exp2(masked_score - m_new)
  l_new = alpha*l_old + reduce_sum(p, dim=1)
  o_new = alpha*o_old + fp32_dot(fp16(p), V)
初始 m=-inf、l=0、o=0，循环结束输出 o/l。
新 IR 使用 deepgengraph.reduce(..., init=...), op=MAX；不是独立 reduce_max op。

适用范围是本模板的 causal attention、有效 launch、有限 FP16 Q/K/V。
每行首轮都包含 key 0，m_new 因而有限；之后即使某行在一个 tile 内全部被
mask，仍沿用已有最大值，不会出现 -inf - -inf。任意全行 masked 的通用
attention 或 NaN/Inf 输入不在这个模板的保证范围内。
保留原有 FP16 Q 预缩放和 FP16 P、FP32 MMAC 累加；稳定化不等于消除混合
精度舍入误差。伴随的高层 @Attn 也改为 FP32 QK 和减行最大值的 softmax。

管线补齐：
  Deepgengraph ReduceType 增加 MAX。
  Deepgengraph -> Frisk 支持 MAX、可选 init、负归约维度；init 按对应的
  add/mul/max 运算合并，不再一律相加。
  Frisk 新增逐元素 max，接入布局分析和两条线程降低实现。
  已有 Frisk reduce(kind="max") 用 -inf 初始化，线程内及 lane 间 maxNum
  归约最终降低为 LLVM maxnum；新增逐元素 max 用于合并历史行最大值。
  split-D 的共享检查跟踪 PV 实际消费的概率 exp2；alpha exp2 不被误算成
  重复 softmax。旧手工流水的固定指令表仅在 pipeline=1 时检查。
  广播行向量不再决定完整矩阵的列布局，避免 score 与输出宽度不同导致
  alpha 的布局传播错误。
  循环状态支持计算得到的初值：修复 block argument 的 memspace 推断、
  computed initializer 的 yield 回写、寄存器化时保留非零初值。否则
  running max 会固定为 -inf 或被错误初始化为零，跨 tile 累计不正确。

当前主路径使用 pipeline=0（与 retained32-baseline 一致）。旧 pipeline=1
的固定 attention 调度表没有扩展到 online softmax；不能把此处的成功
等同于所有历史编译入口都支持 MAX。旧图级 Tiling/DynamicFor/Triton 文本
翻译中的 ADD/ANY 专用路径也未在本次扩展或验证。

验证工具：
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    <新生成的kernel.ll> --bn 128 \
    --legacy generated_ir/q_bk_retention_sweep/retained_32words.ll
BN=64 的矩形配置用 --bn 64，用于检查后续 tile 对部分行全部 masked 的情形。
此工具运行实际 LLVM 的 CPU 模型，比较稳定 FP64 参考，检查有限输出、
输出覆盖、DMA 等待及完整 Q 元素仅加载一次。测试涵盖首/中/末 query tile、
大正值溢出、负值下溢和跨 tile 最大值变化。--legacy 验证旧版确实出现溢出。
CPU 检查及 LLVM verifier 不能代替 gfx936 机器码验证、VGPR/scratch 和性能测量。

本地验证结果（2026-10-23）：
  retained32_online.ll：BM=BN=128、BK=32，group8 reverse1，frag-prefetch1。
  validation-bn128.json / validation-bn64.json：各 6 组实际 LLVM 数值测试通过；
  最大绝对误差 0.001061586，所有输出有限，完整 Q 每元素仅加载一次，
  延迟 DMA 等待检查通过。旧 retained32 在同一大正值测试出现非有限输出。
  LLVM assembly verifier 与 O3+verify 通过；LDS 16384 字节，Q 保留 32 words/thread。
  LowerInfoLayoutTest、check_plan2_thread_tiles.py 及 27 个 Python 前端测试通过。
  无 BK 的 BM=64、BN=32 旧模板入口也通过 LLVM 导出和 verifier
  （release-unsplit/）；该配置未执行实际 LLVM 数值测试。
  legacy-equivalence.json：同配置重新导出的原版，与已测 retained32 原版
  除输入路径/debug 文件名外的 LLVM 文本完全一致。
  本机未进行 GPU 正确性、VGPR/scratch 或性能测量。
  release/、release-unsplit/ 是最终源码导出；其他子目录保留了开发中间
  结果，不应作为可用内核。直接使用 retained32_online.ll。
