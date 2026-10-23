稳定 online softmax：对齐用户指定的 group/fragment-prefetch 最佳方案

旧算法的已测最佳方案：
  generated_ir/retained32_group_prefetch/run-20261023-163255-p16epokv/grid-000001/kernel.ll
  pmc_test.log: median 1.168558955 ms，SDPA median 1.119278491 ms，speedup 0.957828x。
用户反馈先前稳定版约 0.69x；当前 pmc_test.log 没有该稳定版的对应记录。

先前稳定版参数为 group8、reverse1、frag-prefetch1。
本目录 kernel.ll 为 group4、reverse1、frag-prefetch0，保留完整 Q 32 words/thread。
其他调优参数与指定最佳方案相同：BM=BN=128、BK=32、QK/VO tile32、
256 threads、waves4,1、K swizzle8:6、async16/asm128、math=approx、
pipeline0、simplify-loop-control1、coalesce-waits0、LDS16384。

旧模板 query=y/head=x，新模板 query=x/head=y；这两个参数由输入的
parallel_map 自动推导，不能机械照抄旧模板的轴名。生成后已经比较
workgroup ID 到 query/head 的 LLVM 计算，物理分组映射完全一致。

生成命令（仓库根目录）：
  python3 TuneAttention.py \
    --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
    --case retained32-baseline \
    --set workgroup-group-size=4 --set frag-prefetch=0 \
    --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/online_softmax_best_aligned

目标 DCU 上在同一命令追加 --benchmark --warmup 50 --repeat 200，
生成对应 HSACO、校验正确性并记录性能。当前本机没有测量此候选的 GPU 性能。
不修改 retained32-baseline 的历史默认值，也不覆盖前一稳定版，便于 A/B 对照。

验证：
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    generated_ir/online_softmax_best_aligned/kernel.ll --bn 128 --group 4 \
    --output generated_ir/online_softmax_best_aligned/validation.json
validation.json/numerics.log 记录六组实际 LLVM CPU 模型测试，包括跨 tile、
溢出/下溢、最大值变化及完整 Q 每元素只加载一次的检查。
comparison.json 记录参数、物理映射、LDS/Q 保留与静态 LLVM 指令对比。

参数对齐不能消除稳定化的全部代价。两版的 MMAC/DMA128/barrier 静态站点
仍分别为 256/24/17，但稳定版增加 max 归约、score 减法和 O/L 重缩放。
PV 还丢失了原版的直接累加路径：原版 MMAC 使用历史 O 作为 C，稳定版
先从零计算 PV，再做 alpha*O_old + PV。零 C 的 MMAC 起点由16增至32，
四元素向量 fadd 由16增至32。以上是静态 LLVM 计数，不是硬件动态计数。
这一额外临时结果可能增加寄存器压力；是否导致 spill、对性能的贡献，
需要本次候选对应的 gfx936 汇编及 VGPR/scratch/计时数据确认。
