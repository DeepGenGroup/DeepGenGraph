完整 Q 保留：group × 现有 fragment-prefetch 低成本重测

本次按用户确认，基线为 retained32 原版：
  generated_ir/q_bk_retention_sweep/retained_32words.ll
  SHA256 c5627d7e797e5ee93c73ba04d7112a238ba4dbcb52a888267a4c7b30520f9915
  pmc_test.log 中记录 median=1.257279992 ms，VGPR=244，scratch=0。
最新日志里 K0 跨 BN 预取实验为 1.252956510 ms；用户明确选择本次沿用
retained32 原版单独重测现有开关，因此本次不叠加 K0 实验。
以上为历史目标机数据，本目录尚无新 GPU 性能结果。

固定 BHSD=(1,32,4096,128)，BM=BN=128，BK=32，256 threads，waves=4,1，
reverse=1，retain-fragment-words=32，K swizzle=8:6，Q/V swizzle=off，
frag-prefetch-words=4，以及 retained32-baseline 的其余全部参数。
prefetch=1 是另一个 global packet 开关，本次不改。

生成对应 LLVM IR（仓库根目录）：

  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --sweep-base retained32-baseline \
    --sweep 'workgroup-group-size=4;8;16' \
    --sweep 'frag-prefetch=0;1' \
    --set workgroup-reverse=1 \
    --set retain-fragment-words=32 \
    --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/retained32_group_prefetch

不加 --benchmark 时只导出 IR；每次创建新的 run-*，旧结果不覆盖。
--llvm-as 可改成当地 LLVM 工具路径，或省略以跳过这一语法验证步骤。
加 --dry-run 可只查看完整命令。不要用 --case retained32-baseline 代替
--sweep-base，也不要省略 --sweep-base，否则扫描会从历史 retain=0 基线开始。

本次导出目录：run-20261023-163255-p16epokv/
各候选 IR 为 <run目录>/<候选>/kernel.ll：

  候选          group   reverse   frag-prefetch   角色
  grid-000001     4        1           0
  grid-000002     4        1           1
  grid-000003     8        1           0
  grid-000004     8        1           1           retained32 基线
  grid-000005    16        1           0
  grid-000006    16        1           1

每个目录保留 input.mlir、kernel.ll、command.sh、config.json、result.json、
lowering.log。result.json 是编译记录，不是 GPU benchmark 结果。

在原 DTK/gfx936 环境，直接测试这些已生成 IR：

  bash generated_ir/retained32_group_prefetch/run.sh \
    generated_ir/retained32_group_prefetch/run-20261023-163255-p16epokv

脚本依次运行 4 / 1 / 2 / 4 / 3 / 5 / 6 / 4（数字为候选后缀），
基线交错三次，每个候选每次 session 仅编译一次。默认 warmup=50、repeat=200；
可通过 GROUP_PREFETCH_WARMUP、GROUP_PREFETCH_REPEAT 调整。
LLVM_DIR、ROCM_BC、GPU_ARCH、GPU_FEATURES 沿用 GenHsaco.sh 的设置；
使用与原性能记录相同的工具链和后端选项。
输出到唯一 runs/retest-*/，保存每个输入、IR/HSACO 哈希、编译日志、
汇编及静态资源统计、每次正确性与性能 JSON、原始计时样本。
指定候选可只重测该项，例如在上述命令末尾加 grid-000004 grid-000006。

低成本判读：先比较 group=4/8/16 的 prefetch=1，再逐 group 比较 0/1。
只按正确性通过的 kernel_median_ms 排序，计算本轮基线/候选；同时核对
VGPR、scratch 和 LDS。SDPA 比值另列；不要用 hipprof 时间或随 SDPA 漂移的
best.json 排名替代 kernel median。小幅收益需对优胜者和基线交错复测。

也可以在具备 MyTest 和 DTK/GPU 的环境，在生成命令上追加
--benchmark --warmup 50 --repeat 200，完成串行生成/编译/测量；
但该路径不会额外交错三次基线。已有 IR 推荐使用上面的 run.sh。

本地验证复现：
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/retained32_group_prefetch/verify.py \
    generated_ir/retained32_group_prefetch/run-20261023-163255-p16epokv

verify.py 检查六组仅扫描轴不同、实际保留四片 Q、LDS=16384、入口 Q DMA=8、
BN 循环内不再缩放 Q，prefetch=0/1 对应 0/8 个调度 hint，并做 LLVM O3 verify。
基线只归一化源文件路径/debug filename 后必须与历史已测 IR 完全一致。
复用现有延迟 DMA / 跨 wave LDS 冲突 CPU 模型，六组检查 query=0，group8
两组另检查 query=31；核对数值参考、写入范围和全部 Q 元素恰好加载一次。
检查结果保存到 run 目录 validation.json；这些检查不替代目标 GPU 实测。
