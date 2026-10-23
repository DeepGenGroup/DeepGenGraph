一个 block 连续处理两个 BM 分块

产物：group8_reverse_k128_gep_two_bm.ll
基于 pmc_test.log 中已测最快的 group8_reverse_k128_gep.ll（1.779047 ms）。
新版本尚未进行 gfx936 编译和 GPU 性能测试。

固定 BHSD=(1,32,4096,128)，FP16，causal，Q/K/V/O 连续 BHSD。
BM/BN/BK=128/128/32，kernel=Attn_p2。
必须使用 grid=(16,32,1)，block=(256,1,1)，dynamic shared=0。
共 512 个 block，每个 block 串行执行两个相邻 query tile，覆盖 256 行。
静态 LDS 仍为 16384 字节；两次执行分别重置 key 循环、分母和 O 累加器。
K-only swizzle、常量 GEP、128-bit DMA 和原有计算主体沿用源文件。
新增外层循环禁用展开，两块之间增加一次全 block barrier，安全复用 LDS。
这是串行合并两个 BM 任务，K/V 仍各自加载，没有合并为 BM=256 的矩阵计算。

保持原 group8 reverse 的相邻任务配对，映射为：
  p = by * 16 + bx
  l = 2 * p
  head = (l >> 3) & 31
  q0 = 31 - ((l >> 8) * 8 + (l & 7))
  q1 = q0 - 1
例如 (bx,by)=(0,0) 处理 head0 的 q31/q30；(1,0) 处理 q29/q28；
(4,0) 处理 head1 的 q31/q30；(15,31) 处理 head31 的 q1/q0。
由于保留分组重排，物理 by 不直接等于 head。

在 gfx936/DTK 环境编译和运行（从仓库根目录执行）：
  mkdir -p generated_ir/two_bm_per_block/build
  cd generated_ir/two_bm_per_block/build
  bash ../../../GenHsaco.sh ../group8_reverse_k128_gep_two_bm.ll 3 kernel.hsaco
  python3 ../../../launchKernel.py --hsaco kernel.hsaco \
    --config ../group8_reverse_k128_gep_two_bm.launch.json \
    --warmup 50 --repeat 200 --json-output result.json

沿用目标环境中有效的 LLVM_DIR、ROCM_BC、GPU_ARCH 配置。
GenHsaco.sh 的中间产物写在上述独立 build 目录中。

在仓库根目录复现生成和验证：
  python3 generated_ir/two_bm_per_block/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/two_bm_per_block/validate.py

验证包括 LLVM 汇编、O3 verifier、全部 512 个 block 的覆盖映射，以及
首尾和 group8 边界附近的延迟 DMA CPU 数值执行。每个新 block 的两块输出
分别与原 IR 逐位比较，并与 attention 参考比较，检查没有越界输出、没有
未完成 DMA、MMAC 次数正确、仅多一次 block barrier。
CPU 模型不模拟独立 wave 时序或目标 GPU 后端，不能代替真机正确性和计时。

正序版本（不 reverse）

产物：group8_forward_k128_gep_two_bm.ll
启动配置：group8_forward_k128_gep_two_bm.launch.json
仍使用 grid=(16,32,1)，block=(256,1,1)，dynamic shared=0。
保留 group8 分组，q0=(l>>8)*8+(l&7)，q1=q0+1；每块按 q0、q1 顺序执行。
例如 (0,0) 处理 head0 的 q0/q1；(1,0) 处理 q2/q3；
(4,0) 处理 head1 的 q0/q1；(15,31) 处理 head31 的 q30/q31。
计算主体、LDS 复用和两块间的 barrier 与双 BM reverse 版本相同。

复现：
  python3 generated_ir/two_bm_per_block/generate.py --forward
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/two_bm_per_block/validate.py --forward

编译运行（从仓库根目录执行）：
  mkdir -p generated_ir/two_bm_per_block/build-forward
  cd generated_ir/two_bm_per_block/build-forward
  bash ../../../GenHsaco.sh ../group8_forward_k128_gep_two_bm.ll 3 kernel.hsaco
  python3 ../../../launchKernel.py --hsaco kernel.hsaco \
    --config ../group8_forward_k128_gep_two_bm.launch.json \
    --warmup 50 --repeat 200 --json-output result.json

来源和验证记录分别保存为 manifest.forward.json、validation.forward.json。
历史 1.779047 ms 仅指原单 BM reverse 源文件；正序双 BM 版本尚无 GPU 实测。
