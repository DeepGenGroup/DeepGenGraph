BM/BN 四组 attention 对照 IR
===========================

固定条件：BHSD=1,32,4096,128；FP16；causal；Q/K/V/O 连续 BHSD。
kernel=Attn_p2，block=(256,1,1)，dynamic shared=0，wave 布局=4,1。
BK=32，QK/VO 内部分片=32/32，K-only XOR swizzle=8:6，128-bit DMA，
常量 LDS GEP 规范化，group8 reverse。DK 展开，不新增跨循环 pipeline，
coalesce-waits=0。完整参数、编译器/IR 哈希见 manifest.json。

文件                    grid (X,Y,Z)    静态 LDS
bm128_bn128.ll           32,32,1         16 KiB
bm128_bn64.ll            32,32,1         12 KiB
bm64_bn128.ll            64,32,1         16 KiB
bm64_bn64.ll             64,32,1          8 KiB
measured_control.ll     32,32,1         16 KiB

四组 bm*.ll 均由同一版本编译器、同一组优化参数重新导出，用于比较分块。
measured_control.ll 是旧 group8_reverse_k128_gep.ll 的逐字节副本，历史测量
为 1.779047 ms；它是额外控制组，不代表新导出 bm128_bn128.ll 已有该性能。
默认批量脚本包含这五份 IR，便于识别重新导出引起的性能变化。
group=8 固定的是 query tile 数；BM 改变时每组覆盖的 query 行数随之变化。
较小 BM/BN 会同时改变循环次数、访存和 LDS，这是分块实验的固有效应。

单份编译和测试（在 gfx936/DTK 机器、仓库根目录执行）
----------------------------------------------------
沿用原来能编译最优版本的 LLVM_DIR、ROCM_BC、GPU_ARCH 环境。
GenHsaco.sh 会在当前目录写中间文件，每个候选使用独立目录。

  mkdir -p generated_ir/bm_bn_sweep/manual-bm128_bn64
  cd generated_ir/bm_bn_sweep/manual-bm128_bn64
  bash ../../../GenHsaco.sh ../bm128_bn64.ll 3 kernel.hsaco
  python3 ../../../launchKernel.py --hsaco kernel.hsaco \
    --config ../bm128_bn64.launch.json --warmup 50 --repeat 200 \
    --json-output result.json

其他 bm*.ll 使用相应同名 .launch.json，尤其不要给 BM=64 使用 32,32,1 grid。
所有版本的 shared-mem 都是 0；静态 LDS 已写入 IR，不能再作为动态 LDS 叠加。

批量测试（在仓库根目录执行）
----------------------------
  python3 generated_ir/bm_bn_sweep/run.py --dry-run --rounds 1
  python3 generated_ir/bm_bn_sweep/run.py

仅测四组新导出 IR：
  python3 generated_ir/bm_bn_sweep/run.py \
    --variants bm128_bn128 bm128_bn64 bm64_bn128 bm64_bn64

默认先编译所有候选，再轮换顺序测 5 轮；每轮预热 50 次、计时 200 次。
可用 --launch-python 指定 HIP PyTorch 的 Python；--device/--seed 设置设备和输入。
每次新建 run-时间/，不会覆盖旧结果。数值校验失败会停止。
summary.json 每测完一项更新，保留各候选完成轮数；比较时确认轮数一致。
kernel_median_ms 是各轮程序 median 的中位数；sdpa_speedup 是各轮加速比中位数；
speedup_vs_bm128_bn128 以本次重新导出的 128x128 为分块基准。
round-*.json 保存原始样本和正确性；kernel.hsaco.amdgcn 保存目标汇编。
请同时提供 VGPR/SGPR/scratch、实际 LDS、active waves 和程序计时。
hipprof 时间不用于性能排序，PMC 单独采集。

验证及复现
----------
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/bm_bn_sweep/validate.py
  python3 generated_ir/bm_bn_sweep/prepare.py

prepare.py 会重新生成四组输入、导出 LLVM，再更新顶层 IR 和 manifest。
export/run-*/ 保存原始 MLIR、编译命令、参数、lowering 日志和结果。
如重新生成，需重新运行 validate.py；validation.json 中哈希应与 manifest 对应。

validate.py 检查 llvm-as、O3 verifier，并使用延迟 DMA CPU 模型对每份 IR
采样逻辑 (head,query)=(0,0),(1,1),(1,7),(1,8),(1,最后一块)。覆盖 BN>BM
的部分因果 tile、分组边界及完整 key 前缀，检查输出范围、稳定 attention
参考（atol=rtol=1e-2）、未完成 DMA 和动态 MMAC 次数，防止 QK 按 O 分片重算。
CPU 模型不模拟独立 wave 时序或目标后端；未在本机执行 gfx936 编译和 GPU 测试。
沿用现有近似数值路径，没有增加 running-max softmax。
