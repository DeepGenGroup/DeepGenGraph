跨 BN 预取下一轮 K0：32 words 最佳基线上的结构实验

输入：generated_ir/q_bk_retention_sweep/retained_32words.ll
SHA256: c5627d7e797e5ee93c73ba04d7112a238ba4dbcb52a888267a4c7b30520f9915
最近测得 median 1.257279992 ms，SDPA/kernel = 0.889348827；这些是旧基线测量，
本实验性能尚待 gfx936 实测。baseline.ll 与输入逐字节相同。

三个版本：
  baseline.ll             原始 32 words 基线。
  k0_late_control.ll      K0(0) 移到入口；原轮末 barrier 前移到 V3 全部读取之后；
                          在最后 32 条 PV MMAC 之后，有条件发出 K0(n+1)。
  k0_prefetch_pv_tail.ll  与 control 相同，但把 K0(n+1) 提前到最后 32 条 PV MMAC 之前。

预期时序（中间 BN）：
  V3 DMA -> 原 V3 wait/publication -> V3 全部 LDS 读取
  -> 原轮末 reader barrier（前移）-> if (n+1<N) 发出 K0(n+1)
  -> 当前 BN 最后 32 条 PV MMAC -> 下一轮原有 K0 wait/publication -> K0 使用。
首轮在 Q 全部读取及原有交接 barrier 后发出 K0(0)，最后一轮不发出 K0(n+1)。
所有 V 阶段使用低 8 KiB，因此必须在最后 V 读者交接后才覆盖该槽位。

固定条件：BHSD=(1,32,4096,128)，BM=BN=128，BK=32，256 threads，
group8 reverse / K128 swizzle，Q 保留 32 words，静态 LDS 16384 bytes。
每 block 执行 N=query_tile+1 轮，N=1..32。
动态计数保持：Q DMA 8，K DMA 8N，V DMA 8N，vmcnt(0) 4+8N，barrier 5+12N。
实验版提供 32*(N-1) 条 MMAC 的 K0 未完成窗口；不是额外预读，不减少数据字节。
克隆的地址计算沿用原 K0 的完整地址依赖和 swizzle，仅替换 BN 索引；没有假设
“地址固定加一个偏移量”。生成器按输入 SHA 限定范围，本阶段尚未实现为通用 Pass。

测试（仓库根目录；使用原 gfx936 DTK 环境）：
  bash generated_ir/retained_cross_bn_k0/run.sh baseline r1
  bash generated_ir/retained_cross_bn_k0/run.sh k0_late_control r1
  bash generated_ir/retained_cross_bn_k0/run.sh k0_prefetch_pv_tail r1
  bash generated_ir/retained_cross_bn_k0/run.sh baseline r2

默认 warmup=50，repeat=200，可通过 K0_WARMUP / K0_REPEAT 调整。
LLVM_DIR、ROCM_BC、GPU_ARCH 沿用 GenHsaco.sh；默认 GPU_ARCH=gfx936。
每次运行保存到 runs/<variant>-<label>-<随机后缀>/，包括输入 SHA、编译日志、
汇编、assembly_counts.json、benchmark.log、result.json，不覆盖其他实验输出。

建议先看正确性、median、SDPA/kernel、VGPR/SGPR、scratch、LDS，再看 PMC。
baseline vs prefetch 判断整体收益；control vs prefetch 判断提前发出是否有收益。
control 也包含首轮搬运、地址重建、guard 和 barrier 前移，不能只用它取代基线。
当前基线 VGPR 已为 244；新结构的寄存器分配、spill 和后端调度需用目标汇编核对。
这里只隐藏下一轮 K0 的部分延迟，L2 命中率或 SHM operation 数量不必下降。

复现生成及本地检查：
  python3 generated_ir/retained_cross_bn_k0/generate.py
  python3 generated_ir/retained_cross_bn_k0/verify.py
验证输出见 validation.json / validation.log；*.opt.ll 仅用于本地 O3 结构核对，
GPU 编译输入请使用三个原始 .ll（run.sh 已按此设置）。
CPU 模型执行实际输入 IR，检查逐位数值、逐次全局/LDS 地址轨迹、延迟 DMA、
跨 wave LDS 读写冲突、首末轮边界以及预取与 PV 重叠次数。
本地 LLVM verifier/O3 不等于 gfx936 编译或硬件验证；这里不宣称硬件正确性/性能。
