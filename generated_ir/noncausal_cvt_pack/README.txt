cvt/pack 合并实验 IR（non-causal，gfx936）

仅在本目录新增实验文件；不改 C++ Pass，不覆盖已有 IR。
输入来自 ../noncausal_lazy_base/ 的 control.ll 和已实测的 lazy_t8.ll，
不是重新跑当前 Pass 生成，也未叠加后来的 swizzle 等变更。manifest.json 固定源 SHA256。
形状 B,H,S,D = 1,32,4096,128，grid=32,32,1，block=256,1,1，K 为 BHSD。

变体（每个都带同名 .launch.json）
  lazy_t8                原 lazy_t8.ll 的逐字节副本，主对照
  lazy_t8_pk_rne         32 条 v_cvt_pk_f16_f32，最近偶数舍入；需目标支持
  lazy_t8_sdwa_rne       64 条转换，第二次直接写高 16 位；保留原舍入方式
  lazy_t8_pk_rtz         32 条 v_cvt_pkrtz_f16_f32；向零舍入，数值会改变
  control               原 control.ll 的逐字节副本，不启用滞后基准
  control_pk_rne / control_sdwa_rne / control_pk_rtz
                        用来单测转换优化，不叠加滞后基准

改写细节
  只替换 16 个 fptrunc <4 x float> %math.result... to <4 x half>。
  每片按 [0,1]、[2,3] 各打包一个 i32，再 bitcast 回原来的 <4 x half>。
  原 math.result / math.packed 锚点、分母累加、输出转换、MMAC 操作数、
  DMA/wait/barrier 和指令顺序均保留。generate.py 反向替换后校验与源逐字节相同。
  SDWA 的第二条转换使用 dst_unused:UNUSED_PRESERVE 保留低半字；
  inline asm 使用 early-clobber 输出约束，避免第一条覆盖第二条的输入。
  96 -> 32 / 64 是每 wave 每轮 softmax 转换段的预期指令数，
  不含输出转换，且不代表实测性能收益。

本机验证与限制
  所有 .ll 通过 llvm-as 和 opt default<O3>,verify；.o3.ll 是验证产物。
  上机请直接使用非 .o3 的 .ll，由 GenHsaco.sh 按 DTK 再做链接和 O3。
  verify.py 执行实际 IR，检查 RNE 与各自基线逐位相同、RTZ 与参考 attention
  的 atol=rtol=0.01，并比较 LDS 内容/epoch、全局访存、V/K0 调度及慢路径次数。
  同时覆盖 FP16 中点、相邻 FP32 值、下溢和溢出，独立检查 RTZ 舍入模型。
  CPU 模型不验证 gfx936 指令语义、MMAC 内部舍入、寄存器分配或 GPU 时序。
  验证结果在 validation.json / validation.log。

  本机 LLVM 不认识 gfx936，尚未进行目标 GPU 编译或运行。
  本机 gfx900/gfx90a 对象汇编探测：SDWA 和 PKRTZ 成功；PK_RNE 不支持。
  PK_RNE 的汇编语法另已在本机 gfx950 对象编译中验证。
  因此 PK_RNE 是能力探测候选，不应假定 gfx936 一定支持。
  probe_validation.json 与 probes/*.log 记录本机结果；不强开硬件 feature。
  上机脚本会先用目标 DTK 编译小探针到对象文件，不支持的候选自动跳过。

上机测试
  bash generated_ir/noncausal_cvt_pack/run_gpu.sh

  默认顺序：lazy_t8、pk_rne、lazy_t8、sdwa_rne、lazy_t8、pk_rtz、lazy_t8。
  若只测保留舍入方式的候选：
    bash generated_ir/noncausal_cvt_pack/run_gpu.sh lazy_t8 lazy_t8_sdwa_rne lazy_t8
  若单测优化，不叠加 lazy base：
    bash generated_ir/noncausal_cvt_pack/run_gpu.sh control control_sdwa_rne control control_pk_rtz control

  可设置 LLVM_DIR、ROCM_BC（与 GenHsaco.sh 相同），以及 ATTN_WARMUP / ATTN_REPEAT。
  runs/batch-*/ 下每次运行独立目录，含 input.sha256、probe.log、build.log、
  kernel.hsaco.amdgcn、assembly.json、gpu.log、result.json。
  assembly.json 记录全 kernel 静态指令数、VGPR、scratch/spill；
  有 scratch/spill 或目标打包指令数不是 32 的版本会跳过 GPU 测试。
  正确性失败时脚本返回非零；重点比较 result.json 的 passed、kernel_median_ms，
  及 assembly.json 的 VGPR/scratch。RTZ 通过当前容差也不意味着逐位等价。

重现生成与 CPU 验证
  python3 generated_ir/noncausal_cvt_pack/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_cvt_pack/verify.py
  generate.py --llvm-bin 可指定用于 IR 校验的 LLVM 工具目录。
