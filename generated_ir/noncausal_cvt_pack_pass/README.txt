PKRTZ 合并 Pass 复测产物

首选复测文件：pass_pk_rtz.o3.ll（配置：pass_pk_rtz.o3.launch.json）。
该文件由编译好的 SoftmaxCvtPackPass 处理原 lazy_t8.ll 后运行 O3 得到。
与 ../noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll 的完整文本比较，仅以下两处不同：
  1. ModuleID 中的文件路径；
  2. 函数属性 "frisk.softmax_cvt_pack"="rtz"（防止重复应用）。
所有指令、SSA 名、顺序、debug/调度 metadata 都相同。
generate.py 对这一条件做断言；manifest.json 记录所有输入/输出 SHA256。

pmc_test.log 中原实验版数据（用户实测，不是本次 Pass 产物的测量）：
  lazy_t8.o3.ll：median 1.571839 ms
  lazy_t8_sdwa_rne.o3.ll：median 1.538879 ms
  lazy_t8_pk_rtz.o3.ll：median 1.505759 ms，VGPR 228，scratch 0
  PKRTZ 相比原 lazy_t8 耗时下降约 4.2%；日志注明 PK_RNE 指令不支持。

其他产物
  pass_off.o3.ll            关闭新 Pass；与原 lazy_t8.o3.ll 仅 ModuleID 不同。
  pipeline_t8_pk_rtz.o3.ll  从 control.ll 串联 LazySoftmaxBase(T=8) 和新 Pass；
                           包含当前 LazySoftmaxBase 实现的结构差异，供后续流水线复测。
  *.ll                     对应 O3 前的 LLVM22 兼容文本。
  *.raw.ll                 C++ 导出器直接打印的 IR。
  *.pass.log               Pass 命中信息，两个开启版均命中 16 fragments / 32 pairs。

复测（自动交替已测最佳与新产物）
  bash generated_ir/noncausal_cvt_pack_pass/run_gpu.sh
  默认：measured pass_pk_rtz measured pipeline_t8_pk_rtz measured。
  只测精确固化版本：
    bash generated_ir/noncausal_cvt_pack_pass/run_gpu.sh measured pass_pk_rtz measured
  原始对照：
    bash generated_ir/noncausal_cvt_pack_pass/run_gpu.sh pass_off pass_pk_rtz pass_off
  与 GenHsaco.sh 一样可指定 LLVM_DIR、ROCM_BC；可设置 ATTN_WARMUP/ATTN_REPEAT。
  runs/batch-*/ 内保存编译日志、汇编、VGPR/scratch、正确性和耗时 JSON。
  每次都输入 .o3.ll，以复现用户实测的编译入口。

编译器开关
  --softmax-cvt-pack=off|rtz，默认 off。MyTest、LegacyLLVMExportTest 均支持；
  RunFriskSHell.sh 转发该命名参数。
  rtz 明确允许 softmax P 的 FP32->FP16 从最近偶数舍入改为向零舍入。
  分母保持 FP32；输出转换保留原舍入；不修改 DMA、MMAC 顺序、wait/barrier。
  仅匹配带 register_lifetimes 标记、四个 exp2 结果组成的向量转换，
  且 half 结果只通过 packed identity 用作 MMAC A；没有标记或有其他用途时跳过。
  在 LazySoftmaxBase 之后运行，不依赖 workgroup-uniform 控制流。

重现
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 2
  python3 3rd/deepgengraph/test/check_softmax_cvt_pack.py
  python3 generated_ir/noncausal_cvt_pack_pass/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_cvt_pack_pass/verify.py

验证范围
  LLVM/O3 verifier；关闭与幂等；SSA 改名；未标记、非 exp2、B 操作数、
  额外 store/use、错误宽度、strictfp/optnone 等拒绝条件。
  数值模型比较实测 IR、精确 Pass 版和双 Pass 串联版，覆盖随机输入、
  单 wave 重缩放、分数突变；检查输出逐位一致、LDS 内容/epoch、全局访存、
  V/K0 调度、慢路径次数和同步计数。结果在 validation.json。
  本机无 gfx936 GPU/DTK，Pass 新产物尚未实机测量；CPU 模型不模拟硬件时序。
