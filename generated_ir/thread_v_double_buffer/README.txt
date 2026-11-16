Thread IR V 双缓冲迁移后的测试产物

供 GPU 测试：
  causal.o3.ll       对应 causal_mask_lowering/mask_finite_pk_rtz.o3.ll 的调优参数
  noncausal.o3.ll    对应 best_noncausal_early_wave/kernel.o3.ll 的调优参数
  对应的 .o3.launch.json 为 launch 参数；.ll 是 MyTest 原始导出。

共同参数：B/H/S/D=1/32/4096/128，BM/BN/BK=128/128/32，256 threads，
block-warps=4,1，layout=3.3，QK/VO tile=32，async alignment16/useKPack/asm128，
retain-fragment-words=32，PV interleave=4，V double buffer=1，K DMA=64-swizzle，
K LDS prefetch=8，K tail prefetch=4，Softmax CVT=RTZ。
causal：group4/reverse=1，finite mask，lazy-softmax-base=off，uniform-masks=0。
noncausal：group32/reverse=0，lazy-softmax-base=8，uniform-masks=1。
其余完整参数以 causal.command.sh / noncausal.command.sh 为准。

V 双缓冲已在 Thread IR 执行，LLVM 对应实现和导出调用已删除。
两组都命中：4 slices，slot-bytes=8192，displacement=8192，removed-barriers=3，
LDS=16384。不增加 allocation 或容量；ready waits 和最终 reader barrier 保留。
thread.before/after.mlir 保存 Thread 层改写证据。

这里是当前 MyTest 按历史调优参数重新生成的结果，不是历史 LLVM 文件的复制。
*.legacy.ll / *.legacy.o3.ll 是本次修改前的 MyTest 按同样参数重新生成的对照。
causal 的 Cross BN K0 在修改前后都因 requires canonical unsigned loop condition
跳过；其他优化命中报告在修改前后相同。noncausal 的 Cross BN K0 仍命中。

验证：
  *.generation.json：命令、输入/编译器/产物哈希、LLVM/O3 verifier、命中报告。
  validation.json：实际 LLVM 的 CPU 延迟 DMA / LDS 旧读者同步审计和数值对照；
    覆盖 causal/noncausal 的 query 0、1、31，原始及 O3 输出与迁移前逐位一致。
  vdb-tests.log：32 项 Thread pass 接受、拒绝、无部分改写、幂等性回归。
  pv-tests.log：20 项 PV interleave 回归；tune-tests.log：27 项调优参数/报告解析回归。
  final-checks.json：最终编译器/产物哈希核对，真实 Thread IR 的独立执行与幂等性检查。
  GPU 性能、寄存器分配和独立 wave 时序尚未测试。

重现（仓库根目录）：
  python3 generated_ir/thread_v_double_buffer/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/thread_v_double_buffer/verify.py
  python3 3rd/deepgengraph/test/check_v_double_buffer.py
  python3 3rd/deepgengraph/test/check_pv_interleave.py
  python3 generated_ir/thread_v_double_buffer/check_artifacts.py

generate.py --mode legacy 仅用于记录迁移前基线，需要保留本次临时旧编译器
/tmp/MyTest-before-thread-vdb；常规重现默认 mode=thread，不依赖该临时编译器。
