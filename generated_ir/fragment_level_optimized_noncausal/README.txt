Fragment 层提前优化 / 最佳 non-causal 参数的 MyTest 产物

已将 fuse-fragment-accumulator、frisk-frag-ir-reorder 和
frisk-retain-invariant-fragments 放到 convert-fragment-to-thread 之前。
三个 Pass 直接分析 typed Fragment SSA；保留旧 vector/memref 匹配器。
物理 shared packing、DMA 调度、LDS 复用仍由后段 lower-frisk-fragments 完成。

参数完全沿用 flat_fragment_noncausal/command.json 对应的历史最佳 k0_sum_anchor：
  BHSD=1x32x4096x128；non-causal；256 threads；waves=4,1；layout=3.3；approx；
  async/useKPack/asm128；QK/VO tile=32；group32 forward；retain-fragment-words=32；
  PV interleave=4；V double buffer=1；K swizzle=8:6；k-dma=64-swizzle；
  k-lds-prefetch=8；softmax-fragments=pv；k-tail-prefetch=4；cross-bn-k0=1。
全部参数和环境变量见 command.sh / command.json。
历史参考 median=1.663345 ms，本次没有新的目标设备耗时。

产物：
  kernel.ll                    MyTest 导出的最终 gfx936 LLVM IR
  kernel.o3.ll                 LLVM O3 + verifier 产物
  kernel.launch.json           Q/K/V/O ABI、grid/block、non-causal 配置
  01-fragment.mlir             转换后、优化前 typed Fragment IR
  02-accumulator.mlir          accumulator 融合后
  03-reorder.mlir              typed 循环融合与预取后
  04-specialized.mlir          小循环展开后
  05-retained.mlir             不变 operand 保留后
  06-optimized-fragment.mlir   送入 convert-fragment-to-thread 的最终 Fragment IR
  07-thread-bridges.mlir       typed Fragment 转为线程寄存器表示后
  08-thread.mlir               完成 bridge 物化后
  09-storage-planned.mlir      物理 LDS 规划后
  stage-statistics.json        各阶段静态 IR 统计
  lowering.log                完整 MyTest 日志
  validation.json             验证记录与 SHA256
  numerical_validation.json   actual-LLVM 数值/DMA 模型结果
  baseline.ll                 修改前 MyTest 同输入、同参数生成的参考

优化命中：
  QK accumulator add 与 MN producer 融合，K reduction 仍从零开始。
  typed 阶段 affine.for 从 79 减至 37（小循环 specialization 之前）；
  后续 specialization 后为 45，不应把展开后的循环数和融合前直接比较。
  4 个 operand tile 在 typed 阶段移出外层循环，预算合计 32 words/thread。
  以上是静态 IR 结构统计，不是 GPU 指令数或性能测量。

验证结果：
  - MyTest、ThreadTilingTest、FragmentOptTest 构建通过。
  - 新 typed Pass 测试：四组 GEMM 形状、三 Pass 命中/拒绝/幂等性，
    异形 fragment 覆盖、别名写、预算限制、carrier 更新、typed load 预取，
    CPU 数值及零初始化 K reduction 顺序检查通过。
  - 原 typed Fragment、22 项 retention、legacy fragment loop/schedule/
    accumulator 回归通过。schedule 使用 --schedule-only，避开已有的旧地址 oracle。
  - 所有保存阶段可以独立解析；从序列化 01 重新执行提前优化，与 06 一致；
    06 单独转换为线程 IR，与 07 一致。后续 finalization 也独立通过。
  - LLVM verifier / O3 通过。LLVM 文本结构与 baseline 不同，差异见 llvm-diff.log。
  - actual-LLVM wave/DMA 模型通过随机输入和 future-key-dominates 非因果测试；
    baseline 与 kernel 输出逐 bit 一致，随机用例 raw/O3 也逐 bit 一致。
    最大绝对误差约 9.92e-5；延迟 DMA、K0 交接及 Q 每元素读取一次检查通过。
  - 当前机器无 /dev/kfd；没有执行目标 HSACO 编译或 GPU benchmark。
    CPU 模型不模拟独立 wave 时序或真实寄存器分配。

重新生成（仓库根目录）：
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest FragmentOptTest -j 4
  bash generated_ir/fragment_level_optimized_noncausal/command.sh
  python3 generated_ir/fragment_level_optimized_noncausal/save_stages.py
  /data2/xsl/install/bin/opt -S -passes='default<O3>,verify' generated_ir/fragment_level_optimized_noncausal/kernel.ll -o generated_ir/fragment_level_optimized_noncausal/kernel.o3.ll
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/fragment_level_optimized_noncausal/validate.py

目标 DCU 上复测：
  bash generated_ir/fragment_level_optimized_noncausal/run_gpu.sh
  可用 LLVM_DIR、ROCM_BC、GPU_ARCH 指定 DTK 与目标。

baseline-command.sh 是本次修改前二进制 /tmp/fragment-pass-backup/MyTest 的
生成记录；重现当前实现使用 command.sh，不需要该临时二进制。
