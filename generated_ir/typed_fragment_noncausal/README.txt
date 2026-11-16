显式 fragment IR / 最佳 non-causal 配置

MyTest 默认 pipeline：
  block IR -> convert-friskbase-to-fragment -> convert-fragment-to-thread
  -> finalize-thread-tiling -> 原 fragment 调度/优化 -> LLVM

主要产物：
  kernel.ll                 可交给 GenHsaco.sh 的 gfx936 LLVM IR
  kernel.o3.ll              本地 LLVM O3 + verifier 产物
  kernel.launch.json        Q/K/V/O ABI、grid/block、non-causal 配置
  lowering.log              完整 MyTest lowering 日志
  00-block.mlir             fragment 切分前的 block IR
  01-fragment.mlir          !frisk.fragment 与 load/store/extract/insert/gemm_frag
  02-thread-bridges.mlir    fragment 降到寄存器 vector 后，保留 tile bridge
  03-thread.mlir            bridge finalization 与 fragment 调度后的线程 IR
  validation.json           验证汇总与产物 SHA256

配置采用已测 k0_sum_anchor 的完整公开参数：
  BHSD=1x32x4096x128，non-causal online softmax，256 threads，waves=4,1；
  group32 forward，layout=3.3，QK/VO tile=32，approx，async/useKPack/asm128；
  retain-fragment-words=32，PV interleave=4，V double buffer=1；
  K swizzle=8:6，k-dma=64-swizzle，k-lds-prefetch=8；
  softmax-fragments=pv，k-tail-prefetch=4，cross-bn-k0=1。
  所有参数见 command.json / command.sh。历史参考 median=1.663345 ms，
  不代表新产物已完成 GPU benchmark。

重新导出（仓库根目录）：
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j 6
  bash generated_ir/typed_fragment_noncausal/command.sh
  python3 generated_ir/typed_fragment_noncausal/save_stages.py

在配置好 DTK 的目标 DCU 机器上运行：
  bash generated_ir/typed_fragment_noncausal/run_gpu.sh
  可通过 LLVM_DIR、ROCM_BC、GPU_ARCH 覆盖 GenHsaco.sh 默认路径和目标。
  脚本保留每次 HSACO 编译、正确性和性能结果于 runs/。

已完成：
  - typed fragment 单元回归：多种 GEMM 形状、逐元素计算、memref/vector
    边界、非法布局与越界诊断、幂等性、独立序列化。
  - fragment load/store 与 SSA 循环累加经 LLVM 在 CPU 实际执行。
  - 原 plan2 回归通过（legacy-tests.log）。
  - 01-fragment 在独立进程降级，规范化后与 MyTest 线程阶段完全一致。
  - 新旧同参数 LLVM，以及各自 O3 后的 LLVM，均通过 llvm-diff；diff 文件为空。
  - actual-IR wave/DMA 模型：随机输入、future-key-dominates 非因果输入、
    O3 bitwise 对照、延迟 DMA 与 K0 交接检查通过；最大绝对误差 9.92e-5。

当前机器无 /dev/kfd 与 gfx936 DTK，未执行目标 HSACO 编译或 DCU benchmark。
CPU 模型不模拟独立 wave 时序和目标寄存器分配。
另外，旧手工 pipeline=1 回归报 16/17 Op 不匹配，修改前二进制亦失败；
该问题发生在 fragment pass 之前，本次交付使用 pipeline=0。

第一版限制：静态 rank-1/2 和完整可整除的 fragment；动态 fragment 索引
要求调用方保证范围；不转换跨函数 fragment ABI。广播、mask、reduce 和
matrix read 沿用现有 fragment region / 通信实现，详见 FriskFragmentIR.md。
