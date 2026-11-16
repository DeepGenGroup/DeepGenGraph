无 region 包装的 fragment IR / 最佳 non-causal 配置

已删除 frisk.fragment 和 frisk.fragment_yield 两个 Op 的定义与使用。
保留 !frisk.fragment 类型；load/store/extract/insert/gemm_frag/elementwise_frag
直接位于循环体中，用 affine/scf iter_args 和 yield 传递 SSA 累加器。
MyTest：block IR -> fragment IR -> thread IR -> 调度与存储规划 -> LLVM。

产物：
  01-fragment.mlir       可独立解析、重新 lowering 的 fragment IR
  lowering.log           完整 lowering 日志，两个禁用 region Op 均为 0
  kernel.ll              gfx936 LLVM IR，已通过 verifier
  kernel.o3.ll           O3 + verifier 产物
  kernel.launch.json     ABI、grid/block 和 non-causal 配置
  validation.json        验证命令、结果及文件 SHA256
  numerical_validation.json  actual-LLVM wave/DMA 模型结果

采用历史最佳 k0_sum_anchor 的 non-causal 参数，完整命令见 command.sh / command.json：
  BHSD=1x32x4096x128；256 threads；waves=4,1；layout=3.3；approx；
  async/useKPack/asm128；QK/VO tile=32；group32 forward；
  retain-fragment-words=32；PV interleave=4；V double buffer=1；
  K swizzle=8:6；k-dma=64-swizzle；k-lds-prefetch=8；
  softmax-fragments=pv；k-tail-prefetch=4；cross-bn-k0=1。
  历史参考 median=1.663345 ms，不是本次设备实测结果。

验证：
  - typed fragment 多形状、类型校验、幂等性、序列化、CPU 执行通过。
  - flat SSA 调度、预取拒绝条件、CPU 对照和 attention 调度通过。
  - 跨循环保留、shuffle 和 legacy tiling 回归通过。
  - fragment 文件在独立进程重新 lowering，与 MyTest thread 阶段一致。
  - 同参数 raw / O3 LLVM 均与上一版通过 llvm-diff；默认配置也与原二进制一致。
  - actual-LLVM 模型：随机输入、future-key-dominates 非因果用例、O3 bitwise
    对照、延迟 DMA 与 K0 交接均通过；最大绝对误差约 9.92e-5。

重新生成（仓库根目录）：
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j 6
  bash generated_ir/flat_fragment_noncausal/command.sh
  python3 generated_ir/flat_fragment_noncausal/save_stages.py
  /data2/xsl/install/bin/opt -S -passes='default<O3>,verify' generated_ir/flat_fragment_noncausal/kernel.ll -o generated_ir/flat_fragment_noncausal/kernel.o3.ll
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/flat_fragment_noncausal/validate.py

目标 DCU 运行：
  bash generated_ir/flat_fragment_noncausal/run_gpu.sh
  可通过 LLVM_DIR、ROCM_BC、GPU_ARCH 指定 DTK 路径和目标；结果保存在 runs/。
当前机器无 /dev/kfd 与 gfx936 DTK，未执行目标 HSACO 编译或 GPU benchmark。
CPU 模型不模拟独立 wave 时序和目标寄存器分配。

现有测试限制：check_fragment_schedule.py 的完整 MyTest 分支仍使用旧的
静态索引 / asm sideeffect / 物理布局 oracle，在修改前的默认导出上也不通过。
本次通过 --schedule-only 独立运行全部调度、合法性、CPU 和 attention pass 用例；
完整导出另外使用 LLVM 等价比较和上述数值模型验证，详见 validation.json。

当前 fragment 支持静态 rank-1/2 和完整可整除分片；动态索引需由调用方保证范围；
不转换跨函数 fragment ABI。广播、mask、reduction、matrix read 直接使用标量/
vector 操作及现有通信规则。详见 FriskFragmentIR.md。
