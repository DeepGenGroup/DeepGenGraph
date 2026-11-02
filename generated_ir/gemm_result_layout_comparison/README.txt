GEMM 结果 Layout 3.1 / 3.3 对比（2026-10-27）

更新：Exp2OpTiling 源码已修复。请使用 exp2_fixed/layout_3_1/kernel.ll 和
exp2_fixed/layout_3_3/kernel.ll；两者六组 CPU 数值检查均通过。
下面记录的是修复前产物，保留用于失败复现；新产物说明见 exp2_fixed/README.txt。

测试文件
  layout_3_1/kernel.ll：3.1，reg -> LDS -> reg。
  layout_3_3/kernel.ll：3.3，K producer permutation，使 QK 输出直接适配 PV。
各目录还包含 input.mlir、thread.mlir、launch.json、config.json、result.json、
validation.json。thread.mlir 来自同次编译的线程层 lowering 日志。
kernel.o3.ll 仅用于本地 LLVM24 检查；目标 DTK 请使用 kernel.ll。

基准选择
  当前稳定 online softmax 分支最佳已测 case：
  ../online_softmax_deferred_sum/kernel.ll，pmc_test.log median=1.201280 ms。
  历史不稳定 softmax case 曾记录 1.168559 ms，本次沿用稳定 softmax 分支。
  用当前 MyTest 完整流水线重新导出，3.3 参数与 deferred_lane_sum_pass/
  compiler_final 的 BN128 approx case 完全一致；新导出尚无 GPU 测时。

共同设置
  BHSD=(1,32,4096,128)，FP16 causal attention，BM=BN=128，BK=32。
  grid=(32,32,1)，block=(256,1,1)，K 输入 BHSD；group4 reverse。
  async/asm128，alignment=16，K swizzle=8:6，math=approx。
  frag-prefetch=0，retain-fragment-words=32，pipeline=0。
  mask-add-bypass=approx，local-reduce=approx（4 chains），defer-lane-sum=approx。
  两个 config 的 parameters 字段仅 layout 不同。

当前编译器的实际差异
                              3.1                  3.3
  静态 LDS                    65536 bytes          16384 bytes
  Q fragment retention        0 tiles              4 tiles / 32 words
  mask-add bypass             0 groups             1 group / 16 additions
  local reduction             4 groups             4 groups
  deferred lane sum           未命中               2 states / 4 loop shuffles
  CPU 六组数值检查            4 通过，2 未通过      6 全部通过

3.1 的现有导出不是已确认正确的性能候选：random query=1/31 的最大绝对误差
分别为 0.4730803 / 0.0524115（atol=rtol=0.01）。query=0、正溢出、负下溢和
分块常量 max_changes 用例通过。Q 在每次 BN 迭代重新装载，3.3 则只装载一次。
这些是实际 LLVM 的 CPU 模拟结果，未完成目标 GPU 验证或编译器根因修复。
请先核实 3.1 正确性，不能将失败结果用于性能优劣结论。

两份 kernel.ll 均通过 llvm-as 和 opt default<O3>,verify。数值模型继续检查
延迟 DMA、输出范围；本目录 validate.py 为 3.1 增加 f32 LDS 字节访问和
单分支 leader store 模拟，仍拒绝 partial-wave DMA/矩阵读/同步等操作。
模型不能证明 GPU 独立 wave 时序、后端代码生成及性能。

GPU 测试（仓库根目录，沿用已有 LLVM_DIR / ROCM_BC 环境）
  bash generated_ir/gemm_result_layout_comparison/run_gpu.sh 3.3
  bash generated_ir/gemm_result_layout_comparison/run_gpu.sh 3.1
脚本使用 GenHsaco.sh -O3 和 launchKernel.py 正确性校验、warmup=10/repeat=50；
各自 gpu/ 下保存构建日志、HSACO、汇编、test.log 和 result.json。
不带参数时依次运行 3.1、3.3，任一命令失败即停止。

复现导出
  bash generated_ir/gemm_result_layout_comparison/reproduce.sh
新导出保存在 reproduced/run-*，不覆盖本次产物。
原始编译命令及完整 lowering.log 位于 run-20261027-121856-x88fub13/。

复现本地检查
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/gemm_result_layout_comparison/validate.py
当前完整检查返回 1，原因是如上所述的 3.1 数值误差；单独指定 layout_3_3 返回 0。

后续根因诊断：已定位到 Exp2OpTiling 输入/输出 Layout 不一致时错误标注 alpha。
仅绕过错误 alpha 重排的 diagnostic_alpha_identity/kernel.ll 六组 CPU 检查通过。
见 diagnostic_alpha_identity/README.txt；这是定位用 IR，未修改编译器或原始对比产物。
