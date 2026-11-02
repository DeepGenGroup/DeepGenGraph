Exp2OpTiling 源码修复后的 3.1 / 3.3 测试产物（2026-10-27）

请测试本目录下的新导出：
  layout_3_1/kernel.ll
  layout_3_3/kernel.ll
原 ../layout_3_1/kernel.ll 保留为失败回归，不要混用。

源码修复
  3rd/deepgengraph/lib/Conversion/FriskToBase/FriskBaseToTheadlevelIR_plan2.cpp
  Exp2OpTiling 改为按结果 LowerInfo 物化输入，再进行 exp2。
  输入到结果的真实布局转换由已有 bridge/finalization 机制完成，避免按输入
  Layout 计算却按不同输出 Layout 标注，从而将 alpha 分配到错误 query 行。
  不是针对本 case 的 SSA 替换，也没有关闭 safe softmax。

配置
  沿用稳定 softmax 最佳已测配置，BHSD=(1,32,4096,128)，BM=BN=128，BK=32，
  group4 reverse，256 threads；两候选参数仅 layout=3.1/3.3 不同。
  3.1 LDS=65536 bytes，当前 Q retention/mask bypass/deferred sum 仍未命中。
  3.3 LDS=16384 bytes；归一化候选目录路径后，整个 kernel.ll 与修复前一致。
  各自 launch.json 与原配置一致。

验证
  MyTest 和 ThreadTilingTest 构建成功。
  check_plan2_thread_tiles.py 与 check_wmma_layout_shuffle.py 全部通过。
  两份 LLVM 均通过 llvm-as 和 opt default<O3>,verify。
  check_online_softmax.py：每份六组 CPU 实际 LLVM 执行，全部通过；包含随机
  query=0/1/31、正溢出、负下溢、running max 变化，以及延迟 DMA/输出范围检查。
  3.1 random query=1：最大绝对误差 0.4730803 -> 0.0007240294。
  3.1 random query=31：最大绝对误差 0.05241151 -> 0.0000958800。
  两份六组中的最大绝对误差均为 0.001061586。
  同一更新后的测试工具执行旧 3.1 IR，仍在 random query=1 失败，见
  before-fix-negative.log，说明检查没有通过放宽容差掩盖问题。

数值测试工具更新
  check_attention_llvm_numerics.py 支持 f32 LDS 的真实字节存取和只写 LDS 的
  partial-wave leader 分支；仍禁止此类分支内的 DMA/矩阵读/同步/原子操作。
  check_online_softmax.py 新增 --q-loads per-tile，精确检查 3.1 每轮 Q 装载。
  默认仍为 once，保留 3.3 的完整 Q 单次装载检查。

复现数值检查（仓库根目录）
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    generated_ir/gemm_result_layout_comparison/exp2_fixed/layout_3_1/kernel.ll \
    --bn 128 --group 4 --q-loads per-tile
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_online_softmax.py \
    generated_ir/gemm_result_layout_comparison/exp2_fixed/layout_3_3/kernel.ll \
    --bn 128 --group 4 --q-loads once

重新编译导出
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest -j 2
  bash generated_ir/gemm_result_layout_comparison/reproduce.sh
  新产物进入 ../reproduced/run-*，不覆盖本目录。

GPU 测试（沿用 LLVM_DIR / ROCM_BC 环境）
  bash generated_ir/gemm_result_layout_comparison/exp2_fixed/run_gpu.sh
  或追加 3.1 / 3.3 仅测试一个版本。
  产物与日志保存在各自 gpu/ 目录；使用 O3，先正确性校验，warmup=10/repeat=50。
  本轮未运行 GPU，CPU 检查不代表硬件时序或性能结论。
