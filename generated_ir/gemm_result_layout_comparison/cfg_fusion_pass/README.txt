MMAC 融合的 Pass 顺序修复（2026-10-27）

源码
  FuseMMACAccumulator.cpp / LLVMExportUtils.h：融合入口增加 simplifyCFG 参数，
  默认 false，以保留早期数学融合阶段的行为。最终阶段启用后顺序为：
    SimplifyCFG -> InstCombine -> FuseMMACAccumulator -> InstCombine
  test/test_src.cpp：在 lifetime/thread-bound、LDS、DMA 导出改写之后，打印
  LLVM 文本之前执行最终融合。math=strict 不启用；其余已有 MMAC 数学模式启用。
  原有早期融合仍保留，避免改变 reduction/lifetime 匹配所依赖的 CFG。
  test/legacy_llvm_export_test.cpp：独立 --fuse-mmac-accumulator 入口启用 CFG
  简化，直接处理已经导出的 IR。

原因
  OptimizeRegisterLifetimes 才为 thread id 注入范围信息，并将 3.1 的 LDS
  writeback leader guard 折叠为 true。融合器要求 MMAC 与尾部 fadd 同块，
  因此需要在此之后清理 CFG 再融合。普通后端 O3 不会自动重跑自定义融合器。
  不放宽融合器对跨块、load、StrictFP、多用途链的合法性约束。

用户硬件实测（诊断 IR，非本目录新导出的 GPU 测试）
  来源：pmc_test.log 第 2-30 行，已另存 diagnostic_gpu_log_lines_2_30.txt。
  ../spill_analysis/kernel.ll：median 5.346875 ms，scratch 0，VGPR 256，
  LDS 65536 bytes；用户确认数值测试通过。
  相对之前 8.209191 ms，耗时下降约 34.87%。

产物
  layout_3_1/kernel.ll / layout_3_3/kernel.ll：最终源码直接生成的 LLVM IR。
  每个目录含 launch.json、config.json、result.json、lowering.log 和数值报告。
  最终来源：run-20261027-152102-801zvuny/grid-000001、grid-000002。
  run-20261027-151454-z6uud_gi 是开发中提前清理 CFG 的试验，仅用于留档，
  不是最终源码或推荐测试产物。
  原 exp2_fixed 和 spill_analysis 的 IR 均保留。

复现与 GPU 测试（仓库根目录）
  bash generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/reproduce.sh
  上述脚本重新构建、运行 Pass 回归、导出到 reproduced/run-*，不覆盖快照。
  bash generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run_gpu.sh 3.1
  可改为 3.3 或省略参数测试两种；沿用 LLVM_DIR/ROCM_BC，O3，warmup10/repeat50。
  新导出未在本地执行 GPU 测试。

最终验证
  MyTest / LegacyLLVMExportTest 构建成功；check_register_math.py 全部通过，
  包括真实 thread-bound -> lifetime -> CFG cleanup -> MMAC fusion 顺序回归。
  两份导出均通过 llvm-as、LLVM default<O3>,verify 以及各六组 safe-softmax
  实际 LLVM CPU 模型检查；两份最大绝对误差均为 0.001061586。
  3.1 在最后阶段新增 64 处融合，QK 从 64 x 2 链变为 16 x 8 链；终端
  FP32 部分结果从 256 降为 64 元素/线程（逻辑 IR 数量，非物理寄存器数）。
  3.1 的 MMAC/DMA/barrier/bpermute/f32 LDS load/store 数量与测过的诊断 IR
  全部相同；二者经过同一 LLVM O3 后 llvm-diff 返回 0，无结构差异。
  原始文本有常量 GEP 表达形式和 debug metadata 差别，见 diagnostic_llvm_diff.log。
  3.3 早期仍融合 80 处，末尾没有新增融合；额外规范化改变部分地址表达式，
  O3 后与前一份 3.3 并非逐结构相同，需另测硬件性能，不宣称性能不变。
  source.patch 仅记录本次 Pass 顺序与回归测试修改；summary.json、structure.json、
  SHA256SUMS 和各 layout 目录的 validation.json 记录最终产物与验证结果。
