当前编译器生成的 non-causal IR（threadIR 多链规约）

参数来源：../best_noncausal_early_wave/kernel.o3.ll 所在目录的 command.sh。
input.mlir、command.sh 和两份 launch.json 均与参考目录逐字节一致。
使用当前 MyTest，从 MLIR 完整重新 lowering，再运行 default<O3>,verify。
包含当前编译器的其他已有修改，因此并非仅针对多链规约的历史版本 A/B 对比。

主要参数：
  B/H/S/D = 1/32/4096/128；non-causal；BM/BN/BK = 128/128/32。
  WMMA 3.3；256 threads；block-warps 4,1；QK/VO tile 32/32。
  local-reduce=approx，chains=4，min-inputs=16，ops=all。
  其余参数见 command.sh，与 best_noncausal_early_wave 保持一致。

供测试的文件：
  kernel.o3.ll：O3 优化后的 LLVM IR。
  kernel.o3.launch.json：对应 launch 配置。
  kernel.ll：MyTest 原始导出。
  lowering.log / stages.txt：全流程 dump 与阶段索引。
  generation.json：编译器、输入、产物哈希及验证记录。

重新生成（仓库根目录）：
  python3 generated_ir/best_noncausal_thread_multichain/generate.py

已进行 LLVM IR 验证；未进行 GPU 性能、正确性或 spill 实测。
