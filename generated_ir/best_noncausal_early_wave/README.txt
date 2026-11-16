最佳已测 non-causal 配置：当前 MyTest 重新生成

参考：generated_ir/noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll
历史实测中位耗时：1.5057590007781982 ms（来源：noncausal_cvt_pack_pass/manifest.json）。
本目录产物由当前 MyTest 从 input.mlir 全流水线生成，包含提前的 DMA/shuffle lowering
和结构化循环 HoistWaveInvariants；不是历史已测 IR 的复制品，尚未重新进行 GPU 性能测试。

形状及调优参数
  B/H/S/D = 1/32/4096/128；non-causal；BM/BN/BK = 128/128/32。
  WMMA layout 3.3；256 threads；block-warps 4,1；QK/VO tile 32/32。
  workgroup group32，query=x，head=y，reverse=0。
  async DMA，alignment16，useKPack，asm128；carrier-words34816。
  retain-fragment-words32；Q/V swizzle off；K swizzle 8:6。
  math approx；mask-add-bypass、defer-lane-sum、local-reduce 均为 approx。
  LazySoftmaxBase threshold8；SoftmaxCvtPack RTZ（向零舍入）。

流水排布
  采用历史 winner 的 pipeline=0 路径，不启用手工 pipeline.stage/order 调度。
  保留 LLVM 阶段的 PV interleave4、V double buffer、K DMA64-swizzle、
  K LDS prefetch8、softmax-fragments=pv、K tail prefetch4、cross-BN K0 prefetch。
  coalesce-waits=0；frag-prefetch=0。
  LazySoftmaxBase 在 prefetchNextK/elideBackedgeBarrier 之后，SoftmaxCvtPack 最后执行。
  --hoist-lds-bases=1 --hoist-lane-shuffles=1 使用当前提前外提实现。

文件
  kernel.ll：MyTest 导出的 LLVM IR。
  kernel.o3.ll：LLVM default<O3>,verify 后的 IR。
  input.mlir：输入快照；command.sh：完整重现命令。
  lowering.log：全流水线 IR dump；stages.txt：阶段行号。
  generation.json：输入/编译器/产物哈希、检查结果及优化命中记录。
  kernel[.o3].launch.json：对应的 launch 参数。

重现（仓库根目录）
  bash generated_ir/best_noncausal_early_wave/command.sh > generated_ir/best_noncausal_early_wave/lowering.log 2>&1
  /data2/xsl/install/bin/llvm-as generated_ir/best_noncausal_early_wave/kernel.ll -o /dev/null
  /data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' generated_ir/best_noncausal_early_wave/kernel.ll -o generated_ir/best_noncausal_early_wave/kernel.o3.ll
