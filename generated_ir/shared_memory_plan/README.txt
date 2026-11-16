由联合 shared-memory 规划生成的 LLVM IR

测试文件
  noncausal/kernel.o3.ll
  causal/kernel.o3.ll
每份都有同名 .launch.json；kernel.ll 是 MyTest 的原始 LLVM 输出。

配置来源
  non-causal: generated_ir/best_noncausal_early_wave/kernel.o3.ll
  causal: generated_ir/causal_mask_lowering/mask_finite_pk_rtz.o3.ll
采用对应目录记录的 MLIR 输入和 MyTest 调优参数。
共同配置：B/H/S/D=1/32/4096/128，256 threads，block-warps=4,1，
WMMA 3.3，QK/VO tile=32/32，async DMA alignment16，useKPack，asm128。
non-causal: workgroup group32, reverse0, LazySoftmaxBase threshold8。
causal: workgroup group4, reverse1, finite causal mask, LazySoftmaxBase off。
两者均启用 RTZ half 打包。

规划与生成
  计算布局确定后，统一收集 allocation 的全部访问与别名。
  联合选择物理布局、DMA 宽度、参与线程、源映射和目的映射。
  物理布局通过全部 shared 访问的索引改写落实，DMA 使用已发布映射发射。
  原 --k-dma=64-swizzle --k-dma-arg=1 翻译为通用约束
  --shared-copy-constraints=1:8:9:3；8:6 swizzle 仍来自原调优参数。
  8B copy 直接导出 buffer_load_dwordx2，不再经过 LLVM 拆分或二次 swizzle。
  旧 narrowKDMA、分步 packing/DMA/swizzle 规划函数已删除。
  规划器不依赖 attention、Q/K/V 名称或固定参数位置。

复现
  python3 generated_ir/shared_memory_plan/generate.py
或用各子目录 command.sh 重跑 MyTest，然后执行：
  /data2/xsl/install/bin/opt -S '-passes=default<O3>,verify' kernel.ll -o kernel.o3.ll

校验记录
  generation.json：原参数、当前命令、输入/编译器/产物哈希和优化命中记录。
  lowering.log：完整阶段 IR；o3.log：O3 校验输出。
  verification/：规划、地址执行、packing、async、双缓冲和调度回归记录。
  CPU DMA 检查执行实际生成的 LLVM 地址，覆盖 64/128/256 线程、
  完整轮次与末轮部分 wave、两级 XOR、逐字节结果和边界保护区。
  两份最终 LLVM 均需通过 llvm-as 和 default<O3>,verify。
  尚未进行目标 GPU 数值或性能测试。
