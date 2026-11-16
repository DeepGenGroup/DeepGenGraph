当前 MyTest 按指定基准参数重新生成的 attention IR

noncausal/kernel.o3.ll
  参数来源：../best_noncausal_early_wave/command.sh
  group32、reverse=0、lazy-softmax-base=8、softmax-cvt-pack=rtz。

causal/mask_finite_pk_rtz.o3.ll
  参数来源：../causal_mask_lowering/mask_finite_pk_rtz.command.sh
  group4、reverse=1、causal-mask=finite、lazy-softmax-base=off、softmax-cvt-pack=rtz。

输入 MLIR 与基准逐字节一致，全部调优参数保留；仅调整输入/输出路径及工作目录。
两组都从 MLIR 经当前 MyTest 生成，均通过 llvm-as 与 default<O3>,verify。
O3 后 address.thread 空 asm 的范围均为 [0,256)，均有 32 对 RTZ 转换。
尚未进行 GPU 数值或性能测试。

每个子目录保留原始 .ll、.o3.ll、配套 .launch.json、input.mlir、command.sh、
lowering.log、verification.log、generation.json（完整参数、哈希和 pass 命中记录）。
构建日志：build.log。

causal 的 Cross BN K0 与原基准日志一致，未命中：requires canonical unsigned loop condition。
--cross-bn-k0=1 参数仍然开启；当前编译器未完成该变换，不代表重新生成后性能与旧基准相同。

重现（仓库根目录）：
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  python3 generated_ir/thread_bounds_lowering/generate.py

也可单独运行：
  python3 generated_ir/thread_bounds_lowering/generate.py noncausal
  python3 generated_ir/thread_bounds_lowering/generate.py causal
