Causal reverse group4：MyTest 生成的复测 IR

首选：group4_reverse_pk_rtz.o3.ll
对照：group4_reverse_off.o3.ll（只关闭 PKRTZ）。
每个 .ll 均附带同名 .launch.json；causal=true，K 为 BHSD。
shape=(1,32,4096,128)，BM=BN=128，BK=32，256 threads，wave64。
launch grid=(32,32,1)，block=(256,1,1)，kernel=Attn_p2。

日志结论（按 HSACO median，不混用 min / PMC kernel time）
1. causal 全局最快：retained32_group_prefetch/run-20261023-163255-p16epokv/grid-000001/kernel.ll
   median=1.168558955192566 ms，min=1.151519 ms；VGPR=244、scratch=0、LDS=16384。
   pmc_test.log 原文明确归在 unsafe softmax 下；不作为新稳定版的算法输入。
   完整历史参数保存在 historical_best.config.json。
2. 稳定 softmax causal 最快：causal_v_double_buffer_reverse/group4.ll
   median=1.1768704652786255 ms，min=1.160471 ms。
   历史参数及产物哈希保存在 historical_safe_best.config.json。
3. 旧 causal_best_mytest/group4.ll 为 forward，median=2.337099075317383 ms。
4. noncausal_cvt_pack_pass/pass_pk_rtz.o3.ll：median=1.5051189661026 ms。
   该文件是历史 LLVM 经过 Pass/O3 的固化结果，新 causal 产物不声称与它指令级一致。
摘录和 UTF-16 原日志行号见 pmc_evidence.txt；未修改原日志和历史产物。

新生成参数
  pipeline=0, reorder=1, pack=1, layout=3.3, carrier=34816, math=approx
  threads=256, waves=4,1, lifetimes=1, masks=0, prefetch=1, fuse=0, reuse=0
  copy=async, alignment=16, dma-pack=useKPack, dma-export=asm128
  qk-tile=32, vo-tile=32, frag-prefetch=0, frag-prefetch-words=4
  retain-fragment-words=32, swizzle-k=8:6, swizzle-q/v=off
  workgroup-group-size=4, workgroup-reverse=1, query-axis=x, head-axis=y
  pv-interleave=4, v-double-buffer=1, k-dma=64-swizzle, k-dma-arg=1
  k-lds-prefetch=8, softmax-fragments=pv, k-tail-prefetch=4
  mask-add-bypass=approx, defer-lane-sum=approx, local-reduce=approx
  local-reduce-chains=4, local-reduce-min-inputs=16, local-reduce-ops=all
  simplify-loop-control=1, coalesce-waits=0
  simplify-bounded-affine=1, hoist-lds-bases=1, hoist-lane-shuffles=1
  elide-backedge-barrier=1, cross-bn-k0=1（请求值；实际是否命中见下）
  lazy-softmax-base=off, softmax-cvt-pack=rtz/off
全部命令见 *.command.sh 和 manifest.json。

与 non-causal 的必要差异及实际命中
- 恢复 bx 反序：group=4、reverse=1；实际映射 query=31-forward_query。
  当前输入 query/head 为 x/y，历史 unsafe 输入是 y/x，不能照搬其轴名。
- 使用完整 causal online-softmax MLIR，保留逐元素 -inf mask 和 query-dependent keyEnd。
  masks=0 只是关闭 uniform-mask 分支优化，不是取消 causal mask；便于调度 Pass 匹配。
- LazySoftmaxBase 当前实现明确未建模 causal -inf mask，因此关闭。
- cross-bn-k0 虽请求开启，但 reverse 的 signed loop condition 不满足当前 Pass 的
  unsigned 匹配条件，日志明确跳过；本次不修改编译器来放宽证明。
  依赖 cross-bn-k0 的 backedge barrier elision 也不能据此宣称生效。
- V 双缓冲、K 64-swizzle DMA、K LDS 预取、softmax/PV 交错、K tail 预取均命中。
- PKRTZ 只改变 P 的 FP32->FP16 舍入为向零，分母仍为 FP32，输出转换保持原舍入。
  该模式与用户指定的 non-causal 最佳版一致；需要随性能一起实机检查误差。

生成与验证
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  python3 generated_ir/causal_cvt_pack_mytest/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/causal_cvt_pack_mytest/verify.py
每个候选完整走 MLIR -> 当前 MyTest -> LLVM -> O3；不做 LLVM 文本补丁。
生成脚本检查 LLVM/O3 verifier、32 条 PKRTZ 与 causal mask，并记录 SHA256/Pass 日志。
数值验证使用已有 LLVM CPU 解释器，覆盖首/中/尾 query tile、最大值变化、
对角线未来位置占优、正负大分数，并检查 causal K/V 边界、DMA 完成与 LDS 所有权。
验证细节见 validation.json；CPU 模型不能代替 GPU 时序、寄存器分配和性能实测。

GPU 复测
  bash generated_ir/causal_cvt_pack_mytest/run_gpu.sh
默认依次测历史 stable best、off、pk_rtz、历史 stable best。
历史 best 用日志对应的 group4.ll；新候选用 .o3.ll，之后均执行 GenHsaco.sh O3。
只测新候选：
  bash generated_ir/causal_cvt_pack_mytest/run_gpu.sh group4_reverse_pk_rtz
可设置 LLVM_DIR、ROCM_BC、ATTN_WARMUP、ATTN_REPEAT。
每次新建 runs/<variant>-*/，保存输入哈希、HSACO、汇编、编译日志和正确性/性能 JSON。
本机未执行 gfx936 编译或 GPU benchmark，新产物性能待用户实测。

本次验证结果：18 个 case/variant 组合全部通过；最大绝对参考误差 0.00107169441；PKRTZ 相对 off 的最大绝对输出差 0.0009765625。
