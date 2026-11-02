最后一段 QK 的 LDS 预取：通用 C++ Pass 产物

推荐测试 k_tail_prefetch4.ll；另有2/8距离和control原始基线。
来源 ../noncausal_softmax_fragments/softmax_pv.ll，SHA见manifest.json。
对应实测 Python 实验版距离4 median=1.691346ms，原基线1.707992ms。
本目录候选由 LegacyLLVMExportTest --k-tail-prefetch=N 生成，不使用Python重排。
三个候选均核对与已测实验版的完整指令、操作数和顺序一致（忽略metadata和属性编号）。
O3后8组配对和预取距离保持；每份IR配同名launch.json。

GPU测试（仓库根目录，沿用LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_qk_tail_prefetch_pass/run_gpu.sh control k_tail_prefetch4 control
测试全部距离：
  bash generated_ir/noncausal_qk_tail_prefetch_pass/run_gpu.sh
默认warmup=20、repeat=100；ATTN_WARMUP / ATTN_REPEAT可覆盖。
每次输出保存在独立runs/目录，不覆盖原有汇编。
shape=(1,32,4096,128)，grid=(32,32,1)，block=(256,1,1)，non-causal，K=bhsd。

通用入口
  3rd/deepgengraph/build/test/LegacyLLVMExportTest --k-tail-prefetch=4 input.ll > output.raw.ll
  3rd/deepgengraph/build/test/LegacyLLVMExportTest --llvm22-text output.raw.ll > output.ll
MyTest命令追加 --k-tail-prefetch=4，默认0；与 --k-lds-prefetch=8 独立。
TuneAttention.py支持 --set k-tail-prefetch=4 或 --sweep 'k-tail-prefetch=0;2;4;8'。
0不撤销已经施加的调度；比较距离应从原始输入分别生成。

实现
  3rd/deepgengraph/lib/Conversion/ConvertToLLVM/PrefetchPairedLDS.cpp
  3rd/deepgengraph/lib/Conversion/ConvertToLLVM/PairedLDS.md
PrefetchPairedLDSTailPass从MMAC B使用关系、公共基址/常量偏移识别未保护的读取。
允许跨越纯寄存器max/min/sum和向量拆装，但保留它们及MMAC的原顺序。
不依赖SSA名、内核名、固定tile、128字节间隔或最后一段的序号。
现有配对/调度保护、访存副作用、DMA、硬件同步、wave操作、未知调用、
constrained FP均是边界；strictfp/optnone跳过。无匹配时报告0，不修改函数。
复用已有配对/SSA验证算法，独立属性保证幂等，冲突距离报错。

复现
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 3
  python3 3rd/deepgengraph/test/check_paired_lds_tail.py
  python3 generated_ir/noncausal_qk_tail_prefetch_pass/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_qk_tail_prefetch_pass/verify.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_qk_tail_prefetch_pass/verify_mytest.py
manifest.json记录工具/IR哈希、实际命令、匹配统计及与实测IR的一致性。
validation.json/log记录27项原始/O3/极端输入CPU模型检查。
mytest_smoke保存从MLIR完整编译的开关对照及CPU执行结果，不能当作原基线逐字节复制。
CPU模型检查数值、延迟DMA、LDS地址/值/epoch、全局访问和V调度；
不模拟独立wave时间或目标机器码，不代替gfx936 GPU正确性和性能验证。
