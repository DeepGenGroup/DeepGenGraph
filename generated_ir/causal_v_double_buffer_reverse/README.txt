Causal + reverse bx + group4/8/16/32 + V double buffer

底稿是用户指定的当前最佳文件：
  ../noncausal_v_double_buffer_group32/v_double_buffer.ll
  SHA256: 3b2bde804277d68be2926e1ad35a292164d4688a349bedcefa42ab605322888a

测试输入：group4.ll、group8.ll、group16.ll、group32.ll。
每个输入配有同名 .launch.json；.o3.ll 是本地 LLVM O3 验证产物。
建议 GPU 测试使用原始 .ll，让 GenHsaco.sh 使用目标 DTK 做 O3 和代码生成。

共同配置：
  BHSD=(1,32,4096,128)，BM=128，BN=128，BK=32。
  grid=(32,32,1)，block=(256,1,1)，waves=4x1。
  layout=3.3，retain-fragment-words=32，pv-interleave=4，v-double-buffer=1。
  swizzle-k=8:6，Q/V swizzle=off，asm128 异步 DMA，alignment=16。
  静态 LDS=16384 B，V 两个 8192 B 槽复用 Q/K LDS，动态 shared_mem=0。
  其他调优参数逐份记录在 manifest.json 的 parameters 中。

四个版本只在 workgroup-group-size 上不同；均为 reverse=1、query-axis=x、head-axis=y。
物理 (x,y) -> 逻辑 (query,head)：
  t = x + 32*y
  forward = floor(t/(32*G))*G + t%G
  query = 31-forward
  head = floor(t/G)%32
因此 reverse 的是分组后的逻辑 bx，不是先简单对物理 x 做 31-x。

因果语义：key tile 只遍历 0..query（含 query）；每个 score 在 max 和 exp2 之前
执行 key_position <= query_position 的 mask，未来位置设为 -inf。
比较阈值按实际 layout 3.3 的 QK 坐标生成，循环内用 select，不引入分歧 barrier。
这是绑定源文件 SHA 和上述 shape/layout 的实验 IR 生成器，不是通用因果转换 Pass。
直接变换精确底稿，以保留该 case 的 K 分组预取及调度保护，不用重新导出的近似基线。

保留的静态调度：
  每次 key tile 128 条 QK + 128 条 PV MMAC，所有 MMAC 操作数及顺序保留。
  V DMA 的 MMAC 序号：96、128、160、192（每阶段两次 DMA）。
  V fragment 读取序号：128、160、192、224（每阶段八次读取）。
  V 槽顺序 0、1、0、1；所有 K 读取对及预取锚点在 O3 后仍受保护。

GPU 批量测试（仓库根目录执行）：
  bash generated_ir/causal_v_double_buffer_reverse/run_gpu.sh
也可指定顺序、重复对照：
  bash generated_ir/causal_v_double_buffer_reverse/run_gpu.sh group32 group4 group8 group16 group32
默认 warmup=20，repeat=100；可用 ATTN_WARMUP / ATTN_REPEAT 覆盖。
沿用 GenHsaco.sh 的 LLVM_DIR、ROCM_BC、GPU_ARCH 配置，默认目标 gfx936。
脚本显式传 --causal，并保存每次 input.sha256、build.log、gpu.log、result.json。
结果目录：runs/<variant>-<unique>/。

重新生成与 CPU 验证：
  python3 generated_ir/causal_v_double_buffer_reverse/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/causal_v_double_buffer_reverse/verify.py
生成器默认使用 /data2/xsl/install/bin；可用 IR_VERIFY_LLVM_DIR 覆盖。
验证详情见 validation.json / validation.log；覆盖全部物理 block 映射、16384 个 score
的实际坐标、causal 数值、未来 K/V 访问边界、Q 只加载一次、延迟 DMA 和 LDS barrier 所有权。
本地未运行 GPU，CPU 模型不模拟独立 wave 的实际机器调度，性能由目标机器实测。
