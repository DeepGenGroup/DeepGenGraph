Non-causal group-size 扫描：4 / 8 / 16 / 32，reverse=0

基线：../noncausal_grouped_lds_prefetch/group_prefetch2_protected.ll。
group4.ll 与基线逐字节一致；group8/16/32.ll 只修改入口的三个分组常量。
BM=BN=128、BK=32、threads=256、waves=4,1、retain=32、K swizzle=8:6
以及21组预取、全部32组读取保护均保留。形状固定BHSD=(1,32,4096,128)。

文件
  group4.ll / group8.ll / group16.ll / group32.ll：交给GenHsaco的输入。
  *.launch.json：同一grid=(32,32,1)、block=(256,1,1)、non-causal配置。
  *.o3.ll：本地LLVM O3校验产物；GPU测试脚本使用上述原始.ll。
  manifest.json：输入输出哈希、全网格映射和O3调度保护检查。
  validation.json/log：实际LLVM CPU执行模型的数值检查，不替代目标GPU验证。

group-size的含义
  G个相邻query tile（各BM行）作为同一head的一组；不会融合为一个block。
  t=physical_x+32*physical_y
  query=(t//(32*G))*G+t%G；head=(t//G)%32。
  reverse=0，不执行query=31-query。
  生成器对每种G穷举1024个物理block，检查映射双射和Q/O、K/V基址。

复现（仓库根目录）
  python3 generated_ir/noncausal_group_sweep_forward/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_group_sweep_forward/verify.py

GPU测试（沿用已有LLVM_DIR、ROCM_BC环境）
  bash generated_ir/noncausal_group_sweep_forward/run_gpu.sh
默认 group4 → group8 → group4 → group16 → group4 → group32 → group4。
每种只测一次：
  bash generated_ir/noncausal_group_sweep_forward/run_gpu.sh group4 group8 group16 group32
单独测试：
  bash generated_ir/noncausal_group_sweep_forward/run_gpu.sh group16
ATTN_WARMUP默认20，ATTN_REPEAT默认100；结果各自保存在新建的runs子目录。
使用相同launchKernel数值检查和交替SDPA计时，优先比较kernel绝对median。
生成器不依赖当前根目录汇编，旧实验文件保持不变。
