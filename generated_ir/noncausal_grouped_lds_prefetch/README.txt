Non-causal QK：按实际 ds_read2 配对的整组 LDS 预取实验

测试文件
  group_prefetch2_protected.ll 修复全部原有读取组合并保护的新候选
  group_prefetch2.ll          上一版候选，保留供对比（实测 speedup=0.81965）
  control.ll                 使用相同 O3 预处理的关闭预取对照
  *.launch.json              group4/forward，256 threads，non-causal 启动配置

基线为 noncausal_frag_prefetch_forward/control/kernel.ll。
BM=BN=128、BK=32、waves=4,1、retain=32、K swizzle=8:6 均不变。
使用当前 ktest.hsaco.amdgcn 主循环中的 32 个实际 ds_read2 组识别配对。
先做 O3/CSE 消除重复 load，再将每组两次 <4 x half> load 及必要的纯地址
计算一起提前；不是把未合并的所有相邻 load 任意划成两组。

原有 21/32 组预取保持不变：18 组提前到首次使用前两个 MMAC，3 组提前一个。
新增全组保护步骤：插入 sched.barrier 之前，检查全部32组（包括未预取的组）。
若读取组跨过即将插入的屏障，则将后半次读取及纯地址依赖提前到前半次读取旁，
只允许在同一 LDS 数据有效阶段内移动；遇到不安全边界则终止生成。
本次实际修复编号8、16两组（从0编号），只移动 %grp.n244、%grp.n319
及前者的一条 GEP。原有21组预取锚点、MMAC顺序、36个调度提示均不变。
在跨越的独立 MMAC 前后设置 sched.barrier，全部32组内部都没有调度边界。
这些是编译调度约束，不是新增 workgroup barrier，也不手写 lgkmcnt。
最多提前两个 MMAC 是调度距离，不是寄存器预算或最终 VGPR 增量。

约束与验证
  仅移动原来的读取和地址定义，不复制运算，不改变 MMAC 顺序或浮点结合顺序。
  不跨 DMA、vmcnt wait、硬件 barrier/fence、store、branch、opaque call。
  原硬件同步全部保留，group4 reverse=0，不增加 LDS 容量。
  llvm-as 和 opt default<O3>,verify 均通过；再次 O3 后仍有完整读取组位于
  指定 MMAC 前，并对全部32组断言组内没有调度边界。
  普通 LDS load 数仍为 80（入口16+循环64），指令集合与上一版完全相同，仅重排。
  generate.py 是绑定输入与汇编 SHA 的实验生成器，未修改通用编译器 Pass。
  manifest_protected.json 记录新候选哈希和两组修复详情；旧 manifest.json 保留。
  validation_protected.json/log 记录 control、旧候选、新候选的 CPU LLVM 执行
  模型对照；旧 validation.json/log 保留，目标 GPU 正确性仍需实测。
  放大随机 Q/K（各乘8）的额外测试中，候选与 control 逐 bit 一致，但两者
  都存在相对 FP64 参考超过 atol=rtol=0.01 的元素；这是本次确认的基线数值
  限制，调度变换没有修复或加重它。普通随机输入仍需通过该参考阈值。

重新生成和本地数值检查（仓库根目录）
  python3 generated_ir/noncausal_grouped_lds_prefetch/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_grouped_lds_prefetch/verify.py

GPU 测试（沿用 LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_grouped_lds_prefetch/run_gpu.sh
默认 group_prefetch2 → group_prefetch2_protected → group_prefetch2，
warmup=20、repeat=100；也可以显式传入 control 进行关闭预取对照。
单独测试候选：
  bash generated_ir/noncausal_grouped_lds_prefetch/run_gpu.sh group_prefetch2_protected
每次结果独立保存在 runs/，包含汇编、数值检查、计时和 result.json。

目标汇编验收
  先比较同目录 control 与 candidate 的主循环是否确实发生调度变化。
  检查 K 读取是否仍合并为每轮约32条 ds_read2*，是否多出 ds_read_b64。
  检查实际读取/首次消费距离、lgkmcnt 等待、VGPR 和 scratch。
  旧候选 PMC 的 LDS 操作数为9,486,336，比此前9,224,192多262,144，
  恰好对应每wave每轮多2条；检查新版本是否恢复原有合并、消除这部分增量。
  IR 中保留配对机会不等于保证 gfx936 后端生成相同 DS 组合；新候选尚未GPU实测。
