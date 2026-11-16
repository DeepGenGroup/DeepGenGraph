Non-causal 滞后基准（lazy softmax base）候选

状态：lazy_t8 已实测 1.571839 ms（对 SDPA 0.9968），并已固化成编译器 Pass。
  Pass：lib/Conversion/ConvertToLLVM/LazySoftmaxBase.cpp，开关 --lazy-softmax-base=<T>
        （off 关闭，T 取 (0,15]，默认 8）。必须是 LLVM 阶段的最后一步，排在
        prefetchNextK 和 elideBackedgeBarrier 之后，原因见该文件头部说明。
  本目录的 .ll 仍是最早用 generate.py 文本改写得到的版本，作为 Pass 的对照基准保留。

思路
  O/L 的比值与缩放无关，不必每轮按新 M 重缩放。
  状态 M 改为 N = -B（B 是当前基准，初值 N=0），N 直接作 QK 的 C 初值：
    S' = QK                  // S' = S - B，省掉 v_sub
    M' = 跨 lane 行 max        // 保留，每轮都做
    if (首轮 || 本 wave 任一行 M' > T)   // lazy.slow，按 wave 判定
      d = 首轮 ? M' : max(M', 0)
      S' -= d; O, L *= 2^(-d)（首轮不乘）; N -= d
    P = 2^S'
  未进慢路径时 S' <= T，所以 W、P <= 2^T，T <= 15 不会溢出 FP16。
  基准不超过真实 max，P 只会比现状更大，下溢不会变差。
  每行只在 wave 内归约，同行各 lane 的 d 相同，延迟 lane sum 的前提仍成立。
  两条路径的 barrier 数相同，所以不需要跨 wave 同步。

变体
  control      k0_sum_anchor.ll 的逐字节副本（实测 1.6633 ms）
  lazy_t8      T=8，主候选
  lazy_t4      T=4，触发更频繁，P 上界更小
  lazy_always  每轮都进慢路径；算术递推等同 running max（不逐位相同），
               用来单独量出新控制流的开销

相对 control 的变化
  删除：16 个 fsub <4 x float>（64 v_sub）、16 个 fmul <4 x float>（32 v_pk_mul）、
        alpha 的 2 sub / 2 exp2 / L 的 2 fmul、与旧 M 的 maxnum。
  新增：2 个 seed 向量、1 个 maxnum、llvm.amdgcn.fcmp.i64.f32 + 标量分支、
        只在慢路径执行的缩放代码。
  不变：MMAC 顺序及 A/B 操作数、DMA、wait、barrier、K0 分支、分母加法锚点、
        exp2 和 fptrunc 的顺序（generate.py 对此做断言）。
  仅适用 non-causal；causal 的 -inf mask 没有考虑。

生成和 CPU 验证（需要 /data2/xsl/install/bin 的 llvm-as、opt）
  python3 generated_ir/noncausal_lazy_base/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_lazy_base/verify.py
  generate.py 校验 k0_sum_anchor.ll 的 SHA，跑 llvm-as 和 O3 verify，并审计指令计数。
  verify.py 执行实际 IR：与参考值比较（atol=rtol=0.01）；与 control 比较全局访问、
  LDS 值和 epoch、V 调度、K0 发射和 wait/barrier/MMAC/DMA 计数；检查 W 上界；
  并跑两个负向对照（去掉首轮强制、阈值永不触发），它们必须失败。
  输出不要求与 control 逐位相同，validation.json 记录了误差和每 wave 进慢路径的次数。

  模型的局限：解释器让所有 wave 同步执行，所以模型把纯 ALU 的慢路径对所有
  lane 求值，再按 wave 选择 join 处的 phi（这对纯 ALU 块语义等价）。
  不模拟 gfx936 时序、寄存器分配和 MMAC 内部舍入。

上机（目标 gfx936 / DTK 环境）
  bash generated_ir/noncausal_lazy_base/run_gpu.sh
  默认顺序：control lazy_t8 control lazy_t4 control lazy_always control。
  也可以指定：bash generated_ir/noncausal_lazy_base/run_gpu.sh control lazy_t8 control
  结果在 runs/ 下，含 kernel.hsaco.amdgcn、resources.json、gpu.log、result.json。
  有 scratch 或 spill 的变体会跳过测速。

Pass 与手改 IR 的一致性
  python3 generated_ir/noncausal_lazy_base/check_pass.py
  用编译好的 Pass 重新改写 control.ll，得到 pass_t8 / pass_t4 / pass_off，
  检查：off 不改动 IR；Pass 输出与实测的 lazy_t8 结构计数一致（快慢路径分别统计）；
  在同一批数值用例上与 control 的访存轨迹、LDS 值、V 调度一致，且慢路径触发次数与
  lazy_t8 完全相同。结果写入 pass_validation.json。

验收要看
  1. launchKernel.py 的正确性比较是否通过。
  2. median 耗时：lazy_t8 对 control；lazy_always 对 control 是新控制流的开销。
  3. VGPR、scratch 是否变差（check_resources.py）。
  4. 汇编：循环里的 v_sub_f32 和 v_pk_mul_f32 是否只留在慢路径；
     分支是否是 s_cbranch，而不是 exec 掩码（llvm.amdgcn.fcmp 在 DTK 上未验证）；
     phi 是否在快路径上引入了额外的 v_mov。
  5. control 的汇编里，空 asm 前的 v_mov_b64 是否约 5 条（确认拷贝来源的推断）。
