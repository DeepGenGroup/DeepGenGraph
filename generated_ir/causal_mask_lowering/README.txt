Causal maskOp lowering 对照产物

首选复测：mask_finite_pk_rtz.o3.ll
精确语义对照：mask_i32_pk_rtz.o3.ll
原版对照：mask_off_pk_rtz.o3.ll
每个 IR 都配有同名 .launch.json，causal=true，shape=(1,32,4096,128)。
保留 group4 reverse、256 threads、BM=BN=128、BK=32、PKRTZ 及上一版其他参数。

实现
--causal-mask=off|i32|finite，默认 off；MyTest 和 RunFriskSHell.sh 支持。
底层 convert-friskbase-to-thread / convert-friskbase-to-fragment 同名选项。

 i32：匹配无副作用的 select(row + 非负 diagonal <= col, masked, clear)，
      递归证明坐标范围（gpu block/thread 范围、affine/scf 循环及整数表达式）。
      仅当 row、col、row+diagonal 都在非负 signed-i32 范围内，改为 i32 比较。
      保留原 mask 值和 score+mask 运算；不添加有限值或 signed-zero 假设。
 finite：在上述证明成立且 mask 为 -inf/+0、只有一个紧邻的 add 消费者时，
      使用 select(masked, -inf, score)，取消独立 mask 加法。
      显式约定 pre-mask score 为有限值，并允许 signed-zero 差异；
      不承诺 NaN 或 +inf 输入的原始传播行为。这个约定不影响 i32/off 模式。

两条 lowering 路径共用同一范围证明和 emitter。未知坐标、溢出、负起点、
不匹配的 region 都保留通用实现；多用户、隔着 barrier 的 mask 不做直接选择融合。
finite 融合允许 add 结果携带不同的 memory space 标注，但要求形状、元素类型、
memref layout 一致，使用消费者的 LowerInfo，并在替换时保留消费者结果类型。
K-tail 配对 LDS Pass 同时接受纯寄存器 icmp/select，避免新 mask 选择截断预取区域。
比较、选择、MMAC 和 FP 归约自身顺序不变；DMA、barrier、未知调用仍是调度边界。
不根据 SSA 名、固定 shape、固定 lane 映射识别 causal。
本次没有改变 reverse 的地址范围传播、cross-bn-k0 匹配规则或 lazy softmax。

构建及生成
  cmake --build 3rd/deepgengraph/build --target MyTest ThreadTilingTest LegacyLLVMExportTest -j 8
  python3 3rd/deepgengraph/test/check_causal_mask_lowering.py
  python3 3rd/deepgengraph/test/check_paired_lds_tail.py
  python3 generated_ir/causal_mask_lowering/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/causal_mask_lowering/verify.py

generate.py 完整执行 MLIR -> MyTest -> LLVM -> O3，没有导出后的文本补丁。
要求 off.o3.ll 与用户已测的 group4_reverse_pk_rtz.o3.ll 除 ModuleID 外完全一致。
检查新增模式消除 compare.local.base identity；finite 还消除 16 个 vec4 mask add。
同时检查 K-tail 命中：off 为 7 对 LDS 读、移动 5 对、4 个 MMAC 锚点；
i32/finite 为 8 对、移动 7 对、5 个锚点，原来未配对的读也成功配对。
manifest.json 保存输入/编译器/产物 SHA256、完整命令和 Pass 命中信息。

测试
  bash generated_ir/causal_mask_lowering/run_gpu.sh
默认按 off、i32、finite、off 顺序编译并测试。可单独指定一个或多个候选：
  bash generated_ir/causal_mask_lowering/run_gpu.sh mask_finite_pk_rtz
可设置 LLVM_DIR、ROCM_BC、ATTN_WARMUP、ATTN_REPEAT。
runs/ 下保存每轮输入哈希、汇编、编译日志和正确性/耗时 JSON。

验证范围
LLVM/O3 verifier；typed/direct 路径与范围/融合回退条件；实际 LLVM CPU 解释执行；
首、中、尾 query tile、最大值变化、未来列占优、正负大分数；causal K/V 边界，
延迟 DMA、LDS 所有权，原版与新版本的访存/调度比较。
LDS 比较涵盖每次读的地址、值、同步 epoch 和次数，允许同一 epoch 内独立读重排；
另行记录有序读值是否一致。global 访存顺序和 V DMA/读调度必须一致。
数值结果见 validation.json。CPU 模型不模拟独立 wave 时序或目标寄存器分配。
本轮验证：27 次实际 LLVM 执行全部通过，i32/finite 输出均与 off 逐位一致；
相对独立参考实现的最大绝对误差为 0.001071695。
mask/direct/typed 与回退测试、原 mask-add/typed 测试、5 个主预取测试、
8 个尾部预取测试均通过；具体日志保存在本目录的 *_regression.log。
本机没有执行 gfx936 编译或 GPU benchmark，不能据 IR 静态简化声称 spill 已消除。
复测需要检查 VGPR、scratch、spill/reload、median、L2 命中率和输出误差。
