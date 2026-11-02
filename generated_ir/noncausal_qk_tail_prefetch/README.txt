Non-causal 最后一个 QK 子块（K3）LDS 预取实验

基线：../noncausal_softmax_fragments/softmax_pv.ll
SHA256：5183329f5984b2d841ccb1a1b1561cde44708c17123f1a23a192ecbf20c4dc4a
用户记录：HSACO median=1.707992 ms，SDPA median=1.588229 ms。
control.ll 是上述文件的逐字节副本。

测试 IR
  k_tail_prefetch2.ll  最后一段成对 K 读取提前至首次消费前至多 2 条 MMAC
  k_tail_prefetch4.ll  提前至多 4 条 MMAC；建议先测这个中间档
  k_tail_prefetch8.ll  提前至多 8 条 MMAC
  control.ll          当前最佳基线
每个 IR 均有同名 .launch.json。测试使用 .ll，不使用 .o3.ll。
shape=(1,32,4096,128)，grid=(32,32,1)，block=(256,1,1)，non-causal，K=bhsd。

仓库根目录执行（沿用目标机器上的 LLVM_DIR / ROCM_BC）
  bash generated_ir/noncausal_qk_tail_prefetch/run_gpu.sh control k_tail_prefetch4 control
全部距离交替对照：
  bash generated_ir/noncausal_qk_tail_prefetch/run_gpu.sh
默认 warmup=20、repeat=100，可设置 ATTN_WARMUP / ATTN_REPEAT。
每次运行保存独立 runs/ 目录，包含 HSACO、汇编、构建日志、正确性和 result.json。

改动边界
本轮是针对固定 SHA 基线的 Python IR 调度实验，尚未集成为通用编译器 Pass。
只移动最后一个 BK=32 QK 子块的 16 次 <4 x half> LDS load，组成 8 组
同基址、相差128字节的 read2 候选；不改变任何指针、地址计算、载入宽度。
读取仍在 K3-ready barrier 和 V0 DMA 之后，使用高8 KiB；V0 DMA 使用低8 KiB。
新增 sched.barrier(0) 用于固定预取批次与 MMAC 的先后关系，不是硬件 barrier。
新增 frisk.k_tail_pair / frisk.k_tail_prefetch 元数据供审计，原有属性保持不变。
原 frisk.k_lds_prefetch=8 描述继承的预取设置；本轮尾部距离以文件名/manifest为准。
前三段 QK、Q retain32、PV四链交错、softmax/PV调度、V双缓冲、全部硬件同步、
DMA顺序、max和sum表达式及MMAC顺序保持不变，LDS仍为16384字节。
第一组没有更早的独立 MMAC 可用，仍需等待；本实验不宣称消除全部 lgkmcnt(0)。

复现与验证
  python3 generated_ir/noncausal_qk_tail_prefetch/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_qk_tail_prefetch/verify.py
generate.py 检查除16次load移动和新增调度提示外原指令多重集/顺序完全一致；
运行 llvm-as 和 opt default<O3>,verify，核对O3后8组配对及预取距离保持。
IR_CHECK_LLVM_DIR 可覆盖本地验证 LLVM 路径，不影响 GPU 测试工具链。
verify.py 使用现有LLVM CPU执行模型，比较随机首末query、正溢出、负下溢、
running-max变化及未来key主导输入，结果与control逐bit一致并对FP64参考检查。
随机首块还覆盖三个O3版本；检查延迟DMA、LDS地址/值/同步epoch多重集、
全局访问轨迹、V调度、Q读取次数和硬件同步动态计数。
结果以 manifest.json、validation.json 和 validation.log 为准。
CPU模型不模拟独立wave推进或机器指令调度，目标gfx936汇编和GPU性能须实测。

GPU验收
优先看正确性、kernel绝对median、VGPR和scratch；比较尾部是否形成成批ds_read2
及更长的load-to-use间距，并确认原K布局/配对未退化。不要直接删硬件wait/barrier。
