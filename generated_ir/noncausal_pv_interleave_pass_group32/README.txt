PV interleave：编译器Pass产物（group32，reverse=0）

输入：../noncausal_v_handoff_sweep_group32/dma_after4.ll。
生成器调用已编译的LegacyLLVMExportTest --pv-interleave=N；不再用Python
重排MMAC。Python只负责调用、LLVM22文本兼容导出和结果检查。

优先测试：pv_interleave4.ll（四链交错）
其他对照：pv_interleave2.ll（双链）、pv_serial_guarded.ll（同调度提示串行）、
          baseline_after4.ll（原始after4逐字节复制）。
均配有同名launch.json；shape=(1,32,4096,128)，grid=(32,32,1)，block=(256,1,1)。
BM=BN=128、BK=32、16 KiB LDS、Q retain32、K预取/合并保护保持不变。
8次V读取仍提前发出，前三阶段reader barrier/DMA仍在第4条PV MMAC后。

--pv-interleave=0|1|2|4
  0：关闭，MyTest默认值。
  1：保持串行链顺序，但加入与2/4相同的调度提示。
  2：两条累加链交错（A0 B0 A1 B1 ...，随后C/D）。
  4：四条累加链交错（A0 B0 C0 D0 A1 B1 C1 D1 ...）。
这里的2/4是独立输出累加链数，不是tile、线程、wave或group-size。
每条链内部归约顺序不变；调度提示不是新增硬件barrier。

生成及验证（仓库根目录）：
  python3 generated_ir/noncausal_pv_interleave_pass_group32/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_pv_interleave_pass_group32/verify.py
  python3 3rd/deepgengraph/test/check_pv_interleave.py

GPU测试（沿用LLVM_DIR、ROCM_BC）：
  bash generated_ir/noncausal_pv_interleave_pass_group32/run_gpu.sh pv_interleave4
完整对照：
  bash generated_ir/noncausal_pv_interleave_pass_group32/run_gpu.sh

默认warmup=20、repeat=100，可用ATTN_WARMUP/ATTN_REPEAT修改。
runs/各次独立保存HSACO、汇编、数值检查和性能result.json。
commands.sh记录实际Pass导出命令；manifest.json记录输入/工具/输出哈希及
O3校验。*.pass.log应显示4个匹配region；每个region对应一个V/PV阶段。

输出的全部指令和顺序（忽略metadata和属性组编号、统一浮点字面量后）
已核对与上一轮对应Python实验一致。validate日志/JSON包含CPU执行模型的
数值、访存轨迹与DMA位置验证，不能替代GPU数值检查及实际汇编验收。

MyTest现已支持直接追加 --pv-interleave=4，但该开关只负责交错，不会额外
生成K整组预取或改变V handoff位置。mytest_smoke/只验证完整编译入口；
本轮性能测试请使用本目录顶层的pv_interleave4.ll。

实现及完整规则：
  3rd/deepgengraph/lib/Conversion/ConvertToLLVM/PVInterleave.md
