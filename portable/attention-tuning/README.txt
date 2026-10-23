DeepGenGraph attention 一键调优包
================================

1. 快速开始

在兼容的 x86-64 Linux / gfx936 DCU 机器上解压：

  tar -xzf attention-tuning-*.tar.gz
  cd attention-tuning
  ./run_tuning.sh

默认执行：
  BM = 64, 128；BN = 32, 64；BK = 32；
  每份输入测试 baseline / nohint-rolled / pipeline-loop-only /
  group8-reverse / group8-reverse-k128-swizzle，共 20 个候选。
  流程为生成 MLIR -> MyTest 导出 LLVM IR -> GenHsaco.sh 生成 HSACO
  -> launchKernel.py 正确性检查和 GPU 计时。
  默认预热 10 次、计时 50 次、种子 0、opt O3、GPU device 0。
  每候选/每编译阶段超时 300 秒，串行执行。

先跑一个快速测试：

  ./run_tuning.sh --smoke

只检查环境、不生成任何输入或结果：

  ./run_tuning.sh --check

只生成 LLVM IR（不需要 GPU、DTK 或 PyTorch）：

  ./run_tuning.sh --mode ir --smoke

只生成 HSACO、不启动 GPU 测试：

  ./run_tuning.sh --mode hsaco

只打印将要执行的命令：

  ./run_tuning.sh --dry-run

包路径可以包含空格，也可以从其他工作目录通过绝对路径调用 run_tuning.sh。

2. 目标机器需要提供什么

本包包含 MyTest，放在包根目录，与 TuneAttention.py 同目录。
没有捆绑 GPU 驱动、DTK/ROCm 工具链、device bitcode 或 GPU 版 PyTorch。
Python 需要 3.9 或以上；运行 CPU 导出流程只需要标准库。

随包 MyTest 是当前机器编译的 x86-64 Linux ELF：
  glibc >= 2.34；
  libstdc++.so.6 必须提供 GLIBCXX_3.4.30（以及 CXXABI_1.3.13）；
  另需系统 libgcc_s.so.1、libm.so.6、动态加载器。
  不需要额外 libLLVM/libMLIR 动态库。

若出现 GLIBC / GLIBCXX 版本错误，或目标主机不是 x86-64：
  使用同版项目源码在目标环境编译 MyTest；
  将产物 3rd/deepgengraph/build/test/MyTest 放到本包根目录 ./MyTest；
  或在 env.sh 中设置 MYTEST=/目标路径/MyTest。
  不要只替换系统 glibc。此包没有完整 C++ 源码，不能在包内运行 --build。

完整 GPU 流程还需要：
  - 目标机器驱动可访问的 AMD/DCU GPU；
  - 与该 GPU/驱动匹配的 HIP 版 PyTorch 和 NumPy；
  - libamdhip64.so 在动态库搜索路径中；
  - 同一 DTK/ROCm 工具链的 llvm-link、opt、llc、ld.lld；
  - 对应 GPU 的设备库，以及 ABI v4 / wave64 等位码文件。

MyTest 的硬件变换针对当前 DCU 配置，默认 GPU_ARCH=gfx936。
“可在其他机器运行”指兼容该内核和工具链的机器；仅修改 GPU_ARCH
不保证能迁移到任意 AMD/NVIDIA GPU。入口会检查可读出的实际 GPU 架构
与 GPU_ARCH 是否一致。

常见工具路径会自动查找：
  /opt/dtk/llvm/bin
  /opt/dtk/aillvm/bin
  /opt/rocm/llvm/bin
  /workspace/deepgengraph/dtk/aillvm/bin
  以及 PATH 中 llc 所在目录
device bitcode 会查找 /opt/dtk/amdgcn/bitcode、/opt/rocm/amdgcn/bitcode
及工具链附近目录。每个候选路径必须包含所需的整套工具/库。

非标准安装或使用 Python 虚拟环境时：

  cp env.sh.example env.sh
  # 编辑 env.sh 中的 LLVM_DIR、ROCM_BC、GPU_PYTHON、LD_LIBRARY_PATH 等
  ./run_tuning.sh --check
  ./run_tuning.sh

PYTHON 是一键入口的解释器；GPU_PYTHON 是启动器和 GPU 检查的解释器。
默认都使用当前 Python。环境检查不会安装软件或修改系统。
LLVM_AS 可选；设置后还会单独检查 LLVM IR 结构合法性，不能替代 GPU 正确性测试。

3. 配置扫描

新增 swizzle / workgroup 调优（需要同步更新 MyTest 和 TuneAttention.py）：

  ./run_tuning.sh --config tuning.swizzle.json --mode ir
  ./run_tuning.sh --config tuning.swizzle.json

此配置生成 BM=128、BN=128、BK=32 的输入，固定 group=8、reverse=1，
分别扫描 Q/K/V 的 off/on，共 8 个候选。on 为保持 128-bit DMA 的 8:6
字节位 XOR。Q/V 开启是实验配置，收益由实测决定。baseline 仍保持历史配置。

只比较 group/reverse 的对照和 K-only：

  python3 TuneAttention.py --input 3rd/deepgengraph/test/test_input_spidD.mlir \
    --case group8-reverse --case group8-reverse-k128-swizzle --benchmark

--set / --sweep 新增参数：
  swizzle-q / swizzle-k / swizzle-v：off/on 或源位:目标位；on=8:6。
  swizzle-q-arg / swizzle-k-arg / swizzle-v-arg：默认 -1 使用角色注解；
    可指定降低后的函数 ABI 参数编号。K/V 的位置不由脚本猜测。
  workgroup-group-size：0 关闭，正数启用分组，支持非整除尾组。
  workgroup-reverse：0/1 或 false/true；开启要求 group-size>0。
  workgroup-query-axis / workgroup-head-axis：auto/x/y/z，默认 auto。
    脚本按输入 parallel_map 推导；旧输入可为 y/x，当前生成器通常为 x/y。
    无法证明映射时报错，需显式指定两轴；不会按网格大小或 SSA 名猜测。

Swizzle 两位须不同且在 4..30；旧 Q 的 7:3 会拆分 DMA，因此拒绝。
扫描前过滤 pack=0、alignment<16、split32 等已知不兼容组合；
实际 tile 布局、所有 alias 和 DMA 连续性仍由编译器检查。
任一 swizzle 开启会自动执行 LLVM 常量 GEP 规范化，无须额外开关。

config.json / result.json / best_by_shape.json 保存优化参数，实际候选配置
包含解析后的 workgroup 轴；command.sh 可重放。result.json 还包含：
  shared_swizzle_plans：实际 producer 角色、XOR 位、alias 数量；
  workgroup_orders：编译器报告的 [group,query轴,head轴,reverse]；
  buffer_dma128_sites / buffer_dma32_sites：buffer DMA 静态指令点数；
  lds_b64_half_load_sites：IR 中 <4 x half> shared load 指令点数；
  lds_gep_reported_offsets：LLVM Pass 报告的常量偏移规范化数。
这些字段不是最终 ds_read2 数量或 PMC bank 冲突率，性能仍以 GPU 实测为准。


直接编辑 tuning.json，或：

  ./run_tuning.sh --config /path/to/custom.json

字段说明：
  bm / bn / bk：生成器分块参数的笛卡尔积。
    bm、bn 必须为正整数；bk 可以为 0，表示生成时不传 --bk，
    保留 test_input.mlir 模板的非 split-D 结构，不新增 BK 循环。
    例如 "bk": [0, 32, 64] 同时生成非 split-D、BK=32、BK=64 三种输入；
    路径统一汇入 inputs.txt。非 split-D 文件名没有 -BK 后缀。
  cases：TuneAttention 的预设名，可用 glob。
  set：覆盖参数，如 ["math=strict", "threads=128", "waves=2,1"]。
  sweep：扫描轴，如 ["qk-tile=16;32", "vo-tile=16;32"]。
         使用 sweep 时将 cases 设为 []。
  opt_level / device / warmup / repeat / seed / timeout：编译和启动控制。
  output_dir：相对于包根目录的结果目录，也可写绝对路径。

注意：生成器 BK 是原始 IR 的分块值。baseline 的 qk-tile=32、
vo-tile=32 会显式重新分块。若扫描 BK，希望保持 QK/VO 的输入分块，
可以在 set 中指定 dma-pack=noKPack、qk-tile=0、vo-tile=0；
若仍用 useKPack，0 表示让编译器按 kpack 因子重新分块，并不等于原 BK。
也可直接扫描 qk-tile/vo-tile；查看每候选 config.json 和 lowering.log。
bk=0 仅控制输入生成，不会自动修改 cases/set/sweep；非 split-D 输入
建议使用 cases=["legacy-baseline"]。默认 baseline 的 useKPack 和显式
qk-tile/vo-tile 要求 split-D 结构，不宜直接用于 bk=0。
混合 BK 扫描会对每份输入应用同一套候选参数，编译失败仍会记录到汇总。
这里的 bk=0 与 set 中 qk-tile=0/vo-tile=0 的含义不同。

--smoke 会临时改为一份 64x32x32 输入、baseline、预热 1 次、计时 3 次；
不会改写 tuning.json。它适合验证环境，不能替代正式性能采样。

4. 实时 JSON 结果：最大加速比、对应配置及中断保留

4.1 输出位置

入口打印的“本次目录”是本次 session 的绝对路径。默认目录为包根目录的
results/session-*；tuning.combinations.json 使用 results-combinations/session-*。
修改 output_dir 或 --output-dir 后以实际路径为准。每次运行新建目录。
以下路径均相对于本次 session：

  summary.json                      实时进度、状态计数及当前最优配置
  best_by_shape.json                各完整 BHSD 尺寸下加速比最大的正确候选
  settings.json / environment.json  本次扫描参数及检测到的环境
  inputs/ / inputs.txt              生成输入及路径列表
  tuning/run-*/manifest.json        该 run 的实时进度与最优结果
  tuning/run-*/best.json            每份输入 MLIR 各自的最大加速比候选
  tuning/run-*/best_by_shape.json   该 run 按完整 BHSD 分组的最优结果
  tuning/run-*/.../result.json      单候选完整结果：状态、耗时、参数与产物路径
  tuning/run-*/.../config.json      候选优化参数及编译、启动命令
  tuning/run-*/.../input.mlir       候选独立输入副本
  tuning/run-*/.../kernel.ll        LLVM IR
  tuning/run-*/.../kernel.hsaco     GPU 内核
  tuning/run-*/.../launch.json      自动启动参数
  tuning/run-*/.../benchmark.json   数值比较结果和原始计时样本
  tuning/run-*/.../command.sh       单候选重放命令
  tuning/run-*/.../*.log           分阶段诊断日志

结果统一使用标准 JSON，不再生成 manifest.tsv 或 summary.tsv。
直接运行 TuneAttention.py 时，在 run-* 中看 manifest.json 和
best_by_shape.json；一键入口还会同步到 session 根目录。

4.2 选优规则与 SDPA 耗时

SDPA 耗时不是固定常数，也不是整个调优只测一次。
每个候选分别预热，然后交替测量 SDPA 和内核，各采集 repeat 个样本，
分别取中位数，再计算 speedup = sdpa_median_ms / kernel_median_ms。

只有 status=benchmark-passed 的正确候选参与排名。相同完整 BHSD 尺寸下，
选 speedup 最大者；加速比完全相同时，选 kernel_median_ms 更小者。
错误结果即使更快也不会入选，不同完整张量尺寸不混排。

speedup > 1 比本次 SDPA 基线快，< 1 则慢。例如 0.5x 表示约为基线的
两倍耗时，2x 表示约一半耗时。SDPA 实测值会波动，所以“最大加速比”
可能不是“最小内核耗时”。现在按加速比排名，同时保留两个中位数供核对。

best_by_shape.json 的键如 "1x32x4096x128"，表示完整 B,H,S,D，不是 tile。
每条最优记录直接包含：
  speedup / kernel_median_ms / sdpa_median_ms   加速比及两个毫秒中位数
  parameters                                  完整优化配置，无需跳转其他文件
  case                                        预设名或 grid-* 扫描编号
  source_input                                原始切块输入文件路径
  directory / config_json / result_json       候选目录及配置、结果文件路径
  launch                                      shape、grid、block、K 布局等
  hsaco                                       对应 GPU 内核路径
  ranking_metric                              固定为 speedup

文件名 test_input_BxHxSxD-BM-BN-BK.mlir 给出原始切块。qk-tile/vo-tile
及 useKPack 可能重新分块，实际效果结合 parameters 和 lowering.log 判断。
compiler_defaults=true、parameters=null 表示使用 MyTest 自身默认值。

4.3 运行中或中断后查看

将 session_dir 替换为入口打印的实际路径：

  session_dir="/绝对路径/attention-tuning/results-combinations/session-实际编号"
  python3 -m json.tool "$session_dir/best_by_shape.json"
  python3 -m json.tool "$session_dir/summary.json"

每完成一个候选（包括失败项）都会保存检查点并更新 JSON，无须等调优结束。
采用临时文件加原子替换写入，读取时不会看到半份 JSON。尚无正确结果时，
best_by_shape.json 为 {}；ir/hsaco 模式也不会产生性能排名。

summary.json 包含：
  state                  running / completed / interrupted / error
  total_candidates       本次计划候选数
  completed_candidates   已保存完整结果的候选数，包含失败项
  successful_candidates / failed_candidates   成功数 / 失败数
  status_counts          各状态计数，成功到哪个阶段应查看此字段
  last_result            最近完成候选的状态及 result_json 路径
  best_by_shape          同一检查点的最大加速比及配置
  result_files_glob      全部单候选 result.json 的路径模式
  sdpa_baseline          remeasured_per_candidate（每候选重测）
  updated_at             最近检查点的 UTC 时间

Ctrl-C 或 SIGTERM 会停止当前编译/启动子进程，并保存 interrupted 状态。
已完成结果和最优配置保留；当前候选未完成检查点时不计入排名。
kill -9 无法执行退出清理，state 可能仍为 running，但此前的 JSON 检查点
仍可查看。这里是保留结果，不是断点续跑；重新执行会创建新的 session。

只打印当前最大加速比及对应配置：

python3 - "$session_dir" <<'PY'
import json
import sys
from pathlib import Path
best = json.loads((Path(sys.argv[1]) / "best_by_shape.json").read_text())
if not best:
    print("尚无通过正确性检查的 GPU 结果。")
for shape, item in best.items():
    print(f"\nBHSD={shape}, case={item['case']}, 最大加速比={item['speedup']:.6f}x")
    print(f"kernel={item['kernel_median_ms']:.6f} ms, SDPA={item['sdpa_median_ms']:.6f} ms")
    print("输入切块文件:", item["source_input"])
    print("候选输出目录:", item["directory"])
    print("优化配置:", json.dumps(item["parameters"], indent=2, ensure_ascii=False))
    print("启动配置:", json.dumps(item["launch"], ensure_ascii=False))
PY

4.4 所有已完成结果、失败原因及重跑

为避免上万候选时每完成一项就重写巨大的 JSON 数组，summary.json 只保存
进度和最优结果；所有候选的完整明细分别保存在各自 result.json 中。
如需合并“当前所有已完成结果”为一个 JSON 数组，可随时运行：

python3 - "$session_dir" > "$session_dir/completed-results.json" <<'PY'
import json
import sys
from pathlib import Path
session = Path(sys.argv[1])
summary = json.loads((session / "summary.json").read_text())
run = Path(summary["run_directory"])
rows = [json.loads(path.read_text()) for path in run.glob("**/result.json")]
rows.sort(key=lambda row: row["candidate_index"])
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY

completed-results.json 是执行该命令时收集的快照，不会自动刷新。
result.json 中的 seconds 是编译、验证、测试等流程的墙钟时间，不是 GPU
内核时间。benchmark.json 保留 kernel_samples_ms、sdpa_samples_ms、
warmup、repeat、seed 以及设备信息。

常见状态及日志：
  benchmark-passed                 正确性通过，参与加速比排名。
  incorrect                        数值错误；看 benchmark.log 的误差及 K ABI 诊断。
  exported / verified              仅 IR 导出 / llvm-as 验证，不代表 GPU 正确。
  hsaco-built                      仅 HSACO 生成，未完成 GPU 基准测试。
  failed:N / timeout / diagnostic-error / missing-ir / missing-kernel /
  invalid-ir / verify-timeout / invalid-lds-control
                                   查看 lowering.log；N 为进程返回码。
  invalid-launch-config / missing-hsaco /
  hsaco-failed:N / hsaco-timeout / hsaco-spawn-error
                                   查看 hsaco.log。
  benchmark-failed:N / benchmark-timeout / benchmark-spawn-error /
  missing-benchmark-report / invalid-benchmark-report
                                   查看 benchmark.log。

  candidate_dir="/实际候选目录"
  python3 -m json.tool "$candidate_dir/result.json"
  python3 -m json.tool "$candidate_dir/config.json"
  tail -n 80 "$candidate_dir/benchmark.log"

未执行到某阶段时，相应日志/产物可能不存在。最终测试状态看 result.json；
config.json 在编译前写入，其中初始 hardware_tested=false 不代表最终结果。

单候选完整重跑：

  bash "$candidate_dir/command.sh"

重跑前恢复原工具链环境（参考 environment.json / env.sh）。command.sh
会覆盖该候选内核、对应阶段日志，benchmark 模式还会覆盖 benchmark.json，
但不会更新 result.json、manifest.json、summary.json 或最佳排名文件。
需要更新排名时，重新运行一键入口生成新 session。旧结果中的命令包含
绝对路径，移动结果目录后不保证原重放路径仍有效。

数值比较相对 PyTorch SDPA，atol=rtol=1e-2。默认近似数学与模板非稳定
exp2 softmax 的误差由目标机实测判断。有失败候选时入口最终返回非零，
不会丢弃其他已完成的正确结果。

5. 包内容与验证

  MyTest                         当前编译的程序，未裁剪符号
  TuneAttention.py               导出/编译/测试/记录
  generate_attention_ir.py        生成不同分块的输入
  GenHsaco.sh                    LLVM -> HSACO
  launchKernel.py                 GPU 校验和基准测试
  3rd/deepgengraph/test/test_input.mlir
                                 生成器原始模板，保留了它要求的相对路径
  3rd/deepgengraph/test/test_input_spidD.mlir
                                 默认 split-D 输入
  run_tuning.sh / run_tuning.py   一键入口
  tuning.json / env.sh.example    默认扫描和环境配置
  script/test_*.py                CPU 和模拟流程回归测试
  BUILD_INFO.json                二进制来源、哈希、ABI 要求
  SHA256SUMS                     打包文件校验清单

解压后、修改配置之前可检查文件：

  sha256sum -c SHA256SUMS
  python3 -m unittest discover -s script -p 'test_*.py' -v

本包在构建机器上验证了解压后的真实 MyTest 导出流程和 CPU/模拟测试。
构建机器没有可用的 gfx936 GPU/DTK，真实 HSACO/GPU 测试需在目标机器完成。
