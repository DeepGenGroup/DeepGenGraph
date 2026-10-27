#!/usr/bin/env python3
"""生成 split-D attention 的 MyTest 调优候选并导出 LLVM IR。

默认仅导出 IR；--hsaco 生成 HSACO，--benchmark 继续启动、校验并测量性能。
运行 `python3 TuneAttention.py --help` 查看参数和示例。
"""

import argparse
from contextlib import contextmanager
import fnmatch
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import tempfile

# 默认路径相对于脚本位置，支持仓库布局和与 MyTest 一起复制的独立目录。
ROOT = Path(__file__).resolve().parent


def default_paths():
    repo = next((path for path in (ROOT, *ROOT.parents)
                 if (path / "3rd/deepgengraph/test/test_input_spidD.mlir").is_file()), ROOT)
    tool = ROOT / "MyTest"
    if not tool.is_file():
        tool = repo / "3rd/deepgengraph/build/test/MyTest"
    source = ROOT / "test_input_spidD.mlir"
    if not source.is_file():
        source = repo / "3rd/deepgengraph/test/test_input_spidD.mlir"
    return source, tool


def input_paths(args, default_input):
    """Accept generated MLIR files, a directory, or the generator's stdout."""
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise ValueError(f"输入目录不存在：{args.input_dir}")
        paths = sorted(args.input_dir.glob("*.mlir"))
    elif args.input_list is not None:
        content = (sys.stdin.read() if args.input_list == "-"
                   else Path(args.input_list).read_text())
        paths = [Path(line.strip()) for line in content.splitlines() if line.strip()]
    else:
        paths = args.input or [default_input]
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        raise ValueError("没有输入 MLIR；目录须包含 *.mlir，文件列表须每行一个路径")
    return paths


def companion_script(name):
    return next((path / name for path in (ROOT, *ROOT.parents)
                 if (path / name).is_file()), ROOT / name)


def atomic_json(path, value):
    """Readers always see a complete JSON document, including during interruption."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + "-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def checkpoint_signals():
    """Finish a candidate's checkpoint before delivering Ctrl-C/SIGTERM."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def better_speedup(candidate, incumbent):
    return incumbent is None or (
        candidate["speedup"], -candidate["kernel_median_ms"]
    ) > (incumbent["speedup"], -incumbent["kernel_median_ms"])


class LiveResults:
    """Compact live summaries plus one full JSON per completed candidate."""
    SUCCESS = {"exported", "verified", "hsaco-built", "benchmark-passed"}

    def __init__(self, run, total, summary_dir=None):
        self.run, self.total, self.summary_dir = Path(run), total, summary_dir
        self.best, self.best_by_shape = {}, {}
        self.counts, self.completed, self.failed = {}, 0, 0
        self.last_result = None

    def __enter__(self):
        self.save("running")
        return self

    def save(self, state):
        snapshot = {
            "schema_version": 2, "state": state, "ranking_metric": "speedup",
            "sdpa_baseline": "remeasured_per_candidate",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_directory": str(self.run), "total_candidates": self.total,
            "completed_candidates": self.completed, "failed_candidates": self.failed,
            "successful_candidates": self.completed - self.failed,
            "status_counts": self.counts, "last_result": self.last_result,
            "result_files_glob": str(self.run / "**/result.json"),
            "best_by_shape": self.best_by_shape,
        }
        with checkpoint_signals():
            atomic_json(self.run / "best.json", self.best)
            atomic_json(self.run / "best_by_shape.json", self.best_by_shape)
            atomic_json(self.run / "manifest.json", snapshot)
            if self.summary_dir:
                atomic_json(self.summary_dir / "best_by_shape.json", self.best_by_shape)
                atomic_json(self.summary_dir / "summary.json", snapshot)

    def record(self, result):
        with checkpoint_signals():
            directory = Path(result["directory"])
            atomic_json(directory / "result.json", result)
            self.completed += 1
            status = result["status"]
            self.failed += status not in self.SUCCESS
            self.counts[status] = self.counts.get(status, 0) + 1
            self.last_result = {"case": result["case"], "status": status,
                                "directory": str(directory), "result_json": str(directory / "result.json")}
            if status == "benchmark-passed":
                entry = {key: result[key] for key in (
                    "case", "source_input", "directory", "parameters", "compiler_defaults",
                    "launch", "kernel_median_ms", "sdpa_median_ms", "speedup", "hsaco")}
                entry.update(ranking_metric="speedup", result_json=str(directory / "result.json"),
                             config_json=str(directory / "config.json"))
                key = result["source_input"]
                if better_speedup(entry, self.best.get(key)):
                    self.best[key] = entry
                shape = "x".join(map(str, entry["launch"]["shape"]))
                if better_speedup(entry, self.best_by_shape.get(shape)):
                    self.best_by_shape[shape] = entry
                    print(f"  当前最佳 BHSD={shape}: speedup={entry['speedup']:.6f}x，"
                          f"kernel={entry['kernel_median_ms']:.6f} ms，"
                          f"SDPA={entry['sdpa_median_ms']:.6f} ms；{directory}", flush=True)
            self.save("running")

    def __exit__(self, exc_type, exc_value, traceback):
        state = ("completed" if exc_type is None else
                 "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "error")
        self.save(state)
        return False


def interrupted(signum, frame):
    raise KeyboardInterrupt


def launch_config(source, llvm_ir, k_layout="bhsd"):
    """Read the generated attention ABI; use the compiler's actual block size."""
    code = masked(source)
    signatures = re.findall(r"deepgengraph\.kernel\s+@Attn_p2\s*\((.*?)\)\s*->", code, re.S)
    if len(signatures) != 1:
        raise ValueError("自动启动要求唯一 deepgengraph.kernel @Attn_p2")
    shapes = re.findall(r"tensor<(\d+)x(\d+)x(\d+)x(\d+)xf16>", signatures[0])
    if len(shapes) != 3 or len(set(shapes)) != 1:
        raise ValueError("自动启动要求 Q/K/V 为相同尺寸的静态 BSHD f16 tensor")
    b, s, h, d = map(int, shapes[0])
    grids = re.findall(r"deepgengraph_triton\.device_kernel\s+args\s*=\s*\[[^]]*\]"
                       r"\s*,\s*grid\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", code)
    if len(grids) != 1:
        raise ValueError("自动启动要求唯一、静态的 device_kernel grid")
    # DeepgengraphSimplify.cpp maps region arguments to GPU z,y,x.
    grid = list(reversed([int(x) for x in grids[0]]))
    header = re.search(r"^define\s+amdgpu_kernel\s+void\s+@Attn_p2\b[^\n]*"
                       r"!reqd_work_group_size\s+!(\d+)", llvm_ir, re.M)
    if not header:
        raise ValueError("LLVM IR 缺少 Attn_p2 的 !reqd_work_group_size，拒绝猜测 block")
    block = re.search(r"^!" + header[1] + r"\s*=\s*!\{\s*i32\s+(\d+),\s*"
                      r"i32\s+(\d+),\s*i32\s+(\d+)\s*\}", llvm_ir, re.M)
    if not block:
        raise ValueError("无法解析 !reqd_work_group_size")
    block = list(map(int, block.groups()))
    if any(x <= 0 for x in [b, h, s, d, *grid, *block]):
        raise ValueError("启动尺寸必须为正数")
    return {"schema_version": 1, "kernel": "Attn_p2", "shape": [b, h, s, d],
            "grid": grid, "block": block, "shared_mem": 0, "k_layout": k_layout}


def run_stage(command, directory, logpath, timeout):
    """Each backend/launch has its own log and process group, including children."""
    with logpath.open("w") as log:
        try:
            with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    code = process.wait(timeout=timeout)
                    return None if code == 0 else f"failed:{code}"
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    return "timeout"
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
        except OSError as exc:
            log.write(str(exc) + "\n")
            return "spawn-error"


HELP = r"""
默认优先使用脚本同目录的 MyTest 和 test_input_spidD.mlir，缺失时查找上层仓库。
--tool 可显式覆盖工具路径；独立目录中可用 --input 指定生成的 MLIR。
--input 接受一个或多个文件；--input-dir 接受目录下的全部 *.mlir（不递归）。
--input-list FILE 接受每行一个路径的列表；--input-list - 读取标准输入，
可直接接收 generate_attention_ir.py 的标准输出。相对路径以当前工作目录为准。
多输入按“输入 × 候选”编译，独立保存到同一 run-*，结果统一使用 JSON。
baseline 保留历史 nohint 展开配置；不指定 --case/--sweep 时枚举 coverage，
并非只运行 baseline。默认只导出 IR；各配置性能仍需在目标 GPU 上验证。
完整参数对照见同目录《调优参数说明.md》。
生成器默认 --softmax online；本脚本默认输入 test_input_spidD.mlir 仍是历史输入。
online-baseline 只设置编译参数，不会将已有输入转换为 online softmax。

完整流程（可选）：
  --hsaco       每候选调用 GenHsaco.sh：链接设备库、opt/llc、链接 HSACO。
  --benchmark   包含 --hsaco，并调用 launchKernel.py 校验及测时。
  --opt-level   GenHsaco.sh 的优化等级，默认 3。
  GenHsaco.sh 继承 LLVM_DIR、ROCM_BC、GPU_ARCH、GPU_FEATURES、DWARF_VERSION；
  请按目标机器设置（脚本默认 gfx936），相关环境值写入 run.json。
  --gen-hsaco / --launcher / --launch-python 可覆盖脚本和 GPU Python 路径。
  启动 shape 自动从 MLIR 的 BSHD 转为 BHSD，grid 按后端映射转成 X,Y,Z，
  block 从导出 LLVM IR 的 reqd_work_group_size 读取，动态 shared-mem=0。
  K layout 默认 bhsd（当前 MyTest 启用 K normalization）；旧工具可指定 --k-layout bhds。
  --device / --warmup / --repeat / --seed 控制测试，默认 0 / 10 / 50 / 0。
  每候选保存 launch.json、hsaco.log、benchmark.log、benchmark.json。
  每个完成的候选立即保存 result.json、manifest.json、best.json、best_by_shape.json。
  只有校验通过者参与排名，按 speedup 最大选优；相同比值优先 kernel median 小者。
  SDPA 为每候选重新测量，speedup=sdpa_median_ms/kernel_median_ms。
  best.json 按输入分组，best_by_shape.json 按完整 BHSD 分组。
  --summary-dir 可将实时 summary.json 与 best_by_shape.json 同步到一键入口的 session。
  JSON 通过临时文件原子替换；中断后保留已完成候选，未完成候选不参与排名。

三个独立控制：
  pipeline=0|1                 MyTest 的旧位置参数，控制手工 pipeline 调度。
                               0 不会关闭 frisk.pipeline_hint 自动流水线！
  pipeline-stages=none|keep|N   修改目标循环的 frisk.pipeline_hint；none 删除，
                               keep 保留输入，N 为 1..64；1 禁用，2 双缓冲。
  loop-mark=0|1|keep           修改目标循环的 frisk.simplify_loop；1 添加，
                               允许 nohint 循环独立展开；0 删除，keep 保留。
  simplify-loop-control=0|1    是否运行循环简化 Pass（默认 1）。
  coalesce-waits=0|1           是否合并重复 wait/barrier（此处默认 0）。
  mask-add-bypass=off|legal|approx  零 mask 分支直通；默认 off，approx 需 math=approx。
  defer-lane-sum=off|legal|approx  通用循环跨 lane 求和延迟；默认 off，浮点需 approx。
    MyTest 当前固定 wave64 ABI，启用时自动设置；不扫描 defer-lane-sum-wave-size。
  local-reduce=off|legal|approx  通用线程内归约；默认 off，approx 需 math=approx。
  local-reduce-chains=4 local-reduce-min-inputs=16 local-reduce-ops=all
    ops 可为 add,mul,and,or,xor,min,max 的逗号组合；min-inputs 至少为 chains 的两倍。
默认 --loop-iv auto：有 %dk 时修改该循环；没有时保留所有循环标记，并提示跳过。
因此也接受 generate_attention_ir.py 未指定 --bk 的非 split-D 输出。
非 split-D 输入建议使用 --case legacy-baseline；baseline 的 useKPack 和显式
qk-tile/vo-tile 要求 split-D 结构，脚本不会偷偷替换这些编译参数，失败会记录到汇总。
显式 --loop-iv dk 时，只修改唯一的 scf.for %dk；缺失或不唯一时报错。
其他循环的属性保留。每候选保存 input.mlir，绝不改写原始输入。

MyTest 位置参数顺序（命名参数不改变位置；完整命名参数见下文）：
  input pipeline output reorder pack layout carrier math threads waves \
  lifetimes masks prefetch fuse reuse dma-pack dma-export qk-tile vo-tile \
  --frag-prefetch=N --frag-prefetch-words=N --retain-fragment-words=N \
  --simplify-loop-control=N --coalesce-waits=N
另传 mask-add-bypass、defer-lane-sum、local-reduce*、swizzle-*、workgroup-* 命名参数。
copy/alignment 通过环境变量传递；pipeline-stages/loop-mark 修改输入 IR，
lds-bytes 修改导出 IR，不属于 MyTest argv。

--set / --sweep 的参数名及含义：
  reorder/pack/lifetimes/masks/prefetch/fuse：0/1，分别为指令重排、packed LDS、
    存活期优化、有效 tile mask 简化、global packet 寄存器预取、mask/add 融合。
  layout：3.1/shared、3.2/shuffle、3.3/b-permute。
  carrier：非负 64 位整数，单位 words/block，0=不限；不是 VGPR 上限。
  math：strict/mmac/mmac-exp2/mmac-recip/approx；approx 需检查数值误差。
  threads：64..1024 的 64 倍数；waves：M,N，各 1..16，M*N=threads/64。
  reuse：非负 32 位整数，LDS fragment 寄存器复用预算 words/thread，0=关闭；
    它不是 Q/K/V 共享内存空间复用 Pass 的开关。
  copy：legacy/async/sync/hybrid，对应 FRISK_GM_TO_SHM_MODE。
  alignment：0 或正的 2 次幂，字节，对应 FRISK_GM_TO_SHM_ALIGNMENT；
    是调用者对 tile 起始地址的对齐保证，不是自动补齐。当前基线为 16。
  dma-pack：legacy/useKPack/noKPack；后两种要求 copy=async。
  dma-export：auto/split32/hcu128/asm128；auto 随 copy 选择，async/hybrid
    使用 asm128，其他使用 split32；gfx936 当前采用 asm128。
  qk-tile/vo-tile：非负整数，0=自动；独立控制 QK/VO 分块，当前基线为 32/32。
  frag-prefetch：0/1，MMAC fragment 预取，与 prefetch 参数不同。
  frag-prefetch-words：正整数，fragment 预取 words/thread 预算，基线为 4。
  retain-fragment-words：非负 32 位整数，跨循环保留不变 operand 的
    32-bit words/thread 总预算；0=关闭，按完整 fragment 选择，并非实际 VGPR 上限。
    retained32-baseline 使用 32；历史预设保持 0。实际保留数量记录到 retention_reports。
  swizzle-q/swizzle-k/swizzle-v：off/on/源位:目标位，默认 off。
    on 或 1 等价于 8:6；off 或 0 关闭。位编号为 allocation 相对字节地址，
    目标位 XOR 源位；两位须不同且在 4..30，完整保留 16-byte DMA 包。
    任一开启时，MyTest 自动运行 LLVM CanonicalizeLDSGEPsPass；没有独立调优开关。
    旧 Q 的 7:3 会拆包，因此不支持。其他形状/别名/DMA 合法性由编译器证明。
  swizzle-q-arg/swizzle-k-arg/swizzle-v-arg：-1（默认，使用角色注解）或显式 ABI 编号。
  workgroup-group-size：非负整数，0=关闭，正数为 query 分组大小，支持尾组。
  workgroup-reverse：0/1 或 false/true；开启需要正的 group-size。
  workgroup-query-axis/workgroup-head-axis：auto/x/y/z，默认 auto。
    auto 根据每个输入的 BSHD parallel_map 的 res_dims 推导，并写入实际配置；
    不根据 SSA 名或网格大小猜测。旧输入 y/x 与当前输入 x/y 均可处理。
    与 reorder（fragment 指令重排）是不同开关；不改变 launch grid。
  lds-bytes：0=不干预；正数将单个 i8 LDS 池扩大到指定字节数（不允许缩小）。
    仅用于容量控制实验，不改变访存偏移；多 LDS 池时拒绝，须检查目标汇编
    group_segment_fixed_size 确认最终分配。此项不是 MyTest 参数。
环境变量由脚本显式设置，不继承 shell 的同名值。

预设（--case 接受 glob，可以重复）：
  online-baseline：retained32-baseline 改为 group=4、frag-prefetch=0；
    适配 online 输入的编译起点，性能需重新测量；不会更改输入算法。
  retained32-baseline：新性能基线，group8 reverse + K 8:6 + retain=32；
    在已测 BM=BN=128、BK=32 输入上保留完整 Q，LDS 16 KiB。
    其他输入仍需检查实际保留数量、VGPR/scratch 和 GPU 正确性/性能。
  baseline / nohint-unrolled：历史起点，nohint + DK 展开，retain=0。
  group8-reverse：baseline + group=8 + reverse；swizzle 均关闭。
  group8-reverse-k128-swizzle：group8-reverse + 仅 K 8:6，建议先与前者对照。
  group8-reverse-qk128-swizzle / group8-reverse-qkv128-swizzle：Q/K 或 Q/K/V 均为 8:6。
    Q/V 为保留 128-bit DMA 的实验布局，不保证性能收益。
  nohint-rolled：无 hint、不展开，对照旧 nohint。
  nohint-unrolled-32k：同 nohint-unrolled，LDS 扩至 32 KiB。
  pipeline-baseline / pipeline-wait-only / pipeline-loop-only / pipeline-combined：
    hint=2，分别关闭两项清理 / 仅等待合并 / 仅循环简化 / 两项开启。
  pipeline-loop-only-40k：仅循环简化 + LDS 扩至 40 KiB。
  legacy-baseline：保留旧脚本的 strict/legacy 参数，保留输入原有标记。
  k5*：保留历史组合参数；mytest-default：仅传 input、pipeline=0、output，保留输入；
    仍显式设置 copy=legacy、alignment=0，其余 MyTest 参数使用编译器默认值。
    mytest-default 不接受 --set；JSON 列表中 compiler_defaults=true、parameters=null 表示未传参数。
  coverage 还包含单因素对照、合法 threads/waves 组合及部分预算扫描。
  --sweep 只生成指定轴的笛卡尔积；优先级 sweep > set > 预设。
  --sweep-base PRESET 指定扫描基准（例如 retained32-baseline）；只与 --sweep 一起用。
    不指定则沿用历史 baseline；--case 在扫描模式下仍用于筛选 grid-* 候选名。
  不合法的 threads/waves、copy/dma-pack、swizzle/DMA、workgroup 组合自动过滤。
  Swizzle 要求 pack=1、copy=async/hybrid、alignment>=16，且不得用 split32；
  每块 tile 的实际连续性/对齐由编译器进一步检查；lowering 失败记录并继续。

示例：
  python3 TuneAttention.py --case retained32-baseline --dry-run
  python3 TuneAttention.py --sweep-base retained32-baseline \
    --sweep 'retain-fragment-words=0;8;16;24;32' --dry-run
  python3 TuneAttention.py --sweep-base retained32-baseline \
    --sweep 'workgroup-group-size=4;8;16' --benchmark
  python3 TuneAttention.py --case group8-reverse --case group8-reverse-k128-swizzle --dry-run
  python3 TuneAttention.py --set workgroup-group-size=8 --set workgroup-reverse=1 \
    --sweep 'swizzle-q=off;on' --sweep 'swizzle-k=off;on' --sweep 'swizzle-v=off;on'
  python3 TuneAttention.py --case baseline --dry-run
  python3 TuneAttention.py --case baseline --llvm-as /data2/xsl/install/bin/llvm-as
  python3 TuneAttention.py --case 'nohint-*' --case 'pipeline-*'
  python3 TuneAttention.py --sweep 'pipeline-stages=none;2;3' \
    --sweep 'simplify-loop-control=0;1' --sweep 'coalesce-waits=0;1'
  python3 TuneAttention.py --sweep 'qk-tile=16;32;64' --sweep 'vo-tile=16;32;64'
  python3 TuneAttention.py --case baseline --set math=strict
  python3 TuneAttention.py --case mytest-default --build
  python3 TuneAttention.py --case baseline --benchmark
  python3 TuneAttention.py --input generated_ir/tiles/test_input_1x32x4096x128-64-32-32.mlir --case baseline
  python3 TuneAttention.py --input-dir generated_ir/tiles --case baseline
  set -o pipefail
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 16 32 --output-dir generated_ir/tiles \
    | python3 TuneAttention.py --input-list - --case baseline

每次运行新建 run-*；每候选保存 input.mlir、kernel.ll、lowering.log、command.sh、
config.json、result.json。result.json 包含输入/IR SHA256、静态 LDS 字节数、
实际流水线标记数、展开循环数、swizzle 规划、DMA 指令点数、保留 Pass 报告及 GEP 规范化报告。
这些是静态 IR/Pass 信息，不是 PMC bank 冲突或最终 ds_read2 数量。normalized_sha256 仅归一化候选目录路径，
便于发现仅源文件路径不同的重复 IR；哈希不是性能结论。
--llvm-as 验证最终（含容量调整后）IR。性能和数值正确性仍需 GPU 实测。
"""

# 保留原脚本基线，避免历史 k5 预设随新基线一起漂移。
LEGACY_BASE = dict(
    pipeline="0", reorder="1", pack="1", layout="3.3", carrier="34816",
    math="strict", threads="256", waves="4,1", lifetimes="1", masks="0",
    prefetch="0", fuse="1", reuse="0", copy="legacy", alignment="0",
    **{"dma-pack": "legacy", "dma-export": "auto", "qk-tile": "0", "vo-tile": "0",
       "frag-prefetch": "0", "frag-prefetch-words": "8", "retain-fragment-words": "0",
       "pipeline-stages": "keep", "loop-mark": "keep",
       "simplify-loop-control": "1", "coalesce-waits": "1", "lds-bytes": "0",
       "mask-add-bypass": "off",
       "defer-lane-sum": "off", "local-reduce": "off", "local-reduce-chains": "4",
       "local-reduce-min-inputs": "16", "local-reduce-ops": "all",
       "swizzle-q": "off", "swizzle-k": "off", "swizzle-v": "off",
       "swizzle-q-arg": "-1", "swizzle-k-arg": "-1", "swizzle-v-arg": "-1",
       "workgroup-group-size": "0", "workgroup-reverse": "0",
       "workgroup-query-axis": "auto", "workgroup-head-axis": "auto"})
BASE = dict(LEGACY_BASE, math="approx", masks="1", prefetch="1", fuse="0",
            copy="async", alignment="16",
            **{"dma-pack": "useKPack", "dma-export": "asm128", "qk-tile": "32",
               "vo-tile": "32", "frag-prefetch": "1", "frag-prefetch-words": "4",
               "pipeline-stages": "none", "loop-mark": "1", "coalesce-waits": "0"})
# Measured full-Q retention configuration. Keep historical presets stable.
RETAINED32_BASE = dict(BASE, **{"workgroup-group-size": "8", "workgroup-reverse": "1",
                              "swizzle-k": "8:6", "retain-fragment-words": "32"})
# Matches the latest online-softmax native-PV-accumulator export. The softmax
# algorithm is selected by the input generator, not by this compiler preset.
# GPU performance of this new IR must be remeasured.
ONLINE_BASE = dict(RETAINED32_BASE, **{"workgroup-group-size": "4", "frag-prefetch": "0"})
KEYS = list(BASE)
POSITIONAL_KEYS = (
    "reorder", "pack", "layout", "carrier", "math", "threads", "waves",
    "lifetimes", "masks", "prefetch", "fuse", "reuse", "dma-pack", "dma-export",
    "qk-tile", "vo-tile",
)
SWIZZLE_KEYS = ("swizzle-q", "swizzle-k", "swizzle-v")
SWIZZLE_ARG_KEYS = tuple(key + "-arg" for key in SWIZZLE_KEYS)
WORKGROUP_AXIS_KEYS = ("workgroup-query-axis", "workgroup-head-axis")
NAMED_KEYS = ("frag-prefetch", "frag-prefetch-words", "retain-fragment-words",
              "simplify-loop-control", "coalesce-waits", "mask-add-bypass",
              "defer-lane-sum", "local-reduce", "local-reduce-chains", "local-reduce-min-inputs", "local-reduce-ops",
              *SWIZZLE_KEYS, *SWIZZLE_ARG_KEYS, "workgroup-group-size", "workgroup-reverse",
              *WORKGROUP_AXIS_KEYS)
BOOLS = ("pipeline", "reorder", "pack", "lifetimes", "masks", "prefetch", "fuse",
         "frag-prefetch", "simplify-loop-control", "coalesce-waits", "workgroup-reverse")
ENUMS = {"layout": ("3.1", "3.2", "3.3"),
         "mask-add-bypass": ("off", "legal", "approx"),
         "defer-lane-sum": ("off", "legal", "approx"),
         "local-reduce": ("off", "legal", "approx"),
         "math": ("strict", "mmac", "mmac-exp2", "mmac-recip", "approx"),
         "copy": ("legacy", "async", "sync", "hybrid"),
         "dma-pack": ("legacy", "useKPack", "noKPack"),
         "dma-export": ("auto", "split32", "hcu128", "asm128"),
         "loop-mark": ("keep", "0", "1"),
         "workgroup-query-axis": ("auto", "x", "y", "z"),
         "workgroup-head-axis": ("auto", "x", "y", "z")}
ALIASES = {"shared": "3.1", "shuffle": "3.2", "b-permute": "3.3"}

# Mask comments/strings without changing offsets, so braces in them are inert.
LEXICAL = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"')


def masked(text):
    return LEXICAL.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)


def resolve_workgroup_axes(source, cfg):
    """Attention parallel_map's BSHD result dimensions define roles, not SSA names.

    The backend maps the three parallel dimensions to z/y/x in order. Refuse
    unknown maps instead of guessing from equal grid extents or bx/by spelling.
    """
    if cfg is None:
        return None
    cfg = cfg.copy()
    if not int(cfg["workgroup-group-size"]):
        return cfg
    if any(cfg[key] == "auto" for key in WORKGROUP_AXIS_KEYS):
        code = masked(source)
        starts = list(re.finditer(r"deepgengraph\.kernel\s+@Attn_p2\b", code))
        if len(starts) != 1:
            raise ValueError("自动 workgroup 轴需要唯一 Attn_p2 parallel_map；请显式指定 query/head-axis")
        tail = code[starts[0].end():]
        match = re.search(r"\bparallel_map\s*=\s*\[", tail)
        roles = []
        if match:
            begin = match.end()
            depth, end = 1, begin
            while end < len(tail) and depth:
                depth += (tail[end] == "[") - (tail[end] == "]")
                end += 1
            if depth == 0:
                mapping = tail[begin:end - 1]
                dims = re.findall(r"\bres_dims\s*=\s*\[([^]]*)\]", mapping)
                if len(dims) == 3 and sorted(x.strip() for x in dims) == ["0", "1", "2"]:
                    roles = [x.strip() for x in dims]
        if not roles:
            raise ValueError("无法证明 query/head 轴：需要 BSHD parallel_map 的 res_dims=0/1/2；请显式指定两轴")
        for key, role in zip(WORKGROUP_AXIS_KEYS, ("1", "2")):
            if cfg[key] == "auto":
                cfg[key] = "zyx"[roles.index(role)]
    if cfg[WORKGROUP_AXIS_KEYS[0]] == cfg[WORKGROUP_AXIS_KEYS[1]]:
        raise ValueError("workgroup query/head 轴必须不同（包括 auto 解析后）")
    return cfg


def incompatible_layout(cfg):
    """Reject known impossible tuning combinations; tile proofs stay in MyTest."""
    enabled = any(cfg[key] != "off" for key in SWIZZLE_KEYS)
    if enabled:
        export = cfg["dma-export"]
        if (cfg["copy"] not in ("async", "hybrid") or cfg["pack"] != "1" or
                int(cfg["alignment"]) < 16 or export == "split32"):
            return True
        bindings = [cfg[key] for key in SWIZZLE_ARG_KEYS if cfg[key] != "-1"]
        if len(bindings) != len(set(bindings)):
            return True
    # reverse with a disabled reorder pass is a duplicate of the disabled case.
    if not int(cfg["workgroup-group-size"]) and cfg["workgroup-reverse"] == "1":
        return True
    q, h = (cfg[key] for key in WORKGROUP_AXIS_KEYS)
    return bool(int(cfg["workgroup-group-size"]) and q != "auto" and q == h)


def closing_brace(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return i
    raise ValueError("目标循环或属性字典的花括号不完整")


def prepare_input(source, cfg, iv):
    """Edit only the selected SCF loop's trailing attribute dictionary."""
    if cfg is None or (cfg["pipeline-stages"] == cfg["loop-mark"] == "keep"):
        return source
    code = masked(source)
    loops = list(re.finditer(r"\bscf\.for\s+%" + re.escape(iv) + r"\s*=", code))
    if len(loops) != 1:
        raise ValueError(f"需要唯一 scf.for %{iv}，找到 {len(loops)} 个；请设置 --loop-iv")
    # Find the region opener outside parentheses/types in the loop header.
    depth = 0
    start = loops[0].end()
    for start in range(start, len(code)):
        char = code[start]
        if char == "{" and depth == 0:
            break
        if char in "(<[": depth += 1
        if char in ")>]" and code[start - 1] != "-": depth -= 1
    else:
        raise ValueError("目标循环缺少 region")
    end = closing_brace(code, start) + 1
    attr_start = end
    while attr_start < len(code) and code[attr_start].isspace():
        attr_start += 1
    has_attrs = attr_start < len(code) and code[attr_start] == "{"
    attr_end = closing_brace(code, attr_start) + 1 if has_attrs else end
    raw = source[attr_start + 1:attr_end - 1] if has_attrs else ""
    clean = masked(raw)
    # Commas inside nested dictionary/array/type attributes are not separators.
    fields, begin, depth = [], 0, 0
    for i, char in enumerate(clean):
        if char in "({[<": depth += 1
        elif char in ")}]>" and (i == 0 or clean[i - 1] != "-"): depth -= 1
        elif char == "," and depth == 0:
            fields.append(raw[begin:i].strip())
            begin = i + 1
    fields.append(raw[begin:].strip())
    updates = {}
    if cfg["pipeline-stages"] != "keep":
        stages = cfg["pipeline-stages"]
        updates["frisk.pipeline_hint"] = None if stages == "none" else f"frisk.pipeline_hint = {stages} : i64"
    if cfg["loop-mark"] != "keep":
        updates["frisk.simplify_loop"] = "frisk.simplify_loop" if cfg["loop-mark"] == "1" else None
    kept = []
    for field in fields:
        # MLIR attribute names may also be quoted.
        without_comments = LEXICAL.sub(
            lambda m: m[0] if m[0].startswith('"') else " ", field).strip()
        match = re.match(r'(?:"([^"]+)"|([\w.]+))', without_comments)
        key = next((x for x in match.groups() if x is not None), "") if match else ""
        if key not in updates and without_comments:
            kept.append(without_comments)
    kept += [value for value in updates.values() if value is not None]
    attrs = " {" + ", ".join(kept) + "}" if kept else ""
    return source[:end] + attrs + source[attr_end:]


LDS_GLOBAL = re.compile(
    r"^(@[\w.]+ = addrspace\(3\) global )\[(\d+) x (i8|half|float)\]( undef, align (\d+))$",
    re.M)


def lds_bytes(text):
    found = LDS_GLOBAL.findall(text)
    if len(found) != len(re.findall(r"^@.*addrspace\(3\).*global ", text, re.M)) or not found:
        return None
    total = 0
    for _, count, element, _, alignment in found:
        alignment = int(alignment)
        total = (total + alignment - 1) // alignment * alignment
        total += int(count) * {"i8": 1, "half": 2, "float": 4}[element]
    return total


def pad_lds(text, size):
    if not size:
        return text
    found = list(LDS_GLOBAL.finditer(text))
    if len(found) != 1 or found[0][3] != "i8" or lds_bytes(text) is None:
        raise ValueError("lds-bytes 容量控制要求恰好一个 i8 LDS 池")
    match = found[0]
    if size < int(match[2]):
        raise ValueError(f"lds-bytes={size} 小于原生 LDS {match[2]}，不允许缩小")
    return text[:match.start()] + f"{match[1]}[{size} x i8]{match[4]}" + text[match.end():]


def ir_facts(text, diagnostics):
    marker = "---------- after frisk-auto-pipeline ---------\n"
    staged = diagnostics.split(marker, 1)[1].split("\n}\n", 1)[0] if marker in diagnostics else ""
    counts = re.findall(r"SimplifyLoopControl: (\d+) small loops specialized", diagnostics)
    return {"static_lds_bytes": lds_bytes(text),
            "scheduled_loops": staged.count("pipeline.scheduled") if marker in diagnostics else None,
            "simplified_loops": sum(map(int, counts)),
            "defer_lane_sum_reports": [dict(function=fn, states=int(states),
                                              loop_shuffles=int(shuffles), mode=mode)
                for fn, states, shuffles, mode in re.findall(
                    r"DeferLaneSum: ([^\n:]+): (\d+) states, (\d+) loop shuffles deferred \((legal|approx)\)", diagnostics)],
            "local_reduction_reports": [dict(function=fn, groups=int(groups),
                                              inputs=int(inputs), chains=int(chains), mode=mode)
                for fn, groups, inputs, chains, mode in re.findall(
                    r"LocalReduction: ([^\n:]+): (\d+) groups, (\d+) inputs, (\d+) chains \((legal|approx)\)", diagnostics)],
            "mask_add_bypass_reports": [dict(function=fn, groups=int(groups),
                                             additions=int(adds), mode=mode)
                for fn, groups, adds, mode in re.findall(
                    r"MaskAddBypass: ([^\n:]+): (\d+) groups, (\d+) additions bypassed \((legal|approx)\)", diagnostics)],
            "buffer_dma128_sites": text.count("buffer_load_dwordx4 "),
            "buffer_dma32_sites": len(re.findall(r"buffer_load_dword\s", text)),
            "lds_b64_half_load_sites": len(re.findall(r"= load <4 x half>, ptr addrspace\(3\)", text)),
            "retention_reports": [dict(operand_tiles=int(tiles), budget_words_per_thread=int(words))
                for tiles, words in re.findall(
                    r"RetainInvariantFragments: (\d+) operand tiles retained; budget (\d+) words/thread", diagnostics)],
            # MLIR snapshots repeat the same loop attributes. Report distinct
            # observed values, never sum snapshots or infer use from the budget.
            "retained_loop_words_observed": sorted(set(map(int, re.findall(
                r"frisk\.retained_words = (\d+) : i32", diagnostics)))),
            "shared_swizzle_plans": [dict(argument=int(arg), source_bit=int(src),
                                           target_bit=int(dst), dma_bytes=16, aliases=int(aliases))
                for arg, dst, src, aliases in re.findall(
                    r"Shared swizzle: argument (\d+) byte bit (\d+) \^= (\d+), DMA=16 bytes, aliases=(\d+)", diagnostics)],
            "lds_gep_reported_offsets": sum(map(int, re.findall(
                r"CanonicalizeLDSGEPs: [^\n]*: (\d+) proven constant offsets", diagnostics))),
            "workgroup_orders": [list(map(int, values)) for values in sorted(set(re.findall(
                r"frisk\.workgroup_order = array<i64: (\d+), (\d+), (\d+), ([01])>", diagnostics)))]}


def main(argv=None):
    """解析配置、生成候选，再按顺序导出；导入模块时不启动调优。"""
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="TuneAttention.py", description="生成 split-D attention 的 MyTest 调优候选。",
                                epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_input, default_tool = default_paths()
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, nargs="+", help="一个或多个 MLIR 文件")
    inputs.add_argument("--input-dir", type=Path, help="生成结果目录，读取其中全部 *.mlir")
    inputs.add_argument("--input-list", metavar="FILE", help="每行一个输入路径；- 读取生成器标准输出")
    p.add_argument("--tool", type=Path, default=default_tool, help="默认优先使用脚本同目录的 MyTest")
    p.add_argument("--loop-iv", default="auto", help="待修改标记的 scf.for IV；默认 auto 检测 dk，无 dk 则保留标记")
    p.add_argument("--output-dir", type=Path, help="父目录；每次建立独立 run-* 子目录")
    p.add_argument("--summary-dir", type=Path, help="额外实时写入 summary.json / best_by_shape.json 的新目录")
    p.add_argument("--case", action="append", default=[], help="候选名或 glob，可重复")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖候选参数")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1;V2", help="组合扫描轴，可重复")
    p.add_argument("--sweep-base", metavar="PRESET", help="--sweep 的基准预设，默认历史 baseline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="只列出候选，不创建文件")
    mode.add_argument("--dry-run", action="store_true", help="只打印命令，不创建文件")
    mode.add_argument("--prepare-input", type=Path, metavar="OUTPUT", help="只写候选输入副本，供 dry-run/replay 使用")
    mode.add_argument("--pad-lds", type=Path, metavar="IR", help="仅对指定 IR 应用 --set lds-bytes=N")
    mode.add_argument("--write-launch-config", type=Path, metavar="IR",
                      help="从 --input 与 LLVM IR 写入 IR 同目录的 launch.json")
    p.add_argument("--build", action="store_true", help="从 MyTest 所在 build/test 的上级构建 MyTest")
    p.add_argument("--jobs", type=int, default=2, help="构建并发数，默认 2；候选串行生成")
    p.add_argument("--timeout", type=float, default=300, help="每个候选编译超时秒数，默认 300")
    p.add_argument("--llvm-as", type=Path, help="可选的 LLVM IR 语法/验证工具")
    p.add_argument("--hsaco", action="store_true", help="继续调用 GenHsaco.sh 生成 HSACO")
    p.add_argument("--benchmark", action="store_true", help="生成 HSACO 并运行正确性及性能测试")
    p.add_argument("--gen-hsaco", type=Path, default=companion_script("GenHsaco.sh"))
    p.add_argument("--launcher", type=Path, default=companion_script("launchKernel.py"))
    p.add_argument("--launch-python", default=sys.executable, help="含 PyTorch/HIP 的 Python 解释器")
    p.add_argument("--opt-level", choices=("0", "1", "2", "3"), default="3")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-layout", choices=("bhsd", "bhds"), default="bhsd")
    a = p.parse_args(argv)
    a.hsaco = a.hsaco or a.benchmark
    a.gen_hsaco = a.gen_hsaco.resolve()
    a.launcher = a.launcher.resolve()
    if os.sep in a.launch_python:
        # Keep a venv's interpreter symlink: resolving it can select system Python.
        a.launch_python = str(Path(a.launch_python).absolute())
    if not re.fullmatch(r"[\w.$-]+", a.loop_iv):
        p.error("loop-iv 应为不带 % 的 SSA 名")
    if a.jobs < 1 or not (0 < a.timeout < float("inf")):
        p.error("jobs 和 timeout 必须为正数且有限")
    if a.device < 0 or a.warmup < 0 or a.repeat < 1:
        p.error("device/warmup 必须非负，repeat 必须为正数")

    def assignment(text):
        key, sep, value = text.partition("=")
        if not sep or key not in BASE or not value:
            p.error("应为 KEY=VALUE；KEY 可取：" + ", ".join(KEYS))
        return key, value

    def normalize(key, value):
        if key == "local-reduce-ops":
            if value != "all" and not all(v in ("add", "mul", "and", "or", "xor", "min", "max")
                                          for v in value.split(",")):
                p.error("local-reduce-ops 必须为 all 或 add,mul,and,or,xor,min,max 的组合")
        if key in ("local-reduce-chains", "local-reduce-min-inputs"):
            low, high = (2, 16) if key == "local-reduce-chains" else (4, 2048)
            if not value.isascii() or not value.isdigit() or not low <= int(value) <= high:
                p.error(f"{key} 必须为 {low}..{high}")
            value = str(int(value))
        if key == "layout":
            value = ALIASES.get(value, value)
        if key in SWIZZLE_KEYS:
            value = {"0": "off", "1": "8:6", "on": "8:6"}.get(value, value)
            if value != "off":
                pair = re.fullmatch(r"([0-9]+):([0-9]+)", value)
                if not pair or not all(4 <= int(bit) <= 30 for bit in pair.groups()) or pair[1] == pair[2] or int(pair[1]) == int(pair[2]):
                    p.error(f"{key} 必须为 off/on/源位:目标位；两位须不同且在 4..30，保留 128-bit DMA")
                value = ":".join(str(int(bit)) for bit in pair.groups())
        if key in SWIZZLE_ARG_KEYS:
            if not re.fullmatch(r"-1|[0-9]+", value) or int(value) >= 2**31:
                p.error(f"{key} 必须为 -1（角色注解）或非负 32 位参数编号")
            value = str(int(value))
        if key == "workgroup-reverse":
            value = {"false": "0", "true": "1"}.get(value, value)
        if key in BOOLS and value not in ("0", "1"):
            p.error(f"{key} 必须为 0 或 1")
        if key in ENUMS and value not in ENUMS[key]:
            p.error(f"{key} 可取 {ENUMS[key]}")
        if key == "pipeline-stages":
            if value not in ("none", "keep"):
                if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= 64:
                    p.error("pipeline-stages 必须为 none、keep 或 1..64")
                value = str(int(value))
        limits = {"carrier": 2**64, "reuse": 2**32, "threads": 2**32,
                  "alignment": 2**63, "qk-tile": 2**63, "vo-tile": 2**63,
                  "frag-prefetch-words": 2**32, "retain-fragment-words": 2**32,
                  "lds-bytes": 2**32, "workgroup-group-size": 2**63}
        if key in limits:
            if not re.fullmatch(r"[0-9]+", value):
                p.error(f"{key} 必须为非负整数")
            n = int(value)
            if n >= limits[key]:
                p.error(f"{key} 超过 MyTest 整数范围")
            if key == "frag-prefetch-words" and n == 0:
                p.error("frag-prefetch-words 必须为正数")
            if key == "threads" and (not 64 <= n <= 1024 or n % 64):
                p.error("threads 必须是 [64,1024] 内 64 的倍数")
            if key == "alignment" and n and n & (n - 1):
                p.error("alignment 必须为 0 或正的 2 次幂")
            value = str(n)
        if key == "waves":
            if not re.fullmatch(r"[0-9]+,[0-9]+", value):
                p.error("waves 必须为 M,N")
            m, n = map(int, value.split(","))
            if not (1 <= m <= 16 and 1 <= n <= 16):
                p.error("waves 的 M,N 必须在 [1,16]")
            value = f"{m},{n}"
        return value

    overrides = {k: normalize(k, v) for k, v in map(assignment, a.set)}
    axes = {}
    for k, v in map(assignment, a.sweep):
        if k in axes:
            p.error(f"扫描轴重复：{k}")
        axes[k] = list(dict.fromkeys(normalize(k, s) for s in v.split(";")))

    if a.prepare_input or a.pad_lds or a.write_launch_config:
        if a.case or axes or a.sweep_base or a.build or a.hsaco:
            p.error("单步文件操作不接受 --case、--sweep、--sweep-base、--build、--hsaco 或 --benchmark")
        cfg = dict(BASE, **overrides)
        try:
            if a.write_launch_config:
                paths = input_paths(a, default_input)
                if len(paths) != 1:
                    p.error("--write-launch-config 只接受一个输入文件")
                config = launch_config(paths[0].read_text(), a.write_launch_config.read_text(), a.k_layout)
                a.write_launch_config.with_name("launch.json").write_text(json.dumps(config, indent=2) + "\n")
            elif a.prepare_input:
                paths = input_paths(a, default_input)
                if len(paths) != 1:
                    p.error("--prepare-input 只接受一个输入文件")
                input_path = paths[0]
                if a.prepare_input.resolve() == input_path:
                    p.error("不允许覆盖原始输入；请选择不同的 --prepare-input 路径")
                source = input_path.read_text()
                iv = a.loop_iv
                if iv == "auto":
                    iv = "dk" if re.search(r"\bscf\.for\s+%dk\s*=", masked(source)) else None
                content = prepare_input(source, cfg, iv) if iv else source
                a.prepare_input.parent.mkdir(parents=True, exist_ok=True)
                a.prepare_input.write_text(content)
            else:
                a.pad_lds.write_text(pad_lds(a.pad_lds.read_text(), int(cfg["lds-bytes"])))
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        return 0

    def coverage():
        # None 表示省略可选参数，观察编译器自身默认值；它与 BASE 不同。
        yield "mytest-default", None
        yield "baseline", BASE.copy()
        yield "retained32-baseline", RETAINED32_BASE.copy()
        yield "online-baseline", ONLINE_BASE.copy()
        yield "legacy-baseline", LEGACY_BASE.copy()
        grouped = dict(BASE, **{"workgroup-group-size": "8", "workgroup-reverse": "1"})
        yield "group8-reverse", grouped.copy()
        yield "group8-reverse-k128-swizzle", dict(grouped, **{"swizzle-k": "8:6"})
        yield "group8-reverse-qk128-swizzle", dict(grouped, **{"swizzle-q": "8:6", "swizzle-k": "8:6"})
        yield "group8-reverse-qkv128-swizzle", dict(grouped, **{key: "8:6" for key in SWIZZLE_KEYS})
        yield "nohint-unrolled", BASE.copy()
        yield "nohint-rolled", dict(BASE, **{"loop-mark": "0", "simplify-loop-control": "0"})
        yield "nohint-unrolled-32k", dict(BASE, **{"lds-bytes": "32768"})
        for suffix, loop, waits in [("baseline", "0", "0"), ("wait-only", "0", "1"),
                                    ("loop-only", "1", "0"), ("combined", "1", "1"),
                                    ("loop-only-40k", "1", "0")]:
            yield "pipeline-" + suffix, dict(BASE, **{
                "pipeline-stages": "2", "loop-mark": "0", "simplify-loop-control": loop,
                "coalesce-waits": waits, "lds-bytes": "40960" if suffix.endswith("40k") else "0"})
        # 单因素对照：每次只改变一个开关/枚举/预算，方便定位其对 IR 的影响。
        dimensions = {k: ("0", "1") for k in BOOLS if k != "workgroup-reverse"}
        dimensions.update({k: v for k, v in ENUMS.items() if k not in WORKGROUP_AXIS_KEYS})
        dimensions.update({key: ("off", "8:6") for key in SWIZZLE_KEYS})
        dimensions["workgroup-group-size"] = ("1", "4", "8", "16")
        dimensions.update(carrier=("0", "24576", "28672", "34816"), reuse=("0", "4", "8", "16"))
        dimensions.update({"pipeline-stages": ("none", "1", "2", "3", "4"),
                           "qk-tile": ("16", "32", "64"), "vo-tile": ("16", "32", "64"),
                           "frag-prefetch-words": ("4", "8", "16"),
                           "retain-fragment-words": ("0", "8", "16", "24", "32")})
        for key, values in dimensions.items():
            for value in values:
                if value != BASE[key]:
                    cfg = dict(BASE, **{key: value})
                    # Copy mode comparisons need a compatible packing mode.
                    if key == "copy" and value != "async":
                        cfg["dma-pack"] = "legacy"
                    yield f"{'manual-pipeline' if key == 'pipeline' else key}-{value}", cfg
        # 线程数和 wave 排列相互约束，必须成对枚举，不能当作独立单因素。
        for waves in range(1, 17):
            for m in range(1, waves + 1):
                if waves % m == 0:
                    cfg = dict(BASE, threads=str(waves * 64), waves=f"{m},{waves // m}")
                    if cfg != BASE:
                        yield f"threads-{waves * 64}-waves-{m}x{waves // m}", cfg
        for copy in ("async", "sync", "hybrid"):
            yield f"{copy}-align4", dict(BASE, copy=copy, alignment="4",
                                       **{"dma-pack": "useKPack" if copy == "async" else "legacy"})
        # k5 是已有组合实验：近似数学 + mask 简化 + packet 预取，关闭 mask/add 融合。
        # 对应 generate_attention_candidates.py 的 step5；组合收益仍需 GPU 实测。
        k5 = dict(LEGACY_BASE, math="approx", masks="1", prefetch="1", fuse="0")
        yield "k5", k5
        for copy in ("async", "sync", "hybrid"):
            yield f"k5-{copy}", dict(k5, copy=copy, alignment="4")
        yield "k5-hybrid-reuse8", dict(k5, copy="hybrid", alignment="4", reuse="8")

    sweep_base = BASE
    if a.sweep_base:
        if not axes:
            p.error("--sweep-base 必须与 --sweep 一起使用；单独运行预设请用 --case")
        presets = dict(coverage())
        if a.sweep_base not in presets or presets[a.sweep_base] is None:
            p.error("--sweep-base 需要具有显式参数的预设名，例如 retained32-baseline")
        sweep_base = presets[a.sweep_base]

    def candidates():
        # 指定 --sweep 后只生成各扫描轴的笛卡尔积，不再追加默认 coverage。
        if axes:
            for index, values in enumerate(itertools.product(*axes.values()), 1):
                yield f"grid-{index:06d}", dict(sweep_base, **dict(zip(axes, values)))
        else:
            yield from coverage()

    selected = []
    filtered = 0
    for name, cfg in candidates():
        if a.case and not any(fnmatch.fnmatchcase(name, pattern) for pattern in a.case):
            continue
        if cfg is None:
            if overrides:
                if a.case and "mytest-default" in a.case:
                    p.error("mytest-default 保留编译器默认值；请对 baseline 使用 --set")
                continue
        else:
            # 优先级：--sweep > --set > 候选原值；未扫描的参数才应用 --set。
            cfg.update({k: v for k, v in overrides.items() if k not in axes})
            m, n = map(int, cfg["waves"].split(","))
            if (m * n != int(cfg["threads"]) // 64 or
                    (cfg["dma-pack"] != "legacy" and cfg["copy"] != "async") or
                    incompatible_layout(cfg) or
                    (cfg["mask-add-bypass"] == "approx" and cfg["math"] != "approx") or
                    (cfg["defer-lane-sum"] == "approx" and cfg["math"] != "approx") or
                    (cfg["local-reduce"] == "approx" and cfg["math"] != "approx") or
                    int(cfg["local-reduce-min-inputs"]) < 2 * int(cfg["local-reduce-chains"])):
                filtered += 1
                continue
        selected.append((name, cfg))
    if not selected:
        p.error("没有匹配且 threads/waves、copy/dma-pack、swizzle/DMA、workgroup 组合合法的候选；用 --list 查看名称")
    print(f"候选数：{len(selected)}；过滤不合法的参数组合：{filtered}", file=sys.stderr)

    tool = a.tool.resolve()
    def command(cfg, output, source):
        # 显式设置这两个环境变量，避免 shell 中的同名变量污染对照实验。
        env = dict(copy="legacy", alignment="0") if cfg is None else cfg
        cmd = ["env", "FRISK_GM_TO_SHM_MODE=" + env["copy"],
               "FRISK_GM_TO_SHM_ALIGNMENT=" + env["alignment"], str(tool), str(source),
               "0" if cfg is None else cfg["pipeline"], str(output)]
        if cfg is not None:
            explicit = dict(cfg)
            if explicit["dma-export"] == "auto":
                explicit["dma-export"] = "asm128" if cfg["copy"] in ("async", "hybrid") else "split32"
            cmd += [explicit[k] for k in POSITIONAL_KEYS]
            cmd += [f"--{key}={cfg[key]}" for key in NAMED_KEYS
                    if key not in WORKGROUP_AXIS_KEYS or cfg[key] != "auto"]
        return cmd

    def downstream_commands(directory):
        metadata = [sys.executable, str(Path(__file__).resolve()), "--input",
                    str(directory / "input.mlir"), "--write-launch-config",
                    str(directory / "kernel.ll"), "--k-layout", a.k_layout]
        build = ["bash", str(a.gen_hsaco), "kernel.ll", a.opt_level, "kernel.hsaco"]
        launch = [a.launch_python, str(a.launcher), "--hsaco", str(directory / "kernel.hsaco"),
                  "--config", str(directory / "launch.json"),
                  "--json-output", str(directory / "benchmark.json"),
                  "--device", str(a.device), "--warmup", str(a.warmup),
                  "--repeat", str(a.repeat), "--seed", str(a.seed)]
        return metadata, build, launch

    # 预览模式在构建、创建目录之前退出，可先检查配置和实际编译命令。
    if a.list:
        print(json.dumps([{"case": name, "parameters": cfg, "compiler_defaults": cfg is None}
                          for name, cfg in selected], indent=2, ensure_ascii=False))
        return 0
    # 先验证所有输入，避免批处理中途因输入错误留下不完整的运行目录。
    tasks, snapshots = [], []
    try:
        paths = input_paths(a, default_input)
        for input_index, input_path in enumerate(paths, 1):
            original_source = input_path.read_text()
            iv = a.loop_iv
            if iv == "auto":
                iv = "dk" if re.search(r"\bscf\.for\s+%dk\s*=", masked(original_source)) else None
                if iv is None:
                    print(f"{input_path}: 无 %dk，保留循环标记（pipeline-stages/loop-mark 不适用）",
                          file=sys.stderr)
            # 索引避免不同目录的同名文件发生覆盖。
            prefix = f"input-{input_index:04d}-{input_path.stem}" if len(paths) > 1 else ""
            snapshots.append((prefix, input_path, original_source, iv))
            for name, cfg in selected:
                effective = (dict(cfg, **{"pipeline-stages": "keep", "loop-mark": "keep"})
                             if cfg and iv is None else cfg)
                effective = resolve_workgroup_axes(original_source, effective)
                prepared = prepare_input(original_source, effective, iv) if iv else original_source
                relative = Path(prefix) / name
                tasks.append((relative, name, effective, input_path, iv, prepared))
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if a.dry_run:
        for relative, name, cfg, input_path, iv, _ in tasks:
            directory = Path("OUTPUT") / relative
            source, output = directory / "input.mlir", directory / "kernel.ll"
            prepare = [sys.executable, str(Path(__file__).resolve()), "--input", str(input_path),
                       "--loop-iv", iv or "auto", "--prepare-input", str(source),
                       "--set", "pipeline-stages=" + (cfg["pipeline-stages"] if cfg else "keep"),
                       "--set", "loop-mark=" + (cfg["loop-mark"] if cfg else "keep")]
            print(f"# {relative}\n" + shlex.join(prepare) + "\n" + shlex.join(command(cfg, output, source)))
            if cfg and int(cfg["lds-bytes"]):
                print(shlex.join([sys.executable, str(Path(__file__).resolve()), "--pad-lds",
                                  str(output), "--set", "lds-bytes=" + cfg["lds-bytes"]]))
            if a.hsaco:
                metadata, build, launch = downstream_commands(directory)
                print(shlex.join(metadata))
                print("(cd -- " + shlex.quote(str(directory)) + " && " + shlex.join(build) +
                      ") > " + shlex.quote(str(directory / "hsaco.log")) + " 2>&1")
                if a.benchmark:
                    print(shlex.join(launch) + " > " +
                          shlex.quote(str(directory / "benchmark.log")) + " 2>&1")
        return 0

    for enabled, script in ((a.hsaco, a.gen_hsaco), (a.benchmark, a.launcher)):
        if enabled and not script.is_file():
            p.error(f"脚本不存在：{script}")
    summary_dir = a.summary_dir.resolve() if a.summary_dir else None
    if summary_dir:
        if any((summary_dir / name).exists() for name in ("summary.json", "best_by_shape.json")):
            p.error("--summary-dir 已有结果；请选择新的目录，避免覆盖其他运行")
        summary_dir.mkdir(parents=True, exist_ok=True)
    if a.build:
        build_dir = tool.parent.parent
        if tool.name != "MyTest" or tool.parent.name != "test" or not (build_dir / "CMakeCache.txt").is_file():
            p.error("--build 要求 --tool 位于含 CMakeCache.txt 的 build/test/MyTest")
        subprocess.run(["cmake", "--build", str(build_dir), "--target", "MyTest",
                        "-j", str(a.jobs)], check=True)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        p.error(f"MyTest 不可执行：{tool}；可尝试 --build")
    assembler = a.llvm_as.resolve() if a.llvm_as else None
    if assembler and (not assembler.is_file() or not os.access(assembler, os.X_OK)):
        p.error(f"llvm-as 不可执行：{assembler}")
    out = (a.output_dir or ROOT / "generated_ir/tune_attention").resolve()
    out.mkdir(parents=True, exist_ok=True)
    # 独立目录避免覆盖用户手动修改过的 LLVM IR；每候选 cwd 也独立。
    run = Path(tempfile.mkdtemp(prefix=time.strftime("run-%Y%m%d-%H%M%S-"), dir=out))
    input_records = []
    for prefix, input_path, original_source, iv in snapshots:
        snapshot = run / prefix / "input.mlir"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(original_source)
        input_records.append({"input": str(input_path), "snapshot": str(snapshot), "loop_iv": iv,
                              "input_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    (run / "parameters.txt").write_text(HELP, encoding="utf-8")
    (run / "run.json").write_text(json.dumps({
        "argv": argv, "input": str(paths[0]) if len(paths) == 1 else None, "loop_iv": a.loop_iv,
        "sweep_base": (a.sweep_base or "baseline") if axes else None,
        "input_sha256": input_records[0]["input_sha256"] if len(paths) == 1 else None,
        "inputs": input_records,
        "tool": str(tool), "tool_mtime_ns": tool.stat().st_mtime_ns,
        "llvm_as": str(assembler) if assembler else None, "hardware_tested": False,
        "hsaco_requested": a.hsaco, "benchmark_requested": a.benchmark,
        "backend_environment": {key: os.environ.get(key) for key in
                                ("LLVM_DIR", "ROCM_BC", "GPU_ARCH", "GPU_FEATURES", "DWARF_VERSION")},
    }, indent=2) + "\n")
    print(f"输出目录：{run}", flush=True)
    failed = 0
    hardware_tested = False
    # 候选按顺序编译，--jobs 仅控制前面的 CMake 构建并发数。
    # 每候选提交一个完整 JSON 检查点；不反复重写包含全部候选的大列表。
    with LiveResults(run, len(tasks), summary_dir) as live:
        for index, (relative, name, cfg, input_path, iv, prepared) in enumerate(tasks, 1):
            directory = run / relative
            directory.mkdir()
            output, logpath = directory / "kernel.ll", directory / "lowering.log"
            candidate_input = directory / "input.mlir"
            candidate_input.write_text(prepared)
            input_digest = hashlib.sha256(candidate_input.read_bytes()).hexdigest()
            cmd = command(cfg, output, candidate_input)
            replay = directory / "command.sh"
            padding = ([sys.executable, str(Path(__file__).resolve()), "--pad-lds", str(output),
                        "--set", "lds-bytes=" + cfg["lds-bytes"]] if cfg and int(cfg["lds-bytes"]) else [])
            metadata_cmd, build_cmd, launch_cmd = downstream_commands(directory)
            downstream = ""
            if a.hsaco:
                downstream = shlex.join(metadata_cmd) + "\n" + shlex.join(build_cmd) + " > hsaco.log 2>&1\n"
                if a.benchmark:
                    downstream += "rm -f -- benchmark.json\n" + shlex.join(launch_cmd) + " > benchmark.log 2>&1\n"
            replay.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd -- " + shlex.quote(str(directory)) +
                              "\n" + shlex.join(cmd) + " > " + shlex.quote(str(logpath)) + " 2>&1\n" +
                              (shlex.join(padding) + "\n" if padding else "") + downstream)
            replay.chmod(0o755)
            (directory / "config.json").write_text(json.dumps({"case": name, "parameters": cfg,
                "compiler_defaults": cfg is None, "command": cmd, "postprocess_command": padding,
                "input": str(candidate_input), "input_sha256": input_digest,
                "source_input": str(input_path),
                "hsaco_command": build_cmd if a.hsaco else None,
                "launch_command": launch_cmd if a.benchmark else None,
                "loop_iv": iv, "hardware_tested": False}, indent=2) + "\n")
            started = time.monotonic()
            print(f"[{index}/{len(tasks)}] {relative}", flush=True)
            status = "exported"
            facts = {}
            native_lds = None
            config = None
            with logpath.open("w") as log:
                try:
                    result = subprocess.run(cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=a.timeout)
                    if result.returncode:
                        status = f"failed:{result.returncode}"
                    elif not output.is_file() or output.stat().st_size == 0:
                        status = "missing-ir"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                if status == "exported":
                    # MyTest 的部分 PassManager 返回值未传播，不能只看进程退出码。
                    diagnostics = logpath.read_text(errors="replace")
                    if re.search(r"\berror:|LLVM ERROR|Assertion .*failed", diagnostics):
                        status = "diagnostic-error"
                    elif not re.search(r"define\s+amdgpu_kernel\s+void\s+@Attn_p2\b",
                                       output.read_text(errors="replace")):
                        status = "missing-kernel"
                if status == "exported":
                    text = output.read_text()
                    native_lds = lds_bytes(text)
                    try:
                        padded = pad_lds(text, int(cfg["lds-bytes"]) if cfg else 0)
                        if padded != text:
                            (directory / "kernel.native.ll").write_text(text)
                            output.write_text(padded)
                        facts = ir_facts(padded, diagnostics)
                    except ValueError as exc:
                        log.write(f"\nLDS capacity control failed: {exc}\n")
                        status = "invalid-lds-control"
                if status == "exported" and assembler:
                    try:
                        check = subprocess.run([str(assembler), str(output), "-o", os.devnull],
                                               stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                        status = "verified" if check.returncode == 0 else "invalid-ir"
                    except subprocess.TimeoutExpired:
                        status = "verify-timeout"
            measurements = {"hardware_tested": False, "benchmark_attempted": False}
            if status in ("exported", "verified") and a.hsaco:
                try:
                    config = launch_config(prepared, output.read_text(), a.k_layout)
                    (directory / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
                except ValueError as exc:
                    (directory / "hsaco.log").write_text(str(exc) + "\n")
                    status = "invalid-launch-config"
                else:
                    error = run_stage(build_cmd, directory, directory / "hsaco.log", a.timeout)
                    hsaco = directory / "kernel.hsaco"
                    status = "hsaco-" + error if error else "hsaco-built"
                    if not error and (not hsaco.is_file() or hsaco.stat().st_size == 0):
                        status = "missing-hsaco"
                    if status == "hsaco-built":
                        measurements["hsaco"] = str(hsaco)
                        measurements["hsaco_sha256"] = hashlib.sha256(hsaco.read_bytes()).hexdigest()
            if status == "hsaco-built" and a.benchmark:
                measurements["benchmark_attempted"] = True
                error = run_stage(launch_cmd, directory, directory / "benchmark.log", a.timeout)
                report_path = directory / "benchmark.json"
                status = "benchmark-" + error if error else "missing-benchmark-report"
                if report_path.is_file():
                    try:
                        report = json.loads(report_path.read_text())
                        if type(report["passed"]) is not bool:
                            raise ValueError("passed 必须是布尔值")
                        timings = [report["kernel_median_ms"], report["sdpa_median_ms"]]
                        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in timings):
                            raise ValueError("median 耗时必须为有限正数")
                        if not math.isfinite(timings[1] / timings[0]):
                            raise ValueError("speedup 必须为有限值")
                        measurements.update(hardware_tested=True, correct=report["passed"],
                                            kernel_median_ms=timings[0], sdpa_median_ms=timings[1],
                                            speedup=timings[1] / timings[0])
                        hardware_tested = True
                        if not report["passed"]:
                            status = "incorrect"
                        elif not error:
                            status = "benchmark-passed"
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        with (directory / "benchmark.log").open("a") as log:
                            log.write(f"\n无效 benchmark.json: {exc}\n")
                        status = "invalid-benchmark-report"
            # 相同哈希说明导出的 IR 字节相同；不代表不同 IR 一定有性能差异。
            # exported/verified 只反映导出/llvm-as 验证结果，不反映 GPU 正确性或速度。
            digest = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else ""
            normalized_digest = (hashlib.sha256(
                output.read_text().replace(str(directory), "<candidate>").encode()).hexdigest()
                if output.is_file() else "")
            seconds = time.monotonic() - started
            live.record({
                "schema_version": 2, "candidate_index": index, "case": name,
                "directory": str(directory), "source_input": str(input_path),
                "input": str(candidate_input), "llvm_ir": str(output), "log": str(logpath),
                "parameters": cfg, "compiler_defaults": cfg is None, "launch": config,
                "loop_iv": iv,
                "status": status, "seconds": seconds, "sha256": digest, "normalized_sha256": normalized_digest,
                "input_sha256": input_digest, "native_lds_bytes": native_lds, **facts,
                **measurements})
            failed += status not in ("exported", "verified", "hsaco-built", "benchmark-passed")
            print(f"  {status}", flush=True)
    run_record = json.loads((run / "run.json").read_text())
    run_record["hardware_tested"] = hardware_tested
    (run / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    print(f"完成：{len(tasks) - failed} 成功，{failed} 失败；汇总：{run / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n调优已中断；已完成候选及最大加速比保留在 JSON 结果中。", file=sys.stderr)
        sys.exit(130)
