#!/usr/bin/env bash
# MyTest 参数目录与 split-D attention LLVM IR 批量导出工具。
# 用法和所有参数说明：./TuneAttention.sh --help
# Python 仅使用标准库；不编译 HSACO、不运行 GPU benchmark。
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 - "$repo_dir" "$@" <<'PY'
import argparse
import csv
import fnmatch
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(sys.argv[1])
HELP = r"""
默认输入：3rd/deepgengraph/test/test_input_spidD.mlir
该输入固定 BM=128、BN=128、BK=32、D=128、S=4096；这些不是 MyTest 参数。
参数依据：3rd/deepgengraph/test/test_src.cpp。

MyTest 完整调用顺序（下列编号是 argv 下标）：
  MyTest input pipeline output reorder pack layout carrier math threads waves \
         lifetimes masks prefetch fuse reuse

位置  本脚本参数名  MyTest 默认值 / 取值和含义
 1    --input      输入 MLIR 路径。
 2    pipeline     0；0/1：是否运行软件流水调度；仅对带 pipeline 标注的循环生效。
                   MyTest 会尝试标注特定 Attn_p2 seqLen 循环；split-D 的嵌套
                   QK 循环不一定匹配，1 不保证生成不同 IR，须检查 .log。
                   当前实测 baseline 的 0/1 导出 IR 哈希相同。
 3    output       输出 LLVM IR；脚本自动分配为每个候选目录中的 kernel.ll。
 4    reorder      1；0/1：IRDeepOptimize 中展开后的指令重排。
                   不控制其他阶段的 fragment 重排。
 5    pack         1；0/1：规划 GEMM operand 的 packed LDS 布局。
 6    layout       3.2；3.1/shared：P 的 reg→LDS→reg 布局转换；
                   3.2/shuffle：wave 内 shuffle；3.3/b-permute：生产者 B
                   排列策略，不支持的形状回退 shuffle；终端 C 输出保留 shuffle。
 7    carrier      24576；非负 64 位整数，单位 32-bit words/block；
                   tensor 循环变换的 carrier 预算，0=不限；不是 VGPR 硬上限。
 8    math         strict：不开启额外寄存器数学优化；
                   mmac：融合 MMAC 累加，允许浮点累加重结合；
                   mmac-exp2：mmac + 原生 amdgcn.exp2（含次正规数行为差异）；
                   mmac-recip：mmac + 共享倒数乘法替代归一化除法；
                   approx：上述三项全部开启。除 strict 外需要验证数值误差。
 9    threads      128；64,128,...,1024，每 block 线程数；DCU wave=64。
10    waves        省略时自动因数分解；显式 M,N 均为 1..16，
                   M*N=threads/64；代表 GEMM M/N 方向的 wave 排列。
11    lifetimes    1；0/1：优化地址/数学计算调度以缩短寄存器存活期。
12    masks        0；0/1：对可证明完全有效的 tile 跳过逐元素上三角 mask。
13    prefetch     0；0/1：将 global packet 提前加载到普通寄存器；
                   当前内部默认预算为 8 words，MyTest 未暴露该预算参数。
14    fuse         1；0/1：将单次使用的 mask 与相邻加法融合。
15    reuse        0；非负 32 位整数，32-bit words/thread；0=禁用；
                   限制 LDS fragment 复用时同时保留在寄存器中的数据量。

环境变量（脚本用 --set/--sweep 显式控制，不继承 shell 中的同名变量）：
  copy → FRISK_GM_TO_SHM_MODE，默认 legacy：原有同步拷贝路径；
         async：提升为异步拷贝，无法满足条件的情况走回退；
         sync：先提升，再强制同步 lowering，用于对照；
         hybrid：直接 DMA + 不适合直接 DMA 的 packed half 数据走向量同步拷贝。
  alignment → FRISK_GM_TO_SHM_ALIGNMENT，默认 0：推断元素/已有属性对齐；
         也可为正的 2 次幂（字节）。这是所有被提升 tile 起始地址的调用者保证，
         不是运行时检查。4 是已有 split-D DMA 候选的设定；GPU 调用时须保证。
         legacy 不使用此值；FP16 的默认推断不一定能启用 DMA。
  FRISK_SYNC_DMA_PACKETS 是 RunFriskSHell.sh 的后处理开关，不是 MyTest 参数，
  本脚本不使用它；强制替换 DMA packet 与 copy=sync 的编译路径不同。

覆盖范围：
  默认生成 coverage：基线 + 每项枚举/开关的单因素对照 + 全部合法
  threads/waves 对 + carrier={0,24576,28672,34816} + reuse={0,4,8,16}
  + async/sync/hybrid 的 alignment={0,4} + 已有 k5 组合。
  这些是候选，满足命令行约束不保证当前输入能 lowering；失败会记录并继续。
  例如当前 threads=192、waves=1,3 在 frisk.reduce lowering 失败。
  coverage 不是所有参数的笛卡尔积；整数预算和对齐值也不是全部数值。
  --sweep 可重复指定，生成给定轴的完整笛卡尔积（其余参数取基线）。
  值以分号分隔，避免与 waves=M,N 的逗号冲突；应加 shell 引号。
  baseline 使用 256 threads、4,1、3.3、34816、strict，其余为 MyTest 默认。
  mytest-default 省略可选参数，保留编译器自动 wave 选择；不接受 --set。

示例：
  ./TuneAttention.sh --help
  ./TuneAttention.sh --list                         # 全部候选和参数 TSV
  ./TuneAttention.sh --dry-run                      # 查看可复现的完整命令
  ./TuneAttention.sh --case baseline --build        # 构建 MyTest 并生成基线
  ./TuneAttention.sh --case 'k5*'                    # 生成已有 k5 及拷贝变体
  ./TuneAttention.sh                                # 生成整个 coverage
  ./TuneAttention.sh --case baseline --set math=approx --set reuse=8
  ./TuneAttention.sh --sweep 'layout=3.1;3.2;3.3' \
      --sweep 'math=strict;mmac;mmac-exp2;mmac-recip;approx' \
      --sweep 'copy=legacy;async;sync;hybrid' --set alignment=4
  ./TuneAttention.sh --sweep 'threads=128;256' --sweep 'waves=2,1;4,1' --list
      # 线程数与 wave 乘积不匹配的组合自动过滤。
  ./TuneAttention.sh --case baseline --llvm-as /data2/xsl/install/bin/llvm-as

输出：每次运行新建目录；每候选保存 kernel.ll、lowering.log、command.sh、
config.json；manifest.tsv 汇总状态、参数、IR SHA256，哈希相同可识别无效开关。
使用 --llvm-as 时额外验证 IR；生成成功不代表 GPU 数值正确或速度更快。
"""

# 可直接调整这些基线数值，也可用 --set 覆盖，不需要改 MyTest。
BASE = dict(pipeline="0", reorder="1", pack="1", layout="3.3", carrier="34816",
            math="strict", threads="256", waves="4,1", lifetimes="1", masks="0",
            prefetch="0", fuse="1", reuse="0", copy="legacy", alignment="0")
KEYS = list(BASE)
BOOLS = ("pipeline", "reorder", "pack", "lifetimes", "masks", "prefetch", "fuse")
ENUMS = {"layout": ("3.1", "3.2", "3.3"),
         "math": ("strict", "mmac", "mmac-exp2", "mmac-recip", "approx"),
         "copy": ("legacy", "async", "sync", "hybrid")}
ALIASES = {"shared": "3.1", "shuffle": "3.2", "b-permute": "3.3"}
p = argparse.ArgumentParser(prog="TuneAttention.sh", description="生成 split-D attention 的 MyTest 调优候选。",
                            epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--input", type=Path, default=ROOT / "3rd/deepgengraph/test/test_input_spidD.mlir")
p.add_argument("--tool", type=Path, default=ROOT / "3rd/deepgengraph/build/test/MyTest")
p.add_argument("--output-dir", type=Path, help="父目录；每次建立独立 run-* 子目录")
p.add_argument("--case", action="append", default=[], help="候选名或 glob，可重复")
p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖候选参数")
p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1;V2", help="组合扫描轴，可重复")
mode = p.add_mutually_exclusive_group()
mode.add_argument("--list", action="store_true", help="只列出候选，不创建文件")
mode.add_argument("--dry-run", action="store_true", help="只打印命令，不创建文件")
p.add_argument("--build", action="store_true", help="先构建默认 build 目录中的 MyTest")
p.add_argument("--jobs", type=int, default=2, help="构建并发数，默认 2；候选串行生成")
p.add_argument("--timeout", type=float, default=300, help="每个候选编译超时秒数，默认 300")
p.add_argument("--llvm-as", type=Path, help="可选的 LLVM IR 语法/验证工具")
a = p.parse_args(sys.argv[2:])
if a.jobs < 1 or not (0 < a.timeout < float("inf")):
    p.error("jobs 和 timeout 必须为正数且有限")

def assignment(text):
    key, sep, value = text.partition("=")
    if not sep or key not in BASE or not value:
        p.error("应为 KEY=VALUE；KEY 可取：" + ", ".join(KEYS))
    return key, value

def normalize(key, value):
    if key == "layout":
        value = ALIASES.get(value, value)
    if key in BOOLS and value not in ("0", "1"):
        p.error(f"{key} 必须为 0 或 1")
    if key in ENUMS and value not in ENUMS[key]:
        p.error(f"{key} 可取 {ENUMS[key]}")
    if key in ("carrier", "reuse", "threads", "alignment"):
        if not re.fullmatch(r"[0-9]+", value):
            p.error(f"{key} 必须为非负整数")
        n = int(value)
        limit = {"carrier": 2**64, "reuse": 2**32, "threads": 2**32, "alignment": 2**63}[key]
        if n >= limit:
            p.error(f"{key} 超过 MyTest 整数范围")
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

def coverage():
    yield "mytest-default", None
    yield "baseline", BASE.copy()
    dimensions = {k: ("0", "1") for k in BOOLS}
    dimensions.update(ENUMS)
    dimensions.update(carrier=("0", "24576", "28672", "34816"), reuse=("0", "4", "8", "16"))
    for key, values in dimensions.items():
        for value in values:
            if value != BASE[key]:
                yield f"{key}-{value}", dict(BASE, **{key: value})
    for waves in range(1, 17):
        for m in range(1, waves + 1):
            if waves % m == 0:
                cfg = dict(BASE, threads=str(waves * 64), waves=f"{m},{waves // m}")
                if cfg != BASE:
                    yield f"threads-{waves * 64}-waves-{m}x{waves // m}", cfg
    for copy in ("async", "sync", "hybrid"):
        yield f"{copy}-align4", dict(BASE, copy=copy, alignment="4")
    # 对应 generate_attention_candidates.py 的 step5 和已有 hybrid 示例。
    k5 = dict(BASE, math="approx", masks="1", prefetch="1", fuse="0")
    yield "k5", k5
    for copy in ("async", "sync", "hybrid"):
        yield f"k5-{copy}", dict(k5, copy=copy, alignment="4")
    yield "k5-hybrid-reuse8", dict(k5, copy="hybrid", alignment="4", reuse="8")

def candidates():
    if axes:
        for index, values in enumerate(itertools.product(*axes.values()), 1):
            yield f"grid-{index:06d}", dict(BASE, **dict(zip(axes, values)))
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
        # sweep 轴优先于 --set，使 --set 能固定其他参数。
        cfg.update({k: v for k, v in overrides.items() if k not in axes})
        m, n = map(int, cfg["waves"].split(","))
        if m * n != int(cfg["threads"]) // 64:
            filtered += 1
            continue
    selected.append((name, cfg))
if not selected:
    p.error("没有匹配且 threads/waves 合法的候选；用 --list 查看名称")
print(f"候选数：{len(selected)}；过滤不匹配的 threads/waves：{filtered}", file=sys.stderr)

input_path, tool = a.input.resolve(), a.tool.resolve()
def command(cfg, output):
    env = dict(copy="legacy", alignment="0") if cfg is None else cfg
    cmd = ["env", "FRISK_GM_TO_SHM_MODE=" + env["copy"],
           "FRISK_GM_TO_SHM_ALIGNMENT=" + env["alignment"], str(tool), str(input_path),
           "0" if cfg is None else cfg["pipeline"], str(output)]
    if cfg is not None:
        cmd += [cfg[k] for k in KEYS[1:13]]
    return cmd

if a.list:
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["case"] + KEYS)
    for name, cfg in selected:
        defaults = dict(BASE, threads="128", waves="auto", layout="3.2", carrier="24576")
        writer.writerow([name] + [(cfg or defaults)[k] for k in KEYS])
    sys.exit(0)
if a.dry_run:
    for name, cfg in selected:
        print(f"# {name}\n" + shlex.join(command(cfg, Path("OUTPUT") / name / "kernel.ll")))
    sys.exit(0)

if not input_path.is_file():
    p.error(f"输入不存在：{input_path}")
if a.build:
    default_tool = ROOT / "3rd/deepgengraph/build/test/MyTest"
    if tool != default_tool.resolve():
        p.error("--build 仅支持默认 --tool")
    subprocess.run(["cmake", "--build", str(default_tool.parents[1]), "--target", "MyTest",
                    "-j", str(a.jobs)], check=True)
if not tool.is_file() or not os.access(tool, os.X_OK):
    p.error(f"MyTest 不可执行：{tool}；可尝试 --build")
assembler = a.llvm_as.resolve() if a.llvm_as else None
if assembler and (not assembler.is_file() or not os.access(assembler, os.X_OK)):
    p.error(f"llvm-as 不可执行：{assembler}")
out = (a.output_dir or ROOT / "generated_ir/tune_attention").resolve()
out.mkdir(parents=True, exist_ok=True)
# 独立目录避免覆盖用户手动修改过的 LLVM IR；每候选 cwd 也独立。
import tempfile
run = Path(tempfile.mkdtemp(prefix=time.strftime("run-%Y%m%d-%H%M%S-"), dir=out))
shutil.copy2(input_path, run / "input.mlir")
(run / "parameters.txt").write_text(HELP, encoding="utf-8")
(run / "run.json").write_text(json.dumps({
    "argv": sys.argv[2:], "input": str(input_path),
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "tool": str(tool), "tool_mtime_ns": tool.stat().st_mtime_ns,
    "llvm_as": str(assembler) if assembler else None, "hardware_tested": False,
}, indent=2) + "\n")
print(f"输出目录：{run}", flush=True)
failed = 0
with (run / "manifest.tsv").open("w", newline="") as manifest:
    writer = csv.writer(manifest, delimiter="\t", lineterminator="\n")
    writer.writerow(["case", "status", "seconds", "sha256", "llvm_ir", "log"] + KEYS)
    for index, (name, cfg) in enumerate(selected, 1):
        directory = run / name
        directory.mkdir()
        output, logpath = directory / "kernel.ll", directory / "lowering.log"
        cmd = command(cfg, output)
        replay = directory / "command.sh"
        replay.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd -- " + shlex.quote(str(directory)) +
                          "\n" + shlex.join(cmd) + " > " + shlex.quote(str(logpath)) + " 2>&1\n")
        replay.chmod(0o755)
        (directory / "config.json").write_text(json.dumps({"case": name, "parameters": cfg,
            "compiler_defaults": cfg is None, "command": cmd}, indent=2) + "\n")
        started = time.monotonic()
        print(f"[{index}/{len(selected)}] {name}", flush=True)
        status = "exported"
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
            if status == "exported" and assembler:
                try:
                    check = subprocess.run([str(assembler), str(output), "-o", os.devnull],
                                           stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                    status = "verified" if check.returncode == 0 else "invalid-ir"
                except subprocess.TimeoutExpired:
                    status = "verify-timeout"
        digest = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else ""
        writer.writerow([name, status, f"{time.monotonic() - started:.2f}", digest,
                         str(output), str(logpath)] + [(cfg or {}).get(k, "compiler-default") for k in KEYS])
        manifest.flush()
        failed += status not in ("exported", "verified")
        print(f"  {status}", flush=True)
print(f"完成：{len(selected) - failed} 成功，{failed} 失败；汇总：{run / 'manifest.tsv'}")
sys.exit(1 if failed else 0)
PY
