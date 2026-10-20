#!/usr/bin/env python3
"""生成 split-D attention 的 MyTest 调优候选并导出 LLVM IR。

使用 Python 标准库，不编译 HSACO、不运行 GPU benchmark，也不自动评选最快配置。
运行 `python3 TuneAttention.py --help` 查看参数和示例。
"""

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
import tempfile

# 默认路径相对于脚本所在仓库，因而也可以从其他工作目录启动。
ROOT = Path(__file__).resolve().parent
HELP = r"""
默认输入：3rd/deepgengraph/test/test_input_spidD.mlir。
默认 baseline 为当前实测最快的 nohint 展开配置（加速比 0.51737，仅适用于
本次测试环境）；仍需在目标 GPU 上验证。脚本只导出 IR，不自动运行或评选。

三个独立控制：
  pipeline=0|1                 MyTest 的旧位置参数，控制手工 pipeline 调度。
                               0 不会关闭 frisk.pipeline_hint 自动流水线！
  pipeline-stages=none|keep|N   修改目标循环的 frisk.pipeline_hint；none 删除，
                               keep 保留输入，N 为 1..64；1 禁用，2 双缓冲。
  loop-mark=0|1|keep           修改目标循环的 frisk.simplify_loop；1 添加，
                               允许 nohint 循环独立展开；0 删除，keep 保留。
  simplify-loop-control=0|1    是否运行循环简化 Pass（默认 1）。
  coalesce-waits=0|1           是否合并重复 wait/barrier（此处默认 0）。
目标用 --loop-iv dk 指定，只修改唯一的 scf.for %dk；缺失或不唯一时报错。
其他循环的属性保留。每候选保存 input.mlir，绝不改写原始输入。

完整 MyTest 参数（下列顺序固定）：
  input pipeline output reorder pack layout carrier math threads waves \
  lifetimes masks prefetch fuse reuse dma-pack dma-export qk-tile vo-tile \
  --frag-prefetch=N --frag-prefetch-words=N \
  --simplify-loop-control=N --coalesce-waits=N

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
  lds-bytes：0=不干预；正数将单个 i8 LDS 池扩大到指定字节数（不允许缩小）。
    仅用于容量控制实验，不改变访存偏移；多 LDS 池时拒绝，须检查目标汇编
    group_segment_fixed_size 确认最终分配。此项不是 MyTest 参数。
环境变量由脚本显式设置，不继承 shell 的同名值。

预设（--case 接受 glob，可以重复）：
  baseline / nohint-unrolled：当前起点，nohint + DK 展开，原生 LDS 16 KiB。
  nohint-rolled：无 hint、不展开，对照旧 nohint。
  nohint-unrolled-32k：同 nohint-unrolled，LDS 扩至 32 KiB。
  pipeline-baseline / pipeline-wait-only / pipeline-loop-only / pipeline-combined：
    hint=2，分别关闭两项清理 / 仅等待合并 / 仅循环简化 / 两项开启。
  pipeline-loop-only-40k：仅循环简化 + LDS 扩至 40 KiB。
  legacy-baseline：保留旧脚本的 strict/legacy 参数，保留输入原有标记。
  k5*：保留历史组合参数；mytest-default：省略全部可选 MyTest 参数，保留输入。
    mytest-default 不接受 --set，列表里使用 compiler-default 标明未传参数。
  coverage 还包含单因素对照、合法 threads/waves 组合及部分预算扫描。
  --sweep 只生成指定轴的笛卡尔积；优先级 sweep > set > 预设。
  不合法的 threads/waves 或 copy/dma-pack 组合自动过滤；lowering 失败记录并继续。

示例：
  python3 TuneAttention.py --case baseline --dry-run
  python3 TuneAttention.py --case baseline --llvm-as /data2/xsl/install/bin/llvm-as
  python3 TuneAttention.py --case 'nohint-*' --case 'pipeline-*'
  python3 TuneAttention.py --sweep 'pipeline-stages=none;2;3' \
    --sweep 'simplify-loop-control=0;1' --sweep 'coalesce-waits=0;1'
  python3 TuneAttention.py --sweep 'qk-tile=16;32;64' --sweep 'vo-tile=16;32;64'
  python3 TuneAttention.py --case baseline --set math=strict
  python3 TuneAttention.py --case mytest-default --build

每次运行新建 run-*；每候选保存 input.mlir、kernel.ll、lowering.log、command.sh、
config.json、result.json。manifest.tsv 包含输入/IR SHA256、静态 LDS 字节数、
实际流水线标记数和展开循环数。normalized_sha256 仅归一化候选目录路径，
便于发现仅源文件路径不同的重复 IR；哈希不是性能结论。
--llvm-as 验证最终（含容量调整后）IR。性能和数值正确性仍需 GPU 实测。
"""

# 保留原脚本基线，避免历史 k5 预设随新基线一起漂移。
LEGACY_BASE = dict(
    pipeline="0", reorder="1", pack="1", layout="3.3", carrier="34816",
    math="strict", threads="256", waves="4,1", lifetimes="1", masks="0",
    prefetch="0", fuse="1", reuse="0", copy="legacy", alignment="0",
    **{"dma-pack": "legacy", "dma-export": "auto", "qk-tile": "0", "vo-tile": "0",
       "frag-prefetch": "0", "frag-prefetch-words": "8",
       "pipeline-stages": "keep", "loop-mark": "keep",
       "simplify-loop-control": "1", "coalesce-waits": "1", "lds-bytes": "0"})
BASE = dict(LEGACY_BASE, math="approx", masks="1", prefetch="1", fuse="0",
            copy="async", alignment="16",
            **{"dma-pack": "useKPack", "dma-export": "asm128", "qk-tile": "32",
               "vo-tile": "32", "frag-prefetch": "1", "frag-prefetch-words": "4",
               "pipeline-stages": "none", "loop-mark": "1", "coalesce-waits": "0"})
KEYS = list(BASE)
POSITIONAL_KEYS = (
    "reorder", "pack", "layout", "carrier", "math", "threads", "waves",
    "lifetimes", "masks", "prefetch", "fuse", "reuse", "dma-pack", "dma-export",
    "qk-tile", "vo-tile",
)
NAMED_KEYS = ("frag-prefetch", "frag-prefetch-words", "simplify-loop-control", "coalesce-waits")
BOOLS = ("pipeline", "reorder", "pack", "lifetimes", "masks", "prefetch", "fuse",
         "frag-prefetch", "simplify-loop-control", "coalesce-waits")
ENUMS = {"layout": ("3.1", "3.2", "3.3"),
         "math": ("strict", "mmac", "mmac-exp2", "mmac-recip", "approx"),
         "copy": ("legacy", "async", "sync", "hybrid"),
         "dma-pack": ("legacy", "useKPack", "noKPack"),
         "dma-export": ("auto", "split32", "hcu128", "asm128"),
         "loop-mark": ("keep", "0", "1")}
ALIASES = {"shared": "3.1", "shuffle": "3.2", "b-permute": "3.3"}

# Mask comments/strings without changing offsets, so braces in them are inert.
LEXICAL = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"')


def masked(text):
    return LEXICAL.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)


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
            "simplified_loops": sum(map(int, counts))}


def main(argv=None):
    """解析配置、生成候选，再按顺序导出；导入模块时不启动调优。"""
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="TuneAttention.py", description="生成 split-D attention 的 MyTest 调优候选。",
                                epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "3rd/deepgengraph/test/test_input_spidD.mlir")
    p.add_argument("--tool", type=Path, default=ROOT / "3rd/deepgengraph/build/test/MyTest")
    p.add_argument("--loop-iv", default="dk", help="待修改标记的唯一 scf.for IV 名，不带 %%；默认 dk")
    p.add_argument("--output-dir", type=Path, help="父目录；每次建立独立 run-* 子目录")
    p.add_argument("--case", action="append", default=[], help="候选名或 glob，可重复")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖候选参数")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1;V2", help="组合扫描轴，可重复")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="只列出候选，不创建文件")
    mode.add_argument("--dry-run", action="store_true", help="只打印命令，不创建文件")
    mode.add_argument("--prepare-input", type=Path, metavar="OUTPUT", help="只写候选输入副本，供 dry-run/replay 使用")
    mode.add_argument("--pad-lds", type=Path, metavar="IR", help="仅对指定 IR 应用 --set lds-bytes=N")
    p.add_argument("--build", action="store_true", help="先构建默认 build 目录中的 MyTest")
    p.add_argument("--jobs", type=int, default=2, help="构建并发数，默认 2；候选串行生成")
    p.add_argument("--timeout", type=float, default=300, help="每个候选编译超时秒数，默认 300")
    p.add_argument("--llvm-as", type=Path, help="可选的 LLVM IR 语法/验证工具")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[\w.$-]+", a.loop_iv):
        p.error("loop-iv 应为不带 % 的 SSA 名")
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
        if key == "pipeline-stages":
            if value not in ("none", "keep"):
                if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= 64:
                    p.error("pipeline-stages 必须为 none、keep 或 1..64")
                value = str(int(value))
        limits = {"carrier": 2**64, "reuse": 2**32, "threads": 2**32,
                  "alignment": 2**63, "qk-tile": 2**63, "vo-tile": 2**63,
                  "frag-prefetch-words": 2**32, "lds-bytes": 2**32}
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

    if a.prepare_input or a.pad_lds:
        if a.case or axes or a.build:
            p.error("--prepare-input/--pad-lds 不接受 --case、--sweep 或 --build")
        cfg = dict(BASE, **overrides)
        try:
            if a.prepare_input:
                if a.prepare_input.resolve() == a.input.resolve():
                    p.error("不允许覆盖原始输入；请选择不同的 --prepare-input 路径")
                content = prepare_input(a.input.read_text(), cfg, a.loop_iv)
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
        yield "legacy-baseline", LEGACY_BASE.copy()
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
        dimensions = {k: ("0", "1") for k in BOOLS}
        dimensions.update(ENUMS)
        dimensions.update(carrier=("0", "24576", "28672", "34816"), reuse=("0", "4", "8", "16"))
        dimensions.update({"pipeline-stages": ("none", "1", "2", "3", "4"),
                           "qk-tile": ("16", "32", "64"), "vo-tile": ("16", "32", "64"),
                           "frag-prefetch-words": ("4", "8", "16")})
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

    def candidates():
        # 指定 --sweep 后只生成各扫描轴的笛卡尔积，不再追加默认 coverage。
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
            # 优先级：--sweep > --set > 候选原值；未扫描的参数才应用 --set。
            cfg.update({k: v for k, v in overrides.items() if k not in axes})
            m, n = map(int, cfg["waves"].split(","))
            if (m * n != int(cfg["threads"]) // 64 or
                    (cfg["dma-pack"] != "legacy" and cfg["copy"] != "async")):
                filtered += 1
                continue
        selected.append((name, cfg))
    if not selected:
        p.error("没有匹配且 threads/waves、copy/dma-pack 合法的候选；用 --list 查看名称")
    print(f"候选数：{len(selected)}；过滤不合法的参数组合：{filtered}", file=sys.stderr)

    input_path, tool = a.input.resolve(), a.tool.resolve()
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
            cmd += [f"--{key}={cfg[key]}" for key in NAMED_KEYS]
        return cmd

    # 预览模式在构建、创建目录之前退出，可先检查配置和实际编译命令。
    if a.list:
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        writer.writerow(["case"] + KEYS)
        for name, cfg in selected:
            writer.writerow([name] + [cfg[k] if cfg else "compiler-default" for k in KEYS])
        return 0
    if a.dry_run:
        for name, cfg in selected:
            directory = Path("OUTPUT") / name
            source, output = directory / "input.mlir", directory / "kernel.ll"
            prepare = [sys.executable, str(Path(__file__).resolve()), "--input", str(input_path),
                       "--loop-iv", a.loop_iv, "--prepare-input", str(source),
                       "--set", "pipeline-stages=" + (cfg["pipeline-stages"] if cfg else "keep"),
                       "--set", "loop-mark=" + (cfg["loop-mark"] if cfg else "keep")]
            print(f"# {name}\n" + shlex.join(prepare) + "\n" + shlex.join(command(cfg, output, source)))
            if cfg and int(cfg["lds-bytes"]):
                print(shlex.join([sys.executable, str(Path(__file__).resolve()), "--pad-lds",
                                  str(output), "--set", "lds-bytes=" + cfg["lds-bytes"]]))
        return 0

    if not input_path.is_file():
        p.error(f"输入不存在：{input_path}")
    original_source = input_path.read_text()
    prepared = {}
    for name, cfg in selected:
        try:
            prepared[name] = prepare_input(original_source, cfg, a.loop_iv)
        except ValueError as exc:
            p.error(f"{name}: {exc}")
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
    run = Path(tempfile.mkdtemp(prefix=time.strftime("run-%Y%m%d-%H%M%S-"), dir=out))
    shutil.copy2(input_path, run / "input.mlir")
    (run / "parameters.txt").write_text(HELP, encoding="utf-8")
    (run / "run.json").write_text(json.dumps({
        "argv": argv, "input": str(input_path), "loop_iv": a.loop_iv,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "tool": str(tool), "tool_mtime_ns": tool.stat().st_mtime_ns,
        "llvm_as": str(assembler) if assembler else None, "hardware_tested": False,
    }, indent=2) + "\n")
    print(f"输出目录：{run}", flush=True)
    failed = 0
    # 候选按顺序编译，--jobs 仅控制前面的 CMake 构建并发数。
    # 单个候选失败/超时后继续，并将状态及时写入 manifest.tsv。
    with (run / "manifest.tsv").open("w", newline="") as manifest:
        writer = csv.writer(manifest, delimiter="\t", lineterminator="\n")
        writer.writerow(["case", "status", "seconds", "sha256", "normalized_sha256", "input_sha256",
                         "static_lds_bytes", "scheduled_loops", "simplified_loops", "llvm_ir", "log"] + KEYS)
        for index, (name, cfg) in enumerate(selected, 1):
            directory = run / name
            directory.mkdir()
            output, logpath = directory / "kernel.ll", directory / "lowering.log"
            candidate_input = directory / "input.mlir"
            candidate_input.write_text(prepared[name])
            input_digest = hashlib.sha256(candidate_input.read_bytes()).hexdigest()
            cmd = command(cfg, output, candidate_input)
            replay = directory / "command.sh"
            padding = ([sys.executable, str(Path(__file__).resolve()), "--pad-lds", str(output),
                        "--set", "lds-bytes=" + cfg["lds-bytes"]] if cfg and int(cfg["lds-bytes"]) else [])
            replay.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd -- " + shlex.quote(str(directory)) +
                              "\n" + shlex.join(cmd) + " > " + shlex.quote(str(logpath)) + " 2>&1\n" +
                              (shlex.join(padding) + "\n" if padding else ""))
            replay.chmod(0o755)
            (directory / "config.json").write_text(json.dumps({"case": name, "parameters": cfg,
                "compiler_defaults": cfg is None, "command": cmd, "postprocess_command": padding,
                "input": str(candidate_input), "input_sha256": input_digest,
                "loop_iv": a.loop_iv, "hardware_tested": False}, indent=2) + "\n")
            started = time.monotonic()
            print(f"[{index}/{len(selected)}] {name}", flush=True)
            status = "exported"
            facts = {}
            native_lds = None
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
            # 相同哈希说明导出的 IR 字节相同；不代表不同 IR 一定有性能差异。
            # exported/verified 只反映导出/llvm-as 验证结果，不反映 GPU 正确性或速度。
            digest = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else ""
            normalized_digest = (hashlib.sha256(
                output.read_text().replace(str(directory), "<candidate>").encode()).hexdigest()
                if output.is_file() else "")
            seconds = time.monotonic() - started
            (directory / "result.json").write_text(json.dumps({
                "status": status, "seconds": seconds, "sha256": digest, "normalized_sha256": normalized_digest,
                "input_sha256": input_digest, "native_lds_bytes": native_lds, **facts,
                "hardware_tested": False}, indent=2) + "\n")
            writer.writerow([name, status, f"{seconds:.2f}", digest, normalized_digest, input_digest,
                             facts.get("static_lds_bytes"), facts.get("scheduled_loops"),
                             facts.get("simplified_loops"),
                             str(output), str(logpath)] + [(cfg or {}).get(k, "compiler-default") for k in KEYS])
            manifest.flush()
            failed += status not in ("exported", "verified")
            print(f"  {status}", flush=True)
    print(f"完成：{len(selected) - failed} 成功，{failed} 失败；汇总：{run / 'manifest.tsv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
