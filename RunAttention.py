#!/usr/bin/env python3
"""BHSD / causal -> input MLIR -> MyTest LLVM -> GenHsaco -> launchKernel.

Examples (run from any directory):
  python3 RunAttention.py --b 1 --h 32 --s 4096 --d 128 --causal true
  python3 RunAttention.py --s 1024 4096 --d 64 128 --causal both
  python3 RunAttention.py --shape 1,32,4096,128 --shape 2,8,1024,64 --non-causal
  python3 RunAttention.py --stage llvm --case baseline --set threads=256

--b/--h/--s/--d and --bm/--bn/--bk lists form a Cartesian product.
Use repeated --shape BHSD to select exact shapes instead.
Default tiles: BM=128, BN=128, BK=32, online softmax, TuneAttention baseline.
S may be any positive integer; tail tiles use checked IO and key masking.
BK must divide D; backend hardware limits still apply.
GenHsaco inherits LLVM_DIR, ROCM_BC, GPU_ARCH, GPU_FEATURES, DWARF_VERSION.
--launch-python selects the Python environment with HIP PyTorch.
"""
import argparse
import itertools
import json
from pathlib import Path
import signal
import sys
import tempfile

import TuneAttention
from generate_attention_ir import ROOT, TEMPLATE, generate_ir


def parse_shape(value):
    try:
        dimensions = tuple(int(x) for x in value.lower().replace("x", ",").split(","))
    except ValueError:
        dimensions = ()
    if len(dimensions) != 4 or any(x <= 0 for x in dimensions):
        raise argparse.ArgumentTypeError("shape 必须为四个正整数 B,H,S,D，例如 1,32,4096,128")
    return dimensions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=parse_shape, action="append", help="BHSD，可重复")
    for name in ("b", "h", "s", "d"):
        parser.add_argument(f"--{name}", f"--{name.upper()}", type=int, nargs="+")
    for name, default in (("bm", 128), ("bn", 128), ("bk", 32)):
        parser.add_argument(f"--{name}", f"--{name.upper()}", type=int, nargs="+", default=[default])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--causal", "--casual", nargs="?", const="true", default="true",
                      choices=("true", "false", "1", "0", "both"), help="默认 true；兼容 casual 拼写")
    mode.add_argument("--non-causal", dest="causal", action="store_const", const="false")
    parser.add_argument("--softmax", choices=("online", "legacy"), default="online")
    parser.add_argument("--stage", choices=("ir", "llvm", "hsaco", "launch"), default="launch",
                        help="执行到指定阶段，默认 launch（包含正确性校验和计时）")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_ir/attention_runs")
    parser.add_argument("--case", default="baseline", help="TuneAttention 编译预设，默认 baseline")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖编译参数，如 --set qk-tile=64；支持键见 TuneAttention.py --help")
    parser.add_argument("--tool", type=Path, default=TuneAttention.default_paths()[1], help="MyTest 路径")
    parser.add_argument("--build", action="store_true", help="先通过现有 CMake build 构建 MyTest")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300, help="每个编译/启动阶段超时秒数")
    gen_hsaco = next((ROOT / name for name in ("GenHsaco.sh", "genHsaco.sh")
                      if (ROOT / name).is_file()), ROOT / "GenHsaco.sh")
    parser.add_argument("--gen-hsaco", type=Path, default=gen_hsaco)
    parser.add_argument("--launcher", type=Path, default=ROOT / "launchKernel.py")
    parser.add_argument("--launch-python", default=sys.executable)
    parser.add_argument("--opt-level", choices=("0", "1", "2", "3"), default="3")
    parser.add_argument("--llvm-as", type=Path, help="可选 LLVM IR 验证工具")
    parser.add_argument("--k-layout", choices=("bhsd", "bhds"), default="bhsd",
                        help="默认 bhsd；bhds 仅用于未启用 K normalization 的旧版 --tool")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.shape and any(getattr(args, key) is not None for key in ("b", "h", "s", "d")):
        parser.error("--shape 与 --b/--h/--s/--d 不能同时使用")
    if args.jobs < 1 or not 0 < args.timeout < float("inf"):
        parser.error("jobs 和 timeout 必须为有限正数")
    if args.device < 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("device/warmup 必须非负，repeat 必须为正数")
    shapes = args.shape or itertools.product(args.b or [1], args.h or [32],
                                             args.s or [4096], args.d or [128])
    modes = [True, False] if args.causal == "both" else [args.causal in ("true", "1")]
    # Validate every input before creating a run directory; duplicate values
    # must not silently overwrite a configuration in a Cartesian sweep.
    generated = []
    try:
        source = TEMPLATE.read_text()
        for shape, bm, bn, bk, causal in itertools.product(
                dict.fromkeys(shapes), dict.fromkeys(args.bm), dict.fromkeys(args.bn),
                dict.fromkeys(args.bk), modes):
            b, h, s, d = shape
            ir, _ = generate_ir(source, bm, bn, bk, args.softmax,
                                b=b, h=h, s=s, d=d, causal=causal)
            mode_name = "causal" if causal else "noncausal"
            name = f"{'x'.join(map(str, shape))}-{bm}-{bn}-{bk}-{args.softmax}-{mode_name}"
            generated.append((name, ir, dict(shape=shape, bm=bm, bn=bn, bk=bk, causal=causal)))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
        inputs = run / "inputs"
        inputs.mkdir()
        records = []
        for name, ir, config in generated:
            path = inputs / (name + ".mlir")
            path.write_text(ir)
            records.append(dict(config, input=str(path)))
        (run / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"运行目录：{run}\n输入数：{len(records)}；阶段：{args.stage}", flush=True)
    if args.stage == "ir":
        return 0

    # Reuse the established compiler flags, isolated per-candidate working
    # directories, diagnostic checks, timeout handling and benchmark reports.
    command = ["--input-dir", str(inputs), "--output-dir", str(run / "compiled"),
               "--summary-dir", str(run / "summary")]
    for key in ("case", "tool", "jobs", "timeout", "gen_hsaco", "launcher", "launch_python",
                "opt_level", "k_layout", "device", "warmup", "repeat", "seed"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    for assignment in args.set:
        command += ["--set", assignment]
    if args.build:
        command.append("--build")
    if args.llvm_as:
        command += ["--llvm-as", str(args.llvm_as)]
    if args.stage == "hsaco":
        command.append("--hsaco")
    elif args.stage == "launch":
        command.append("--benchmark")
    (run / "pipeline.json").write_text(json.dumps(
        dict(stage=args.stage, tune_attention_argv=command), indent=2) + "\n")
    return TuneAttention.main(command)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, TuneAttention.interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n运行已中断，已完成的产物和日志保留在运行目录。", file=sys.stderr)
        sys.exit(130)
