#!/usr/bin/env python3
"""Build the supplied IR with DTK and compare in rotated, correctness-checked rounds."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(manifest["variants"]),
                        default=list(manifest["variants"]))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launch-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.repeat) < 1 or args.warmup < 0:
        parser.error("rounds/repeat must be positive; warmup must be nonnegative")
    run = HERE / ("run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not args.dry_run:
        run.mkdir()
        shutil.copy2(HERE / "manifest.json", run / "manifest.json")
        (run / "run.json").write_text(json.dumps(dict(arguments=vars(args),
            environment={k: os.environ.get(k) for k in
                         ("LLVM_DIR", "ROCM_BC", "GPU_ARCH", "GPU_FEATURES")},
            launcher_sha256=digest(ROOT / "launchKernel.py"),
            builder_sha256=digest(ROOT / "GenHsaco.sh")), indent=2) + "\n")

    def execute(command, directory, log):
        print(shlex.join(map(str, command)), flush=True)
        if args.dry_run:
            return
        with log.open("w") as stream:
            subprocess.run(command, cwd=directory, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    for name in args.variants:
        item = manifest["variants"][name]
        source = HERE / item["ir"]
        if digest(source) != item["sha256"]:
            raise RuntimeError(f"IR hash mismatch: {source}")
        directory = run / name
        if not args.dry_run:
            directory.mkdir()
            shutil.copy2(source, directory / "kernel.ll")
        execute(["bash", ROOT / "GenHsaco.sh", "kernel.ll", "3", "kernel.hsaco"],
                directory, directory / "build.log")
        if not args.dry_run:
            (directory / "binary.json").write_text(json.dumps(dict(
                ir_sha256=digest(source), hsaco_sha256=digest(directory / "kernel.hsaco")),
                indent=2) + "\n")

    samples = {name: [] for name in args.variants}
    for round_index in range(args.rounds):
        offset = round_index % len(args.variants)
        order = args.variants[offset:] + args.variants[:offset]
        if round_index % 2:
            order.reverse()
        for name in order:
            item = manifest["variants"][name]
            directory = run / name
            report = directory / f"round-{round_index + 1}.json"
            execute([args.launch_python, ROOT / "launchKernel.py", "--hsaco",
                     directory / "kernel.hsaco", "--kernel", "Attn_p2",
                     "--shape", "1,32,4096,128", "--grid", ",".join(map(str, item["grid"])),
                     "--block", "256,1,1", "--shared-mem", "0", "--k-layout", "bhsd",
                     "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                     "--device", str(args.device), "--seed", str(args.seed),
                     "--json-output", report], directory, report.with_suffix(".log"))
            if args.dry_run:
                continue
            result = json.loads(report.read_text())
            if result.get("passed") is not True:
                raise RuntimeError(f"Correctness failed: {report}")
            samples[name].append(result)
            summary = {key: dict(completed_rounds=len(values),
                kernel_median_ms=statistics.median(v["kernel_median_ms"] for v in values),
                sdpa_median_ms=statistics.median(v["sdpa_median_ms"] for v in values),
                sdpa_speedup=statistics.median(v["speedup"] for v in values))
                for key, values in samples.items() if values}
            control = summary.get("bm128_bn128", {}).get("kernel_median_ms")
            if control:
                for value in summary.values():
                    value["speedup_vs_bm128_bn128"] = control / value["kernel_median_ms"]
            temporary = run / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(run / "summary.json")
    print("Results:", run)


if __name__ == "__main__":
    main()
