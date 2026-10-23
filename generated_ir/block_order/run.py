#!/usr/bin/env python3
"""Build with target DTK, then benchmark block-order variants in rotated rounds."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys

from generate import HERE, SOURCE, SOURCE_SHA256, VARIANTS, DEFAULT_VARIANTS, digest

REPO = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launch-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("rounds/repeat must be positive; warmup must be nonnegative")
    names = list(dict.fromkeys(args.variants))
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest.get("source_sha256") != SOURCE_SHA256 or digest(SOURCE) != SOURCE_SHA256:
        parser.error("Source/manifest baseline mismatch; regenerate and validate first")
    if digest(HERE / "baseline.ll") != SOURCE_SHA256:
        parser.error("baseline.ll must be byte-identical to baseline_swap_bx_by.ll")
    for name in names:
        if digest(HERE / f"{name}.ll") != manifest["variants"][name]["sha256"]:
            parser.error(f"IR hash mismatch: {name}; regenerate and validate first")
    run = HERE / ("run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    env = {**os.environ, "GPU_ARCH": os.environ.get("GPU_ARCH", "gfx936")}
    if not args.dry_run:
        run.mkdir()
        shutil.copy2(HERE / "manifest.json", run / "manifest.json")
        (run / "run.json").write_text(json.dumps({
            "arguments": vars(args), "environment": {k: env.get(k) for k in
                ("GPU_ARCH", "LLVM_DIR", "ROCM_BC", "GPU_FEATURES", "DWARF_VERSION")},
            "gen_hsaco_sha256": digest(REPO / "GenHsaco.sh"),
            "launcher_sha256": digest(REPO / "launchKernel.py"),
            "baseline_source": manifest["source"], "baseline_sha256": SOURCE_SHA256,
        }, indent=2) + "\n")
    print("Results:", run, flush=True)

    def execute(command, directory, log):
        print(shlex.join(map(str, command)), flush=True)
        if args.dry_run:
            return
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {log}")

    # Build everything before timing, with isolated directories for GenHsaco's scratch files.
    for name in names:
        directory = run / name
        if not args.dry_run:
            directory.mkdir()
            shutil.copy2(HERE / f"{name}.ll", directory / "kernel.ll")
        execute(["bash", REPO / "GenHsaco.sh", "kernel.ll", "3", "kernel.hsaco"],
                directory, directory / "build.log")
        if not args.dry_run:
            assembly = (directory / "kernel.hsaco.amdgcn").read_text()
            match = re.search(r"(?:\.amdhsa_group_segment_fixed_size\s+|\.group_segment_fixed_size:\s*)(\d+)", assembly)
            if not match or int(match[1]) != 16384:
                raise RuntimeError(f"Unexpected LDS allocation for {name}; inspect target assembly")
            (directory / "binary.json").write_text(json.dumps({
                "ir_sha256": digest(directory / "kernel.ll"),
                "hsaco_sha256": digest(directory / "kernel.hsaco"),
                "assembly_sha256": digest(directory / "kernel.hsaco.amdgcn"),
                "static_lds_bytes": int(match[1]),
            }, indent=2) + "\n")

    samples = {name: [] for name in names}
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index % 2:
            order.reverse()
        print(f"Round {round_index + 1}: {', '.join(order)}", flush=True)
        for name in order:
            directory = run / name
            result_file = directory / f"round-{round_index+1}.json"
            execute([args.launch_python, REPO / "launchKernel.py",
                     "--hsaco", directory / "kernel.hsaco", "--kernel", "Attn_p2",
                     "--shape", "1,32,4096,128", "--grid", "32,32,1", "--block", "256,1,1",
                     "--shared-mem", "0", "--k-layout", "bhsd",
                     "--device", str(args.device), "--seed", str(args.seed),
                     "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                     "--json-output", result_file], directory,
                    directory / f"round-{round_index+1}.log")
            if args.dry_run:
                continue
            result = json.loads(result_file.read_text())
            if result.get("passed") is not True:
                raise RuntimeError(f"Correctness failed: {result_file}")
            samples[name].append(result)
            summary = {}
            for key, values in samples.items():
                if values:
                    medians = [v["kernel_median_ms"] for v in values]
                    summary[key] = {
                        "baseline_sha256": SOURCE_SHA256,
                        "completed_rounds": len(values), "passed": True,
                        "median_of_round_medians_ms": statistics.median(medians),
                        "min_round_median_ms": min(medians), "max_round_median_ms": max(medians),
                        "median_sdpa_ms": statistics.median(v["sdpa_median_ms"] for v in values),
                        "median_sdpa_speedup": statistics.median(v["speedup"] for v in values),
                    }
            if "baseline" in summary:
                base = summary["baseline"]["median_of_round_medians_ms"]
                for value in summary.values():
                    value["speedup_vs_baseline"] = base / value["median_of_round_medians_ms"]
            (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(name, f"{result['kernel_median_ms']:.6f} ms", flush=True)
    if not args.dry_run:
        print("Summary:", run / "summary.json")


if __name__ == "__main__":
    main()
