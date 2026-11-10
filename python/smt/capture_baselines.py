#!/usr/bin/env python3
"""Capture/replay current Base IR for the two measured historical best configs.

Historical LLVM instruction schedules are provenance, not Base op schedules.
All compiler scratch files are isolated in temporary directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

try:
    from .schedule_schema import read_problem, validate_schedule, write_json
except ImportError:
    from schedule_schema import read_problem, validate_schedule, write_json

ROOT = Path(__file__).resolve().parents[2]
POSITIONAL = ("reorder", "pack", "layout", "carrier", "math", "threads", "waves",
              "lifetimes", "masks", "prefetch", "fuse", "reuse", "dma-pack", "dma-export",
              "qk-tile", "vo-tile")
INPUT_ONLY = {"pipeline", "copy", "alignment", "pipeline-stages", "loop-mark", "lds-bytes"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    causal_config = ROOT / "generated_ir/causal_cvt_pack_mytest/historical_safe_best.config.json"
    causal = json.loads(causal_config.read_text())
    noncausal = dict(causal["parameters"], **{
        "workgroup-group-size": "32", "workgroup-reverse": "0",
        "k-dma": "64-swizzle", "k-dma-arg": "1", "k-lds-prefetch": "8",
        "softmax-fragments": "pv", "k-tail-prefetch": "4", "cross-bn-k0": "1",
        "lazy-softmax-base": "8", "softmax-cvt-pack": "rtz",
    })
    return {
        "causal": dict(
            input="generated_ir/causal_cvt_pack_mytest/causal.input.mlir",
            artifact="generated_ir/causal_v_double_buffer_reverse/group4.ll",
            median_ms=1.1768704652786255, parameters=causal["parameters"],
            config_source=str(causal_config.relative_to(ROOT)),
            historical_late_schedule=causal["schedule"]),
        "noncausal": dict(
            input="generated_ir/async_task_noncausal/input.mlir",
            artifact="generated_ir/noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll",
            median_ms=1.5057590007781982, parameters=noncausal,
            config_source="generated_ir/noncausal_cvt_pack_pass/mytest_lowering/command.sh",
            historical_manifest="generated_ir/noncausal_cvt_pack_pass/manifest.json"),
    }


def run(command, env, work, log):
    with Path(log).open("w") as stream:
        result = subprocess.run(command, env=env, cwd=work, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        raise RuntimeError(f"compiler failed ({result.returncode}); see {log}")


def capture(name, case, output, mytest, verify_lowering):
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    cfg = case["parameters"]
    source = ROOT / case["input"]
    artifact = ROOT / case["artifact"]
    (dest / "input.mlir").write_bytes(source.read_bytes())
    options = [str(cfg[k]) for k in POSITIONAL]
    options += [f"--{key}={value}" for key, value in cfg.items()
                if key not in POSITIONAL and key not in INPUT_ONLY]
    environment = {"FRISK_GM_TO_SHM_MODE": cfg["copy"],
                   "FRISK_GM_TO_SHM_ALIGNMENT": cfg["alignment"]}
    env = dict(os.environ, **environment)
    prefix = [str(mytest), str(dest / "input.mlir"), "0"]
    with tempfile.TemporaryDirectory(prefix=f"cpsat-{name}-") as scratch:
        command = prefix + [str(Path(scratch) / "unused.ll")] + options
        run(command + [f"--cpsat-export={dest / 'problem.json'}",
                       f"--cpsat-baseline={dest / 'schedule.json'}",
                       f"--cpsat-dump-ir={dest / 'base.mlir'}", "--cpsat-stop-after=1"],
            env, scratch, dest / "export.log")
        problem = read_problem(dest / "problem.json")
        schedule = json.loads((dest / "schedule.json").read_text())
        validate_schedule(problem, schedule)
        run(command + [f"--cpsat-import={dest / 'schedule.json'}",
                       f"--cpsat-dump-ir={Path(scratch) / 'replayed.mlir'}", "--cpsat-stop-after=1"],
            env, scratch, dest / "replay.log")
        if (dest / "base.mlir").read_bytes() != (Path(scratch) / "replayed.mlir").read_bytes():
            raise RuntimeError(f"{name}: identity replay changed Base IR")
        lowering_equal = None
        if verify_lowering:
            finals = []
            for variant in ("original", "replayed"):
                work = Path(scratch) / variant
                work.mkdir()
                llvm_file = work / "kernel.ll"
                flags = [f"--cpsat-import={dest / 'schedule.json'}"] if variant == "replayed" else []
                run(prefix + [str(llvm_file)] + options + flags, env, work, dest / f"{variant}.log")
                finals.append(llvm_file.read_bytes())
            lowering_equal = finals[0] == finals[1]
            if not lowering_equal:
                raise RuntimeError(f"{name}: identity replay changed final LLVM")
    provenance = dict(
        schema="deepgengraph.cpsat.baseline-provenance", version=1, case=name,
        phase=problem["phase"], loop_id=problem["loop"]["id"],
        historical_measurement=dict(artifact=case["artifact"], sha256=sha256(artifact),
            median_ms=case["median_ms"], source="pmc_test.log", measured_in_this_run=False),
        input_source=case["input"], input_sha256=sha256(source),
        config_source=case["config_source"], parameters=cfg, environment=environment,
        mytest_sha256=sha256(mytest), mytest_arguments_after_output=options,
        fingerprint=problem["fingerprint"], base_sha256=sha256(dest / "base.mlir"),
        identity_replay_exact=True, final_llvm_identity_replay_exact=lowering_equal,
        note="schedule.json records the current compiler's original Base order with this config. "
             "Historical winners include later fragment/LLVM transformations and are not "
             "reproduced solely by this Base schedule. No new GPU timing is claimed.")
    for key in ("historical_late_schedule", "historical_manifest"):
        if key in case:
            provenance[key] = case[key]
    write_json(dest / "provenance.json", provenance)
    # Verbose compiler dumps are disposable; retain only useful artifacts.
    for log in dest.glob("*.log"):
        log.unlink()
    print(f"{name}: {len(problem['operations'])} ops; {dest / 'schedule.json'}; replay exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "baselines")
    parser.add_argument("--mytest", type=Path, default=ROOT / "3rd/deepgengraph/build/test/MyTest")
    parser.add_argument("--verify-lowering", action="store_true",
                        help="also compile original/replayed inputs through LLVM and compare bytes")
    args = parser.parse_args()
    for name, case in cases().items():
        capture(name, case, args.output.resolve(), args.mytest.resolve(), args.verify_lowering)


if __name__ == "__main__":
    main()
