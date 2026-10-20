#!/usr/bin/env python3
"""A/B cleanup exports: resource usage and delayed-DMA numerical execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from check_auto_pipeline_attention import DelayedDMA, module_after
from check_attention_llvm_numerics import reference, report

HERE = Path(__file__).resolve().parent


class CountedDMA(DelayedDMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vm_waits = self.barriers = 0

    def call(self, body):
        self.vm_waits += "s_waitcnt vmcnt(0)" in body
        self.barriers += "@llvm.amdgcn.s.barrier(" in body
        return super().call(body)


def check(work, tool, llvm, existing):
    rng = np.random.default_rng(42)
    q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16)
               for _ in range(3)]
    results = {}
    for name, loop, waits in [("baseline", 0, 0), ("wait_only", 0, 1),
                              ("loop_only", 1, 0), ("combined", 1, 1)]:
        out, log = work / (name + ".ll"), work / (name + ".log")
        command = [str(tool), str(HERE / "test_input_spidD.mlir"), "0", str(out),
                   "1", "1", "3.3", "34816", "approx", "256", "4,1",
                   "1", "1", "1", "0", "0", "useKPack", "asm128", "32", "32",
                   "--frag-prefetch=1", "--frag-prefetch-words=4",
                   f"--simplify-loop-control={loop}", f"--coalesce-waits={waits}"]
        if not existing:
            with log.open("w") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=stream,
                               env={**os.environ, "FRISK_GM_TO_SHM_MODE": "async",
                                    "FRISK_GM_TO_SHM_ALIGNMENT": "16"}, timeout=180)
        text = out.read_text()
        diagnostic = log.read_text()
        assert "0 staged, 0 fallback" in diagnostic
        storages = re.findall(r"^@[\w.]+ = .*addrspace\(3\) global "
                              r"\[(\d+) x (i8|half|float)\]", text, re.M)
        size = sum(int(n) * {"i8": 1, "half": 2, "float": 4}[ty] for n, ty in storages)
        assert size == (32768 if loop else 40960), (name, size)
        if loop:
            specialized = module_after(
                diagnostic, "---------- after frisk-simplify-loop-control ---------")
            assert "pipeline.scheduled" not in specialized
            assert "SimplifyLoopControl: 1 small loops specialized" in diagnostic
        subprocess.run([llvm / "llvm-as", out, "-o", work / (name + ".bc")], check=True)
        subprocess.run([llvm / "opt", "-passes=default<O3>,verify",
                        work / (name + ".bc"), "-disable-output"], check=True)
        cases = []
        for head, block in ([(0, 0), (1, 1), (1, 31)] if name == "combined"
                            else [(0, 0), (1, 1)]):
            model = CountedDMA(text, q, k, v, threads=256, head=head, block=block)
            actual = model.run().reshape(q.shape)[head, block * 128:(block + 1) * 128]
            expected = reference(q, k, v, head, block)
            assert report(f"{name} head={head} block={block}", actual, expected)
            assert not model.pending and model.overlapped_mma > 0
            cases.append({
                "head": head, "block": block,
                "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected))),
                "vm_waits": model.vm_waits, "barriers": model.barriers,
                "mmac_with_pending_dma": model.overlapped_mma,
            })
        results[name] = {
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "static_lds_bytes": size, "llvm_verify": True, "llvm_o3_verify": True,
            "cpu_delayed_dma": cases,
            "mytest_flags": command[4:],
        }
        print("PASS", name, json.dumps(cases), flush=True)
    for n in range(2):
        baseline = results["baseline"]["cpu_delayed_dma"][n]
        wait_only = results["wait_only"]["cpu_delayed_dma"][n]
        loop_only = results["loop_only"]["cpu_delayed_dma"][n]
        both = results["combined"]["cpu_delayed_dma"][n]
        assert wait_only["vm_waits"] < baseline["vm_waits"]
        assert wait_only["barriers"] < baseline["barriers"]
        assert both["vm_waits"] < loop_only["vm_waits"]
        assert both["barriers"] < loop_only["barriers"]
    results["gpu_tested"] = False
    results["note"] = ("CPU model checks numerical execution and deferred DMA storage hazards; "
                       "it does not model independently progressing waves or GPU performance. "
                       "Dynamic counts are per modeled workgroup, not static instruction counts.")
    (work / "validation.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mytest", type=Path, default=HERE.parent / "build/test/MyTest")
    parser.add_argument("--llvm", type=Path, default=Path("/data2/xsl/install/bin"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--existing", action="store_true",
                        help="Validate four existing exports instead of regenerating them")
    args = parser.parse_args()
    if args.existing and not args.output_dir:
        parser.error("--existing requires --output-dir")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check(args.output_dir.resolve(), args.mytest.resolve(), args.llvm, args.existing)
    else:
        with tempfile.TemporaryDirectory(prefix="pipeline-cleanup-") as directory:
            check(Path(directory), args.mytest.resolve(), args.llvm, False)


if __name__ == "__main__":
    main()
