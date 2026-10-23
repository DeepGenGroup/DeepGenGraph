#!/usr/bin/env python3
"""Generate and validate two control experiments against loop_only.ll.

1. loop_only_40k.ll: enlarge the existing, used LDS global to 40 KiB.
   Every instruction and accessed offset is identical to loop_only.ll.
   This isolates allocated capacity, not the address changes from disabling reuse.
2. nohint_unrolled.ll: remove pipeline_hint, mark only the DK loop for the same
   simplification pass. Keep coalesce-waits=0, matching loop_only.ll.
   nohint_unrolled_32k.ll additionally matches the pipeline's allocated capacity.

Padding enlarges the used global, rather than adding a dead allocation. Verify
that it survives O3 here; run.sh also checks the target assembly's LDS metadata.
No GPU performance or independent-wave execution is modeled here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
TEST = REPO / "3rd/deepgengraph/test"
sys.path.insert(0, str(TEST))

import numpy as np
from check_pipeline_cleanup_attention import CountedDMA
from check_auto_pipeline_attention import module_after
from check_attention_llvm_numerics import reference, report


GLOBAL = re.compile(r"^@shm_0 = addrspace\(3\) global \[(\d+) x i8\] undef, align 16$", re.M)


def lds(text):
    allocations = re.findall(r"^@[\w.]+ = .*addrspace\(3\) global ", text, re.M)
    matches = GLOBAL.findall(text)
    assert len(allocations) == len(matches) == 1
    return int(matches[0])


def pad(text, size):
    assert lds(text) <= size
    return GLOBAL.sub(f"@shm_0 = addrspace(3) global [{size} x i8] undef, align 16", text)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--llvm", type=Path, default=Path("/data2/xsl/install/bin"))
    args = parser.parse_args()
    source = (TEST / "test_input_spidD.mlir").read_text()
    source, count = re.subn(r"\{frisk\.pipeline_hint = 2 : i64\}",
                            "{frisk.simplify_loop}", source)
    assert count == 1 and "frisk.pipeline_hint" not in source
    mlir = WORK / "nohint_unrolled.mlir"
    out = WORK / "nohint_unrolled.ll"
    log = WORK / "nohint_unrolled.log"
    command = [str(REPO / "3rd/deepgengraph/build/test/MyTest"), str(mlir), "0", str(out),
               "1", "1", "3.3", "34816", "approx", "256", "4,1", "1", "1", "1",
               "0", "0", "useKPack", "asm128", "32", "32",
               "--frag-prefetch=1", "--frag-prefetch-words=4",
               "--simplify-loop-control=1", "--coalesce-waits=0"]
    if not args.existing:
        mlir.write_text(source)
        with log.open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=stream,
                           env={**os.environ, "FRISK_GM_TO_SHM_MODE": "async",
                                "FRISK_GM_TO_SHM_ALIGNMENT": "16"}, timeout=180)
        (WORK / "loop_only_40k.ll").write_text(pad((WORK / "loop_only.ll").read_text(), 40960))
        (WORK / "nohint_unrolled_32k.ll").write_text(pad(out.read_text(), 32768))
    assert mlir.read_text() == source
    diagnostic = log.read_text()
    before = module_after(diagnostic, "---------- after createFinalizeThreadTilingPass ---------")
    after = module_after(diagnostic, "---------- after frisk-simplify-loop-control ---------")
    scheduled = module_after(diagnostic, "---------- after frisk-auto-pipeline ---------")
    assert "pipeline.scheduled" not in scheduled
    assert before.count("frisk.simplify_loop") == 1
    assert "frisk.simplify_loop" not in after
    assert "SimplifyLoopControl: 1 small loops specialized" in diagnostic
    assert "0 staged, 0 fallback" in diagnostic
    assert lds((WORK / "loop_only.ll").read_text()) == 32768
    assert lds(out.read_text()) == 16384
    # Instruction-for-instruction identity of both capacity control pairs.
    pairs = [("loop_only", "loop_only_40k", 40960),
             ("nohint_unrolled", "nohint_unrolled_32k", 32768)]
    for base, padded, size in pairs:
        assert pad((WORK / f"{base}.ll").read_text(), size) == (WORK / f"{padded}.ll").read_text()

    names = ["loop_only", "loop_only_40k", "nohint_unrolled", "nohint_unrolled_32k"]
    optimized = {}
    for name in names:
        path = WORK / f"{name}.ll"
        subprocess.run([args.llvm / "llvm-as", path, "-o", WORK / f"{name}.bc"], check=True)
        opt = WORK / f"{name}.control_o3.ll"
        subprocess.run([args.llvm / "opt", "-passes=default<O3>,verify", "-S",
                        WORK / f"{name}.bc", "-o", opt], check=True)
        optimized[name] = re.sub(r"^; ModuleID = .*\n", "", opt.read_text())
        assert lds(optimized[name]) == lds(path.read_text())
    for base, padded, size in pairs:
        assert pad(optimized[base], size) == optimized[padded], (base, padded)

    rng = np.random.default_rng(42)
    q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    results = {}
    for name in names[1:]:
        path = WORK / f"{name}.ll"
        text = path.read_text()
        cases = []
        for head, block in [(0, 0), (1, 1), (1, 31)]:
            model = CountedDMA(text, q, k, v, threads=256, head=head, block=block)
            actual = model.run().reshape(q.shape)[head, block * 128:(block + 1) * 128]
            expected = reference(q, k, v, head, block)
            assert report(f"{name} head={head} block={block}", actual, expected)
            assert not model.pending
            cases.append({"head": head, "block": block,
                          "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected))),
                          "vm_waits": model.vm_waits, "barriers": model.barriers,
                          "mmac_with_pending_dma": model.overlapped_mma})
        results[name] = {"sha256": digest(path), "static_lds_bytes": lds(text),
                         "llvm_verify": True, "llvm_o3_verify": True,
                         "cpu_delayed_dma": cases}
        print("PASS", name, json.dumps(cases), flush=True)
    assert results["nohint_unrolled"]["cpu_delayed_dma"] == results["nohint_unrolled_32k"]["cpu_delayed_dma"]
    results["reference"] = {"file": "loop_only.ll", "sha256": digest(WORK / "loop_only.ll"),
                            "static_lds_bytes": 32768, "reported_speedup": 0.509}
    results["nohint_mytest_command"] = command
    results["padding_pairs_identical_except_global_size_before_and_after_o3"] = True
    results["gpu_tested"] = False
    results["note"] = ("CPU deferred-DMA model checks numerical execution and pending-destination hazards. "
                       "It does not model independent wave progress or target backend scheduling. "
                       "Padding controls allocated LDS capacity; all accessed offsets remain unchanged.")
    (WORK / "controls.validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
