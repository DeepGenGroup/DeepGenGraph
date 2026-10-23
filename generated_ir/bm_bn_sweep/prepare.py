#!/usr/bin/env python3
"""Generate four tile configurations, or collect an existing TuneAttention export."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from TuneAttention import launch_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path)
    args = parser.parse_args()
    if args.collect:
        run = args.collect.resolve()
    else:
        subprocess.run([sys.executable, ROOT / "generate_attention_ir.py", "--bm", "128", "64",
                        "--bn", "128", "64", "--bk", "32", "--output-dir", HERE / "inputs"], check=True)
        existing = set((HERE / "export").glob("run-*"))
        subprocess.run([sys.executable, ROOT / "TuneAttention.py", "--input-dir", HERE / "inputs",
                        "--case", "group8-reverse-k128-swizzle", "--set", "threads=256",
                        "--set", "waves=4,1", "--output-dir", HERE / "export",
                        "--llvm-as", "/data2/xsl/install/bin/llvm-as"], check=True)
        created = set((HERE / "export").glob("run-*")) - existing
        assert len(created) == 1
        run = created.pop()
    variants = {}
    for result_path in sorted(run.glob("input-*/*/result.json")):
        result = json.loads(result_path.read_text())
        assert result["status"] == "verified", result
        bm, bn = map(int, re.search(r'-(\d+)-(\d+)-32\.mlir$', result["source_input"]).groups())
        name = f"bm{bm}_bn{bn}"
        target = HERE / f"{name}.ll"
        shutil.copy2(result["llvm_ir"], target)
        launch = launch_config(Path(result["input"]).read_text(), target.read_text())
        assert launch["grid"] == [4096 // bm, 32, 1]
        assert launch["block"] == [256, 1, 1]
        (HERE / f"{name}.launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        variants[name] = dict(bm=bm, bn=bn, bk=32, ir=target.name, sha256=digest(target),
            grid=launch["grid"], block=launch["block"], static_lds_bytes=result["static_lds_bytes"],
            export_result=str(result_path.relative_to(HERE)),
            dma128_sites=result["buffer_dma128_sites"], dma32_sites=result["buffer_dma32_sites"],
            gep_offsets=result["lds_gep_reported_offsets"], swizzle=result["shared_swizzle_plans"],
            parameters=result["parameters"])
    assert set(variants) == {f"bm{m}_bn{n}" for m in (128,64) for n in (128,64)}
    control = ROOT / "generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll"
    shutil.copy2(control, HERE / "measured_control.ll")
    variants["measured_control"] = dict(bm=128, bn=128, bk=32, ir="measured_control.ll",
        sha256=digest(control), grid=[32,32,1], block=[256,1,1], static_lds_bytes=16384,
        historical_median_ms=1.7790470123291016,
        note="Byte-identical copy of the previously measured IR; remeasure alongside regenerated variants.")
    manifest = dict(shape=[1,32,4096,128], dtype="fp16", causal=True, k_layout="bhsd",
        generator_sha256=digest(ROOT / "generate_attention_ir.py"),
        tuner_sha256=digest(ROOT / "TuneAttention.py"),
        compiler_sha256=digest(ROOT / "3rd/deepgengraph/build/test/MyTest"),
        gpu_tested=False, variants=variants)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, item in variants.items():
        print(name, "grid=", item["grid"], "LDS=", item["static_lds_bytes"])


if __name__ == "__main__":
    main()
