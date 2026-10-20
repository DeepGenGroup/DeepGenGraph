#!/usr/bin/env python3
"""BHSD attention: physical producer selection, automatic DMA and CPU numerics.

This checks emitted LLVM, not GPU throughput or hardware synchronization.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def check(tool, directory):
    env = {**os.environ, "FRISK_GM_TO_SHM_MODE": "async",
           "FRISK_GM_TO_SHM_ALIGNMENT": "16", "OPENBLAS_NUM_THREADS": "1"}
    for layout, mode in (("3.2", "legacy"), ("3.2", "useKPack"),
                         ("3.2", "noKPack"), ("3.3", "useKPack")):
        name = f"layout-{layout}-{mode}"
        ir, log = directory / (name + ".ll"), directory / (name + ".log")
        command = [str(tool), str(HERE / "test_input_spidD.mlir"), "0", str(ir),
                   "1", "1", layout, "34816", "approx", "256", "4,1",
                   "1", "1", "1", "0", "0", mode]
        # Deliberately omit the export argument: automatic wide DMA must be
        # executable through the default export, including mixed 4B/16B calls.
        with log.open("w") as output:
            subprocess.run(command, cwd=directory, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=240)
        text, diagnostics = ir.read_text(), log.read_text()
        assert not re.search(r"error:|LLVM ERROR|Assertion .*failed", diagnostics)
        assert re.search(r"LowerFriskAsyncCopy: \d+ direct, 0 staged, 0 fallback; staging bytes 0", diagnostics)
        assert "@async_lds_staging_" not in text
        assert "buffer_load_dwordx4" in text
        assert "s_waitcnt vmcnt(0)" in text
        matrix_reads = len(re.findall(r"call .*@llvm.hcu.ds.read.m32x16", text))
        assert matrix_reads == 32, matrix_reads  # V only; K uses vector reads.
        narrow = len(re.findall(r"call void @llvm.amdgcn.global.load.async.lds\(", text))
        assert (narrow > 0) == (mode == "legacy"), narrow
        assert "call void @llvm.hcu.global.load" not in text
        assert not re.search(r"call void @llvm.amdgcn.(?:asyncmark|wait.asyncmark)\(", text)
        assert "load <4 x half>" in text
        with (directory / (name + ".numerics.log")).open("w") as output:
            subprocess.run([sys.executable, str(HERE / "check_attention_llvm_numerics.py"), str(ir)],
                           env=env, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        print(f"PASS {name}: direct DMA, K vector/V matrix, default export, four CPU numerical cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mytest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check(args.mytest.resolve(), args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="attention-physical-copy-") as directory:
            check(args.mytest.resolve(), Path(directory))


if __name__ == "__main__":
    main()
