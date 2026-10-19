#!/usr/bin/env python3
"""Check split-D BK inference, LLVM export and lowered-IR CPU numerics.

Usage: OPENBLAS_NUM_THREADS=1 python3 check_attention_kpack.py build/test/MyTest
Requires NumPy; this does not execute GPU machine code.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from check_attention_split_d import check_numerics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mytest", type=Path)
    parser.add_argument("--wmma-layout", choices=("3.1", "3.2", "3.3"), default="3.3")
    args = parser.parse_args()
    tool = args.mytest.resolve()
    fixtures = Path(__file__).resolve().parent
    env = {**os.environ, "FRISK_GM_TO_SHM_MODE": "async",
           "FRISK_GM_TO_SHM_ALIGNMENT": "16"}
    with tempfile.TemporaryDirectory(prefix="attention-kpack-") as directory:
        work = Path(directory)

        def compile_ir(name, source):
            path = work / (name + ".mlir")
            output = path.with_suffix(".ll")
            path.write_text(source)
            proc = subprocess.run(
                [str(tool), str(path), "0", str(output), "1", "1", args.wmma_layout,
                 "34816", "approx", "256", "4,1", "1", "1", "1", "0", "0",
                 "useKPack", "asm128"], cwd=work, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180)
            return proc, output

        for bk, fixture in ((16, "test_input_sd16.mlir"),
                            (32, "test_input_spidD.mlir")):
            proc, output = compile_ir(f"bk{bk}", (fixtures / fixture).read_text())
            assert proc.returncode == 0, proc.stdout[-5000:]
            assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", proc.stdout), proc.stdout[-5000:]
            match = re.search(r"kpack=(\d+), BK=(\d+) -> (\d+)", proc.stdout)
            assert match, "Missing BK inference diagnostic"
            factor, old, new = map(int, match.groups())
            assert old == bk and new == bk * factor and 128 % new == 0
            llvm = output.read_text()
            assert "define amdgpu_kernel void @Attn_p2" in llvm
            assert "buffer_load_dwordx4" in llvm
            error = check_numerics(proc.stdout, 128, seed=1,
                                   blocks=[(0, 0, 0), (0, 13, 15), (0, 31, 31)])
            print(f"PASS BK={old} -> {new}: LLVM export, lowered-IR CPU numerics; max_abs={error:.6g}", flush=True)

        source = (fixtures / "test_input_sd16.mlir").read_text()
        mismatched = source.replace("%cBK = arith.constant 16", "%cBK = arith.constant 32")
        wrong_tiles = mismatched.replace("to 128 step 16", "to 128 step 32")
        oversized = source.replace("to 128 step 16", "to 128 step 256")
        for name, text, diagnostic in (
            ("mismatched-steps", mismatched, "operand tile extents must match their own loop steps"),
            ("wrong-tiles", wrong_tiles, "operand tile extents must match their own loop steps"),
            ("oversized-bk", oversized, "input QK/VO steps must divide their respective D extents"),
        ):
            proc, output = compile_ir(name, text)
            assert proc.returncode == 1 and diagnostic in proc.stdout, proc.stdout[-5000:]
            assert not output.exists(), "Invalid input reached LLVM export"
            print(f"PASS rejected {name}: {diagnostic}", flush=True)


if __name__ == "__main__":
    main()
