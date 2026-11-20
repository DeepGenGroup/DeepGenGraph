#!/usr/bin/env python3
"""Check block tiling hint inference, LLVM export and lowered-IR CPU numerics.

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
from generate_attention_ir import TEMPLATE, generate_ir


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
                 "approx", "256", "4,1", "1", "1", "1", "0", "0",
                 "useKPack", "asm128"], cwd=work, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180)
            return proc, output

        for bk in (16, 32):
            proc, output = compile_ir(f"bk{bk}", generate_ir(TEMPLATE.read_text(),128,128,bk,softmax="legacy")[0])
            assert proc.returncode == 0, proc.stdout[-5000:]
            assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", proc.stdout), proc.stdout[-5000:]
            match = re.search(r"frisk.attention_tiled = array<i64: (\d+), (\d+)>", proc.stdout)
            assert match, "Missing effective block tiling attribute"
            qk, vo = map(int, match.groups())
            assert qk == vo == bk*2
            old, new = bk, qk
            llvm = output.read_text()
            assert "define amdgpu_kernel void @Attn_p2" in llvm
            assert "buffer_load_dwordx4" in llvm
            error = check_numerics(proc.stdout, 128, seed=1,
                                   blocks=[(0, 0, 0), (0, 13, 15), (0, 31, 31)])
            print(f"PASS BK={old} -> {new}: LLVM export, lowered-IR CPU numerics; max_abs={error:.6g}", flush=True)

        source = generate_ir(TEMPLATE.read_text(),128,128,16,softmax="legacy")[0]
        for name, hints in (("invalid-qk", "24, 16"), ("invalid-vo", "16, 256"),
                            ("negative-hint", "-1, 16")):
            text = source.replace("frisk.attention_tiles = array<i64: 16, 16>",
                                  f"frisk.attention_tiles = array<i64: {hints}>")
            proc, output = compile_ir(name, text)
            assert proc.returncode == 1 and "tiles" in proc.stdout, proc.stdout[-5000:]
            assert not output.exists(), "Invalid hint reached LLVM export"
            print(f"PASS rejected {name}", flush=True)


if __name__ == "__main__":
    main()
