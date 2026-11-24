#!/usr/bin/env python3
"""Validate explicit launch strides through frontend, tiling and LLVM execution.

Uses the CPU wave model, not GPU timing or asynchronous hazard simulation.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from check_attention_tails import TEMPLATE, generate_ir, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mytest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="strided-attention-") as tmp:
        directory = args.output_dir.resolve() if args.output_dir else Path(tmp)
        directory.mkdir(parents=True, exist_ok=True)
        cases = [(s, causal, "legacy") for s in (33, 64) for causal in (True, False)]
        cases.append((64, False, "async"))
        for s, causal, mode in cases:
            # Multiple heads/batches, S==D, nonzero tile origins and tails.
            b, h, d, bm, bn = 2, 3, 64, 32, 32
            source, _ = generate_ir(TEMPLATE.read_text(), bm, bn, 32,
                                    b=b, h=h, s=s, d=d, causal=causal)
            stem = f"s{s}-causal{int(causal)}-{mode}"
            inp, out, log = [directory / (stem + suffix)
                             for suffix in (".mlir", ".ll", ".log")]
            inp.write_text(source)
            command = [str(args.mytest.resolve()), f"--input={inp}",
                       f"--output={out}", "--math=mmac", "--llvm-opt=0",
                       "--threads=128", "--block-warps=2,1", "--wmma-layout=3.2",
                       "--fuse-mask-add=0", "--uniform-masks=0"]
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=directory,
                    env={**os.environ, "FRISK_GM_TO_SHM_MODE": mode, "FRISK_GM_TO_SHM_ALIGNMENT": "16"},
                    stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            diagnostics = log.read_text()
            assert result.returncode == 0, diagnostics[-6000:]
            assert "createNormalizeAttentionKLayoutPass" not in diagnostics
            assert "memref.transpose" not in diagnostics
            assert f"memref<{b}x{h}x{d}x{s}xf16, strided<[{h*s*d}, {s*d}, 1, {d}]>, 1>" in diagnostics
            for role in ("q", "k", "v"):
                assert f'frisk.shared_role = "{role}"' in diagnostics
            assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", diagnostics)
            check(out.read_text(), s, bm, d, b, h, causal)
            print(f"PASS explicit strides: S={s}, causal={causal}, B={b}, H={h}, copy={mode}", flush=True)


if __name__ == "__main__":
    main()
