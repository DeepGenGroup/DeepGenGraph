#!/usr/bin/env python3
"""Regression checks for SSA affine loop carriers lowered by MyTest.

Usage: python3 check_loop_thread_tiles.py [path/to/MyTest]
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def main():
    test_dir = Path(__file__).resolve().parent
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
        test_dir.parent / "build/test/MyTest"
    )
    source = (test_dir / "test_friskBaseDebug.mlir").read_text()
    live = "\n".join(
        line for line in source.splitlines() if "%qkmerAcc_filled = " not in line
    ).replace("%qkmerAcc_filled", "%19")
    zero = "%cst_3 = arith.constant dense<0.000000e+00>"
    swapped = replace_once(
        live,
        "iter_args(%oAcc = %cst_2, %qkmerAcc = %cst_3) -> "
        "(vector<64x128xf32>, vector<64x1xf32>)",
        "iter_args(%qkmerAcc = %cst_3, %oAcc = %cst_2) -> "
        "(vector<64x1xf32>, vector<64x128xf32>)",
    )
    swapped = replace_once(
        swapped,
        "affine.yield %oAcc_next, %qkmerAcc_next : "
        "vector<64x128xf32>, vector<64x1xf32>",
        "affine.yield %qkmerAcc_next, %oAcc_next : "
        "vector<64x1xf32>, vector<64x128xf32>",
    )
    swapped = replace_once(swapped, "frisk.div %9#0, %9#1", "frisk.div %9#1, %9#0")
    cases = {
        "constant_fill": source,
        "live_accumulator": live,
        "nonzero_init": replace_once(live, zero, zero.replace("0.000000", "3.000000")),
        "nonsplat_init": replace_once(
            live, zero, "%cst_3 = arith.constant dense<["
            + ",".join(f"[{float(i)}]" for i in range(64)) + "]>"
        ),
        "swapped_carriers": swapped,
    }
    with tempfile.TemporaryDirectory(prefix="frisk-thread-tiles-") as temp:
        for name, ir in cases.items():
            work = Path(temp) / name
            work.mkdir()
            input_file = work / "input.mlir"
            input_file.write_text(ir)
            result = subprocess.run(
                [str(binary), str(input_file)], cwd=work, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
            )
            log = result.stdout
            assert result.returncode == 0, f"{name}: MyTest failed\n{log[-4000:]}"
            # MyTest currently ignores some PassManager return values.
            assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", log), log
            stage = log.split("---------- after createConvertFriskBaseToThreadLevelIRPass", 1)[1]
            stage = stage.split("---- after threadIR legalize", 1)[0]
            expected = "vector<2x32xf32>, vector<2x1xf32>"
            if name == "swapped_carriers":
                expected = "vector<2x1xf32>, vector<2x32xf32>"
            assert f"-> ({expected})" in stage, name
            assert re.search(r"affine\.yield [^\n]+ : " + re.escape(expected), stage), name
            assert not re.search(r"affine\.yield[^\n]*vector<64x(?:1|128)xf32>", stage), name
            assert "thread_tile_insert" not in stage, name
            # Single-column vector/memref accesses must always select column 0.
            zeros = set(re.findall(r"(%[\w]+) = arith.constant 0 : index", stage))
            for line in stage.splitlines():
                if ("vector.extract" in line or "affine.load" in line) and re.search(
                    r"(?:vector|memref)<(?:64|2)x1xf32", line
                ):
                    column = re.search(r"\[[^\]]*,\s*([^,\]]+)\]", line).group(1)
                    assert column == "0" or column in zeros, f"{name}: {line}"
            assert "[d] llvmIR" in log, name
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
