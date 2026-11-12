#!/usr/bin/env python3
"""Check GEMM loop markers and selective full unrolling with ThreadTilingTest."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TOOL = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parent.parent / "build/test/ThreadTilingTest").resolve()
MARKER = "frisk.loopUnrollFull"


def run(source, directory, name, *passes):
    src = directory / (name + ".mlir")
    dst = directory / (name + ".out.mlir")
    src.write_text(source)
    proc = subprocess.run([str(TOOL), str(src), "-o", str(dst), *passes],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return dst.read_text()


def main():
    with tempfile.TemporaryDirectory(prefix="gemm-loop-unroll-") as temp:
        work = Path(temp)
        source = """
func.func @gemm(%a: memref<64x128xf16, 3>, %b: memref<128x32xf16, 3>,
                %out: memref<64x32xf32, 3>) attributes {thread_num = 64 : i32} {
  %g_init = frisk.zero : memref<64x32xf32>
  %g = frisk.gemm(%a, %b, %g_init) {transA = false, transB = false}
    : memref<64x128xf16, 3>, memref<128x32xf16, 3>, memref<64x32xf32> -> memref<64x32xf32>
  frisk.copy %g to %out [affine_map<() -> (2)>]
    : memref<64x32xf32>, memref<64x32xf32, 3>
  return
}
"""
        tiled = run(source, work, "gemm-tiled", "--convert-friskbase-to-thread=matrix-loads=false")
        bridges = [line for line in tiled.splitlines() if "frisk.to_threadTile" in line]
        assert sum("frisk.fragment_source" in line for line in bridges) == 2, tiled
        k_count = int(re.findall(r"affine.for .* = 0 to (\d+) iter_args",
                                tiled.split("frisk.warp_mma_rr", 1)[0])[-1])
        assert k_count > 1, tiled
        mma_count = tiled.count("frisk.warp_mma_rr")
        finalized = run(tiled, work, "gemm-final", "--finalize-thread-tiling", "--lower-frisk-fragments")
        assert MARKER not in finalized, finalized
        assert 'iterLabel = "gemm_k"' not in finalized, finalized
        assert not re.search(r'iterLabel = "load\d+"', finalized), finalized
        assert finalized.count("frisk.warp_mma_rr") == mma_count * k_count, finalized
        assert "memref.load" in finalized and "affine.store" in finalized, finalized
        print("PASS GEMM A/B assembly and MMA loops fully unrolled")

        # Exercise nested loops and loop-carried values independently of GEMM.
        nested = """
func.func @nested(%init: f32, %x: f32) -> f32 attributes {thread_num = 64 : i32} {
  %r = affine.for %i = 0 to 2 iter_args(%a = %init) -> f32 {
    %s = affine.for %j = 0 to 3 iter_args(%b = %a) -> f32 {
      %v = arith.addf %b, %x : f32
      affine.yield %v : f32
    } {frisk.loopUnrollFull = true}
    affine.yield %s : f32
  } {frisk.loopUnrollFull = true}
  return %r : f32
}
"""
        ir = run(nested, work, "nested", "--finalize-thread-tiling", "--lower-frisk-fragments")
        assert "affine.for" not in ir and ir.count("arith.addf") == 6, ir
        # False and absent markers must preserve both loops.
        for name, attrs in [("false", " {frisk.loopUnrollFull = false}"),
                            ("absent", "")]:
            ir = run(nested.replace(" {frisk.loopUnrollFull = true}", attrs),
                     work, name, "--finalize-thread-tiling", "--lower-frisk-fragments")
            assert ir.count("affine.for") == 2, ir
        print("PASS nested iter_args, false markers and unmarked loops")

        dynamic = nested.replace("%init: f32, %x: f32", "%init: f32, %x: f32, %n: index")
        dynamic = dynamic.replace("%i = 0 to 2", "%i = 0 to %n")
        ir = run(dynamic, work, "dynamic", "--finalize-thread-tiling", "--lower-frisk-fragments")
        assert ir.count("affine.for") == 1 and MARKER in ir, ir
        assert ir.count("arith.addf") == 3, ir
        print("PASS dynamic outer loop retained, static inner loop unrolled")


if __name__ == "__main__":
    main()
