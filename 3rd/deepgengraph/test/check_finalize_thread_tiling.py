#!/usr/bin/env python3
"""Check finalization with MyTest (IR verification, not GPU execution).

Usage: python3 check_finalize_thread_tiling.py [path/to/MyTest]
"""
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
BINARY = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          TEST_DIR.parent / "build/test/MyTest").resolve()
STAGE = "---------- after createConvertFriskBaseToThreadLevelIRPass ---------"


def run(path, work):
    proc = subprocess.run([str(BINARY), str(path)], cwd=work, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=60)
    assert proc.returncode == 0, proc.stdout[-6000:]
    assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", proc.stdout), proc.stdout[-6000:]
    ir = proc.stdout.split(STAGE, 1)[1].strip()
    assert not re.search(r"frisk\.(?:to_threadTile|from_threadTile|buffer_view|cast)\b", ir), ir
    assert "unrealized_conversion_cast" not in ir, ir
    return ir


def check_copy_addresses(ir):
    # Evaluate the generated address maps for every owner, not just their
    # spelling: Q uses row-major registers, K/V use column-major registers.
    cases = [(64, 128, 2, 32, 128), (128, 32, 32, 2, 64), (32, 128, 8, 8, 64)]
    for rows, cols, tile_rows, tile_cols, threads in cases:
        load = re.search(r"memref.load (%\w+)\[(%\w+), (%\w+)\] : memref<"
                         + f"{rows}x{cols}xf16, strided<", ir)
        assert load, (rows, cols)
        maps = []
        for index in load.group(2, 3):
            definition = re.findall(re.escape(index) + r" = affine.apply affine_map<[^\n]+", ir[:load.start()])[-1]
            expr = re.search(r" -> \((.*)\)>", definition)[1]
            expr = expr.replace("floordiv", "//").replace("mod", "%")
            tree = ast.parse(expr, mode="eval")
            assert all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub,
                ast.Name, ast.Load, ast.Constant)) for node in ast.walk(tree))
            maps.append(compile(tree, "<generated affine map>", "eval"))
        addresses = []
        for tid in range(threads):
            for i in range(tile_rows):
                for j in range(tile_cols):
                    addresses.append(tuple(eval(code, {"__builtins__": {}},
                        {"d0": iv, "s0": tid}) for code, iv in zip(maps, (i, j))))
        expected = {(i, j) for i in range(rows) for j in range(cols)}
        assert len(addresses) == len(set(addresses)), (rows, cols, "duplicate addresses")
        assert set(addresses) == expected, (rows, cols, "incomplete or out-of-bounds copy")


def main():
    with tempfile.TemporaryDirectory(prefix="finalize-thread-tiling-") as temp:
        work = Path(temp)
        ir = run(TEST_DIR / "test_input.mlir", work)
        assert "memref<128x32xf16, strided<[4096, 1], offset: ?>, 1>" in ir
        assert "memref<64x128xf16, strided<[128, 1], offset: ?>, 1>" in ir
        assert re.search(r"arith.truncf .*vector<2x8xf32> to vector<2x8xf16>", ir)
        assert re.search(r"arith.truncf .*vector<2x32xf32> to vector<2x32xf16>", ir)
        assert "-> (vector<2x32xf32>, vector<2x1xf32>)" in ir
        assert not re.search(r"vector<64x(?:128|1)xf32>", ir)
        check_copy_addresses(ir)
        # Q/K/V/output copy scratch is forwarded directly as SSA vectors.
        # Physical block loads and writebacks are checked separately above.
        for name in ("memref.alloca", "vector.store", "vector.load"):
            assert name not in ir, name
        # Shared layout exchange is ordered before reads and before reuse.
        for name in re.findall(r"(%\w+) = memref.alloc\(\) : memref<64x1xf32, 3>", ir):
            after = ir.split(name + " = memref.alloc()", 1)[1]
            store = re.search(r"affine.store[^\n]*" + re.escape(name) + r"\[", after)
            load = re.search(r"memref.load " + re.escape(name) + r"\[", after)
            assert store and load and store.start() < load.start()
            assert "gpu.barrier" in after[store.end():load.start()]
        print("PASS test_input: strides, register loads/casts, initialized copies, loop carriers")

        # Re-running the entire MyTest pipeline must not re-tile the result.
        finalized = work / "finalized.mlir"
        finalized.write_text(ir)
        repeated = run(finalized, work)
        # Earlier MyTest passes add loop markers and simplify affine maps.
        # Finalization must not introduce new loads, copies or conversions.
        for name in ("memref.load", "memref.alloca", "arith.truncf", "gpu.barrier"):
            assert repeated.count(name) == ir.count(name), name
        assert "-> (vector<2x32xf32>, vector<2x1xf32>)" in repeated
        print("PASS repeated pipeline preserves finalized operations")

        cases = {
            "copy_to_reg": ("""
func.func @copy_to_reg(%src: memref<32x32xf16, 3>, %i: index) -> vector<1x4xf32>
    attributes {thread_num = 64 : i32} {
  %v = frisk.copy_to_reg %src at (%i) [affine_map<(d0) -> (d0, 4)>]
       : memref<32x32xf16, 3> -> vector<1x4xf32>
  return %v : vector<1x4xf32>
}
""", ["memref.load", "arith.extf", "vector<1x4xf32>"]),
            "numeric_casts": ("""
func.func @casts(%v: vector<2x4xf32>, %i: i32) -> (vector<2x4xf16>, f32)
    attributes {thread_num = 64 : i32} {
  %a = frisk.cast %v : vector<2x4xf32> -> vector<2x4xf16>
  %b = frisk.cast %i : i32 -> f32
  return %a, %b : vector<2x4xf16>, f32
}
""", ["arith.truncf", "arith.sitofp"]),
            "nested_view": ("""
func.func @views(%src: memref<4x8x16xf32, 1>, %i: index) -> vector<1x4xf32>
    attributes {thread_num = 64 : i32} {
  %v = frisk.buffer_view %src[1, %i, 2], ranges = [4, 8]
       : memref<4x8x16xf32, 1> -> memref<4x8xf32, 1>
  %s = frisk.buffer_view %v[1, 1], ranges = [1, 4]
       : memref<4x8xf32, 1> -> memref<1x4xf32, 1>
  %r = frisk.to_threadTile %s : memref<1x4xf32, 1> -> vector<1x4xf32>
  return %r : vector<1x4xf32>
}
""", ["memref.subview", "strided<[16, 1]", "memref.load"]),
        }
        for name, (source, expected) in cases.items():
            path = work / (name + ".mlir")
            path.write_text(source)
            result = run(path, work)
            for token in expected:
                assert token in result, (name, token, result)
            print("PASS", name)

        scratch_cases = {
            "forward_multiple_loads": ("""
func.func @forward(%v: vector<4xf32>, %flag: i1) -> vector<4xf32>
    attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %tmp = memref.alloca() : memref<4xf32>
  vector.store %v, %tmp[%c0] : memref<4xf32>, vector<4xf32>
  %a = vector.load %tmp[%c0] : memref<4xf32>, vector<4xf32>
  %result = scf.if %flag -> vector<4xf32> {
    %b = vector.load %tmp[%c0] : memref<4xf32>, vector<4xf32>
    %sum = arith.addf %a, %b : vector<4xf32>
    scf.yield %sum : vector<4xf32>
  } else {
    scf.yield %a : vector<4xf32>
  }
  return %result : vector<4xf32>
}
""", False),
            "preserve_intervening_write": ("""
func.func @overwrite(%v: vector<4xf32>, %x: f32) -> vector<4xf32>
    attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %tmp = memref.alloca() : memref<4xf32>
  vector.store %v, %tmp[%c0] : memref<4xf32>, vector<4xf32>
  memref.store %x, %tmp[%c0] : memref<4xf32>
  %r = vector.load %tmp[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
""", True),
            "preserve_escaping_storage": ("""
func.func private @mutate(memref<4xf32>)
func.func @escape(%v: vector<4xf32>) -> vector<4xf32>
    attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %tmp = memref.alloca() : memref<4xf32>
  vector.store %v, %tmp[%c0] : memref<4xf32>, vector<4xf32>
  func.call @mutate(%tmp) : (memref<4xf32>) -> ()
  %r = vector.load %tmp[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
""", True),
            "preserve_different_indices": ("""
func.func @different(%v: vector<4xf32>, %i: index, %j: index) -> vector<4xf32>
    attributes {thread_num = 64 : i32} {
  %tmp = memref.alloca() : memref<8xf32>
  vector.store %v, %tmp[%i] : memref<8xf32>, vector<4xf32>
  %r = vector.load %tmp[%j] : memref<8xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
""", True),
        }
        for name, (source, retained) in scratch_cases.items():
            path = work / (name + ".mlir")
            path.write_text(source)
            result = run(path, work)
            for token in ("memref.alloca", "vector.store", "vector.load"):
                assert (token in result) == retained, (name, token, result)
            if not retained:
                assert "arith.addf %arg0, %arg0" in result, result
            print("PASS", name)


if __name__ == "__main__":
    main()
