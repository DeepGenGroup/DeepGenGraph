#!/usr/bin/env python3
"""Regression checks for --ir-deep-optimize using FragmentOptTest."""
import re
import subprocess
import sys
from pathlib import Path


TOOL = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parent.parent / "build/test/FragmentOptTest")


def run(source):
    result = subprocess.run([str(TOOL), "--ir-deep-optimize"], input=source,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def check_static(ir):
    for line in ir.splitlines():
        if re.search(r"vector\.(insert|extract)\s", line):
            assert "%" not in line.split("[", 1)[1].split("]", 1)[0], line


def main():
    nested = """
func.func @nested(%src: vector<2x4xf32>, %init: vector<2x4xf32>) -> vector<2x4xf32> {
  %r = affine.for %i = 0 to 2 iter_args(%a = %init) -> vector<2x4xf32> {
    %s = affine.for %j = 0 to 4 iter_args(%b = %a) -> vector<2x4xf32> {
      %rev = affine.apply affine_map<(d0) -> (3 - d0)>(%j)
      %x = vector.extract %src[%i, %rev] : f32 from vector<2x4xf32>
      %y = vector.insert %x, %b[%i, %j] : f32 into vector<2x4xf32>
      affine.yield %y : vector<2x4xf32>
    }
    affine.yield %s : vector<2x4xf32>
  }
  return %r : vector<2x4xf32>
}
"""
    ir = run(nested)
    assert "affine.for" not in ir, ir
    assert ir.count("vector.extract ") == 8, ir
    assert ir.count("vector.insert ") == 8, ir
    check_static(ir)
    print("PASS nested IVs, affine.apply and vector iter_args")

    carried = """
func.func @carried(%src: vector<8xf32>, %init: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %r:2 = affine.for %i = 0 to 4 iter_args(%p = %c0, %v = %init)
      -> (index, vector<4xf32>) {
    %x = vector.extract %src[%p] : f32 from vector<8xf32>
    %y = vector.insert %x, %v[%i] : f32 into vector<4xf32>
    %next = arith.addi %p, %c2 : index
    affine.yield %next, %y : index, vector<4xf32>
  }
  return %r#1 : vector<4xf32>
}
"""
    # Use constant insert positions so only the carried index triggers unrolling.
    carried = carried.replace("%v[%i]", "%v[0]")
    ir = run(carried)
    assert "affine.for" not in ir, ir
    check_static(ir)
    assert "[6]" in ir, ir
    print("PASS constant-seeded index recurrence")

    dynamic = carried.replace("%init: vector<4xf32>",
                              "%init: vector<4xf32>, %start: index")
    ir = run(dynamic.replace("%p = %c0", "%p = %start"))
    assert "affine.for" in ir and "arith.addi" in ir, ir
    print("PASS runtime-seeded index recurrence stays dynamic")

    stepped = """
func.func @stepped(%src: vector<16xf32>, %init: f32) -> f32 {
  %r = affine.for %i = 2 to 10 step 2 iter_args(%s = %init) -> f32 {
    %p = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
    %x = vector.extract %src[%p] : f32 from vector<16xf32>
    %next = arith.addf %s, %x : f32
    affine.yield %next : f32
  }
  return %r : f32
}
"""
    ir = run(stepped)
    assert "affine.for" not in ir, ir
    check_static(ir)
    for index in (3, 5, 7, 9):
        assert f"[{index}]" in ir, ir
    print("PASS nonzero lower bound and nonunit step")

    shifted = stepped.replace("%init: f32", "%init: f32, %offset: index")
    shifted = shifted.replace("2 to 10 step 2", "%offset to #end(%offset)")
    ir = run("#end = affine_map<(d0) -> (d0 + 4)>\n" + shifted)
    assert "affine.for" in ir, ir
    print("PASS constant trip count with runtime bounds stays dynamic")

    bounded = stepped.replace("vector<16xf32>", "vector<128xf32>")
    ir = run(bounded.replace("2 to 10 step 2", "0 to 65"))
    assert "affine.for" in ir, ir
    print("PASS trip-count growth limit")

    outer = nested.replace("  %r = affine.for", """
  %result = affine.for %k = 0 to 128 iter_args(%acc = %init) -> vector<2x4xf32> {
  %r = affine.for""", 1).replace("%a = %init", "%a = %acc")
    outer = outer.replace("  return %r : vector<2x4xf32>", """
    affine.yield %r : vector<2x4xf32>
  }
  return %result : vector<2x4xf32>""")
    ir = run(outer)
    assert ir.count("affine.for") == 1 and "to 128" in ir, ir
    check_static(ir)
    print("PASS unrelated outer accumulation loop is preserved")


if __name__ == "__main__":
    main()
