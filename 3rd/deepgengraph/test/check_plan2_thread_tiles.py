#!/usr/bin/env python3
"""Verify the plan2 tiling pass using an opt driver linked to plan2.cpp.

Usage: python3 check_plan2_thread_tiles.py /path/to/plan2-opt
The opt driver must link the plan2 implementation selected by CMake.
Checks IR/type/dominance invariants, not GPU numerical execution.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(sys.argv[1]).resolve()
TEST_DIR = Path(__file__).resolve().parent


def run(source, directory, name, lower=True):
    src = directory / (name + ".mlir")
    dst = directory / (name + ".out.mlir")
    src.write_text(source)
    command = [str(TOOL), str(src), "-o", str(dst)]
    if lower:
        command.append("--convert-friskbase-to-thread")
    proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, name + "\n" + proc.stderr[-6000:]
    return dst.read_text()


def check(source, directory, name):
    ir = run(source, directory, name)
    assert "unrealized_conversion_cast" not in ir, name
    # All temporary bridges retain an explicit processing marker.
    for line in ir.splitlines():
        if "frisk.to_threadTile" in line or "frisk.from_threadTile" in line:
            assert "tiled = true" in line, line
    twice = run(ir, directory, name + "-twice")
    assert ir == twice, name + ": repeated tiling changed IR"
    return ir


COMBINED = r"""module {
func.func @combined(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>, %out: memref<32x32xf32, 3>, %red: memref<32x1xf32, 3>) -> memref<32x32xf32, 3> attributes {thread_num = 64 : i32} {
  %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32, 3>
  %e = frisk.exp2 %g : memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  %s = frisk.add %e, %g : memref<32x32xf32, 3>, memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  %d = frisk.sub %s, %g : memref<32x32xf32, 3>, memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  %t = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x32xf32, 3>
  %f = frisk.fill %t {value = 2.0 : f32} : memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  %m = frisk.mul %d, %f : memref<32x32xf32, 3>, memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  frisk.reduce %m, %red {dim = 1 : i64, kind = "add"} : memref<32x32xf32, 3>, memref<32x1xf32, 3>
  %v = frisk.div %m, %red : memref<32x32xf32, 3>, memref<32x1xf32, 3> -> memref<32x32xf32, 3>
  %r = frisk.copy %v to %out [affine_map<() -> (2)>] : memref<32x32xf32, 3>, memref<32x32xf32, 3> -> memref<32x32xf32, 3>
  return %r : memref<32x32xf32, 3>
}
}
"""

BLOCK = r"""module {
func.func @block(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>, %out: memref<32x32xf32, 3>) attributes {thread_num = 64 : i32} {
  %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32>
  "frisk.block"() ({
  ^bb0(%i: index, %j: index):
    %x = affine.load %g[%i, %j] : memref<32x32xf32>
    %v = arith.addf %x, %x : f32
    affine.store %v, %out[%i, %j] : memref<32x32xf32, 3>
    "frisk.end"() : () -> ()
  }) {ranges = array<i64: 32, 32>} : () -> ()
  return
}
}
"""

def gemm(m, n, k, threads, space=3):
    a, b, c = f"memref<{m}x{k}xf16, {space}>", f"memref<{k}x{n}xf16, 3>", f"memref<{m}x{n}xf32>"
    return f"""func.func @gemm(%a: {a}, %b: {b}) -> {c} attributes {{thread_num = {threads} : i32}} {{
      %g = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
      return %g : {c}
    }}"""


LOOP = """
func.func @loop(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>, %init: vector<32x32xf32>) -> vector<32x32xf32> attributes {thread_num = 64 : i32} {
  %out = affine.for %i = 0 to 3 iter_args(%acc = %init) -> vector<32x32xf32> {
    %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32>
    %next = frisk.add %acc, %g : vector<32x32xf32>, memref<32x32xf32> -> memref<32x32xf32>
    frisk.copy %next to %acc [affine_map<() -> (2)>] : memref<32x32xf32>, vector<32x32xf32>
    affine.yield %acc : vector<32x32xf32>
  }
  return %out : vector<32x32xf32>
}
"""

COPY_TO_REG = """
func.func @copy_to_reg(%src: memref<32x32xf16, 3>, %i: index) -> vector<1x4xf32> attributes {thread_num = 64 : i32} {
  %v = frisk.copy_to_reg %src at (%i) [affine_map<(d0) -> (d0, 4)>] : memref<32x32xf16, 3> -> vector<1x4xf32>
  return %v : vector<1x4xf32>
}
"""

# Read/write selectors share storage but have distinct SSA values and layouts.
# In particular K is 128x32 and V is 32x128: inferring all scf.if results from
# the first known result would mix their element ownership.
PIPELINE_SLOTS = """
func.func @pipeline_slots(%q: memref<64x128xf16, 3>,
    %kg: memref<128x32xf16, 1>, %vg: memref<32x128xf16, 1>, %even: i1)
    -> vector<64x128xf32> attributes {thread_num = 128 : i32} {
  %k0 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
  %k1 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
  %v0 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
  %v1 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
  %p0 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x32xf16, 3>
  %p1 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x32xf16, 3>
  frisk.copy %kg to %k0 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
  frisk.copy %vg to %v0 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
  %s:6 = scf.if %even -> (memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<64x32xf16, 3>, memref<64x32xf16, 3>) {
    scf.yield %k0, %v1, %k1, %v0, %p1, %p0 : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<64x32xf16, 3>, memref<64x32xf16, 3>
  } else {
    scf.yield %k1, %v0, %k0, %v1, %p0, %p1 : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<64x32xf16, 3>, memref<64x32xf16, 3>
  }
  scf.if %even {
    frisk.copy %kg to %s#0 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
  }
  frisk.copy %vg to %s#1 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
  %qk = frisk.gemm(%q, %s#2) {transA = false, transB = false} : memref<64x128xf16, 3>, memref<128x32xf16, 3> -> memref<64x32xf32>
  %pv = frisk.gemm(%s#5, %s#3) {transA = false, transB = false} : memref<64x32xf16, 3>, memref<32x128xf16, 3> -> memref<64x128xf32>
  %p = frisk.exp2 %qk : memref<64x32xf32> -> memref<64x32xf32, 3>
  frisk.copy %p to %s#4 [affine_map<() -> (2)>] : memref<64x32xf32, 3>, memref<64x32xf16, 3>
  %result = frisk.copy_to_reg %pv [affine_map<() -> (2)>] : memref<64x128xf32> -> vector<64x128xf32>
  return %result : vector<64x128xf32>
}
"""

LOCAL_FILL = """
func.func @local_fill(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>) -> memref<32x32xf32> attributes {thread_num = 64 : i32} {
  %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32>
  %t = frisk.alloc_buffer {scope = "local", alignment = 16} -> memref<32x32xf32>
  %f = frisk.fill %t {value = 2.0 : f32} : memref<32x32xf32> -> memref<32x32xf32>
  %sum = frisk.add %g, %f : memref<32x32xf32>, memref<32x32xf32> -> memref<32x32xf32>
  return %sum : memref<32x32xf32>
}
"""


PARTIAL_COPY = """
func.func @partial_copy(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>, %dst: memref<64x64xf32, 3>) -> memref<64x64xf32, 3> attributes {thread_num = 64 : i32} {
  %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32>
  %copy = frisk.copy %g to %dst [affine_map<() -> (16, 16)>] : memref<32x32xf32>, memref<64x64xf32, 3> -> memref<64x64xf32, 3>
  return %copy : memref<64x64xf32, 3>
}
"""

BRIDGE_PAIR = """
func.func @bridges(%x: vector<2x8xf32>) -> vector<2x8xf32> attributes {thread_num = 64 : i32} {
  %b = frisk.from_threadTile %x {tiled = true, warp_layout = array<i64: 8, 8>} : vector<2x8xf32> -> memref<32x32xf32>
  %r = frisk.to_threadTile %b {tiled = true, warp_layout = array<i64: 8, 8>} : memref<32x32xf32> -> vector<2x8xf32>
  return %r : vector<2x8xf32>
}
"""


def check_bridge_pairs(directory):
    ir = check(BRIDGE_PAIR, directory, "bridge-pair")
    assert "frisk." not in ir and "return %arg0 : vector<2x8xf32>" in ir
    # Both thread consumers are replaced, but the escaping block value keeps
    # FromThreadTile alive. Erasing the pair unconditionally would leave a use
    # of an erased value here.
    multi = BRIDGE_PAIR.replace(
        "-> vector<2x8xf32> attributes", "-> (vector<2x8xf32>, vector<2x8xf32>, memref<32x32xf32>) attributes")
    multi = multi.replace("  return %r : vector<2x8xf32>", """
  %r2 = frisk.to_threadTile %b {tiled = true, warp_layout = array<i64: 8, 8>} : memref<32x32xf32> -> vector<2x8xf32>
  return %r, %r2, %b : vector<2x8xf32>, vector<2x8xf32>, memref<32x32xf32>""")
    ir = check(multi, directory, "bridge-multiple-users")
    assert "frisk.to_threadTile" not in ir and ir.count("frisk.from_threadTile") == 1
    assert "return %arg0, %arg0," in ir
    chain = BRIDGE_PAIR.replace("  return %r : vector<2x8xf32>", """
  %b2 = frisk.from_threadTile %r {tiled = true, warp_layout = array<i64: 8, 8>} : vector<2x8xf32> -> memref<32x32xf32>
  %r2 = frisk.to_threadTile %b2 {tiled = true, warp_layout = array<i64: 8, 8>} : memref<32x32xf32> -> vector<2x8xf32>
  return %r2 : vector<2x8xf32>""")
    assert "frisk." not in check(chain, directory, "bridge-chain")
    different_type = BRIDGE_PAIR.replace("-> vector<2x8xf32>", "-> vector<4x4xf32>").replace(
        "return %r : vector<2x8xf32>", "return %r : vector<4x4xf32>")
    ir = check(different_type, directory, "bridge-different-type")
    assert "frisk.to_threadTile" in ir and "frisk.from_threadTile" in ir
    different_layout = BRIDGE_PAIR.replace(
        "%b {tiled = true, warp_layout = array<i64: 8, 8>}",
        "%b {tiled = true, warp_layout = array<i64: 4, 16>}")
    ir = check(different_layout, directory, "bridge-different-layout")
    assert "frisk.to_threadTile" in ir and "frisk.from_threadTile" in ir
    print("PASS bridge cancellation, multiple users, chains, type/layout guards")


def check_memref_bridge_pairs(directory):
    source = """
func.func @memref_bridges(%x: memref<2x32xf16>, %v: vector<2x32xf16>) -> (vector<2x32xf16>, memref<64x128xf16>) attributes {thread_num = 128 : i32} {
  %c0 = arith.constant 0 : index
  %b = frisk.from_threadTile %x {tiled = true, warp_layout = array<i64: 16, 4>} : memref<2x32xf16> -> memref<64x128xf16>
  vector.store %v, %x[%c0, %c0] : memref<2x32xf16>, vector<2x32xf16>
  %r = frisk.to_threadTile %b {tiled = true, warp_layout = array<i64: 16, 4>} : memref<64x128xf16> -> vector<2x32xf16>
  return %r, %b : vector<2x32xf16>, memref<64x128xf16>
}
"""
    ir = check(source, directory, "memref-bridge-live-block")
    assert "frisk.to_threadTile" not in ir
    assert ir.count("frisk.from_threadTile") == 1
    assert "vector.load %arg0[" in ir
    assert ir.index("vector.store") < ir.index("vector.load"), "load moved before write"

    dead_block = source.replace(
        "-> (vector<2x32xf16>, memref<64x128xf16>) attributes",
        "-> vector<2x32xf16> attributes").replace(
        "return %r, %b : vector<2x32xf16>, memref<64x128xf16>",
        "return %r : vector<2x32xf16>")
    ir = check(dead_block, directory, "memref-bridge-dead-block")
    assert "frisk." not in ir and ir.count("vector.load") == 1

    repeated = dead_block.replace("  return %r : vector<2x32xf16>", """
  vector.store %v, %x[%c0, %c0] : memref<2x32xf16>, vector<2x32xf16>
  %r2 = frisk.to_threadTile %b {tiled = true, warp_layout = array<i64: 16, 4>} : memref<64x128xf16> -> vector<2x32xf16>
  %sum = arith.addf %r, %r2 : vector<2x32xf16>
  return %sum : vector<2x32xf16>""")
    ir = check(repeated, directory, "memref-bridge-repeated-read")
    assert "frisk." not in ir
    accesses = re.findall(r"vector\.(store|load)", ir)
    assert accesses == ["store", "load", "store", "load"], accesses

    # Equal memref types still fold directly, without loading memory.
    same_type = BRIDGE_PAIR.replace("vector<2x8xf32>", "memref<2x8xf32>")
    ir = check(same_type, directory, "memref-bridge-same-type")
    assert "frisk." not in ir and "vector.load" not in ir
    assert "return %arg0 : memref<2x8xf32>" in ir

    guards = {
        "layout": source.replace(
            "%b {tiled = true, warp_layout = array<i64: 16, 4>}",
            "%b {tiled = true, warp_layout = array<i64: 4, 16>}"),
        "shape": source.replace("memref<2x32xf16>", "memref<4x16xf16>").replace(
            "  vector.store %v, %x[%c0, %c0] : memref<4x16xf16>, vector<2x32xf16>", ""),
        "element": source.replace("memref<2x32xf16>", "memref<2x32xf32>").replace(
            "  vector.store %v, %x[%c0, %c0] : memref<2x32xf32>, vector<2x32xf16>", ""),
        "stride": source.replace("memref<2x32xf16>",
            "memref<2x32xf16, strided<[64, 2]>>").replace(
            "  vector.store %v, %x[%c0, %c0] : memref<2x32xf16, strided<[64, 2]>>, vector<2x32xf16>", ""),
    }
    for name, guarded in guards.items():
        ir = check(guarded, directory, "memref-bridge-guard-" + name)
        assert "frisk.to_threadTile" in ir and "frisk.from_threadTile" in ir
        assert "vector.load" not in ir
    print("PASS memref bridge loads, write ordering, live users and guards")


def main():
    with tempfile.TemporaryDirectory(prefix="plan2-tiles-") as temp:
        directory = Path(temp)
        check_bridge_pairs(directory)
        check_memref_bridge_pairs(directory)
        for m, n, k, threads, space in [(16,16,16,64,3), (64,32,128,64,3),
                (64,128,32,128,3), (64,128,32,256,0)]:
            name = f"gemm-{m}-{n}-{k}-{threads}"
            ir = check(gemm(m,n,k,threads,space), directory, name)
            assert "frisk.gemm" not in ir
            assert ir.count("frisk.warp_mma_rr") == m*n//(256*(threads//64))
            assert f"-> memref<{m}x{n}xf32>" in ir
            print("PASS", name)
        for kind in ("add", "mul", "min", "max"):
            ir = check(COMBINED.replace('kind = "add"', f'kind = "{kind}"'), directory, "combined-"+kind)
            assert "gpu.shuffle" in ir
            assert "affine.store" in ir and "gpu.barrier" in ir
            assert not re.search(r"frisk\.(?:fill|copy\b|reduce|exp2|add|sub|mul|div|alloc_buffer)", ir)
            print("PASS combined-"+kind)
        zero = COMBINED.replace("%s = frisk.add %e, %g", "%z = frisk.zero : memref<32x32xf32, 3>\n  %s = frisk.add %e, %z")
        assert "frisk.zero" not in check(zero, directory, "zero")
        ir = check(BLOCK, directory, "block")
        assert "frisk.block" not in ir and "affine.store" in ir
        ir = check(LOOP, directory, "loop")
        loop_arg = re.search(r"affine.for .* = 0 to 3 iter_args\((%\w+) =", ir)[1]
        yielded = re.search(r"affine.yield (%\w+) : vector<32x32xf32>", ir)[1]
        assert loop_arg != yielded, "legacy vector copy did not update yield"
        assert f"frisk.to_threadTile {loop_arg} " in ir, "pre-copy accumulator read lost"
        chained = LOOP.replace(
            "    affine.yield %acc",
            "    frisk.copy %acc to %acc [affine_map<() -> (2)>] : "
            "vector<32x32xf32>, vector<32x32xf32>\n    affine.yield %acc")
        assert "frisk.copy " not in check(chained, directory, "chained-updates")
        local = check(LOCAL_FILL, directory, "local-fill")
        assert "memref.alloca" in local and "memref<2x8xf32>" in local
        assert "frisk.fill" not in local and "memref.copy" not in local
        print("PASS block, zero and loop-carried block-vector update")
        ir = check(COPY_TO_REG, directory, "copy-to-reg")
        assert "frisk.copy_to_reg" not in ir and "arith.extf" in ir
        assert "frisk.buffer_view" in ir and "vector<1x4xf32>" in ir
        ir = check(PIPELINE_SLOTS, directory, "pipeline-slots")
        assert "scf.if" in ir and "scf.yield" in ir
        assert not re.search(r"frisk\.(?:copy\b|copy_to_reg|gemm\b)", ir)
        assert "vector<64x128xf32>" in ir and "frisk.warp_mma_rr" in ir
        print("PASS pipeline slot layout propagation and whole-tile copy_to_reg")
        partial = check(PARTIAL_COPY, directory, "partial-copy")
        assert "frisk.buffer_view" in partial and "[16, 16]" in partial
        assert "return %arg2 : memref<64x64xf32, 3>" in partial, "copy result must alias full destination"
        # A result-bearing slice copy must also support a higher-rank backing
        # buffer. Keep the result live so dropping an unused result cannot hide
        # an incorrect result type or loss of the untouched destination region.
        rank4 = PARTIAL_COPY.replace("memref<64x64xf32, 3>", "memref<1x2x64x64xf32, 1>").replace(
            "affine_map<() -> (16, 16)>", "affine_map<() -> (0, 1, 16, 16)>")
        ir = check(rank4, directory, "rank4-copy-result")
        assert "frisk.copy " not in ir
        assert "[0, 1, 16, 16]" in ir and "affine.store" in ir
        assert "return %arg2 : memref<1x2x64x64xf32, 1>" in ir
        assert not re.search(r"frisk.to_threadTile %arg2\b", ir), "only the copied slice has a tile layout"
        # A fully marked operation must be skipped, including analysis.
        marked = gemm(16,16,16,64).replace("transA = false", "tiled = true, transA = false")
        ir = check(marked, directory, "already-tiled")
        assert "frisk.gemm" in ir and "frisk.warp_mma_rr" not in ir
        attention = (TEST_DIR / "test_friskBase.mlir").read_text()
        ir = check(attention, directory, "attention")
        assert "frisk.mask" not in ir and "frisk.convert_layout" not in ir
        assert "vector<64x128xf32>, vector<64x1xf32>" in ir
        assert "affine.yield" in ir and "frisk.from_threadTile" in ir
        assert "memref.copy" not in ir, "writeback must use per-thread stores"
        debug = check((TEST_DIR / "test_friskBaseDebug.mlir").read_text(), directory, "attention-debug")
        assert "frisk.copy " not in debug and "frisk.gemm" not in debug
        assert "frisk.from_threadTile" in debug and "affine.store" in debug
        print("PASS copy-to-reg, tiled guard and attention/idempotence")
        print("PASS partial-copy aliases, rank4-copy-result and attention-debug/idempotence")


if __name__ == "__main__":
    main()
