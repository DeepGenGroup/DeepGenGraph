#!/usr/bin/env python3
"""Check view composition and naive GM->LDS copies, including CPU numerics.

Usage: check_affine_copy.py [ThreadTilingTest] [LLVM bin directory]
CPU execution serializes thread IDs and removes the block barrier; this checks
addresses, coverage and conversion, not GPU synchronization at runtime.
"""
import itertools
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


def run(command, source=None):
    proc = subprocess.run(list(map(str, command)), input=source, text=True,
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, f"{command}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return proc.stdout


def memref(shape, elem="f32", space=None):
    return "memref<" + "x".join(map(str, shape)) + "x" + elem + (
        f", {space}" if space else "") + ">"


def flatten(coords, shape):
    result = 0
    for coordinate, extent in zip(coords, shape):
        result = result * extent + coordinate
    return result


def cpu_main(src_shape, dst_shape, tile, origin, dst_origin, threads, elem):
    src_type, dst_type = memref(src_shape), memref(dst_shape, elem)
    lines = ["func.func @main() -> i32 {",
             f"%src = memref.alloc() : {src_type}",
             f"%dst = memref.alloc() : {dst_type}",
             "%offset = arith.constant 2 : index",
             "%ok = arith.constant true", "%zero = arith.constant 0 : i32",
             "%one = arith.constant 1 : i32", "%minus = arith.constant -1.0 : " + elem]
    for axis, extent in enumerate(src_shape):
        lines.append(f"affine.for %i{axis} = 0 to {extent} {{")
    dims = ", ".join(f"d{i}" for i in range(len(src_shape)))
    ivs = ", ".join(f"%i{i}" for i in range(len(src_shape)))
    terms = " + ".join(f"d{i} * {math.prod(src_shape[i+1:])}" for i in range(len(src_shape)))
    lines += [f"%linear = affine.apply affine_map<({dims}) -> ({terms})>({ivs})",
              "%int = arith.index_cast %linear : index to i32",
              "%value = arith.sitofp %int : i32 to f32",
              f"memref.store %value, %src[{ivs}] : {src_type}"]
    lines += ["}"] * len(src_shape)
    lines.append(f"%all = affine.for %tid = 0 to {threads} iter_args(%prior = %ok) -> i1 {{")
    for axis, extent in enumerate(dst_shape):
        lines.append(f"affine.for %j{axis} = 0 to {extent} {{")
    dst_ivs = ", ".join(f"%j{i}" for i in range(len(dst_shape)))
    lines += [f"memref.store %minus, %dst[{dst_ivs}] : {dst_type}"]
    lines += ["}"] * len(dst_shape)
    lines += [f"func.call @copy(%src, %dst, %offset, %tid) : ({src_type}, {dst_type}, index, index) -> ()"]
    previous = "%prior"
    per_thread = (math.prod(tile) + threads - 1) // threads
    for number, coord in enumerate(itertools.product(*(range(n) for n in dst_shape))):
        local = [a - b for a, b in zip(coord, dst_origin)]
        inside = all(0 <= a < n for a, n in zip(local, tile))
        source = list(origin)
        if inside:
            for axis, value in enumerate(local):
                source[len(source) - len(local) + axis] += value
        expected = flatten(source, src_shape) if inside else -1
        lines += [f"%v{number} = affine.load %dst[{', '.join(map(str, coord))}] : {dst_type}",
                  f"%e{number} = arith.constant {expected}.0 : {elem}"]
        reference = f"%e{number}"
        if inside:
            owner = flatten(local, tile) // per_thread
            lines += [f"%owner{number} = arith.constant {owner} : index",
                      f"%owns{number} = arith.cmpi eq, %tid, %owner{number} : index",
                      f"%expected{number} = arith.select %owns{number}, %e{number}, %minus : {elem}"]
            reference = f"%expected{number}"
        lines += [f"%c{number} = arith.cmpf oeq, %v{number}, {reference} : {elem}",
                  f"%ok{number} = arith.andi {previous}, %c{number} : i1"]
        previous = f"%ok{number}"
    lines += [f"affine.yield {previous} : i1", "}",
              "%status = arith.select %all, %zero, %one : i32",
              f"memref.dealloc %src : {src_type}", f"memref.dealloc %dst : {dst_type}",
              "return %status : i32", "}"]
    return "\n".join(lines)


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TEST_DIR.parent / "build/test/ThreadTilingTest"
    llvm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data2/xsl/install/bin")
    pipeline = "builtin.module(func.func(convert-friskbase-to-thread,finalize-thread-tiling))"
    # No GEMM or LowerInfo anchor: every copy must lower from thread_num alone.
    cases = [
        ("rank4_K_stride", (1, 2, 16, 64), (8, 16), (8, 16), (0, 1, 3, 16), (0, 0), 8, "f32", False),
        ("nested_source_destination", (2, 12, 20), (8, 12), (4, 8), (1, 3, 4), (2, 2), 8, "f32", True),
        ("partial_thread_interval", (8, 16), (3, 5), (3, 5), (2, 3), (0, 0), 8, "f32", False),
        ("idle_threads_and_cast", (8, 16), (1, 3), (1, 3), (2, 3), (0, 0), 32, "f16", False),
        ("interval_spans_rows", (8, 16), (4, 6), (4, 6), (2, 3), (0, 0), 2, "f32", False),
    ]
    with tempfile.TemporaryDirectory(prefix="affine-copy-") as temp:
        work = Path(temp)
        for name, src_shape, dst_shape, tile, origin, dst_origin, threads, elem, nested in cases:
            src_type, dst_type = memref(src_shape, space=1), memref(dst_shape, elem, 3)
            view_type = memref(tile, space=1)
            if nested:
                views = f"""
  %outer = frisk.buffer_view %src[1, symbol(%offset), 3], ranges = [8, 12]
    : {src_type} -> memref<8x12xf32, 1>
  %view = frisk.buffer_view %outer[1, 1], ranges = [4, 8]
    : memref<8x12xf32, 1> -> {view_type}
  %dstview = frisk.buffer_view %dst[%offset, %offset], ranges = [4, 8]
    : {dst_type} -> memref<4x8xf32, 3>
  %alias = frisk.copy %view to %dstview [affine_map<() -> (0, 0)>]
    : {view_type}, memref<4x8xf32, 3> -> memref<4x8xf32, 3>
  %probe = memref.load %alias[%c0, %c0] : memref<4x8xf32, 3>
  memref.store %probe, %alias[%c0, %c0] : memref<4x8xf32, 3>
"""
            else:
                # Exercise copy's own offset map, including dims and symbols.
                expressions = list(map(str, origin))
                expressions[-2] = f"s0 + {origin[-2] - 2}"
                amap = "affine_map<()[s0] -> (" + ", ".join(expressions) + ")>"
                views = f"frisk.copy %src to %dst at (%offset) [{amap}] : {src_type}, {dst_type}"
            source = f"""module {{
func.func @copy(%src: {src_type}, %dst: {dst_type}, %offset: index, %tid: index)
    attributes {{thread_num = {threads} : i32}} {{
  %c0 = arith.constant 0 : index
  {views}
  return
}}
}}"""
            path = work / f"{name}.mlir"
            path.write_text(source)
            output = work / "lowered.mlir"
            run([binary, path, "--pass-pipeline=" + pipeline, "-o", output])
            ir = output.read_text()
            assert not re.search(r"frisk\.(?:buffer_view|copy|to_threadTile|from_threadTile)\b|memref.subview|unrealized_conversion_cast", ir), ir
            assert "vector.store" in ir and ir.count("frisk.sync_threads_in_block") == 1, ir
            assert re.search(r"^    frisk.sync_threads_in_block", ir, re.M), "barrier must be outside the per-thread guard"
            assert "gpu.thread_id" in ir and "tile_layout" not in ir, ir
            # Check exact coverage and per-thread contiguous intervals in the
            # lowered code by executing one thread at a time. Other elements
            # must remain untouched; this also detects replicated writes.
            cpu = re.sub(r"(%[\w]+) = gpu.thread_id\s+x",
                         r"\1 = affine.apply affine_map<(d0) -> (d0)>(%arg3)", ir)
            cpu = re.sub(r"\s*frisk.sync_threads_in_block[^\n]*", "", cpu)
            cpu = re.sub(r", [13](?=>)", "", cpu)
            cpu = cpu.rstrip().rsplit("}", 1)[0] + cpu_main(src_shape, dst_shape, tile, origin, dst_origin, threads, elem) + "\n}"
            lowered = run([llvm / "mlir-opt", "--lower-affine", "--convert-scf-to-cf",
                           "--convert-vector-to-llvm", "--finalize-memref-to-llvm",
                           "--convert-arith-to-llvm", "--convert-cf-to-llvm",
                           "--convert-func-to-llvm", "--convert-ub-to-llvm",
                           "--reconcile-unrealized-casts"], cpu)
            llvm_ir = run([llvm / "mlir-translate", "--mlir-to-llvmir"], lowered)
            ll = work / "copy.ll"
            ll.write_text(llvm_ir)
            run([llvm / "lli", ll])
            print(f"PASS {name}: direct original-buffer accesses, vector stores, sync, CPU values")

        # Both memref and affine consumers must compose their own indices with
        # nested view origins, including non-identity affine load/store maps.
        source = """
func.func @main() -> i32 attributes {thread_num = 64 : i32} {
  %src = memref.alloc() : memref<2x8x16xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32
  %outer = frisk.buffer_view %src[1, 2, 3], ranges = [4, 8]
    : memref<2x8x16xi32> -> memref<4x8xi32>
  %view = frisk.buffer_view %outer[1, 1], ranges = [2, 4]
    : memref<4x8xi32> -> memref<2x4xi32>
  affine.store %one, %view[%c0 + 1, %c1 * 2] : memref<2x4xi32>
  %a = affine.load %src[1, 4, 6] : memref<2x8x16xi32>
  memref.store %zero, %view[%c1, %c2] : memref<2x4xi32>
  %b = affine.load %view[%c0 + 1, %c1 * 2] : memref<2x4xi32>
  %c = memref.load %view[%c1, %c2] : memref<2x4xi32>
  %err = arith.subi %one, %a : i32
  %err2 = arith.addi %err, %b : i32
  %err3 = arith.addi %err2, %c : i32
  memref.dealloc %src : memref<2x8x16xi32>
  return %err3 : i32
}
"""
        path.write_text(source)
        run([binary, path, "--pass-pipeline=" + pipeline, "-o", output])
        ir = output.read_text()
        assert "frisk.buffer_view" not in ir and "memref.subview" not in ir, ir
        lowered = run([llvm / "mlir-opt", "--lower-affine", "--finalize-memref-to-llvm",
                       "--convert-arith-to-llvm", "--convert-func-to-llvm",
                       "--reconcile-unrealized-casts"], ir)
        ll.write_text(run([llvm / "mlir-translate", "--mlir-to-llvmir"], lowered))
        run([llvm / "lli", ll])
        print("PASS nested affine/memref reads and writes: CPU values at original-buffer coordinates")


if __name__ == "__main__":
    main()
