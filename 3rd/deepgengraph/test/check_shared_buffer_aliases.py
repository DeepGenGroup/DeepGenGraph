#!/usr/bin/env python3
"""CPU semantics and conservative fallback for selected/loop-carried LDS.

Usage: check_shared_buffer_aliases.py [ThreadTilingTest] [LLVM bin directory]
"""
import tempfile
from pathlib import Path

from check_shared_packing import TOOL, run, cpu, gemm


def fixture(rows, cols, axis):
    ty = f"memref<{rows}x{cols}xf32, 3>"
    attrs = f"frisk.shared_pack = 2 : i64, frisk.shared_pack_axis = {axis} : i64"
    return f"""func.func @probe(%choose: i1, %n: index, %r: index, %c: index) -> i32 {{
  %a = memref.alloc() alignment = 16 {{{attrs}}} : {ty}
  %b = memref.alloc() alignment = 16 {{{attrs}}} : {ty}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %offset = arith.constant 10000.0 : f32
  affine.for %i = 0 to {rows} {{
    affine.for %j = 0 to {cols} {{
      %idx = affine.apply affine_map<(d0,d1) -> (d0*37+d1*3)>(%i,%j)
      %int = arith.index_cast %idx : index to i32
      %v = arith.sitofp %int : i32 to f32
      %w = arith.addf %v, %offset : f32
      affine.store %v, %a[%i,%j] : {ty}
      affine.store %w, %b[%i,%j] : {ty}
    }}
  }}
  %slots:2 = scf.if %choose -> ({ty}, {ty}) {{
    scf.yield %a, %b : {ty}, {ty}
  }} else {{
    scf.yield %b, %a : {ty}, {ty}
  }}
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%buf = %slots#0) -> {ty} {{
    %next = arith.select %choose, %slots#1, %buf : {ty}
    scf.yield %next : {ty}
  }}
  %selected = affine.for %i = 0 to 2 iter_args(%buf = %loop) -> {ty} {{
    affine.yield %buf : {ty}
  }}
  %v = memref.load %selected[%r,%c] : {ty}
  %av = memref.load %a[%r,%c] : {ty}
  %bv = memref.load %b[%r,%c] : {ty}
  %empty = arith.cmpi eq, %n, %c0 : index
  %isa = arith.andi %choose, %empty : i1
  %expected = arith.select %isa, %av, %bv : f32
  %ok0 = arith.cmpf oeq, %v, %expected : f32
  // Writing a selected descriptor must update the corresponding direct alias.
  %changed = arith.addf %v, %offset : f32
  memref.store %changed, %selected[%r,%c] : {ty}
  %aftera = memref.load %a[%r,%c] : {ty}
  %afterb = memref.load %b[%r,%c] : {ty}
  %wanta = arith.select %isa, %changed, %av : f32
  %wantb = arith.select %isa, %bv, %changed : f32
  %ok1 = arith.cmpf oeq, %aftera, %wanta : f32
  %ok2 = arith.cmpf oeq, %afterb, %wantb : f32
  %ok01 = arith.andi %ok0, %ok1 : i1
  %ok = arith.andi %ok01, %ok2 : i1
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one : i32
  memref.dealloc %a : {ty}
  memref.dealloc %b : {ty}
  return %status : i32
}}
func.func @main() -> i32 {{
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  %false = arith.constant false
  %sum = affine.for %n = 0 to 4 iter_args(%acc = %zero) -> i32 {{
    %rs = affine.for %r = 0 to {rows} iter_args(%acc2 = %acc) -> i32 {{
      %cs = affine.for %c = 0 to {cols} iter_args(%acc3 = %acc2) -> i32 {{
        %a = func.call @probe(%true,%n,%r,%c) : (i1,index,index,index) -> i32
        %b = func.call @probe(%false,%n,%r,%c) : (i1,index,index,index) -> i32
        %s = arith.addi %a, %b : i32
        %t = arith.addi %acc3, %s : i32
        affine.yield %t : i32
      }}
      affine.yield %cs : i32
    }}
    affine.yield %rs : i32
  }}
  return %sum : i32
}}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="shared-aliases-") as tmp:
        work = Path(tmp)
        for rows, cols, axis in [(12, 7, 0), (7, 12, 1)]:
            source = fixture(rows, cols, axis)
            packed = run([TOOL, "--pack-shared-memory", "--verify-each"], source)
            assert packed.count("frisk.packed_width") == 2
            assert f"memref<{rows}x{cols}xf32, 3>" not in packed
            cpu(source, work, f"base-{axis}")
            cpu(packed, work, f"packed-{axis}")
            assert run([TOOL, "--pack-shared-memory"], packed) == packed
            print(f"PASS CPU axis {axis}: both branches, SCF/affine iter_args, zero-trip, every coordinate, alias writes")

        source = fixture(12, 7, 0)
        ty = "memref<12x7xf32, 3>"
        for name, text in [
            ("mixed-contract", source.replace("frisk.shared_pack = 2", "frisk.shared_pack = 4", 1)),
            ("unmarked-slot", source.replace("frisk.shared_pack = 2 : i64, ", "", 1)),
            ("escape-selected", f"func.func private @escape({ty})\n" + source.replace(
                "  %v = memref.load %selected", f"  func.call @escape(%selected) : ({ty}) -> ()\n  %v = memref.load %selected")),
            ("escape-other-slot", f"func.func private @escape({ty})\n" + source.replace(
                "  %v = memref.load %selected", f"  func.call @escape(%b) : ({ty}) -> ()\n  %v = memref.load %selected")),
            ("unknown-origin", source.replace(
                f"  %b = memref.alloc() alignment = 16 {{frisk.shared_pack = 2 : i64, frisk.shared_pack_axis = 0 : i64}} : {ty}",
                f"  %external = memref.alloc() : {ty}\n  %b = memref.cast %external : {ty} to {ty}")),
        ]:
            output = run([TOOL, "--pack-shared-memory", "--verify-each"], text)
            assert "frisk.packed_width" not in output, name
            print("PASS atomic fallback", name)

        # Automatic planning must see GEMM users through a nested selector,
        # including a live destination-aliasing copy result.
        for m, n, k, threads in [(32, 32, 16, 64), (64, 48, 32, 128)]:
            ty = f"memref<{k}x{n}xf16, 3>"
            src = gemm(m, n, k, threads).replace("@matmul(%a:", "@matmul(%choose: i1, %a:")
            src = src.replace("  frisk.copy", "  %written = frisk.copy", 1).replace(
                f", {ty}\n", f", {ty} -> {ty}\n", 1)
            src = src.replace("  %c = frisk.gemm", f'''  %other = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {ty}
  frisk.copy %global to %other [affine_map<() -> (2)>] : memref<{k}x{n}xf16, 1>, {ty}
  %branch = scf.if %choose -> ({ty}) {{
    scf.yield %written : {ty}
  }} else {{
    scf.yield %other : {ty}
  }}
  %selected = arith.select %choose, %branch, %other : {ty}
  %c = frisk.gemm''').replace("gemm(%a, %b)", "gemm(%a, %selected)")
            path, out = work/"gemm.mlir", work/"gemm.out.mlir"
            path.write_text(src)
            run([TOOL, path, "--convert-friskbase-to-thread", "--finalize-thread-tiling",
                 "--pack-shared-memory", "--verify-each", "-o", out])
            text = out.read_text()
            assert text.count("frisk.packed_width") == 2, text
            print(f"PASS generic GEMM {m}x{n}x{k}: selected storage, live copy result")
            carried = src.replace("  %c = frisk.gemm", f'''  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %carried = scf.for %i = %zero to %two step %one iter_args(%slot = %selected) -> {ty} {{
    %next = arith.select %choose, %slot, %other : {ty}
    scf.yield %next : {ty}
  }}
  %c = frisk.gemm''').replace("gemm(%a, %selected)", "gemm(%a, %carried)")
            path.write_text(carried)
            run([TOOL, path, "--convert-friskbase-to-thread", "--finalize-thread-tiling",
                 "--pack-shared-memory", "--verify-each", "-o", out])
            assert out.read_text().count("frisk.packed_width") == 2
            print(f"PASS generic GEMM {m}x{n}x{k}: SCF loop-carried storage inference")

        # Computed producers store directly into selected allocations. The
        # destination's consumers must not force an extra FP32 layout scratch.
        a, p, c = "memref<32x32xf16, 3>", "memref<32x64xf16, 3>", "memref<32x64xf32>"
        src = f'''func.func @computed(%choose: i1, %a: {a}, %b: {p}, %rhs: memref<64x32xf16, 3>, %out: memref<32x32xf32, 1>) attributes {{thread_num = 64 : i32}} {{
  %value = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {p} -> {c}
  %first = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {p}
  %second = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {p}
  %selected = arith.select %choose, %first, %second : {p}
  frisk.copy %value to %selected [affine_map<() -> (2)>] : {c}, {p}
  %r = frisk.gemm(%selected, %rhs) {{transA = false, transB = false}} : {p}, memref<64x32xf16, 3> -> memref<32x32xf32>
  frisk.copy %r to %out [affine_map<() -> (2)>] : memref<32x32xf32>, memref<32x32xf32, 1>
  return
}}'''
        path.write_text(src)
        run([TOOL, path, "--convert-friskbase-to-thread", "--finalize-thread-tiling",
             "--pack-shared-memory", "--verify-each", "-o", out])
        text = out.read_text()
        assert text.count("frisk.packed_width") == 2
        assert text.count("memref.alloc()") == 2
        assert text.count("gpu.barrier") == 1
        print("PASS generic computed producer: selected packed destinations, no FP32 scratch")


if __name__ == "__main__":
    main()
