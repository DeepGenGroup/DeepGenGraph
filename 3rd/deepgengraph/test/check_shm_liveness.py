#!/usr/bin/env python3
"""Storage-lifetime, alias, packing and barrier regressions (no GPU required).

Usage: python3 check_shm_liveness.py [build/test/ThreadTilingTest] [build/log.log]
The optional log checks the actual post-thread-tiling attention IR as well.
"""
import re
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
BINARY = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          TEST_DIR.parent / "build/test/ThreadTilingTest").resolve()


def run(source, pipeline="test-shm-reuse"):
    proc = subprocess.run(
        [str(BINARY), "--allow-unregistered-dialect",
         f"--pass-pipeline=builtin.module(func.func({pipeline}))"],
        input=source, text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, proc.stderr


def kernel(body, declarations="", result="f32"):
    return f"""module {{
      {declarations}
      func.func @test(%cond: i1, %n: index) -> {result}
          attributes {{thread_num = 128 : i32}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %x = arith.constant 1.0 : f32
        {body}
      }}
    }}"""


def module_after(log, marker):
    stage = log.split(marker, 1)[1]
    start = stage.index("module {")
    depth = 0
    for end in range(start, len(stage)):
        depth += (stage[end] == "{") - (stage[end] == "}")
        if depth == 0 and stage[end] == "}":
            return stage[start:end + 1]
    raise AssertionError("unterminated module")


ALLOCS = """
  %a = memref.alloc() {alignment = 16 : i64} : memref<16xf32, 3>
  %b = memref.alloc() {alignment = 16 : i64} : memref<8xf32, 3>
"""
A = """
  memref.store %x, %a[%c0] : memref<16xf32, 3>
  %av = memref.load %a[%c0] : memref<16xf32, 3>
"""
B = """
  memref.store %x, %b[%c0] : memref<8xf32, 3>
  %bv = memref.load %b[%c0] : memref<8xf32, 3>
"""
RETURN = """
  %r = arith.addf %av, %bv : f32
  return %r : f32
"""


def check(name, source, before, after, barriers):
    ir, report = run(source)
    expected = f"shm bytes {before} -> {after}, barriers {barriers}"
    assert expected in report, f"{name}: expected {expected}\n{report}"
    if before > after:
        assert f"memref<{after}xi8, 3>" in ir, ir
        assert ir.count("frisk.shm_pool") == 1, ir
    # A second application must leave the IR identical, including barriers.
    repeated, _ = run(ir)
    assert repeated == ir, f"{name}: reuse is not idempotent"
    print(f"PASS {name}: {expected}")
    return ir, report


def main():
    ir, _ = check("different_sizes", kernel(ALLOCS + A + B + RETURN), 96, 64, 1)
    assert ir.index("gpu.barrier") < ir.index("memref.store", ir.index("gpu.barrier"))
    check("existing_gpu_barrier", kernel(ALLOCS + A + "gpu.barrier\n" + B + RETURN), 96, 64, 0)
    check("existing_frisk_barrier", kernel(ALLOCS + A + "frisk.sync_threads_in_block\n" + B + RETURN), 96, 64, 0)
    check("overlapping", kernel(ALLOCS + B + A + """
      %again = memref.load %b[%c0] : memref<8xf32, 3>
    """ + RETURN), 96, 96, 0)

    check("subview_extends_lifetime", kernel(ALLOCS + """
      %alias = memref.subview %a[0] [8] [1] : memref<16xf32, 3>
          to memref<8xf32, strided<[1]>, 3>
    """ + A + B + """
      %again = memref.load %alias[%c0] : memref<8xf32, strided<[1]>, 3>
    """ + RETURN), 96, 96, 0)

    same = ALLOCS.replace("8xf32", "16xf32")
    check("same_operation_interferes", kernel(same + """
      memref.copy %a, %b : memref<16xf32, 3> to memref<16xf32, 3>
      return %x : f32
    """), 128, 128, 0)
    check("if_result_aliases", kernel(same + A + """
      memref.store %x, %b[%c0] : memref<16xf32, 3>
      %alias = scf.if %cond -> (memref<16xf32, 3>) {
        scf.yield %a : memref<16xf32, 3>
      } else {
        scf.yield %b : memref<16xf32, 3>
      }
      %v = memref.load %alias[%c0] : memref<16xf32, 3>
      return %v : f32
    """), 128, 128, 0)

    for loop in ("scf.for %i = %c0 to %c4 step %c1", "affine.for %i = 0 to 4"):
        dialect = loop.split(".")[0]
        check(f"{dialect}_backedge", kernel(ALLOCS + f"""
          {loop} {{
            {A}
            {B}
          }}
          return %x : f32
        """), 96, 96, 0)
        # Allocation inside the iteration gives fresh storage. Sharing needs
        # both the A->B handoff and the B->A backedge to synchronize warps.
        local_body = ALLOCS + A + B
        ir, _ = check(f"{dialect}_iteration_local", kernel(f"""
          {loop} {{ {local_body} }}
          return %x : f32
        """), 96, 64, 2)
        assert ir.index("gpu.barrier") > ir.index(f"{dialect}.for"), ir
        check(f"{dialect}_existing_loop_barriers", kernel(f"""
          {loop} {{ {ALLOCS} {A} gpu.barrier {B} gpu.barrier }}
          return %x : f32
        """), 96, 64, 0)
        check(f"{dialect}_prefix_barrier", kernel(f"""
          {loop} {{ {ALLOCS} gpu.barrier {A} gpu.barrier {B} }}
          return %x : f32
        """), 96, 64, 0)
        check(f"{dialect}_iter_arg", kernel(same + A + f"""
          %out = {loop} iter_args(%carried = %a) -> (memref<16xf32, 3>) {{
            {dialect}.yield %carried : memref<16xf32, 3>
          }}
          memref.store %x, %b[%c0] : memref<16xf32, 3>
          %v = memref.load %out[%c0] : memref<16xf32, 3>
          return %v : f32
        """), 128, 128, 0)

    full_allocs = """
      %a = memref.alloc() : memref<128xf32, 3>
      %b = memref.alloc() : memref<128xf32, 3>
      %tid = gpu.thread_id x
    """
    full_a = """
      memref.store %x, %a[%tid] : memref<128xf32, 3>
      %av = memref.load %a[%tid] : memref<128xf32, 3>
    """
    full_b = full_a.replace("%a", "%b")
    check("full_overwrite_each_iteration", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{ {full_a} {full_b} }}
      return %x : f32
    """), 1024, 512, 2)
    check("divergent_loop_bounds", kernel(full_allocs + f"""
      scf.for %i = %c0 to %tid step %c1 {{ {full_a} {full_b} }}
      return %x : f32
    """), 1024, 1024, 0)
    check("read_before_overwrite", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{
        %old = memref.load %a[%tid] : memref<128xf32, 3>
        {full_a} {full_b}
      }}
      return %x : f32
    """), 1024, 1024, 0)
    check("use_after_loop_preserves_storage", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{ {full_a} {full_b} }}
      %v = memref.load %a[%tid] : memref<128xf32, 3>
      return %v : f32
    """), 1024, 1024, 0)
    check("conditional_loop_not_collective", kernel(full_allocs + f"""
      scf.if %cond {{
        scf.for %i = %c0 to %n step %c1 {{ {full_a} {full_b} }}
      }}
      return %x : f32
    """), 1024, 1024, 0)
    check("partial_write_inside_full_sized_buffer", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{
        {full_a.replace('%a[%tid]', '%a[%c0]')} {full_b}
      }}
      return %x : f32
    """), 1024, 1024, 0)
    check("overlapping_loop_lifetimes", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{
        {full_a} {full_b}
        %again = memref.load %a[%tid] : memref<128xf32, 3>
      }}
      return %x : f32
    """), 1024, 1024, 0)
    check("conditional_initialization", kernel(full_allocs + f"""
      scf.for %i = %c0 to %n step %c1 {{
        scf.if %cond {{ {full_a} }}
        {full_b}
      }}
      return %x : f32
    """), 1024, 1024, 0)
    # The vector stores have an affine per-thread mapping, like the real K/V
    # copies. Merely seeing a store or a tiled marker must not imply coverage.
    vector_allocs = """
      %a = memref.alloc() : memref<128x4xf32, 3>
      %b = memref.alloc() : memref<128x4xf32, 3>
      %tid = gpu.thread_id x
      %v = arith.constant dense<1.0> : vector<4xf32>
    """
    vector_body = """
      %row = affine.apply affine_map<(d0) -> (d0 mod 128)>(%tid)
      vector.store %v, %a[%row, %c0] : memref<128x4xf32, 3>, vector<4xf32>
      %av = vector.load %a[%row, %c0] : memref<128x4xf32, 3>, vector<4xf32>
      vector.store %v, %b[%row, %c0] : memref<128x4xf32, 3>, vector<4xf32>
      %bv = vector.load %b[%row, %c0] : memref<128x4xf32, 3>, vector<4xf32>
    """
    for name, body, after, barriers in (
        ("complete_vector_mapping", vector_body, 2048, 2),
        ("incomplete_vector_mapping", vector_body.replace("mod 128", "mod 64"), 4096, 0),
        ("rotating_vector_mapping", vector_body.replace("(d0) -> (d0 mod 128)>(%tid)",
             "(d0, d1) -> ((d0 + d1) mod 128)>(%tid, %i)"), 4096, 0),
    ):
        check(name, kernel(vector_allocs + f"""
          scf.for %i = %c0 to %n step %c1 {{ {body} }}
          return %x : f32
        """), 4096, after, barriers)
    check("loop_local_conditional_storage", kernel(f"""
      scf.for %i = %c0 to %n step %c1 {{
        {ALLOCS}
        scf.if %cond {{ {A} }}
        scf.if %cond {{ {B} }}
      }}
      return %x : f32
    """), 96, 64, 2)
    check("nested_collective_loops", kernel(f"""
      scf.for %i = %c0 to %n step %c1 {{
        affine.for %j = 0 to 4 {{ {ALLOCS} {A} {B} }}
      }}
      return %x : f32
    """), 96, 64, 2)

    ir, _ = check("barrier_outside_divergent_if", kernel(ALLOCS + A + f"""
      scf.if %cond {{ {B} }}
      return %av : f32
    """), 96, 64, 1)
    assert ir.index("gpu.barrier") < ir.index("scf.if"), ir
    check("nested_barrier_not_sufficient", kernel(ALLOCS + A + """
      scf.if %cond { gpu.barrier }
    """ + B + RETURN), 96, 64, 1)

    check("escaping_call", kernel(ALLOCS + A + """
      func.call @escape(%a) : (memref<16xf32, 3>) -> ()
    """ + B + RETURN, "func.func private @escape(memref<16xf32, 3>)"), 32, 32, 0)
    check("unknown_async_use", kernel(ALLOCS + A + """
      "test.async_read"(%a) : (memref<16xf32, 3>) -> ()
    """ + B + RETURN), 32, 32, 0)
    check("async_region", kernel(ALLOCS + A + """
      "test.async_execute"() ({
        %v = memref.load %a[%c0] : memref<16xf32, 3>
        "test.yield"() : () -> ()
      }) : () -> ()
    """ + B + RETURN), 32, 32, 0)
    check("explicit_deallocation", kernel(ALLOCS + A + """
      memref.dealloc %a : memref<16xf32, 3>
    """ + B + RETURN), 32, 32, 0)
    for name, old, new in (
        ("global", "16xf32, 3", "16xf32, 1"),
        ("strided", "16xf32, 3", "16xf32, strided<[2]>, 3"),
        ("dynamic", "16xf32, 3", "?xf32, 3"),
    ):
        body = (ALLOCS + A + B + RETURN).replace(old, new)
        if name == "dynamic":
            body = body.replace("%a = memref.alloc()", "%a = memref.alloc(%n)")
        check(name, kernel(body), 32, 32, 0)

    ir, report = check("mixed_types_alignment", kernel("""
      %a = memref.alloc() {alignment = 16 : i64} : memref<128xi8, 3>
      %b = memref.alloc() {alignment = 64 : i64} : memref<4xf32, 3>
      %c = memref.alloc() {alignment = 32 : i64} : memref<4xf32, 3>
      %byte = arith.constant 1 : i8
      memref.store %byte, %a[%c0] : memref<128xi8, 3>
      memref.store %x, %b[%c0] : memref<4xf32, 3>
      memref.store %x, %c[%c0] : memref<4xf32, 3>
      %v = memref.load %b[%c0] : memref<4xf32, 3>
      return %v : f32
    """), 160, 128, 1)
    assert "placement 2 @ 32" in report, report
    assert "alignment = 64" in ir, ir

    check("multi_block", """module {
      func.func @cfg() attributes {thread_num = 128 : i32} {
        %a = memref.alloc() : memref<16xf32, 3>
        cf.br ^next
      ^next:
        return
      }
    }""", 0, 0, 0)

    # Exercise the actual integration point without block operations to retile.
    integrated, _ = run(kernel(ALLOCS + A + B + RETURN),
                        "convert-friskbase-to-thread")
    assert "frisk.shm_pool" in integrated, integrated

    if len(sys.argv) > 2:
        log = Path(sys.argv[2]).read_text()
        source = module_after(log, "---------- after createConvertFriskBaseToThreadLevelIRPass ---------")
        # Bridges defer pooling until finalization exposes every physical use.
        deferred, _ = run(source)
        if "frisk.to_threadTile" in source:
            assert "frisk.shm_pool" not in deferred
        ir, report = run(source, "finalize-thread-tiling,test-shm-reuse")
        pools = re.findall(r"memref.alloc\(\)[^\n]*frisk.shm_pool[^\n]*memref<(\d+)xi8, 3>", ir)
        assert pools, report
        if "pipeline.scheduled" not in source:
            assert pools == ["16384"], pools
        repeated, _ = run(ir)
        assert repeated == ir
        print(f"PASS attention log: finalized pools {pools} bytes; reuse is idempotent")
        marker = "---------- after frisk-pipeline-schedule ---------"
        if marker in log:
            scheduled = module_after(log, marker)
            integrated, _ = run(scheduled, "convert-friskbase-to-thread,finalize-thread-tiling")
            assert "frisk.shm_pool" in integrated
            assert "frisk.to_threadTile" not in integrated
            print("PASS scheduled attention through conversion and bridge finalization: SHM pool applied")


if __name__ == "__main__":
    main()
