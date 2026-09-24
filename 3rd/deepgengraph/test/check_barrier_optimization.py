#!/usr/bin/env python3
"""Barrier dependency regressions; no GPU is required.

Usage: check_barrier_optimization.py [FragmentOptTest] [build/log.log]
The optional log verifies the attention IR produced before LLVM lowering.
"""
import re
import subprocess
import sys
from pathlib import Path


TOOL = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parent.parent / "build/test/FragmentOptTest")


def run(source):
    proc = subprocess.run([str(TOOL), "--allow-unregistered-dialect",
                           "--barrier-optimize"], input=source, text=True,
                          capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def kernel(body, declarations="", attrs="attributes {thread_num = 128 : i32}"):
    return f"""module {{
      {declarations}
      func.func @test(%a: memref<16xf32, 3>, %b: memref<16xf32, 3>,
                      %out: memref<16xf32, 1>, %cond: i1, %n: index) {attrs} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %x = arith.constant 1.0 : f32
        {body}
        return
      }}
    }}"""


def barrier(name):
    return f'gpu.barrier {{test.id = "{name}"}}\n'


def check(name, body, expected, **kwargs):
    source = kernel(body, **kwargs)
    result = run(source)
    actual = re.findall(r'test.id = "([^"]+)"', result)
    assert actual == expected, f"{name}: {actual} != {expected}\n{result}"
    assert run(result) == result, f"{name}: pass is not idempotent"
    # This pass must only remove barriers, never reorder or change memory ops.
    generic = subprocess.run([str(TOOL), "--allow-unregistered-dialect"],
                             input=source, text=True, capture_output=True,
                             timeout=60)
    assert generic.returncode == 0, generic.stderr
    strip = lambda text: re.sub(r'^.*gpu\.barrier.*\n', '', text, flags=re.M)
    assert strip(generic.stdout) == strip(result), name
    print(f"PASS {name}: retained {expected}")


READ = "%v = memref.load %a[%c0] : memref<16xf32, 3>\n"
WRITE = "memref.store %x, %a[%c0] : memref<16xf32, 3>\n"
OUTPUT = "memref.store %v, %out[%c0] : memref<16xf32, 1>\n"
POOL = """
  %pool = memref.alloc() : memref<128xi8, 3>
  %left = memref.view %pool[%c0][] : memref<128xi8, 3> to memref<16xf32, 3>
  %right = memref.view %pool[%c64][] : memref<128xi8, 3> to memref<16xf32, 3>
"""


def main():
    check("empty", barrier("a") + barrier("b"), [])
    check("RAW", WRITE + barrier("raw") + READ + OUTPUT, ["raw"])
    check("WAR", READ + barrier("war") + WRITE + OUTPUT, ["war"])
    check("WAW", WRITE + barrier("waw") + WRITE, ["waw"])
    check("read_only", READ + barrier("rr") +
          READ.replace("%v =", "%w =") + OUTPUT, [])
    check("adjacent_keep_one", WRITE + barrier("a") + barrier("b") +
          READ + OUTPUT, ["b"])
    check("tail_shared_to_global", WRITE + barrier("ready") + READ +
          barrier("tail") + OUTPUT, ["ready"])
    check("global_RAW", (WRITE + barrier("raw") + READ + OUTPUT)
          .replace("%a[", "%out[").replace("16xf32, 3", "16xf32, 1"), ["raw"])
    check("arguments_may_alias", WRITE + barrier("alias") +
          READ.replace("%a[", "%b[") + OUTPUT, ["alias"])
    private = (WRITE + barrier("private") + READ + OUTPUT)
    private = private.replace("%a[", "%private[").replace("16xf32, 3", "16xf32, 5")
    check("private_memory", "%private = memref.alloca() : memref<16xf32, 5>\n" + private, [])

    copy = (WRITE.replace("%a[", "%left[") + barrier("K") +
            WRITE.replace("%a[", "%right[") + barrier("KV") +
            READ.replace("%a[", "%left[") +
            READ.replace("%a[", "%right[").replace("%v =", "%w =") + OUTPUT)
    check("disjoint_pooled_views", POOL + copy, ["KV"])
    check("overlapping_pooled_views", POOL.replace("%pool[%c64]", "%pool[%c32]") +
          copy, ["K", "KV"])
    check("dynamic_pooled_views", POOL.replace("%pool[%c64]", "%pool[%n]") +
          copy, ["K", "KV"])
    padded = (POOL + copy.replace(OUTPUT, "")).replace("f32", "i24")
    padded = padded.replace("%x,", "%narrow,")
    check("padded_element_width", "%narrow = arith.constant 1 : i24\n" +
          padded, ["K", "KV"])
    check("different_allocations", """
      %left = memref.alloc() : memref<16xf32, 3>
      %right = memref.alloc() : memref<16xf32, 3>
    """ + copy, ["KV"])
    # Different lanes can select different shifted views of the same pool.
    # Nonoverlapping offsets relative to that selected base may still race.
    check("thread_dependent_base", """
      %pool = memref.alloc() : memref<192xi8, 3>
      %lo = memref.view %pool[%c0][] : memref<192xi8, 3> to memref<128xi8, 3>
      %hi = memref.view %pool[%c64][] : memref<192xi8, 3> to memref<128xi8, 3>
      %lane = gpu.thread_id x
      %choose = arith.cmpi eq, %lane, %c0 : index
      %base = arith.select %choose, %lo, %hi : memref<128xi8, 3>
      %left = memref.view %base[%c0][] : memref<128xi8, 3> to memref<16xf32, 3>
      %right = memref.view %base[%c64][] : memref<128xi8, 3> to memref<16xf32, 3>
    """ + copy, ["K", "KV"])
    check("subview_alias", WRITE + """
      %sub = memref.subview %a[0] [16] [1] : memref<16xf32, 3>
          to memref<16xf32, strided<[1]>, 3>
    """ + barrier("alias") + READ.replace("%a[", "%sub[")
          .replace("16xf32, 3", "16xf32, strided<[1]>, 3") + OUTPUT, ["alias"])
    check("cast_alias", WRITE + """
      %cast = memref.cast %a : memref<16xf32, 3> to memref<?xf32, 3>
    """ + barrier("alias") + READ.replace("%a[", "%cast[")
          .replace("16xf32, 3", "?xf32, 3") + OUTPUT, ["alias"])
    check("select_alias", WRITE + """
      %selected = arith.select %cond, %a, %b : memref<16xf32, 3>
    """ + barrier("alias") + READ.replace("%a[", "%selected[") + OUTPUT, ["alias"])
    check("same_global_symbol", """
      %g1 = memref.get_global @shared : memref<16xf32, 3>
      %g2 = memref.get_global @shared : memref<16xf32, 3>
    """ + WRITE.replace("%a[", "%g1[") + barrier("alias") +
          READ.replace("%a[", "%g2[") + OUTPUT, ["alias"],
          declarations='memref.global "private" @shared : memref<16xf32, 3> = uninitialized')
    check("different_lane_indices", "%tid = gpu.thread_id x\n" +
          WRITE.replace("[%c0]", "[%tid]") + barrier("raw") + READ + OUTPUT, ["raw"])

    for loop in ("scf.for %i = %c0 to %n step %c1", "affine.for %i = 0 to %n"):
        dialect = loop.split(".")[0]
        body = WRITE + barrier("ready") + READ + OUTPUT + barrier("backedge")
        check(dialect + "_backedge_and_exit", WRITE + barrier("entry") +
              loop + " {\n" + body + "}\n" + barrier("exit") + WRITE,
              ["entry", "ready", "backedge"])
        check(dialect + "_zero_trip", READ.replace("%v =", "%initial =") +
              loop + " {\n" + body + "}\n" + barrier("zero") + WRITE,
              ["ready", "backedge", "zero"])
        # A single remaining barrier must account for access across iterations.
        check(dialect + "_single_barrier", loop + " {\n" + READ + OUTPUT +
              barrier("war") + WRITE + "}\n", ["war"])
        check(dialect + "_divergent_loop", "%tid = gpu.thread_id x\n" +
              loop.replace("%n", "%tid") + " {\n" + barrier("inside") + "}\n" +
              barrier("outside"), ["inside", "outside"])

    check("conditional_write", "scf.if %cond {\n" + WRITE + "}\n" +
          barrier("raw") + READ + OUTPUT, ["raw"])
    check("conditional_barrier", WRITE + "scf.if %cond {\n" + barrier("conditional") +
          "}\n" + barrier("required") + READ + OUTPUT, ["conditional", "required"])
    check("unknown_call", 'func.call @unknown() : () -> ()\n' + barrier("call") + READ,
          ["call"], declarations="func.func private @unknown()")
    check("unknown_async", '"test.async_copy"(%a) : (memref<16xf32, 3>) -> ()\n' +
          barrier("async") + READ, ["async"])
    check("atomic", """
      %old = memref.atomic_rmw addf %x, %a[%c0] : (f32, memref<16xf32, 3>) -> f32
    """ + barrier("atomic") + READ, ["atomic"])
    check("opaque_asm", 'llvm.inline_asm has_side_effects "", "" : () -> ()\n' +
          barrier("asm") + READ, ["asm"])
    check("callable_boundary", barrier("entry") + WRITE + barrier("exit"),
          ["entry", "exit"], attrs="")

    if len(sys.argv) > 2:
        log = Path(sys.argv[2]).read_text()
        section = log.split("---- after createIRDeepOptimizePass -----", 1)[1]
        start = section.index("module {")
        depth = 0
        for end in range(start, len(section)):
            depth += (section[end] == "{") - (section[end] == "}")
            if depth == 0 and section[end] == "}":
                source = section[start:end + 1]
                break
        result = run(source)
        before, after = source.count("gpu.barrier"), result.count("gpu.barrier")
        assert before == 16 and after == 13, (before, after)
        assert run(result) == result
        print(f"PASS attention: {before} -> {after} barriers")


if __name__ == "__main__":
    main()
