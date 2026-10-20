#!/usr/bin/env python3
"""Automatic prefetch: CPU recurrence/slot checks and conservative rejection.

CPU execution replaces async transfers by immediate copies. It checks values,
slot rotation and boundary guards; it does not simulate GPU completion timing.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "build/test/ThreadTilingTest"
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data2/xsl/install/bin")
PASS = "--frisk-auto-pipeline"


def run(args, source=None, ok=True):
    p = subprocess.run(list(map(str, args)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr[-8000:]
    return p.stdout, p.stderr


def fixture(hint="2 : i64", name="probe", asynchronous=True):
    src = "memref<64xf32, 1>"
    view = "memref<1xf32, strided<[1], offset: ?>, 1>"
    dst = "memref<1xf32, 3>"
    loads = ""
    for suffix in ("a", "b"):
        loads += f"""    %{suffix} = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {dst}
    %v{suffix} = memref.subview %src{suffix}[%offset] [1] [1] : {src} to {view}
"""
        if asynchronous:
            loads += f"""    frisk.async_copy_gm_to_shm %v{suffix} to %{suffix} : {view}, {dst}
    frisk.wait_buffer_ready %{suffix} : {dst}
"""
        else:
            loads += f"""    %h{suffix} = frisk.copy %v{suffix} to %{suffix} [affine_map<() -> (1)>] : {view}, {dst} -> {dst}
"""
    return f"""func.func @{name}(%srca: {src}, %srcb: {src}, %lb: index, %ub: index) -> (f32, f32) {{
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %three = arith.constant 3.0 : f32
  %r:2 = affine.for %i = %lb to %ub step 3 iter_args(%acc = %zero, %sum = %zero) -> (f32, f32) {{
    %offset = affine.apply affine_map<(d0, d1) -> (d0 - d1 + 16)>(%i, %lb)
{loads}
    %x = memref.load %{'a' if asynchronous else 'ha'}[%c0] : {dst}
    %y = memref.load %{'b' if asynchronous else 'hb'}[%c0] : {dst}
    %p = arith.mulf %acc, %three : f32
    %q = arith.addf %x, %y : f32
    %next = arith.addf %p, %q : f32
    %sum_next = arith.addf %sum, %y : f32
    affine.yield %next, %sum_next : f32, f32
  }}{'' if hint is None else ' {frisk.pipeline_hint' + ('' if hint == 'unit' else ' = ' + hint) + '}'}
  return %r#0, %r#1 : f32, f32
}}
"""


def cpu_ir(text):
    # Remove only the target memory spaces and block synchronization.
    text = re.sub(r", [13]>", ">", text)
    text = re.sub(r'frisk\.alloc_buffer \{[^}]*\} -> (memref<[^\n]+>)',
                  r'memref.alloc() : \1', text)
    text = re.sub(
        r"frisk\.async_copy_gm_to_shm (%[\w#]+) to (%[\w#]+)(?: \{[^}]*\})? : (memref<.*>), (memref<.*>)",
        r"memref.copy \1, \2 : \3 to \4", text)
    text = re.sub(r"^.*frisk\.(?:wait_buffer_ready|sync_threads_in_block).*\n",
                  "", text, flags=re.M)
    return text


def main_ir():
    src = "memref<64xf32, 1>"
    s = f"""func.func @main() -> i32 {{
  %a = memref.alloc() : {src}
  %b = memref.alloc() : {src}
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %two = arith.constant 2.0 : f32
  affine.for %i = 0 to 64 {{
    %x = arith.index_cast %i : index to i32
    %f = arith.sitofp %x : i32 to f32
    %g = arith.mulf %f, %two : f32
    affine.store %f, %a[%i] : {src}
    affine.store %g, %b[%i] : {src}
  }}
"""
    previous = "%c0"
    # Zero, negative trip count, one tile, partial step, multiple ring wraps.
    for n, (lb, ub) in enumerate([(0, 0), (4, 1), (-3, -2), (-3, 5),
                                  (-3, 16), (2, 3), (2, 17),
                                  ((1 << 63) - 7, (1 << 63) - 1),
                                  (-(1 << 63), -(1 << 63) + 6)]):
        s += f"""  %lb{n} = arith.constant {lb} : index
  %ub{n} = arith.constant {ub} : index
  %p{n}:2 = func.call @probe(%a, %b, %lb{n}, %ub{n}) : ({src}, {src}, index, index) -> (f32, f32)
  %r{n}:2 = func.call @reference(%a, %b, %lb{n}, %ub{n}) : ({src}, {src}, index, index) -> (f32, f32)
  %ok{n}a = arith.cmpf oeq, %p{n}#0, %r{n}#0 : f32
  %ok{n}b = arith.cmpf oeq, %p{n}#1, %r{n}#1 : f32
  %ok{n} = arith.andi %ok{n}a, %ok{n}b : i1
  %bad{n} = arith.select %ok{n}, %c0, %c1 : i32
  %total{n} = arith.addi {previous}, %bad{n} : i32
"""
        previous = f"%total{n}"
    return s + f"  return {previous} : i32\n}}\n"


def execute_cpu(text, work, name):
    src, low, ll = [work / (name + ext) for ext in (".mlir", ".low.mlir", ".ll")]
    src.write_text(cpu_ir(text))
    run([LLVM / "mlir-opt", src, "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
         "--finalize-memref-to-llvm", "--convert-arith-to-llvm",
         "--convert-func-to-llvm", "--convert-cf-to-llvm",
         "--reconcile-unrealized-casts", "-o", low])
    run([LLVM / "mlir-translate", "--mlir-to-llvmir", low, "-o", ll])
    run([LLVM / "lli", ll])


def main():
    with tempfile.TemporaryDirectory(prefix="auto-pipeline-") as directory:
        work = Path(directory)
        for hint, stages in [("unit", 2), ("2 : i64", 2), ("3 : i64", 3), ("4 : i64", 4)]:
            source = fixture(hint) + fixture(None, "reference") + main_ir()
            out, err = run([TOOL, PASS, "--verify-each"], source)
            assert "skipped" not in err, err
            assert out.count("pipeline.scheduled") == 1
            assert f"pipeline.num_stages = {stages} : i64" in out
            # N slots per input plus two untouched reference allocations.
            assert out.count("frisk.alloc_buffer") == 2 * stages + 2
            assert run([TOOL, PASS], out)[0] == out, "not idempotent"
            execute_cpu(out, work, f"stages{stages}-{hint == 'unit'}")
            print(f"PASS stages={stages}: guards, negative/nonzero/extreme bounds, step=3, short loops, ring wraps, two recurrences")

        # Both synchronous copy handles and explicit async transfers are accepted.
        out, err = run([TOOL, PASS, "--verify-each"], fixture(asynchronous=False))
        assert "pipeline.scheduled" in out and "skipped" not in err
        assert "frisk.copy " not in out
        multi, _ = run([TOOL, PASS], fixture() + fixture("3 : i64", "other"))
        assert multi.count("pipeline.scheduled") == 2
        no_carriers = fixture().replace(" -> (f32, f32)", "", 1)
        no_carriers = no_carriers.replace("%r:2 = affine.for", "affine.for")
        no_carriers = no_carriers.replace(
            " iter_args(%acc = %zero, %sum = %zero) -> (f32, f32)", "")
        no_carriers = re.sub(r"%acc\b|%sum\b", "%zero", no_carriers)
        no_carriers = no_carriers.replace("affine.yield %next, %sum_next : f32, f32",
                                          "affine.yield")
        no_carriers = no_carriers.replace("return %r#0, %r#1 : f32, f32", "return")
        out, _ = run([TOOL, PASS, "--verify-each"], no_carriers)
        assert "pipeline.scheduled" in out
        print("PASS synchronous result aliases, multiple hinted loops, no iter_args, idempotence")

        plain = fixture(None)
        assert run([TOOL], plain)[0] == run([TOOL, PASS], plain)[0]
        disabled = fixture("1 : i64")
        assert run([TOOL], disabled)[0] == run([TOOL, PASS], disabled)[0]
        for hint in ["0 : i64", "-2 : i64", '"two"', "2.0 : f32"]:
            _, err = run([TOOL, PASS], fixture(hint), ok=False)
            assert "frisk.pipeline_hint" in err

        bad = {
            "global-write": fixture().replace(
                "    %p = arith.mulf",
                "    memref.store %x, %srca[%offset] : memref<64xf32, 1>\n    %p = arith.mulf"),
            "escaping-alias": fixture().replace(
                "    %x = memref.load",
                "    %alias = memref.cast %a : memref<1xf32, 3> to memref<?xf32, 3>\n    %x = memref.load"),
            "missing-wait": fixture().replace(
                "    frisk.wait_buffer_ready %a : memref<1xf32, 3>\n", ""),
            "early-consumer": fixture().replace(
                "    frisk.wait_buffer_ready %a : memref<1xf32, 3>",
                "    %early = memref.load %a[%c0] : memref<1xf32, 3>\n    frisk.wait_buffer_ready %a : memref<1xf32, 3>"),
            "opaque-call": "func.func private @opaque()\n" + fixture().replace(
                "    %p = arith.mulf", "    func.call @opaque() : () -> ()\n    %p = arith.mulf"),
            "recurrence-address": fixture().replace(
                "    %offset = affine.apply affine_map<(d0, d1) -> (d0 - d1 + 16)>(%i, %lb)",
                "    %int = arith.fptosi %acc : f32 to i64\n    %offset = arith.index_cast %int : i64 to index"),
            "divergent-bounds": fixture().replace(
                "  %r:2 = affine.for %i = %lb",
                "  %tid = gpu.thread_id x\n  %r:2 = affine.for %i = %tid"),
            "too-many-stages": fixture("65 : i64"),
            "explicit-schedule": fixture().replace(
                "{frisk.pipeline_hint = 2 : i64}",
                "{frisk.pipeline_hint = 2 : i64, pipeline.stage = array<i64: 0>}"),
        }
        for name, source in bad.items():
            out, err = run([TOOL, PASS, "--verify-each"], source)
            assert "automatic pipeline skipped" in err, (name, err)
            assert out == run([TOOL], source)[0], name
            print("PASS unchanged rejection:", name)
        print("PASS unmarked/disabled no-op and invalid attribute diagnostics")


if __name__ == "__main__":
    main()
