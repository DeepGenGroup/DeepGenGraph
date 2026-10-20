#!/usr/bin/env python3
"""Synchronization safety, bounded specialization and CPU recurrence checks."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "build/test/ThreadTilingTest"
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data2/xsl/install/bin")
SYNC = "--frisk-coalesce-waits-and-barriers"
LOOP = "--frisk-simplify-loop-control"


def run(source, *args, ok=True):
    p = subprocess.run([str(TOOL), "--allow-unregistered-dialect",
                        "--verify-each", *args], input=source, text=True,
                       capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, p.stderr[-8000:]
    return p.stdout


def wait(name, count=0):
    return f'rocdl.wait.asyncmark {count} {{test.id = "{name}"}}\n'


def barrier(name):
    return f'gpu.barrier {{test.id = "{name}"}}\n'


def sync_checks():
    def check(name, body, retained):
        source = """func.func @probe(%a: memref<4xi32, 3>, %cond: i1, %n: index) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %x = arith.constant 1 : i32
        """ + body + "\nreturn\n}"
        out = run(source, SYNC)
        assert re.findall(r'test.id = "([^"]+)"', out) == retained, (name, out)
        assert run(out, SYNC) == out, name
        print("PASS sync:", name)

    seq = wait("w0") + barrier("b0") + wait("w1") + barrier("b1")
    check("wait-all publication", seq, ["w0", "b0"])
    check("adjacent barriers", barrier("a") + barrier("b"), ["a"])
    check("pure index expressions", wait("a") + barrier("b") +
          "%j = arith.addi %n, %c1 : index\n" +
          wait("c") + barrier("d"), ["a", "b"])
    check("wait after earlier barrier must be published",
          barrier("a") + wait("b") + barrier("c"), ["a", "b", "c"])
    for label, middle in [
        ("store", "memref.store %x, %a[%c0] : memref<4xi32, 3>\n"),
        ("read", "%v = memref.load %a[%c0] : memref<4xi32, 3>\n"),
        ("new async group", "rocdl.asyncmark\n"),
        ("unknown effects", '"test.opaque"() : () -> ()\n'),
        ("opaque assembly", 'llvm.inline_asm has_side_effects "", "" : () -> ()\n'),
        ("named buffer wait", "frisk.wait_buffer_ready %a : memref<4xi32, 3>\n"),
        ("conditional", "scf.if %cond { rocdl.asyncmark }\n"),
    ]:
        check(label, wait("a") + barrier("b") + middle +
              wait("c") + barrier("d"), ["a", "b", "c", "d"])
    check("partial wait", wait("a", 1) + barrier("b") +
          wait("c", 1) + barrier("d"), ["a", "b", "c", "d"])
    check("loop boundary", seq +
          "scf.for %i = %c0 to %n step %c1 {\n" +
          wait("inner0") + barrier("inner1") +
          "}\n" + wait("tail0") + barrier("tail1"),
          ["w0", "b0", "inner0", "inner1", "tail0", "tail1"])
    check("branch-local merge", "scf.if %cond {\n" + seq + "}\n" +
          wait("tail0") + barrier("tail1"), ["w0", "b0", "tail0", "tail1"])


def loop_fixture(kind, lb=2, ub=13, step=3, marker=True, name="probe"):
    header = (f"affine.for %i = {lb} to {ub} step {step}" if kind == "affine"
              else "scf.for %i = %lb to %ub step %step")
    return f"""func.func @{name}(%seed: i64) -> i64 {{
      %lb = arith.constant {lb} : index
      %ub = arith.constant {ub} : index
      %step = arith.constant {step} : index
      %two = arith.constant 2 : index
      %three = arith.constant 3 : i64
      %r = {header} iter_args(%acc = %seed) -> (i64) {{
        %slot = arith.remui %i, %two : index
        %nexti = arith.addi %i, %step : index
        %valid = arith.cmpi slt, %nexti, %ub : index
        %addr = arith.select %valid, %slot, %i : index
        %v = arith.index_cast %addr : index to i64
        %p = arith.muli %acc, %three : i64
        %next = arith.addi %p, %v : i64
        {kind}.yield %next : i64
      }}{ " {frisk.simplify_loop}" if marker else ""}
      return %r : i64
    }}"""


def loop_checks():
    for kind in ("affine", "scf"):
        for lb, ub, step in [(2, 13, 3), (-5, 4, 2), (4, 4, 1), (4, 2, 1),
                              (9, 10, 2)]:
            source = loop_fixture(kind, lb, ub, step)
            out = run(source, LOOP)
            assert kind + ".for" not in out, out
            assert "arith.remui" not in out and "arith.select" not in out, out
            assert run(out, LOOP) == out
            # Compare to untouched recurrence through CPU LLVM execution.
            reference = loop_fixture(kind, lb, ub, step, False, "reference")
            body = out.strip()
            assert body.startswith("module {") and body.endswith("}")
            combined = body[len("module {"):-1] + reference + """
              func.func @main() -> i32 {
                %seed = arith.constant 7 : i64
                %a = func.call @probe(%seed) : (i64) -> i64
                %b = func.call @reference(%seed) : (i64) -> i64
                %bad = arith.cmpi ne, %a, %b : i64
                %code = arith.extui %bad : i1 to i32
                return %code : i32
              }"""
            with tempfile.TemporaryDirectory(prefix="loop-control-") as directory:
                work = Path(directory)
                (work / "input.mlir").write_text(combined)
                subprocess.run([LLVM / "mlir-opt", work / "input.mlir",
                                "--lower-affine", "--convert-scf-to-cf",
                                "--convert-arith-to-llvm", "--convert-func-to-llvm",
                                "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
                                "-o", work / "low.mlir"], check=True, capture_output=True)
                subprocess.run([LLVM / "mlir-translate", "--mlir-to-llvmir",
                                work / "low.mlir", "-o", work / "out.ll"],
                               check=True, capture_output=True)
                subprocess.run([LLVM / "lli", work / "out.ll"],
                               check=True, capture_output=True)
            print("PASS loop CPU:", kind, lb, ub, step)
        for label, source, options in [
            ("unmarked", loop_fixture(kind, marker=False), LOOP),
            ("trip budget", loop_fixture(kind, ub=100), LOOP),
            ("growth budget", loop_fixture(kind), LOOP + "=max-expanded-ops=1"),
        ]:
            assert kind + ".for" in run(source, options), label
            print("PASS preserved:", kind, label)
        assert kind + ".for" not in run(loop_fixture(kind, marker=False),
                                       LOOP + "=only-marked=false")
    dynamic = loop_fixture("scf").replace(
        "@probe(%seed: i64)", "@probe(%seed: i64, %ub: index)").replace(
        "%ub = arith.constant 13 : index", "")
    assert "scf.for" in run(dynamic, LOOP)
    for kind in ("affine", "scf"):
        wrapping = loop_fixture(kind, (1 << 63) - 5, (1 << 63) - 1, 3)
        assert kind + ".for" in run(wrapping, LOOP)
    narrow = """func.func @narrow() {
      %lb = arith.constant 120 : i8
      %ub = arith.constant 127 : i8
      %step = arith.constant 3 : i8
      scf.for %i = %lb to %ub step %step : i8 {
        "test.effect"(%i) : (i8) -> ()
      } {frisk.simplify_loop}
      return
    }"""
    assert "scf.for" in run(narrow, LOOP)
    nested = """func.func @nested() {
      affine.for %i = 0 to 4 {
        affine.for %j = 1 to 9 step 2 {
          "test.effect"(%i, %j) : (index, index) -> ()
        } {frisk.simplify_loop}
      } {frisk.simplify_loop}
      return
    }"""
    out = run(nested, LOOP)
    assert "affine.for" not in out and out.count('"test.effect"') == 16
    assert run(out, LOOP) == out
    print("PASS nested loops preserve all ordered effects")
    run(loop_fixture("scf"), LOOP + "=max-trip-count=0", ok=False)
    print("PASS dynamic bounds, wrapping increments and invalid budget")


if __name__ == "__main__":
    sync_checks()
    loop_checks()
