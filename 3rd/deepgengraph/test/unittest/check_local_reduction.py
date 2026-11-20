#!/usr/bin/env python3
"""Target-independent reduction legality, def-use, options and native CPU tests."""
import argparse
import ctypes
import difflib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def fixture(name, op="fadd", ty="float", n=32, flags="", attrs="",
            right=False, escaped=False, split=False, interleaved=False, mixed=False):
    lines = [f"define void @{name}(ptr %input, ptr %output) {attrs} {{", "entry:"]
    def load(i):
        lines.extend([f"  %p{i} = getelementptr {ty}, ptr %input, i64 {i}",
                      f"  %x{i} = load {ty}, ptr %p{i}, align 1"])
    if not interleaved:
        for i in range(n): load(i)
    else: load(0)
    previous = "%x0"
    for i in range(1, n):
        if split and i == n // 2:
            lines.extend(["  br label %next", "next:"])
        if interleaved: load(i)
        lhs, rhs = (f"%x{i}", previous) if right and i % 2 else (previous, f"%x{i}")
        use_flags = "" if mixed and i == n // 2 else flags
        if op.startswith("llvm."):
            expression = f"call {use_flags}{ty} @{op}({ty} {lhs}, {ty} {rhs})"
        else:
            expression = f"{op} {use_flags}{ty} {lhs}, {rhs}"
        lines.append(f"  %r{i} = {expression}")
        if escaped and i == n // 2:
            lines.append(f"  store {ty} %r{i}, ptr %output, align 1")
        previous = f"%r{i}"
    lines.extend([f"  store {ty} {previous}, ptr %output, align 1", "  ret void", "}"])
    return "\n".join(lines) + "\n"


def body(text, name):
    return re.search(r"define [^\n]+@" + name + r"\([^\n]*\).*?\n}", text, re.S)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tool", type=Path)
    p.add_argument("--llvm", type=Path, default=Path("/data2/xsl/install/bin"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    cases, legal, approx, executable = {}, set(), set(), {}
    declarations = set()
    def add(name, expected="legal", dtype=None, **kw):
        cases[name] = fixture(name, **kw)
        if expected == "legal": legal.add(name)
        if expected in ("legal", "approx"): approx.add(name)
        if dtype: executable[name] = dtype
        if kw.get("op", "").startswith("llvm."):
            ty = kw.get("ty", "float")
            declarations.add(f"declare {ty} @{kw['op']}({ty}, {ty})")
    add("plain_add", "approx", "float32")
    add("legal_add", dtype="float32", flags="reassoc nsz ")
    add("reassoc_only", "approx", flags="reassoc ")
    add("nsz_only", "approx", flags="nsz ")
    add("double_add", dtype="float64", ty="double", flags="reassoc nsz ", n=37)
    add("half_add", ty="half", flags="reassoc nsz ", n=19)
    add("bfloat_add", ty="bfloat", flags="reassoc nsz ", n=17)
    add("vector_float", ty="<3 x float>", flags="reassoc nsz ", n=23)
    add("vector_int", op="xor", ty="<7 x i16>", n=27)
    add("scalable", "none", op="add", ty="<vscale x 4 x i32>")
    add("alternating_operands", dtype="float32", flags="reassoc nsz ", right=True, n=47)
    add("late_loads", dtype="float32", flags="reassoc nsz ", interleaved=True, n=29)
    add("short", "none", flags="reassoc nsz ", n=7)
    add("strict", "none", flags="reassoc nsz ", attrs="strictfp")
    add("optnone", "none", flags="reassoc nsz ", attrs="noinline optnone")
    # These split the spine into pieces shorter than the default threshold.
    add("escaped", "none", flags="reassoc nsz ", n=24, escaped=True)
    add("blocks", "none", flags="reassoc nsz ", n=24, split=True)
    add("mixed_flags", "approx", flags="reassoc nsz ", n=24, mixed=True)
    for op in ("add", "mul", "and", "or", "xor"):
        add("integer_" + op, dtype="uint32", ty="i32", op=op, n=35)
    add("nowrap", ty="i32", op="add", flags="nsw nuw ")
    add("disjoint", ty="i32", op="or", flags="disjoint ")
    add("float_mul", dtype="float32", op="fmul", flags="reassoc nsz ")
    for op in ("smin", "smax", "umin", "umax"):
        add(op, dtype="uint32", ty="i32", op=f"llvm.{op}.i32", n=21)
    for op in ("maxnum", "minnum", "maximum", "minimum"):
        add(op, dtype="float32", op=f"llvm.{op}.f32", flags="nnan nsz ")
        add(op + "_plain", "approx", dtype="float32", op=f"llvm.{op}.f32")
    add("subtract", "none", op="fsub", flags="reassoc nsz ")
    add("divide", "none", op="fdiv", flags="reassoc nsz ")
    add("long", op="add", ty="i64", n=257)
    # A loop-carried value is an opaque leaf, never followed through its PHI.
    loop = fixture("loop", op="add", ty="i32", n=23)
    loop = loop.replace("entry:\n", "entry:\n  br label %loop\nloop:\n"
                        "  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]\n"
                        "  %carry = phi i32 [ 0, %entry ], [ %r22, %loop ]\n")
    loop = loop.replace("%r1 = add i32 %x0,", "%r1 = add i32 %carry,")
    loop = loop.replace("  ret void", "  %next = add i32 %iteration, 1\n"
                        "  %again = icmp ult i32 %next, 2\n"
                        "  br i1 %again, label %loop, label %exit\nexit:\n  ret void")
    cases["loop"] = loop
    legal.add("loop"); approx.add("loop")
    constrained = fixture("constrained", n=24)
    constrained = re.sub(r"fadd float ([^,]+), ([^\n]+)",
        r'call float @llvm.experimental.constrained.fadd.f32(float \1, float \2, metadata !"round.dynamic", metadata !"fpexcept.strict")', constrained)
    cases["constrained"] = constrained
    declarations.add("declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)")
    for name, op, ty, identity, flags in (
        ("seed_sum", "fadd", "float", "0.0", "reassoc nsz "),
        ("seed_mul", "mul", "i32", "1", ""),
        ("seed_and", "and", "i32", "-1", ""),
    ):
        fn = fixture(name, op=op, ty=ty, flags=flags, n=33)
        start = f"  %r1 = {op} {flags}{ty} %x0,"
        fn = fn.replace(start, f"  %seed = {op} {flags}{ty} %x0, {identity}\n"
                              f"  %r1 = {op} {flags}{ty} %seed,")
        cases[name] = fn
        legal.add(name); approx.add(name)
    source = "\n".join(sorted(declarations)) + "\n" + "\n".join(cases.values())
    comparisons = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "source.ll"
        src.write_text(source)
        def run(mode, path, extra=()):
            return subprocess.run([str(a.tool.resolve()), "--local-reduce=" + mode, *extra, str(path)],
                                  capture_output=True, text=True, check=True).stdout
        outputs = {}
        for mode in ("off", "legal", "approx"):
            output = run(mode, src)
            expected = {"off": set(), "legal": legal, "approx": approx}[mode]
            for name in cases:
                actual = body(output, name)
                assert ("!frisk.local_reduction" in actual) == (name in expected), (mode, name, actual)
                if name not in expected and mode != "off":
                    assert actual == body(outputs["off"], name), (mode, name, "fallback changed")
            path = tmp / (mode + ".ll")
            path.write_text(output)
            again = run(mode, path)
            # LLVM may renumber metadata even on an off-mode parse/print.
            roundtrip = run("off", path)
            clean = lambda s: re.sub(r"^; ModuleID.*\n", "", s)
            assert clean(roundtrip) == clean(again), (mode, "not idempotent", "\n".join(
                list(difflib.unified_diff(clean(roundtrip).splitlines(), clean(again).splitlines()))[:100]))
            subprocess.run([str(a.llvm / "opt"), "-passes=default<O3>,verify", str(path), "-disable-output"], check=True)
            outputs[mode] = output
            print("PASS", mode, len(expected), "transformed fixtures", flush=True)

        for chains in (2, 3, 5, 8, 16):
            out = run("approx", src, [f"--local-reduce-chains={chains}",
                                     f"--local-reduce-min-inputs={2*chains}"])
            path = tmp / f"chains{chains}.ll"
            path.write_text(out)
            subprocess.run([str(a.llvm / "opt"), "-passes=verify", str(path), "-disable-output"], check=True)
            assert "!frisk.local_reduction" in body(out, "long")
        for operation in ("add", "max", "add,max", "mul", "and", "or", "xor", "min"):
            out = run("approx", src, ["--local-reduce-ops=" + operation])
            assert ("!frisk.local_reduction" in body(out, "plain_add")) == ("add" in operation.split(","))
            assert ("!frisk.local_reduction" in body(out, "maxnum_plain")) == ("max" in operation.split(","))
        for option in ("--local-reduce=bad", "--local-reduce-chains=0", "--local-reduce-chains=17",
                       "--local-reduce-chains=9", "--local-reduce-min-inputs=3",
                       "--local-reduce-min-inputs=2049", "--local-reduce-ops=", "--local-reduce-ops=max,",
                       "--local-reduce-ops=sub", "--local-reduce-chains=-1", "--local-reduce-chains=4x"):
            result = subprocess.run([str(a.tool.resolve()), "--local-reduce=approx", option, str(src)], capture_output=True)
            assert result.returncode != 0, option
        assert "nsw" not in body(outputs["legal"], "nowrap")
        assert "nuw" not in body(outputs["legal"], "nowrap")
        assert "disjoint" not in body(outputs["legal"], "disjoint").split("{", 1)[1]

        # Execute arbitrary CPU functions rather than a model of the rewrite.
        # Compile before/after code together; discard marker/debug metadata only.
        functions = []
        for prefix, text in (("before", outputs["off"]), ("after", outputs["approx"])):
            for name in executable:
                fn = body(text, name).replace("@" + name + "(", "@" + prefix + "_" + name + "(")
                fn = re.sub(r", !frisk.local_reduction !\d+", "", fn)
                functions.append(fn)
        path = tmp / "execute.ll"
        path.write_text("\n".join(sorted(declarations)) + "\n" + "\n".join(functions))
        obj, lib = tmp / "execute.o", tmp / "execute.so"
        subprocess.run([str(a.llvm / "llc"), "-filetype=obj", "-relocation-model=pic", str(path), "-o", str(obj)], check=True)
        subprocess.run(["cc", "-shared", str(obj), "-o", str(lib)], check=True)
        library = ctypes.CDLL(str(lib))
        rng = np.random.default_rng(17)
        for name, dtype in executable.items():
            funcs = [getattr(library, prefix + "_" + name) for prefix in ("before", "after")]
            for fn in funcs: fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; fn.restype = None
            for trial in range(40):
                if dtype == "uint32": values = rng.integers(0, 2**32, size=64, dtype=np.uint32)
                elif name == "float_mul": values = rng.uniform(.95, 1.05, size=64).astype(dtype)
                else: values = rng.uniform(-2, 2, size=64).astype(dtype)
                outputs_native = [np.zeros(1, dtype=dtype) for _ in funcs]
                for fn, out in zip(funcs, outputs_native): fn(values.ctypes.data, out.ctypes.data)
                x, y = outputs_native
                if dtype == "uint32": np.testing.assert_array_equal(x, y)
                else: np.testing.assert_allclose(x, y, atol=2e-5, rtol=2e-5)
                comparisons += 1
        print("PASS", comparisons, "native CPU comparisons", flush=True)
    report = dict(fixtures=len(cases), modes=["off", "legal", "approx"],
                  legal_transformed=len(legal), approx_transformed=len(approx),
                  cpu_comparisons=comparisons, idempotent=True, O3_verified=True,
                  tested_chain_counts=[2, 3, 4, 5, 8, 16])
    if a.output: a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
