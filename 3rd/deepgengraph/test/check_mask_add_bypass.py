#!/usr/bin/env python3
"""Mask bypass legality, fallbacks, idempotence and native CPU edge cases."""
import argparse
import ctypes
import json
from pathlib import Path
import re
import subprocess
import tempfile


def scalar(name, flags="", attrs="", left=False, extra="", score="%score"):
    operands = f"%m, {score}" if left else f"{score}, %m"
    return f"""define float @{name}(float %score, float %mask, i1 %visible, ptr %out) {attrs} {{
entry:
  br i1 %visible, label %clear, label %masked
clear:
  br label %join
masked:
  br label %join
join:
  %m = phi float [ 0.0, %clear ], [ %mask, %masked ]
{extra}  %result = fadd {flags}float {operands}
  ret float %result
}}
"""


def aggregate(name, width, poison=False):
    # Different shapes from the attention fixture, two consumers of a common
    # aggregate mask, plus an empty join-to-consumer block.
    ty = f"[2 x <{width} x float>]"
    indices = "<i32 0, i32 poison>" if poison else "<i32 0, i32 1>"
    return f"""define <2 x float> @{name}(<2 x float> %score, {ty} %mask, i1 %visible) {{
entry:
  br i1 %visible, label %clear, label %masked
clear:
  br label %join
masked:
  br label %join
join:
  %m = phi {ty} [ zeroinitializer, %clear ], [ %mask, %masked ]
  br label %use
use:
  %a = extractvalue {ty} %m, 0
  %b = extractvalue {ty} %m, 1
  %av = shufflevector <{width} x float> %a, <{width} x float> poison, <2 x i32> {indices}
  %bv = shufflevector <{width} x float> %b, <{width} x float> poison, <2 x i32> <i32 0, i32 1>
  %x = fadd nsz <2 x float> %score, %av
  %y = fadd nsz <2 x float> %bv, %score
  %z = fmul <2 x float> %x, %y
  ret <2 x float> %z
}}
"""


def body(module, name):
    return re.search(r"define [^\n]+@" + name + r"\([^\n]*\).*?\n}", module, re.S)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tool", type=Path)
    p.add_argument("--llvm", type=Path, default=Path("/data2/xsl/install/bin"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    a.tool = a.tool.resolve()
    declarations = ("declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)\n"
                    "declare float @llvm.fabs.f32(float)\n")
    cases = {
        "plain": scalar("plain"),
        "nsz": scalar("nsz", "nsz "),
        "lhs": scalar("lhs", "nsz ", left=True),
        "strict": scalar("strict", "nsz ", "strictfp"),
        "optnone": scalar("optnone", "nsz ", "noinline optnone"),
        "multi_use": scalar("multi_use", "nsz ", extra="  store float %m, ptr %out\n"),
        "late_score": scalar("late_score", "nsz ", extra="  %late = fmul float %score, 2.0\n", score="%late"),
        "nonzero": scalar("nonzero", "nsz ").replace("[ 0.0, %clear ]", "[ 1.0, %clear ]"),
        "aggregate4": aggregate("aggregate4", 4),
        "aggregate8": aggregate("aggregate8", 8),
        "poison": aggregate("poison", 4, poison=True),
        "constrained": scalar("constrained").replace("fadd float %score, %m", "call float @llvm.experimental.constrained.fadd.f32(float %score, float %m, metadata !\"round.dynamic\", metadata !\"fpexcept.strict\")"),
        "loop": """define float @loop(float %score, float %mask, i1 %again) {
entry:
  br label %join
join:
  %m = phi float [ 0.0, %entry ], [ %mask, %latch ]
  %r = fadd nsz float %score, %m
  br i1 %again, label %latch, label %exit
latch:
  br label %join
exit:
  ret float %r
}
""",
    }
    # A masked predecessor with another exit must not receive a sunk add.
    cases["other_exit"] = scalar("other_exit", "nsz ").replace(
        "masked:\n  br label %join", "masked:\n  br i1 %visible, label %join, label %exit").replace(
        "  ret float %result\n}", "  ret float %result\nexit:\n  ret float %score\n}")
    cases["double"] = scalar("double", "nsz ").replace("float", "double")
    cases["half"] = scalar("half", "nsz ").replace("float", "half")
    cases["known_positive"] = scalar("known_positive", score="%positive").replace(
        "entry:\n", "entry:\n  %positive = call float @llvm.fabs.f32(float %score)\n")
    cases["dynamic_index"] = """define float @dynamic_index(float %score, <4 x float> %mask, i1 %visible, i32 %index) {
entry:
  br i1 %visible, label %clear, label %masked
clear:
  br label %join
masked:
  br label %join
join:
  %m = phi <4 x float> [ zeroinitializer, %clear ], [ %mask, %masked ]
  %element = extractelement <4 x float> %m, i32 %index
  %result = fadd nsz float %score, %element
  ret float %result
}
"""
    source = declarations + "\n".join(cases.values())
    expected_legal = {"nsz", "lhs", "aggregate4", "aggregate8", "double", "half", "known_positive"}
    outputs = {}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        src = directory / "input.ll"
        src.write_text(source)

        def run(mode, path):
            result = subprocess.run([str(a.tool), "--mask-add-bypass=" + mode, str(path)],
                                    capture_output=True, text=True, check=True)
            return result.stdout

        original = run("off", src)
        for mode in ("off", "legal", "approx"):
            out = run(mode, src)
            expected = set() if mode == "off" else expected_legal | ({"plain"} if mode == "approx" else set())
            for name in cases:
                fn = body(out, name)
                transformed = ".masked = fadd" in fn
                assert transformed == (name in expected), (mode, name, fn)
                if name not in expected:
                    assert fn == body(original, name), (mode, name, "fallback changed")
                else:
                    # No floating additions on the clear branch; aggregate
                    # extraction and shuffles must remain in the masked arm.
                    assert "fadd" not in fn.split("clear:", 1)[1].split("masked:", 1)[0]
                    assert not re.search(r"phi .*zeroinitializer", fn), (mode, name)
            path = directory / (mode + ".ll")
            path.write_text(out)
            again = run(mode, path)
            # Ignore only LLVM's path-derived ModuleID comment.
            clean = lambda s: re.sub(r"^; ModuleID.*\n", "", s)
            assert clean(out) == clean(again), (mode, "not idempotent")
            subprocess.run([str(a.llvm / "opt"), "-passes=default<O3>,verify", str(path),
                            "-disable-output"], check=True)
            outputs[mode] = out
            print("PASS", mode, "transformed", sorted(expected), flush=True)

        # Execute unoptimized and transformed scalar slices on native CPU,
        # passing raw FP bits to preserve signed zero, subnormals and NaNs.
        functions = []
        for prefix, text in (("before", original), ("after", outputs["approx"])):
            for name in ("plain", "nsz", "lhs"):
                renamed = prefix + "_" + name
                functions.append(body(text, name).replace("@" + name + "(", "@" + renamed + "("))
                functions.append(f"""define i32 @{renamed}_bits(i32 %s, i32 %m, i1 %visible) {{
  %sf = bitcast i32 %s to float
  %mf = bitcast i32 %m to float
  %r = call float @{renamed}(float %sf, float %mf, i1 %visible, ptr null)
  %bits = bitcast float %r to i32
  ret i32 %bits
}}""")
        path = directory / "execute.ll"
        path.write_text("\n".join(functions))
        obj, library = directory / "execute.o", directory / "execute.so"
        subprocess.run([str(a.llvm / "llc"), "-filetype=obj", "-relocation-model=pic", str(path), "-o", str(obj)], check=True)
        subprocess.run(["cc", "-shared", str(obj), "-o", str(library)], check=True)
        lib = ctypes.CDLL(str(library))
        edge = [0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000,
                0x7fc12345, 0x3f800000, 0xbf800000, 0x7f7fffff]
        comparisons = 0
        for name in ("plain", "nsz", "lhs"):
            before, after = [getattr(lib, prefix + "_" + name + "_bits") for prefix in ("before", "after")]
            for fn in (before, after):
                fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
                fn.restype = ctypes.c_uint32
            for s in edge:
                for m in edge:
                    for visible in (False, True):
                        x, y = before(s, m, visible), after(s, m, visible)
                        nan = lambda v: (v & 0x7f800000) == 0x7f800000 and bool(v & 0x7fffff)
                        assert x == y or (nan(x) and nan(y)) or (
                            visible and (x & 0x7fffffff) == 0 and (y & 0x7fffffff) == 0), (name, s, m, visible, hex(x), hex(y))
                        comparisons += 1
        print("PASS", comparisons, "native CPU FP edge comparisons", flush=True)
    report = {"cases": len(cases), "modes": ["off", "legal", "approx"],
              "cpu_comparisons": comparisons, "o3_verified": True, "idempotent": True}
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
