#!/usr/bin/env python3
"""One-off, hash-bound mask bypass experiment on the measured native-acc IR."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "online_softmax_native_acc"
EXPECTED = "2ebae1fa1f4366d3c4072d0aec4bd2fe2bcec623cc667bf154a409de0d358b1e"


def generate():
    original = (BASE / "kernel.ll").read_text()
    assert hashlib.sha256((BASE / "kernel.ll").read_bytes()).hexdigest() == EXPECTED
    # Moving numbered SSA definitions would invalidate LLVM's implicit slot
    # numbering. Keep ABI arguments and implicit entry block 4 unchanged.
    text = re.sub(r"%(\d+)\b", lambda m: "%v" + m[1] if int(m[1]) >= 5 else m[0], original)
    text = re.sub(r"^(\d+):", lambda m: "v" + m[1] + ":", text, flags=re.M)
    start = text.index("  %v849 = extractvalue")
    end = text.index("  %v947 = call float @llvm.maxnum")
    moved, remaining, phis = [], [], []
    for line in text[start:end].splitlines():
        if " = extractvalue " in line or " = shufflevector " in line:
            moved.append(line.replace("%v847,", "%v845,"))
        elif " = fadd <4 x float> " in line:
            match = re.fullmatch(r"  (%v\d+) = fadd <4 x float> (%v\d+), (%v\d+)(, !dbg !\d+)", line)
            assert match, line
            result, score, mask, dbg = match.groups()
            masked = result + ".masked"
            moved.append(f"  {masked} = fadd <4 x float> {score}, {mask}{dbg}")
            phis.append(f"  {result} = phi <4 x float> [ {masked}, %v702 ], [ {score}, %v701 ]{dbg}")
        else:
            assert " = extractelement <4 x float> " in line, line
            remaining.append(line)
    assert len(phis) == 16 and len(remaining) == 64
    text = text[:start] + "\n".join(remaining) + "\n" + text[end:]
    old_merge = "  %v847 = phi [2 x <32 x float>] [ %v845, %v702 ], [ zeroinitializer, %v701 ], !dbg !13"
    assert text.count(old_merge) == 1
    text = text.replace(old_merge, "\n".join(phis))
    anchor = "  br label %v846, !dbg !13\n\nv846:"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "\n".join(moved) + "\n" + anchor)
    # Verify the bypass block contains no arithmetic; no score add is left in
    # the common continuation. The diagonal arm keeps original mask arithmetic.
    assert text.split("v701:", 1)[1].split("v702:", 1)[0].strip().splitlines()[-1].strip() == "br label %v846, !dbg !13"
    assert "fadd <4 x float>" not in text.split("v848:", 1)[1]
    assert text.count("fadd <4 x float>") == original.count("fadd <4 x float>") == 16
    calls = lambda s: [line.split(" = ", 1)[-1] for line in s.splitlines() if "call " in line]
    renamed_original = re.sub(r"%(\d+)\b", lambda m: "%v" + m[1] if int(m[1]) >= 5 else m[0], original)
    assert calls(text) == calls(renamed_original), "Call/DMA/MMAC sequence changed"
    (HERE / "kernel.ll").write_text(text)
    shutil.copyfile(BASE / "launch.json", HERE / "launch.json")
    report = {
        "baseline": str(BASE / "kernel.ll"), "baseline_sha256": EXPECTED,
        "experiment": str(HERE / "kernel.ll"),
        "experiment_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "transformation": "Move 16 vector4 score+mask adds into masked arm; PHI raw QK on fully visible arm",
        "condition_unchanged": True, "call_sequence_unchanged": True,
        "masked_vector4_adds": 16, "unmasked_vector4_adds": 0,
        "common_vector4_adds": 0, "static_lds_bytes": 16384,
        "semantic_scope": "Approximate attention experiment: bypassing +0 may preserve negative zero; no global fast-math flags added",
        "gpu_tested": False,
    }
    (HERE / "structure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    generate()
