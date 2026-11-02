#!/usr/bin/env python3
"""Hash-bound experiment: carry lane-local denominator, all-reduce on exit."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "local_reduction_pass"
EXPECTED = "d335c9c97bc82127273b007d53b27b514109ab736f03fcd2ab141fdec0a8532f"


def generate():
    source = (BASE / "kernel_dtk.ll").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
    # Keep ABI arguments %0..%3 and implicit entry block %4 for LLVM and the
    # existing execution model. Name every other slot before deleting nodes.
    original = re.sub(r"%(\d+)\b", lambda m: "%v" + m[1] if int(m[1]) >= 5 else m[0], source)
    original = re.sub(r"^(\d+):", lambda m: "v" + m[1] + ":", original, flags=re.M)
    text = original
    removed = []
    for first, end, local, update, scaled in (
        (1192, 1199, 1086, 1200, 973), (1203, 1210, 1191, 1211, 981)
    ):
        # Delete precisely two shuffle/add stages per row. The existing alpha
        # multiply and final local+scaled-old update keep their original order.
        for index in range(first, end + 1):
            pattern = rf"^  %v{index} = .*\n"
            match = re.search(pattern, text, re.M)
            assert match
            removed.append(match[0])
            text = text[:match.start()] + text[match.end():]
        old = f"  %v{update} = fadd float %v{end}, %v{scaled}, !dbg !33"
        assert text.count(old) == 1
        text = text.replace(old, f"  %v{update} = fadd float %v{local}, %v{scaled}, !dbg !33")

    inserted = []
    for row, extraction in enumerate((1748, 1966)):
        old = f"  %v{extraction} = extractvalue [2 x <1 x float>] %v207, {row}, !dbg !19"
        assert text.count(old) == 1
        p = f"%denom.exit.r{row}"
        lines = [
            f"  {p}.packed = extractvalue [2 x <1 x float>] %v207, {row}, !dbg !19",
            f"  {p}.local = extractelement <1 x float> {p}.packed, i64 0, !dbg !19",
            f"  {p}.bits16 = bitcast float {p}.local to i32, !dbg !19",
            f"  {p}.peer16 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 {p}.bits16), !dbg !19",
            f"  {p}.float16 = bitcast i32 {p}.peer16 to float, !dbg !19",
            f"  {p}.sum16 = fadd float {p}.local, {p}.float16, !dbg !19",
            f"  {p}.bits32 = bitcast float {p}.sum16 to i32, !dbg !19",
            f"  {p}.peer32 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 {p}.bits32), !dbg !19",
            f"  {p}.float32 = bitcast i32 {p}.peer32 to float, !dbg !19",
            f"  {p}.sum32 = fadd float {p}.sum16, {p}.float32, !dbg !19",
            f"  %v{extraction} = insertelement <1 x float> poison, float {p}.sum32, i64 0, !dbg !19",
        ]
        inserted.extend(lines)
        text = text.replace(old, "\n".join(lines))

    # No other users of the loop-carried denominator: scaled recurrence and
    # final normalization only. Running max, exp2, DMA and numerator stay intact.
    uses = [line for line in original.splitlines() if re.search(r"%v207\b", line)]
    assert len(uses) == 5 and "zeroinitializer" in uses[0]
    assert all("extractvalue [2 x <1 x float>] %v207" in line for line in uses[1:])
    split = lambda s: s.split("\nv1623:", 1)
    before_loop, before_exit = split(original)
    after_loop, after_exit = split(text)
    count = lambda s: s.count("call i32 @llvm.amdgcn.ds.bpermute(")
    assert count(before_loop) == 8 and count(after_loop) == 4
    assert count(after_exit) == count(before_exit) + 4
    assert count(text) == count(original)
    other_calls = lambda s: [line for line in s.splitlines() if "call " in line and
                             "@llvm.amdgcn.ds.bpermute(" not in line]
    assert other_calls(text) == other_calls(original)
    # Undo the entire edit and demand exact text equality, including unrelated
    # shuffle calls, stores, flags, attributes, metadata and arithmetic.
    undo = text
    for row, extraction in enumerate((1748, 1966)):
        block = "\n".join(inserted[row*11:(row+1)*11])
        undo = undo.replace(block, f"  %v{extraction} = extractvalue [2 x <1 x float>] %v207, {row}, !dbg !19")
    for row, (local, update, scaled, end) in enumerate(((1086, 1200, 973, 1199), (1191, 1211, 981, 1210))):
        new = f"  %v{update} = fadd float %v{local}, %v{scaled}, !dbg !33"
        old = f"  %v{update} = fadd float %v{end}, %v{scaled}, !dbg !33"
        undo = undo.replace(new, "".join(removed[row*8:(row+1)*8]) + old)
    assert undo == original
    (HERE / "kernel.ll").write_text(text)
    (HERE / "baseline.ll").write_text(original)
    shutil.copyfile(BASE / "launch.json", HERE / "launch.json")
    report = dict(baseline=str(BASE / "kernel_dtk.ll"), baseline_sha256=EXPECTED,
                  baseline_ir_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  experiment_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  loop_denominator_shuffles_before=4, loop_denominator_shuffles_after=0,
                  exit_denominator_shuffles_before=0, exit_denominator_shuffles_after=4,
                  loop_max_shuffles_before=4, loop_max_shuffles_after=4,
                  total_static_shuffles=count(text), all_other_ir_unchanged=True,
                  lane_group_xor=[16, 32], workgroup_size=256, bn=128,
                  numerical_scope="Approximate: change sum/alpha multiplication order; running max and alpha retained",
                  gpu_tested=False)
    (HERE / "structure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    generate()
