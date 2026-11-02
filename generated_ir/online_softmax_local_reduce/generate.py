#!/usr/bin/env python3
"""Hash-bound experiment: four independent thread-local reduction chains."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "online_softmax_mask_bypass"
EXPECTED = "68410c5a5ad3bc28dda29769d670d05fd6d9421e051833a6b57cf9a8d9eaa863"
ROOTS = {"max": ("%v978", "%v1033"), "sum": ("%v1213", "%v1348")}
PATTERNS = {
    "max": re.compile(r"  (%[\w.]+) = call float @llvm.maxnum.f32\(float ([^,]+), float ([^)]+)\)(, !dbg !\d+)"),
    "sum": re.compile(r"  (%[\w.]+) = fadd float ([^,]+), ([^,]+)(, !dbg !\d+)"),
}


def graph(text, kind):
    return {m[1]: (i, m[2], m[3], m[4])
            for i, line in enumerate(text.splitlines())
            if (m := PATTERNS[kind].fullmatch(line))}


def signature(nodes, root):
    if root not in nodes:
        return Counter([root]), 0, 0
    _, left, right, _ = nodes[root]
    a, da, na = signature(nodes, left)
    b, db, nb = signature(nodes, right)
    return a + b, max(da, db) + 1, na + nb + 1


def rewrite(text, kinds):
    lines = text.splitlines()
    replacements = {}
    reports = []
    for kind in kinds:
        nodes = graph(text, kind)
        for row, root in enumerate(ROOTS[kind]):
            chain = []
            current = root
            while current in nodes:
                chain.append(current)
                current = nodes[current][1]
            chain.reverse()
            assert len(chain) == (32 if kind == "max" else 31)
            # All internal values must belong exclusively to this local chain.
            for value in chain[:-1]:
                assert len(re.findall(re.escape(value) + r"(?![\w.])", text)) == 2
            first = nodes[chain[0]]
            if kind == "max":
                assert first[2] == "0xFFF0000000000000"
                leaves = [first[1]] + [nodes[v][2] for v in chain[1:]]
            else:
                leaves = [first[1], first[2]] + [nodes[v][2] for v in chain[1:]]
            assert len(leaves) == len(set(leaves)) == 32
            prefix = f"%local.{kind}.r{row}"
            partial = leaves[:4]
            dbg = first[3]

            def emit(dst, lhs, rhs):
                if kind == "max":
                    return f"  {dst} = call float @llvm.maxnum.f32(float {lhs}, float {rhs}){dbg}"
                return f"  {dst} = fadd float {lhs}, {rhs}{dbg}"

            for value in chain:
                replacements[nodes[value][0]] = []
            if kind == "max":
                # Retain the original -inf seed (including all-NaN behavior).
                partial[0] = prefix + ".seed"
                replacements[first[0]].append(emit(partial[0], leaves[0], first[2]))
            # Place each add at its original input's accumulation position:
            # exp2/packing/lifetime fences stay in exactly the original order.
            for index in range(4, 32):
                bank = index % 4
                dst = f"{prefix}.p{index}"
                old = chain[index if kind == "max" else index - 1]
                replacements[nodes[old][0]].append(emit(dst, partial[bank], leaves[index]))
                partial[bank] = dst
            replacements[nodes[root][0]].extend([
                emit(prefix + ".merge01", partial[0], partial[1]),
                emit(prefix + ".merge23", partial[2], partial[3]),
                emit(root, prefix + ".merge01", prefix + ".merge23"),
            ])
            reports.append({"kind": kind, "row": row, "root": root})
    output = "\n".join(new for i, line in enumerate(lines)
                       for new in replacements.get(i, [line])) + "\n"
    # Every original line outside the four selected reductions is untouched.
    assert [s for i, s in enumerate(lines) if i not in replacements] == [
        s for s in output.splitlines()
        if not any(PATTERNS[k].fullmatch(s) and
                   (PATTERNS[k].fullmatch(s)[1].startswith("%local.") or
                    PATTERNS[k].fullmatch(s)[1] in ROOTS[k]) for k in kinds)]
    for report in reports:
        old = signature(graph(text, report["kind"]), report["root"])
        new = signature(graph(output, report["kind"]), report["root"])
        assert old[0] == new[0] and old[2] == new[2]
        report.update(inputs=32, chains=4, operations_before=old[2],
                      operations_after=new[2], depth_before=old[1], depth_after=new[1])
    return output, reports


def main():
    source = (BASE / "kernel.ll").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
    report = {"baseline": str(BASE / "kernel.ll"), "baseline_sha256": EXPECTED,
              "gpu_tested": False, "variants": {}}
    for name, kinds in (("baseline", ()), ("max4", ("max",)),
                        ("sum4", ("sum",)), ("max_sum4", ("max", "sum"))):
        output, reductions = rewrite(source, kinds)
        target = HERE / name
        target.mkdir(exist_ok=True)
        (target / "kernel.ll").write_text(output)
        shutil.copyfile(BASE / "launch.json", target / "launch.json")
        report["variants"][name] = {
            "sha256": hashlib.sha256(output.encode()).hexdigest(),
            "reductions": reductions,
            "all_other_ir_lines_unchanged": True,
        }
    (HERE / "structure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
