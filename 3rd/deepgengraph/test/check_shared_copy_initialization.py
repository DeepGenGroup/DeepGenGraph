#!/usr/bin/env python3
"""Regress QK -> shared -> f16 PV copies, including the path without debug fills.

Usage: python3 check_shared_copy_initialization.py [path/to/MyTest]
These are LLVM dataflow/address checks, not a GPU execution test.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_gemm_pv_addresses import Addresses


def check(text):
    defs = dict(re.findall(r"^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$", text, re.M))

    def root(ptr):
        expr = defs.get(ptr, "")
        if expr.startswith("getelementptr "):
            return root(re.search(r"ptr(?: addrspace\(\d+\))? ([%@][\w.]+)", expr)[1])
        return ptr

    stores, loads = {}, {}
    for pos, line in enumerate(text.splitlines()):
        match = re.search(r"\b(store|load) .*, ptr addrspace\([35]\) ([%@][\w.]+),", line)
        if match:
            table = stores if match[1] == "store" else loads
            table.setdefault(root(match[2]), []).append((pos, line))
    for buffer, reads in loads.items():
        assert buffer in stores, f"Uninitialized buffer: {buffer}"
        assert stores[buffer][0][0] < reads[0][0], f"Read before first write: {buffer}"

    qk = re.search(r"^(@\w+) = addrspace\(3\) global \[64 x \[32 x float\]\]", text, re.M)[1]
    lines = text.splitlines()
    between = "\n".join(lines[stores[qk][-1][0]:loads[qk][0][0]])
    assert "@llvm.amdgcn.s.barrier()" in between, "QK shared copy requires a barrier"

    # Check all addresses, not merely the presence of stores: a wrong thread
    # layout can leave holes or silently overwrite another thread's elements.
    ptrs = [re.search(r"ptr addrspace\(3\) (%\w+),", line)[1]
            for _, line in stores[qk]]
    phis, seen = set(), set()

    def visit(name):
        if name in seen or name not in defs:
            return
        seen.add(name)
        expr = defs[name]
        if expr.startswith("phi i64 "):
            phis.add(name)
            return
        for operand in re.findall(r"%[\w.]+", expr):
            visit(operand)

    for ptr in ptrs:
        visit(ptr)
    assert len(phis) == 2, phis
    bounds = {p: int(re.search(r"icmp slt i64 " + re.escape(p) + r", (\d+)\b", text)[1])
              for p in phis}
    assert sorted(bounds.values()) == [2, 2], bounds
    a, b = sorted(phis)
    tid = next(n for n, expr in defs.items() if "@llvm.amdgcn.workitem.id.x()" in expr)
    covered = set()
    for thread in range(128):
        for i in range(bounds[a]):
            for j in range(bounds[b]):
                evaluator = Addresses(defs, {tid: thread, a: i, b: j})
                for ptr in ptrs:
                    buffer, offset = evaluator.value(ptr)
                    assert buffer == qk and 0 <= offset < 64 * 32
                    assert offset not in covered, f"Duplicate QK write: {offset}"
                    covered.add(offset)
    assert covered == set(range(64 * 32)), "Incomplete QK shared tile"

    # The f32 -> f16 copy must consume QK shared memory, never an uninitialized
    # private scalar. Follow the actual value stored to LDS through truncf/load.
    conversions = 0
    for writes in stores.values():
        for _, line in writes:
            match = re.search(r"store half (%\w+), ptr addrspace\(3\)", line)
            if not match:
                continue
            trunc = re.fullmatch(r"fptrunc float (%\w+) to half", defs[match[1]])
            if not trunc:
                continue
            load = re.match(r"load float, ptr addrspace\(3\) (%\w+),", defs[trunc[1]])
            if load and root(load[1]) == qk:
                conversions += 1
    assert conversions == 4, "PV conversion lost its QK shared source"


def main():
    test_dir = Path(__file__).resolve().parent
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else test_dir.parent / "build/test/MyTest"
    source = (test_dir / "test_friskBaseDebug.mlir").read_text()
    no_fill = "\n".join(line for line in source.splitlines()
                        if "%pvWeights_fill =" not in line).replace("%pvWeights_fill,", "%pvWeights_copied,")
    live = "\n".join(line for line in no_fill.splitlines()
                     if "%qkmerAcc_filled =" not in line).replace("%qkmerAcc_filled", "%19")
    with tempfile.TemporaryDirectory(prefix="frisk-shared-copy-") as temp:
        work = Path(temp)
        for name, ir in {"debug_fill": source, "real_qk_weights": no_fill,
                         "real_qk_and_denominator": live}.items():
            input_path, output_path = work / f"{name}.mlir", work / f"{name}.ll"
            input_path.write_text(ir)
            result = subprocess.run([str(binary), str(input_path), str(output_path)],
                                    cwd=work, capture_output=True, text=True, timeout=60)
            log = result.stdout + result.stderr
            assert result.returncode == 0 and not re.search(r"error:|LLVM ERROR|Assertion .*failed", log), log[-4000:]
            check(output_path.read_text())
            print(f"PASS {name}: initialized reads, complete QK stores, barrier, and QK -> f16 dataflow")


if __name__ == "__main__":
    main()
