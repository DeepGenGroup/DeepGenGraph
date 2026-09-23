#!/usr/bin/env python3
"""Check that every Attn_p2 output uses its own row's softmax denominator.

Usage: check_attention_normalization.py path/to/finalLLVMText.ll
       check_attention_normalization.py --mytest path/to/MyTest
Evaluates the generated LLVM address graph without executing GPU instructions.
The fixture has shape [1,32,4096,128], 64-row tiles and 128 threads.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from check_gemm_pv_addresses import Addresses, split_args


class DivisorAddresses(Addresses):
    """Track only the denominator through division, casts and vector packing."""

    def value(self, name):
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            if expr.startswith("fdiv "):
                self.cache[name] = self.value(split_args(expr[5:])[1])
                return self.cache[name]
            if expr.startswith("fptrunc "):
                self.cache[name] = self.value(expr.split(" to ")[0].split()[-1])
                return self.cache[name]
        return super().value(name)


def check(text):
    definitions = dict(re.findall(
        r"^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$", text, re.M))
    tid = next(n for n, e in definitions.items()
               if "@llvm.amdgcn.workitem.id.x()" in e)
    head = next(n for n, e in definitions.items()
                if "@llvm.amdgcn.workgroup.id.x()" in e)
    tile = next(n for n, e in definitions.items()
                if "@llvm.amdgcn.workgroup.id.y()" in e)
    signature = re.search(r"define amdgpu_kernel void @Attn_p2\((.*)\) #", text)
    args = re.findall(r"ptr addrspace\(1\) (%\w+)", signature.group(1))
    assert len(args) == 4, "Expected Q/K/V/O pointer arguments"
    stores = re.findall(
        r"store half (%\w+), ptr addrspace\(1\) (%\w+)", text)
    assert len(stores) == 64, "Expected 64 output elements per thread"
    row_buffers = set(re.findall(
        r"^(@\w+) = addrspace\(3\) global \[64 x \[1 x float\]\]",
        text, re.M))
    assert row_buffers, "Expected shared row-sum buffer"
    failures = []
    total = 0
    for h, block in ((0, 0), (1, 1), (31, 63)):
        covered = set()
        base = h * 4096 * 128 + block * 64 * 128
        for thread in range(128):
            context = {arg: (arg, 0) for arg in args}
            context.update({tid: thread, head: h, tile: block})
            evaluator = DivisorAddresses(definitions, context)
            for value, pointer in stores:
                output, offset = evaluator.value(pointer)
                assert output == args[-1], "Store does not target O"
                offset -= base
                covered.add(offset)
                buffer, row = evaluator.value(value)
                assert buffer in row_buffers, "Divisor is not a row sum"
                expected = offset // 128
                if row != expected:
                    failures.append((h, block, thread, expected, row))
                total += 1
        assert covered == set(range(8192)), "Missing or out-of-range output stores"
    assert not failures, (
        f"{len(failures)}/{total} outputs use the wrong denominator; "
        f"(head, tile, thread, output row, denominator row): {failures[:8]}")
    print(f"PASS: {total} output/divisor coordinate pairs across three tiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="LLVM IR file or MyTest executable")
    parser.add_argument("--mytest", action="store_true", help="Compile the attention fixture first")
    args = parser.parse_args()
    if args.mytest:
        fixture = Path(__file__).resolve().parent / "test_input.mlir"
        with tempfile.TemporaryDirectory(prefix="attention-normalization-") as temp:
            output = Path(temp) / "attention.ll"
            proc = subprocess.run(
                [str(args.path.resolve()), str(fixture), str(output)],
                cwd=temp, capture_output=True, text=True, timeout=60)
            log = proc.stdout + proc.stderr
            assert proc.returncode == 0 and not re.search(
                r"error:|Assertion .*failed|LLVM ERROR", log), log[-6000:]
            check(output.read_text())
    else:
        check(args.path.read_text())
