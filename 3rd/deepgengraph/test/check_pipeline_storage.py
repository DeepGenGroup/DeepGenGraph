#!/usr/bin/env python3
"""Trace actual LLVM pipeline stores to the next step's MMA operand addresses.

Checks both slot parities, all lanes, and distinct symbolic P values. This is
not a GPU execution test. Usage: check_pipeline_storage.py pipeline.ll
"""
import re
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args


class PipelineTrace(Trace):
    def typed(self, text):
        if text.startswith("{"):
            end = text.index("}")
            value = text[end + 1:].strip()
            if value == "poison":
                return [None] * 5  # ranked memref descriptor
            return self.value(value)
        if text.startswith("ptr ") and "inttoptr" in text:
            return ("unused-allocation-pointer", 0)
        return super().typed(text)


def loop_text(text):
    start = re.search(r"^  (%\w+) = phi i64 ", text, re.M)
    assert start
    exit_label = re.search(r"br i1 .*label %\d+, label %(\d+)", text[start.start():])[1]
    end = re.search(r"^" + exit_label + ":", text, re.M)
    return start[1], text[start.start():end.start()]


def check(text):
    definitions = dict(re.findall(r"^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$", text, re.M))
    iv, body = loop_text(text)
    debug = dict(re.findall(r"!(\d+) = !DILocation\(line: (\d+)", text))
    def builtin(name):
        return next(n for n, e in definitions.items() if "@llvm.amdgcn." + name + "()" in e)
    tid, head, tile = map(builtin, ["workitem.id.x", "workgroup.id.x", "workgroup.id.y"])
    args = re.findall(r"ptr addrspace\(1\) (%\w+)",
                      re.search(r"define amdgpu_kernel void @Attn_p2\((.*)\) #", text)[1])
    mmac = {"62": [], "78": []}
    for name, expr, loc in re.findall(r"^  (%\w+) = (.*asm sideeffect.*), !dbg !(\d+)$", body, re.M):
        mmac[debug[loc]].append(name)
    assert [len(mmac[k]) for k in ("62", "78")] == [32, 32]
    stores = []
    for ty, value, ptr, loc in re.findall(
            r"^  store (half|<\d+ x half>) (%\w+), ptr addrspace\(3\) (%\w+),.*!dbg !(\d+)$", body, re.M):
        line = debug[loc]
        if line in ("60", "61", "77"):
            stores.append((line, ty, value, ptr))
    assert stores

    def context(lane, step, h):
        values = {a: (a, 0) for a in args}
        values.update({tid: lane, head: h, tile: 63, iv: step})
        for i, name in enumerate(mmac["62"]):
            fragment = i // 8
            row = fragment // 2 * 32 + lane // 64 * 16 + lane % 16
            col = fragment % 2 * 16 + lane % 64 // 16
            values[name] = [(row, col + 4*r) for r in range(4)]
        return values

    for step, h in [(1, 0), (2, 7), (63, 31), (126, 31)]:
        memory = {}
        counts = {"60": 0, "61": 0, "77": 0}
        traces = []
        for lane in range(128):
            traces.append(PipelineTrace(definitions, context(lane, step - 1, h), traces))
        for trace in traces:
            for line, ty, value, ptr in stores:
                data = trace.value(value)
                data = [data] if ty == "half" else data
                base, offset = trace.value(ptr)
                for j, element in enumerate(data):
                    address = (base, offset + 2*j)
                    assert address not in memory, ("overlapping slots/writers", line, address)
                    memory[address] = element
                    counts[line] += 1
        assert counts == {"60": 4096, "61": 4096, "77": 2048}, counts
        for lane in range(128):
            trace = PipelineTrace(definitions, context(lane, step, h))
            for kind, k_count, n_count in [("62", 8, 2), ("78", 2, 8)]:
                for i, name in enumerate(mmac[kind]):
                    fragment, k = divmod(i, k_count)
                    row = fragment // n_count * 32 + lane // 64 * 16 + lane % 16
                    col = fragment % n_count * 16 + lane % 16
                    kk = k * 16 + lane % 64 // 16 * 4
                    operands = split_args(re.search(r'"\((.*)\)', definitions[name])[1])
                    rhs = [memory[p] for p in trace.typed(operands[1])]
                    if kind == "62":
                        expected = [(args[1], 2*(h*4096*128 + ((step+1)*32+col)*128 + kk+r)) for r in range(4)]
                    else:
                        expected = [(args[2], 2*(h*4096*128 + (step*32+kk+r)*128+col)) for r in range(4)]
                        lhs = [memory[p] for p in trace.typed(operands[0])]
                        assert lhs == [("f16", ("exp2", (row, kk+r))) for r in range(4)], (step, lane, i, lhs)
                    assert rhs == expected, (step, lane, i, rhs, expected)
        print(f"PASS pipeline storage: step={step}, head={h}, all K/V/P stores and MMA reads")


if __name__ == "__main__":
    check(Path(sys.argv[1]).read_text())
