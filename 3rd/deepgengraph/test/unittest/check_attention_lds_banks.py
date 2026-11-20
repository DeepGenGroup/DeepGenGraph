#!/usr/bin/env python3
"""Exhaustive exported LDS addresses for bounded-jam 3.3 attention.

Models 32 four-byte banks: b64 accesses use contiguous 16-lane phases;
b128 reads use AMD's paired four-lane groups. This is not gfx936 profiling.
Unlike the older fixture, includes Q and rejects all multi-way accesses.
"""
import collections
import re
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_packed_lds_banks import multiplicity


def check(text, minimum_locations=6):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    threads = int(re.search(r'"amdgpu-flat-work-group-size"="\d+,(\d+)"', text)[1])
    stats = collections.Counter()
    seen = set()
    # Include nonzero tile and loop contexts; allocation permutations must
    # work throughout the outer D, key and QK-reduction loops.
    for value in (0, 1, 3):
        context = {n: value for n, e in definitions.items()
                   if e.startswith('phi i64 ') or '@llvm.amdgcn.workgroup.id.' in e}
        traces = [Trace(definitions, {**context, tid: lane}) for lane in range(threads)]
        for line, instruction in enumerate(text.splitlines(), 1):
            if not re.search(r'(?:= load |^  store ).*ptr addrspace\(3\)', instruction):
                continue
            match = re.search(r'(load|store) <(\d+) x half>(?:, | %[\w.]+, )ptr addrspace\(3\) (%[\w.]+)', instruction)
            assert match, ('unexpected scalar/unmodeled LDS access', line, instruction)
            op, count, pointer = match.groups()
            width = int(count) * 2
            assert width in (8, 16), (line, width)
            addresses = [trace.value(pointer)[1] for trace in traces]
            if op == 'load' and width == 16:
                phases = [[wave + upper + x for x in [*range(a, a+4), *range(b, b+4)]]
                          for wave in range(0, threads, 64) for upper in (0, 32)
                          for a, b in ((0, 20), (4, 16), (8, 28), (12, 24))]
            else:
                phases = [range(start, start + 128 // width)
                          for start in range(0, threads, 128 // width)]
            for lanes in phases:
                concentration = multiplicity([addresses[lane] for lane in lanes], width)
                assert concentration == 1, (line, op, width, value, concentration)
            stats[op, width] += 1
            seen.add(re.search(r'!dbg !(\d+)', instruction)[1])
    assert len(seen) >= minimum_locations, 'Insufficient Q/K/V access coverage'
    print(f'PASS all Q/K/V LDS packets, all {threads} lanes, three loop contexts:', dict(stats))


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text())
