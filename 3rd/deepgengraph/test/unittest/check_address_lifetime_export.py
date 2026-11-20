#!/usr/bin/env python3
"""Compare every memory address before/after thread-index simplification."""
from pathlib import Path
import re
import sys
from check_direct_shared_copy import Trace

parsed = []
for file in sys.argv[1:3]:
    text = Path(file).read_text()
    defs = {name: re.sub(r', ![\w.]+ !\d+', '', expr)
            for name, expr in re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M)}
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    accesses = []
    for line in text.splitlines():
        match = re.search(r'\b(load|store) (<\d+ x half>).*?ptr addrspace\(([13])\) (%[\w.]+)', line)
        if match:
            accesses.append(match.groups())
    parsed.append((defs, tid, accesses))
assert [(a, b, c) for a, b, c, _ in parsed[0][2]] == [(a, b, c) for a, b, c, _ in parsed[1][2]]
for head, query, output, key, dk in [(0, 0, 0, 0, 0), (7, 31, 1, 30, 3), (31, 31, 1, 31, 3)]:
    traces = []
    for defs, tid, accesses in parsed:
        context = {'%' + str(i): ('%' + str(i), 0) for i in range(4)}
        for axis, value in [('x', head), ('y', query), ('z', 0)]:
            context[next(n for n, e in defs.items() if '@llvm.amdgcn.workgroup.id.' + axis + '()' in e)] = value
        for (name, _), value in zip([(n, e) for n, e in defs.items() if e.startswith('phi i64 ')], [output, key, dk]):
            context[name] = value
        traces.append([Trace(defs, {**context, tid: t}) for t in range(256)])
    for a, b in zip(parsed[0][2], parsed[1][2]):
        for t in range(256):
            assert traces[0][t].value(a[3]) == traces[1][t].value(b[3]), (t, a, b)
print('PASS all global/LDS load/store addresses including O: 256 threads, 3 loop/head contexts')
