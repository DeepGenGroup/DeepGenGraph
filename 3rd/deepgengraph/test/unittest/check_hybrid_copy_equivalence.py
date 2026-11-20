#!/usr/bin/env python3
"""Compare exported hybrid Q/K/V copy addresses with the synchronous k5 IR.

All 256 lanes and three loop/head contexts, including DMA's implicit lane
offset. This checks logical data movement, not gfx936 timing or allocation.
"""
import re
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args


class CopyTrace(Trace):
    def value(self, name):
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            if expr.startswith('ptrtoint '):
                base, offset = self.typed(expr.split(' to ')[0][9:])
                assert base == '@shm_0'
                self.cache[name] = offset
            elif expr.startswith('inttoptr '):
                self.cache[name] = ('@shm_0', self.typed(expr.split(' to ')[0][9:]))
            elif '@llvm.amdgcn.readfirstlane.i32(' in expr:
                operand = re.search(r'\(i32 (%[\w.]+)\)', expr)[1]
                first = self.thread // 64 * 64
                self.cache[name] = self.peers[first].value(operand)
                assert self.value(operand) == self.cache[name], 'DMA base must be wave-uniform'
        return super().value(name)


def copies(text, head, query, key, dk):
    defs = {name: re.sub(r', ![\w.]+ !\d+', '', expr)
            for name, expr in re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M)}
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    context = {f'%{i}': (f'%{i}', 0) for i in range(4)}
    for axis, value in [('x', head), ('y', query)]:
        context[next(n for n, e in defs.items() if '@llvm.amdgcn.workgroup.id.' + axis + '()' in e)] = value
    context.update({n: 0 for n, e in defs.items() if '@llvm.amdgcn.workgroup.id.z()' in e})
    ivs = [n for n, e in defs.items() if e.startswith('phi i64 ')]
    assert len(ivs) == 2, ivs
    context.update(zip(ivs, (key, dk)))
    peers = []
    peers.extend(CopyTrace(defs, {**context, tid: t}, peers) for t in range(256))
    mapping = {}
    consumers = []
    for line in text.splitlines():
        store = re.search(r'^  store <(\d+) x half> (%[\w.]+), ptr addrspace\(3\) (%[\w.]+)', line)
        dma = re.search(r'call void @llvm.amdgcn.global.load.async.lds\((.*)\)', line)
        load = re.search(r' = load <\d+ x half>, ptr addrspace\(3\) (%[\w.]+)', line)
        if load:
            consumers.append([trace.value(load[1]) for trace in peers])
        if not (store or dma):
            continue
        for lane, trace in enumerate(peers):
            if store:
                values = trace.value(store[2])
                base, offset = trace.value(store[3])
                assert len(values) == int(store[1])
            else:
                source, destination, size, soffset, aux = split_args(dma[1])
                assert (size, soffset, aux) == ('i32 4', 'i32 0', 'i32 0')
                src, start = trace.typed(source)
                values = [(src, start), (src, start + 2)]
                base, offset = trace.typed(destination)
                offset += lane % 64 * 4
            assert base == '@shm_0' and 0 <= offset < 16384
            for i, source in enumerate(values):
                assert source[0] in ('%0', '%1', '%2'), source
                assert source not in mapping, ('duplicate global element', source)
                mapping[source] = (offset + 2 * i) % 8192
    assert len(mapping) == 6 * 4096, len(mapping)
    return mapping, consumers


reference, candidate = [Path(p).read_text() for p in sys.argv[1:3]]
for context in [(0, 0, 0, 0), (7, 31, 3, 1), (31, 31, 31, 3)]:
    assert copies(reference, *context) == copies(candidate, *context), context
print('PASS all Q/K/V global-to-packed-LDS mappings and consumer addresses match k5: 256 lanes, three contexts')
