#!/usr/bin/env python3
"""Evaluate exported attention rowsum recurrence for every lane, without a GPU.

QK coordinate tags establish which P element each exp2 produces. Inject distinct
FP32 values at those exp2 calls, then evaluate the actual shuffle/add graph and
loop-carried result. No LDS load is allowed in the forwarded recurrence.
"""
import random
import re
import struct
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import Addresses, split_args


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def check(text):
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    mma = [n for n, e in defs.items() if 'asm sideeffect "' in e and 'v_mmac_' in e]
    exp = [n for n, e in defs.items() if '@__ocml_exp2_f32(' in e]
    phis = [n for n, e in defs.items() if e.startswith('phi [2 x <1 x float>]')]
    assert len(mma) == 64 and len(exp) == 16 and len(phis) == 1
    phi = phis[0]
    recurrence = re.findall(r'\[ (%[\w.]+), %\w+ \]', defs[phi])
    assert len(recurrence) == 1
    coordinates = []
    for thread in range(128):
        context = {tid: thread}
        for i, name in enumerate(mma[:32]):
            tile = i // 8
            row = tile // 2 * 32 + thread // 64 * 16 + thread % 16
            col = tile % 2 * 16 + thread % 64 // 16
            context[name] = [(row, col + 4 * r) for r in range(4)]
        trace = Trace(defs, context)
        coords = {name: trace.value(name) for name in exp}
        assert all(value[0] == 'exp2' for value in coords.values())
        coordinates.append({name: value[1] for name, value in coords.items()})

    class Lane(Addresses):
        def __init__(self, thread, values, previous, peers):
            context = {tid: thread, phi: [[previous[thread][0]], [previous[thread][1]]]}
            context.update({name: values[r][c] for name, (r, c) in coordinates[thread].items()})
            super().__init__(defs, context)
            self.thread, self.peers = thread, peers

        def value(self, name):
            if name in self.cache:
                return self.cache[name]
            if not name.startswith(('%', '@')) and ('.' in name or 'e' in name.lower()):
                return float(name)
            if name in defs:
                expr = defs[name]
                opcode, body = expr.split(' ', 1)
                if opcode == 'fadd':
                    left, right = split_args(body)
                    a = self.typed(left)
                    b = self.value(right)
                    self.cache[name] = ([f32(x + y) for x, y in zip(a, b)]
                                        if isinstance(a, list) else f32(a + b))
                elif opcode == 'bitcast':
                    self.cache[name] = self.value(body.split()[1])
                elif '@llvm.amdgcn.mbcnt.' in expr:
                    self.cache[name] = self.thread % (32 if '.lo(' in expr else 64)
                elif '@llvm.amdgcn.ds.bpermute(' in expr:
                    args = split_args(re.search(r'@llvm.amdgcn.ds.bpermute\((.*?)\)', expr)[1])
                    byte = self.typed(args[0])
                    assert byte % 4 == 0 and 0 <= byte < 256
                    peer = self.thread // 64 * 64 + byte // 4
                    self.cache[name] = self.peers[peer].typed(args[1])
                elif opcode in ('and', 'xor', 'shl'):
                    a, b = split_args(body)
                    a, b = self.typed(a), self.value(b)
                    self.cache[name] = {'and': lambda: a & b, 'xor': lambda: a ^ b,
                                        'shl': lambda: a << b}[opcode]()
                elif opcode == 'load':
                    raise AssertionError('rowsum still depends on memory: ' + expr)
            return super().value(name)

    for seed in range(3):
        rng = random.Random(seed)
        previous = [[f32(rng.random()) for _ in range(2)] for _ in range(128)]
        for iteration in range(3):
            values = [[f32(2 ** rng.uniform(-12, 8)) for _ in range(32)] for _ in range(64)]
            peers = []
            for thread in range(128):
                peers.append(Lane(thread, values, previous, peers))
            local = []
            for thread in range(128):
                row = thread // 64 * 16 + thread % 16
                column = thread % 64 // 16
                sums = []
                for r in (row, row + 32):
                    acc = 0.0
                    for c in range(column, 32, 4):
                        acc = f32(acc + values[r][c])
                    sums.append(acc)
                local.append(sums)
            for offset in (16, 32):
                local = [[f32(local[t][r] + local[t ^ offset][r]) for r in range(2)]
                         for t in range(128)]
            expected = [[f32(local[t][r] + previous[t][r]) for r in range(2)]
                        for t in range(128)]
            actual = [[x[0] for x in peer.value(recurrence[0])] for peer in peers]
            assert actual == expected, ('rowsum mismatch', seed, iteration)
            previous = actual
    print('PASS rowsum: 128 lanes x 2 rows x 9 iterations, exact FP32 shuffle/add recurrence, no LDS')


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text())
