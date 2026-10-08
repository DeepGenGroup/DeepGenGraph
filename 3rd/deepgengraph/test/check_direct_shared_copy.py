#!/usr/bin/env python3
"""Trace attention QK results through exp2/f16 stores to PV LDS addresses.

Every QK output coordinate is a distinct symbolic value. This checks all 128
lanes, store guards, complete nonoverlapping coverage and the publication
barrier without assuming equal numerical inputs. It is not a GPU timing test.
"""
import re
import sys
from pathlib import Path

from check_gemm_pv_addresses import Addresses, split_args


class Trace(Addresses):
    def value(self, name):
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            opcode, body = expr.split(' ', 1)
            if opcode == 'getelementptr':
                ty, ptr, index = split_args(body)
                base, offset = self.typed(ptr)
                size = {'i8': 1, 'half': 2, 'float': 4}[ty.split()[-1]]
                self.cache[name] = (base, offset + size * self.typed(index))
            elif opcode == 'fadd':
                # QK + causal mask preserves the QK coordinate.
                self.cache[name] = self.typed(split_args(body)[0])
            elif '@__ocml_exp2_f32(' in expr:
                arg = re.search(r'@__ocml_exp2_f32\(float (%\w+)\)', expr)[1]
                self.cache[name] = ('exp2', self.value(arg))
            elif opcode == 'fptrunc':
                value = self.typed(body.split(' to ')[0])
                self.cache[name] = ([('f16', x) for x in value] if isinstance(value, list)
                                    else ('f16', value))
            elif opcode == 'icmp' and body.startswith('ult '):
                a, b = split_args(body.split(' ', 2)[2])
                self.cache[name] = self.value(a) < self.value(b)
            elif opcode == 'load' and body.startswith('float,'):
                raise AssertionError('P still takes an FP32 LDS round trip')
        return super().value(name)


def check(text):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    mma = [n for n, e in definitions.items() if 'asm sideeffect "' in e and 'v_mmac_' in e]
    assert len(mma) == 64
    pv_start = text.index(f'  {mma[32]} =')
    stores = list(re.finditer(r'^  store half (%\w+), ptr addrspace\(3\) (%\w+),', text[:pv_start], re.M))
    assert len(stores) == 16, 'Expected 16 scalar P stores per thread'
    label = re.findall(r'^(\d+):', text[:stores[0].start()], re.M)[-1]
    guard = re.findall(r'br i1 (%\w+), label %' + label + r', label %(\w+)', text)
    assert len(guard) == 1
    assert not re.search(r'^\d+:', text[stores[0].start():stores[-1].end()], re.M)
    published = text[stores[-1].end():pv_start]
    first_read = re.search(r'load half, ptr addrspace\(3\)', published)
    assert first_read and '@llvm.amdgcn.s.barrier()' in published[:first_read.start()]
    assert re.search(r'^' + guard[0][1] + ':', published, re.M), 'Barrier must follow reconvergence'


    # Obtain the physical P base from the first actual PV operand at lane zero.
    pv_a = split_args(re.search(r'"\((.*)\)', definitions[mma[32]])[1])[0]
    base, offset = Trace(definitions, {tid: 0}).typed(pv_a)[0]
    covered = set()
    for thread in range(128):
        context = {tid: thread}
        for i, name in enumerate(mma[:32]):
            tile = i // 8
            row = tile // 2 * 32 + thread // 64 * 16 + thread % 16
            col = tile % 2 * 16 + thread % 64 // 16
            context[name] = [(row, col + 4 * r) for r in range(4)]
        trace = Trace(definitions, context)
        if not trace.value(guard[0][0]):
            continue
        for store in stores:
            value, ptr = store.groups()
            result = trace.value(value)
            assert result[0] == 'f16' and result[1][0] == 'exp2', result
            row, col = result[1][1]
            address = trace.value(ptr)
            assert address == (base, offset + 2 * (row * 32 + col)), (thread, result, address)
            assert address not in covered, ('duplicate store', thread, address)
            covered.add(address)
    assert covered == {(base, offset + 2 * i) for i in range(64 * 32)}, 'P contains holes'
    print('PASS direct P copy: 2048 unique exp2/f16 values, all lane guards, PV addresses and barrier')


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text())
