#!/usr/bin/env python3
"""Trace attention QK results through exp2/f16 stores to PV LDS addresses.

Every QK output coordinate is a distinct symbolic value. This checks all 128
lanes, store guards, complete nonoverlapping coverage and the publication
barrier without assuming equal numerical inputs. It is not a GPU timing test.
"""
import re
import sys
from pathlib import Path

from check_gemm_pv_addresses import split_args
from check_packed_attention_copy import ByteAddresses


class Trace(ByteAddresses):
    def __init__(self, definitions, context, peers=None):
        super().__init__(definitions, context)
        self.peers = peers
        tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
        self.thread = context[tid]

    def value(self, name):
        if name in ('poison', 'undef'):
            return None
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            opcode, body = expr.split(' ', 1)
            if 'asm sideeffect "", "=v,0"(' in expr:
                operand = expr.split('"=v,0"(', 1)[1].split(')', 1)[0]
                self.cache[name] = self.typed(operand)
            elif opcode == 'bitcast':
                self.cache[name] = self.value(body.split()[1])
            elif opcode == 'shufflevector':
                a, b, indices = map(self.typed, split_args(body))
                self.cache[name] = [None if i is None else (a + b)[i] for i in indices]
            elif opcode in ('sext', 'zext', 'trunc'):
                src, value, _, dst = body.split()
                bits = int(src[1:])
                unsigned = self.value(value) % (1 << bits)
                if opcode == 'sext':
                    result = unsigned - (1 << bits) if unsigned >> (bits - 1) else unsigned
                elif opcode == 'trunc':
                    result = unsigned % (1 << int(dst[1:]))
                else:
                    result = unsigned
                self.cache[name] = result
            elif '@llvm.amdgcn.mbcnt.' in expr:
                self.cache[name] = self.thread % (32 if '.lo(' in expr else 64)
            elif '@llvm.amdgcn.ds.bpermute(' in expr:
                args = split_args(re.search(r'@llvm.amdgcn.ds.bpermute\((.*?)\)', expr)[1])
                byte = self.typed(args[0])
                assert self.peers is not None and byte % 4 == 0 and 0 <= byte < 256
                peer = self.thread // 64 * 64 + byte // 4
                self.cache[name] = self.peers[peer].typed(args[1])
            elif opcode in ('and', 'or', 'xor', 'shl', 'lshr', 'ashr'):
                a_text, b = split_args(body)
                a, b = self.typed(a_text), self.value(b)
                bits = int(re.search(r'\bi(\d+)\b', a_text)[1])
                unsigned = a & ((1 << bits) - 1)
                signed = unsigned - (1 << bits) if unsigned >> (bits - 1) else unsigned
                self.cache[name] = {'and': lambda: a & b, 'or': lambda: a | b,
                                    'xor': lambda: a ^ b, 'shl': lambda: a << b,
                                    'lshr': lambda: unsigned >> b,
                                    'ashr': lambda: signed >> b}[opcode]()
            elif opcode == 'icmp':
                pred, ty, operands = body.split(' ', 2)
                a, b = map(self.value, split_args(operands))
                bits = int(ty[1:])
                ua, ub = a % (1 << bits), b % (1 << bits)
                sa = ua - (1 << bits) if ua >> (bits - 1) else ua
                sb = ub - (1 << bits) if ub >> (bits - 1) else ub
                self.cache[name] = {'eq': ua == ub, 'ne': ua != ub,
                    'slt': sa < sb, 'sle': sa <= sb, 'sgt': sa > sb, 'sge': sa >= sb,
                    'ult': ua < ub, 'ule': ua <= ub, 'ugt': ua > ub, 'uge': ua >= ub}[pred]
            elif opcode == 'getelementptr':
                ty, ptr, index = split_args(body)
                base, offset = self.typed(ptr)
                size = {'i8': 1, 'half': 2, 'float': 4}[ty.split()[-1]]
                self.cache[name] = (base, offset + size * self.typed(index))
            elif opcode == 'fadd':
                # QK + causal mask preserves the QK coordinate.
                self.cache[name] = self.typed(split_args(body)[0])
            elif '@__ocml_exp2_f32(' in expr or '@llvm.amdgcn.exp2.f32(' in expr:
                arg = re.search(r'@(?:__ocml_exp2_f32|llvm.amdgcn.exp2.f32)\(float (%[\w.]+)\)', expr)[1]
                self.cache[name] = ('exp2', self.value(arg))
            elif opcode == 'fptrunc':
                value = self.typed(body.split(' to ')[0])
                self.cache[name] = ([('f16', x) for x in value] if isinstance(value, list)
                                    else ('f16', value))
            elif opcode == 'load' and body.startswith('float,'):
                raise AssertionError('P still takes an FP32 LDS round trip')
        return super().value(name)


def check(text, packed=True):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    mma = [n for n, e in definitions.items() if 'asm sideeffect "' in e and 'v_mmac_' in e]
    assert len(mma) == 64
    pv_start = text.index(f'  {mma[32]} =')
    qk_end = text.index(f'  {mma[31]} =')
    stores = [m for m in re.finditer(r'^  store (half|<\d+ x half>) (%\w+), ptr addrspace\(3\) (%\w+),', text[:pv_start], re.M)
              if m.start() > qk_end]
    widths = [1 if m[1] == 'half' else int(m[1].split()[0][1:]) for m in stores]
    assert sum(widths) == 16, 'Expected exactly 16 P elements per thread'
    if packed:
        assert widths in ([8, 8], [4, 4, 4, 4]), 'Expected packed P register fragments'
    label = re.findall(r'^(\d+):', text[:stores[0].start()], re.M)[-1]
    guard = re.findall(r'br i1 (%\w+), label %' + label + r', label %(\w+)', text)
    assert len(guard) == 1
    assert not re.search(r'^\d+:', text[stores[0].start():stores[-1].end()], re.M)
    published = text[stores[-1].end():pv_start]
    first_read = re.search(r'load (?:half|<\d+ x half>), ptr addrspace\(3\)', published)
    assert first_read and '@llvm.amdgcn.s.barrier()' in published[:first_read.start()]
    assert re.search(r'^' + guard[0][1] + ':', published, re.M), 'Barrier must follow reconvergence'


    # Obtain the physical P base from the first actual PV operand at lane zero.
    pv_a = split_args(re.search(r'"\((.*)\)', definitions[mma[32]])[1])[0]
    base, offset = Trace(definitions, {tid: 0}).typed(pv_a)[0]
    covered = {}
    traces = []
    for thread in range(128):
        context = {tid: thread}
        for i, name in enumerate(mma[:32]):
            tile = i // 8
            row = tile // 2 * 32 + thread // 64 * 16 + thread % 16
            col = tile % 2 * 16 + thread % 64 // 16
            context[name] = [(row, col + 4 * r) for r in range(4)]
        traces.append(Trace(definitions, context, traces))
    for thread, trace in enumerate(traces):
        if not trace.value(guard[0][0]):
            continue
        for store in stores:
            ty, value, ptr = store.groups()
            values = trace.value(value)
            values = [values] if ty == 'half' else values
            begin = trace.value(ptr)
            for i, result in enumerate(values):
                assert result[0] == 'f16' and result[1][0] == 'exp2', result
                row, col = result[1][1]
                address = (begin[0], begin[1] + 2*i)
                if not packed:
                    assert address == (base, offset + 2 * (row*32+col)), (thread, result, address)
                assert address not in covered, ('duplicate store', thread, address)
                covered[address] = result
    assert set(covered) == {(base, offset + 2 * i) for i in range(64 * 32)}, 'P contains holes'
    # Validate actual consumer addresses, allowing the packing planner to
    # choose either strided-C packets or contiguous-C packets.
    for thread, trace in enumerate(traces):
        for i, name in enumerate(mma[32:]):
            fragment, k = divmod(i, 2)
            row = fragment // 8 * 32 + thread // 64 * 16 + thread % 16
            col = k * 16 + thread % 64 // 16 * 4
            operand = split_args(re.search(r'"\((.*)\)', definitions[name])[1])[0]
            assert [covered[p] for p in trace.typed(operand)] == [
                ('f16', ('exp2', (row, col + r))) for r in range(4)]
    print('PASS direct P copy: 2048 unique exp2/f16 values, all lane guards, PV addresses and barrier')


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text())
