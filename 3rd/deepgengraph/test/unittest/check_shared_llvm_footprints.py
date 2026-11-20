#!/usr/bin/env python3
"""Compare all wave64 LDS read addresses in two exported attention kernels.

This executes the LLVM address expressions, not a copy of the planner's maps.
It checks per-lane footprints (including multiplicity), independently of SSA
names and scheduling. It does not assert target timing or instruction pairing.
Usage: check_shared_llvm_footprints.py reference.ll candidate.ll
"""
from collections import Counter
from pathlib import Path
import re
import sys

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args


class AddressTrace(Trace):
    def value(self, name):
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            if expr.startswith('ptrtoint '):
                pointer = self.typed(expr.split(' to ')[0][9:])
                assert pointer[0] == '@shm_0', pointer
                self.cache[name] = pointer[1]
            elif expr.startswith('inttoptr '):
                self.cache[name] = ('@shm_0', self.typed(expr.split(' to ')[0][9:]))
            elif '@llvm.amdgcn.readfirstlane.i32(' in expr:
                operand = re.search(r'\(i32 (%[\w.]+)\)', expr)[1]
                self.cache[name] = self.peers[self.thread // 64 * 64].value(operand)
        return super().value(name)


def footprints(path):
    text = path.read_text()
    definitions = {}
    for name, expr in re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M):
        expr = re.sub(r', ![\w.]+ !\d+', '', expr)
        expr = expr.replace('icmp samesign ', 'icmp ')
        expr = re.sub(r'trunc (?:(?:nsw|nuw) )+', 'trunc ', expr)
        definitions[name] = expr
    tids = [name for name, expr in definitions.items()
            if '@llvm.amdgcn.workitem.id.x()' in expr]
    assert tids
    # These fixtures have static LDS addresses across workgroups and BN loops.
    context = {name: 0 for name, expr in definitions.items()
               if '@llvm.amdgcn.workgroup.id.' in expr or expr.startswith('phi i64 ')}
    peers = []
    peers.extend(AddressTrace(definitions, {**context, **dict.fromkeys(tids, tid)}, peers)
                 for tid in range(256))
    result = {'matrix': Counter(), 'ordinary': Counter()}
    for expr in definitions.values():
        if 'call <4 x i32> @llvm.hcu.ds.read.m32x16' in expr:
            args = re.search(r'@llvm.hcu.ds.read[^ (]*\((.*)\)', expr)[1]
            operand = split_args(args)[0]
            kind = 'matrix'
        elif expr.startswith('load <4 x half>, ptr addrspace(3)'):
            operand = split_args(expr[5:])[1]
            kind = 'ordinary'
        else:
            continue
        addresses = tuple(trace.typed(operand) for trace in peers)
        assert all(base == '@shm_0' and 0 <= byte < 16384 for base, byte in addresses)
        result[kind][addresses] += 1
    assert all(result.values()), 'expected both ordinary and collective LDS reads'
    return result


def main():
    reference, candidate = (footprints(Path(path)) for path in sys.argv[1:])
    for kind in reference:
        before, after = reference[kind], candidate[kind]
        assert before == after, (kind, 'reference-only', sum((before - after).values()),
                                 'candidate-only', sum((after - before).values()))
        print(f'PASS {sum(before.values())} {kind} LDS reads: identical addresses for all 256 threads')


if __name__ == '__main__':
    main()
