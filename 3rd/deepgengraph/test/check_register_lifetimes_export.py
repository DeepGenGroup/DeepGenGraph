#!/usr/bin/env python3
"""Compare complete exported operand DAGs, ignoring only register identities.

PHIs are numbered symbolic roots. Their incoming expressions are compared
separately, preserving recurrence structure and floating-point association.
"""
import collections
import hashlib
import re
import sys
from pathlib import Path


def describe(text):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M))
    phis = {name: f'phi:{i}' for i, (name, _) in enumerate(
        (item for item in definitions.items() if item[1].startswith('phi ')))}
    args = re.findall(r'%[\w.]+', re.search(r'define amdgpu_kernel .*?\((.*?)\)',
                                           text.replace('addrspace(1)', 'AS1'))[1])
    cache = {name: f'arg:{i}' for i, name in enumerate(args)} | phis

    def expression(expr):
        expr = re.sub(r', ![\w.]+ !\d+', '', expr)
        expr = re.sub(r' #\d+', '', expr)
        return re.sub(r'%[\w.]+', lambda m: value(m[0]), expr)

    def value(name):
        if name not in cache:
            expr = definitions[name]
            if 'asm sideeffect "", "=v,0"(' in expr:
                operand = re.search(r'"=v,0"\(.*? (%[\w.]+)\)', expr)[1]
                cache[name] = value(operand)
            else:
                cache[name] = hashlib.sha256(expression(expr).encode()).hexdigest()
        return cache[name]

    memory = [expression(line.strip()) for line in text.splitlines()
              if line.startswith('  store ')]
    loads = [value(n) for n, e in definitions.items() if e.startswith('load ')]
    math = collections.Counter(value(n) for n, e in definitions.items()
        if re.match(r'f(add|sub|mul|div|rem|neg|ptrunc|pext)\b', e) or
        '@__ocml_exp2_f32' in e or 'v_mmac_' in e)
    recurrences = [[value(v) if v.startswith('%') else v
                    for v in re.findall(r'\[ ([^,]+), %[\w.]+ \]', definitions[n])]
                   for n in phis]
    return memory, loads, math, recurrences


def main():
    before, after = (Path(p).read_text() for p in sys.argv[1:3])
    old, new = describe(before), describe(after)
    for name, left, right in zip(('stores', 'loads', 'FP/MMAC DAGs', 'loop recurrences'), old, new):
        assert left == right, f'{name} changed beyond identity materialization'
    packs = len(re.findall(r'%math.packed[\w.]* =', after))
    assert packs > 0 and 'address.thread' in after
    # Each pack consumes at most one small group of fresh exp2 outputs;
    # previously every cast followed the entire exp2 block.
    pending = 0
    for line in after.splitlines():
        if 'call float @__ocml_exp2_f32(' in line:
            pending += 1
        if re.search(r'%math.packed[\w.]* =', line):
            assert 0 < pending <= 4, pending
            pending = 0
    assert pending == 0
    print(f'PASS complete address/FP/MMAC operand graphs and loop recurrences unchanged; {packs} early packs, at most 4 exp2 calls between packs')


if __name__ == '__main__':
    main()
