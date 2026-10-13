#!/usr/bin/env python3
"""Check each fused seed against the unfused attention's actual loop carrier.

Usage: check_mmac_accumulator_export.py baseline.ll fused.ll [LLVM bin dir]
This checks register coordinates and chains; GPU rounding/performance are not
emulated. Baseline and fused files must use the same tile and thread layout.
"""
import re
import subprocess
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args


def describe(text):
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    calls = {n: split_args(re.search(r'"\((.*)\)', e)[1])
             for n, e in defs.items() if 'asm sideeffect ' in e and 'v_mmac_' in e}
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    context = {tid: 0}
    shapes = []
    for n, e in defs.items():
        m = re.match(r'phi \[(\d+) x <(\d+) x float>\]', e)
        if m:
            rows, cols = map(int, m.groups())
            context[n] = [[('carrier', len(shapes), r, c) for c in range(cols)]
                          for r in range(rows)]
            shapes.append((rows, cols))
    return defs, calls, Trace(defs, context), shapes


def main():
    baseline, fused = map(Path, sys.argv[1:3])
    llvm = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/data2/xsl/install/bin')
    before = subprocess.check_output([str(llvm / 'opt'), '-passes=instcombine',
                                      '-S', str(baseline), '-o', '-'], text=True)
    after = fused.read_text()
    old_defs, old_calls, old_trace, old_shapes = describe(before)
    new_defs, new_calls, new_trace, new_shapes = describe(after)
    assert old_shapes == new_shapes
    old_ids, new_ids = list(old_calls), list(new_calls)
    assert len(old_ids) == len(new_ids)
    expected_seeds = {}
    for expression in old_defs.values():
        if not expression.startswith('fadd '):
            continue
        operands = split_args(expression[len('fadd '):])
        for side, operand in enumerate(operands):
            last = operand.split()[-1]
            if last not in old_calls:
                continue
            first = last
            while old_calls[first][2].split()[-1] in old_calls:
                first = old_calls[first][2].split()[-1]
            assert old_calls[first][2].endswith('zeroinitializer')
            expected_seeds[old_ids.index(first)] = old_trace.typed(operands[1 - side])
    assert len(expected_seeds) == 24, expected_seeds.keys()
    for i, (old, new) in enumerate(zip(old_ids, new_ids)):
        if i in expected_seeds:
            assert new_trace.typed(new_calls[new][2]) == expected_seeds[i], i
        else:
            old_prev = old_calls[old][2].split()[-1]
            new_prev = new_calls[new][2].split()[-1]
            assert old_ids.index(old_prev) == new_ids.index(new_prev), i
    for expression in new_defs.values():
        if expression.startswith('fadd '):
            assert not any(op.split()[-1] in new_calls
                           for op in split_args(expression[len('fadd '):]))
    for opcode in ['call float @__ocml_exp2_f32', 'fdiv <4 x float>']:
        assert before.count(opcode) == after.count(opcode)
    assert '@llvm.amdgcn.exp2' not in after and 'fdiv arcp' not in after
    assert not re.search(r'\b(disjoint|nusw|nneg|samesign)\b', after)
    print(f'PASS {len(expected_seeds)} exact carrier-fragment seeds, {len(old_ids)} MMAC calls, '
          'unchanged inner chains/OCML/division, no terminal MMAC fadd')


if __name__ == '__main__':
    main()
