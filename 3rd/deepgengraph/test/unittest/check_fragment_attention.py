#!/usr/bin/env python3
"""Validate static QK/PV fragment addresses and zero-seeded PV K reductions.

Usage: check_fragment_attention.py path/to/fragment.ll
Uses the existing address evaluator; no GPU is required.
"""
import re
import sys
from pathlib import Path

from check_gemm_pv_addresses import Addresses


def main():
    text = Path(sys.argv[1]).read_text()
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    mma = [name for name, expr in definitions.items()
           if 'asm sideeffect "' in expr and 'v_mmac_' in expr]
    # QK: 4 static MN fragments x 8 unrolled K steps. PV: 16 static
    # MN fragments, each retaining its two-iteration K reduction loop.
    assert len(mma) == 48, len(mma)
    for name in mma:
        asm = definitions[name].split('asm sideeffect "', 1)[1].split('"', 1)[0]
        prefix, suffix = asm.split('v_mmac_f32_16x16x16_f16', 1)
        assert suffix.startswith(' $0, $1, $2, $3')
        assert prefix.count('s_nop 7') == suffix.count('s_nop 7') == 8
    for line in text.splitlines():
        if 'extractelement' in line or 'insertelement' in line:
            assert not re.search(r', i(?:32|64) %[^ ,]+(?:, !dbg.*)?$', line), line
    assert 'fadd <32 x float>' not in text, 'Full-tile O addition remains'
    assert text.count('fadd <4 x float>') == 16, 'Expected one update per O fragment'
    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)

    for tile, name in enumerate(mma[32:]):
        pv = definitions[name]
        operands = re.search(r'"\(<4 x half> (%\w+), <4 x half> (%\w+),', pv).groups()
        before = text[:text.index(f'  {name} =')]
        k = re.findall(r'^  (%\w+) = phi i64', before, re.M)[-1]
        assert re.search(rf'icmp slt i64 {re.escape(k)}, 2\b', before)
        accumulator = re.search(r', <4 x float> (%\w+)\)', pv).group(1)
        carried = re.search(r'extractvalue \[1 x <4 x float>\] (%\w+), 0',
                            definitions[accumulator]).group(1)
        phi = definitions[carried]
        assert phi.startswith('phi [1 x <4 x float>]') and '[ zeroinitializer,' in phi
        updated = re.search(r'\[ (%\w+),', phi).group(1)
        assert f'<4 x float> {name}, 0' in definitions[updated]
        p_buffer = Addresses(definitions, {tid: 0, k: 0}).value(operands[0])[0][0]
        assert re.search(re.escape(p_buffer) + r' = addrspace\(3\) global \[64 x \[32 x half\]\]', text)
        for thread in range(128):
            lane = thread % 64
            for reduction in range(2):
                evaluator = Addresses(definitions, {tid: thread, k: reduction})
                a, b = map(evaluator.value, operands)
                row = tile // 8 * 32 + thread // 64 * 16 + lane % 16
                col = tile % 8 * 16 + lane % 16
                kk = reduction * 16 + lane // 16 * 4
                assert a == [(p_buffer, row * 32 + kk + j) for j in range(4)], (thread, tile, reduction, a)
                assert b == [('@shm_2', (kk + j) * 128 + col) for j in range(4)], (thread, tile, reduction, b)
    print('PASS PV: 32768 operand addresses, zero-seeded two-step K reductions, 16 fragment adds')

    scaled_q = [n for n, e in definitions.items() if e.startswith('fmul <32 x half>')]
    assert len(scaled_q) == 2
    for thread in range(128):
        lane = thread % 64
        context = {tid: thread}
        for block_m, name in enumerate(scaled_q):
            row = block_m * 32 + thread // 64 * 16 + lane % 16
            context[name] = [('Q', row, j // 4 * 16 + lane // 16 * 4 + j % 4)
                             for j in range(32)]
        for i, name in enumerate(mma[:32]):
            tile, reduction = divmod(i, 8)
            q_operand = re.search(r'"\(<4 x half> (%\w+),', definitions[name]).group(1)
            actual = Addresses(definitions, context).value(q_operand)
            row = tile // 2 * 32 + thread // 64 * 16 + lane % 16
            expected = [('Q', row, reduction * 16 + lane // 16 * 4 + j) for j in range(4)]
            assert actual == expected, (thread, tile, reduction, actual, expected)
    print('PASS QK: 16384 scaled-Q elements retain M/K coordinates; no dynamic vector indices')
    print('PASS MMAC: operand order and hardware-validated nop padding preserved')


if __name__ == '__main__':
    main()
