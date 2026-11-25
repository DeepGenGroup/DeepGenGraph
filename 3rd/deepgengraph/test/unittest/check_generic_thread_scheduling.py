#!/usr/bin/env python3
"""Generic scheduling: register MMA chains, operand sources, read budgets and ALU windows."""
import argparse
from pathlib import Path
import re
import subprocess

from check_pv_interleave import fixture as matrix_fixture, ids, result_graph
from check_mma_interleave import fixture as lds_fixture
from check_thread_fragment_schedules import paired, graph


def register_chains(chains=2, separated=False):
    lines = ['module {', 'func.func private @unknown()',
             'func.func @ordinary_gemm(%a: vector<4xf16>, %b: vector<4xf16>, '
             '%seed: vector<4xf32>) -> (' + ', '.join(['vector<4xf32>'] * chains) + ') {']
    for c in range(chains):
        for k in range(3):
            previous = '%seed' if k == 0 else f'%c{c}_{k-1}'
            lines.append(f'%c{c}_{k} = frisk.warp_mma_rr %a, %b, {previous} '
                         f'{{inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", '
                         f'inst_constraints = "=v,v,v,0", test.id = "{c}:{k}"}} : '
                         'vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>')
        if separated and c == 1:
            lines.append('func.call @unknown() : () -> ()')
    lines += ['return ' + ', '.join(f'%c{c}_2' for c in range(chains)) + ' : ' +
              ', '.join(['vector<4xf32>'] * chains), '}', '}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path,
                        default=Path(__file__).resolve().parents[2]/'build/test/ThreadTilingTest')
    args = parser.parse_args()
    cases = 0

    def run(source, *passes, ok=True):
        result = subprocess.run([str(args.tool), '--verify-each', *passes],
                                input=source, text=True, capture_output=True, timeout=120)
        assert (result.returncode == 0) == ok, result.stderr
        return result.stdout if ok else result.stderr

    interleave = '--frisk-thread-interleave-mma-chains=width=2 guard-group=0'
    original = run(register_chains())
    result = run(original, interleave)
    assert ids(result) == ['0:0', '1:0', '0:1', '1:1', '0:2', '1:2'], result
    assert result_graph(result) == result_graph(original)
    assert run(result, interleave) == result
    cases += 1

    original = run(register_chains(chains=4, separated=True))
    result = run(original, interleave)
    assert ids(result) == ['0:0', '1:0', '0:1', '1:1', '0:2', '1:2',
                           '2:0', '3:0', '2:1', '3:1', '2:2', '3:2'], result
    assert result_graph(result) == result_graph(original)
    assert ids(result.split('call @unknown()')[0]) == ['0:0', '1:0', '0:1', '1:1', '0:2', '1:2']
    cases += 1

    # A ready LDS operand can be reused at several steps of either chain.
    shared = register_chains().replace('%b: vector<4xf16>', '%sm: memref<4096xf16, 3>')
    shared = shared.replace('%c0_0 =', '%i = arith.constant 0 : index\n'
        '%b = frisk.lds_load %sm[%i] {alignment = 8 : i64, byte_offset = 0 : i64} : '
        'memref<4096xf16, 3> -> vector<4xf16>\n%c0_0 =')
    original = run(shared)
    result = run(original, interleave+' operand-source=lds')
    assert ids(result) == ['0:0', '1:0', '0:1', '1:1', '0:2', '1:2'], result
    assert result_graph(result) == result_graph(original)
    cases += 1

    for source, kind in [(matrix_fixture(chains=2, steps=3), 'matrix'),
                         (lds_fixture(chains=2, steps=3, distinct=True), 'lds')]:
        original = run(source)
        motion = 'none' if kind == 'matrix' else 'bounded'
        option = interleave + f' operand-source={kind} load-motion={motion} prefetch-words=12'
        result = run(original, option)
        assert ids(result) == ['0:0', '1:0', '0:1', '1:1', '0:2', '1:2'], result
        assert result_graph(result) == result_graph(original)
        assert run(result, option) == result
        cases += 1

    source = run(lds_fixture(chains=2, steps=3, distinct=True))
    assert run(source, interleave+' operand-source=lds load-motion=bounded prefetch-words=11') == source
    assert run(source, interleave+' operand-source=register') == source
    blocked = lds_fixture(chains=2, steps=3, distinct=True).replace('// boundary', 'gpu.barrier')
    assert run(blocked, interleave+' operand-source=lds load-motion=bounded') == run(blocked)
    cases += 3

    # Ordinary GEMM reads its A operands from LDS. add/mul are pure register
    # computations, rather than an attention-specific reduction pattern.
    source = paired(count=8, tail=True).replace('arith.maxnumf', 'arith.addf')
    for i in range(8):
        source = source.replace(f'%a, %b{i},', f'%b{i}, %a,')
    source = source.replace('%r1 = arith.addf', '%r1 = arith.mulf')
    original = run(source)
    prefetch = '--frisk-thread-prefetch-paired-lds=distance=4 window=register'
    result = run(original, prefetch)
    assert 'frisk.lds_pair' in result, result
    assert graph(result) == graph(original)
    assert run(result, prefetch) == result
    cases += 1

    # Add a live continuation to an already scheduled function. A function
    # marker must not prevent new pairs, or reuse old pair identifiers.
    header = next(line for line in result.splitlines() if 'func.func @kernel(' in line)
    sm, a, _, _ = re.findall(r'(%\w+): ', header)
    index = re.search(r'(%\w+) = arith.constant 0 : index', result)[1]
    returned = re.search(r'return (%\w+), (%\w+) :', result)
    seed, scalar = returned.groups()
    extra = []
    for i in range(4):
        extra.append(f'%newb{i} = frisk.lds_load {sm}[{index}] '
                     f'{{alignment = 8 : i64, byte_offset = {32768+i*128} : i64}} : '
                     'memref<65536xf16, 3> -> vector<4xf16>')
        previous = seed if i == 0 else f'%newm{i-1}'
        extra.append(f'%newm{i} = llvm.inline_asm has_side_effects convergent '
                     '"v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0" '
                     f'{a}, %newb{i}, {previous} : '
                     '(vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>')
    augmented = result[:returned.start()] + '\n'.join(extra) + '\n' + \
        result[returned.start():].replace(f'return {seed}, {scalar}', f'return %newm3, {scalar}', 1)
    original = run(augmented)
    replayed = run(original, prefetch)
    pairs = re.findall(r'frisk.lds_pair = (\d+) : i64', replayed)
    assert sorted(set(map(int, pairs))) == [0, 1, 2, 3, 4, 5], replayed
    assert all(pairs.count(str(i)) == 2 for i in range(6)), replayed
    assert graph(replayed) == graph(original)
    assert run(replayed, prefetch) == replayed
    cases += 1

    for boundary in ['gpu.barrier', 'func.call @unknown() : () -> ()',
                     'vector.store %a, %sm[%i] : memref<65536xf16, 3>, vector<4xf16>']:
        source = run(paired(tail=True, boundary=boundary))
        assert run(source, prefetch) == source
        cases += 1
    for bad in ['operand-source=unknown', 'load-motion=unknown']:
        run(register_chains(), interleave+' '+bad, ok=False)
    run(paired(), '--frisk-thread-prefetch-paired-lds=distance=4 window=unknown', ok=False)
    cases += 3
    print(f'PASS {cases} generic scheduling contracts (ordinary GEMM, SSA, budgets, boundaries, replay)')


if __name__ == '__main__':
    main()
