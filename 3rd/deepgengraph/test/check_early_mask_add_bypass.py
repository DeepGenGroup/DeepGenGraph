#!/usr/bin/env python3
"""Early mask bypass: both lowering paths, serialization, FP and fusion guards."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(sys.argv[1]).resolve()
SOURCE = Path(__file__).with_name('test_friskBase.mlir').read_text()
SOURCE = re.sub(r'%25 = scf.if %24 -> \(f32\) \{.*?\} else \{.*?\n        \}',
                '%25 = arith.select %24, %cst_0, %cst : f32', SOURCE, flags=re.S)
CONSUMER = '%16 = frisk.add %qk, %maskTIle'


def region(text, start):
    assert text[start] == '{'
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if not depth:
            return text[start + 1:end], end + 1
    raise AssertionError('unterminated region')


def uniform_branches(text):
    for match in re.finditer(r'scf.if[^\n{]*\{', text):
        clear, end = region(text, match.end() - 1)
        if not text[end:].startswith(' else {'):
            continue
        masked, end = region(text, end + len(' else '))
        attrs = text[end:].split('\n', 1)[0]
        if 'frisk.uniform_mask' in attrs:
            yield clear, masked, attrs


def main():
    multi = SOURCE.replace('%maskTIle = frisk.mask',
        '%score2 = frisk.mul %qk, %cst_1 : memref<64x32xf32>, f32 -> memref<64x32xf32>\n'
        '      %maskTIle = frisk.mask')
    multi = multi.replace(CONSUMER, '%extra = frisk.add %score2, %maskTIle : '
        'memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>\n      ' + CONSUMER)
    multi = multi.replace('%qkme = frisk.exp2 %16',
        '%sum = frisk.add %extra, %16 : memref<64x32xf32>, memref<64x32xf32> '
        '-> memref<64x32xf32>\n      %qkme = frisk.exp2 %sum')
    cases = [
        ('rhs', SOURCE, 'approx', True, True, 'bypassed'),
        ('lhs', SOURCE.replace(CONSUMER, '%16 = frisk.add %maskTIle, %qk'),
         'approx', True, True, 'bypassed'),
        ('off', SOURCE, 'off', True, True, None),
        ('no-uniform', SOURCE, 'approx', False, True, 'select-bypass'),
        ('no-fusion', SOURCE, 'approx', True, False, 'unfused-bypass'),
        ('multi-use', multi, 'approx', True, True, 'unfused-bypass'),
        ('barrier', SOURCE.replace(CONSUMER, 'gpu.barrier\n      ' + CONSUMER),
         'approx', True, True, 'no-fused-add'),
        ('negative-zero', SOURCE.replace('0.000000e+00 : f32', '-0.000000e+00 : f32'),
         'approx', True, True, 'clear-is-not-positive-zero'),
        ('nonzero', SOURCE.replace('0.000000e+00 : f32', '1.000000e+00 : f32'),
         'approx', True, True, 'clear-is-not-positive-zero'),
        ('signed-predicate', SOURCE.replace('cmpi ule', 'cmpi sle'),
         'approx', True, True, 'select-bypass'),
        ('negative-diagonal', SOURCE.replace('%c1 = arith.constant 1 : index',
                                             '%c1 = arith.constant -1 : index'),
         'approx', True, True, 'select-bypass'),
        ('64-threads', SOURCE.replace('thread_num = 128', 'thread_num = 64'),
         'approx', True, True, 'bypassed'),
        ('256-threads', SOURCE.replace('thread_num = 128 : i32',
            'thread_num = 256 : i32, frisk.block_warps = array<i64: 4, 1>'),
         'approx', True, True, 'bypassed'),
    ]
    for attr in ('strictfp', 'llvm.strictfp', 'optnone', 'llvm.optnone',
                 'optimize_none', 'llvm.optimize_none',
                 'passthrough = ["strictfp"]', 'passthrough = ["optnone"]',
                 'passthrough = [["denormal-fp-math", "preserve-sign"]]'):
        cases.append((attr, SOURCE.replace('attributes {thread_num',
            'attributes {' + attr + ', thread_num'), 'approx', True, True, 'protected-fp'))
    with tempfile.TemporaryDirectory(prefix='early-mask-bypass-') as tmp:
        tmp = Path(tmp)
        src, out = tmp / 'input.mlir', tmp / 'out.mlir'

        def run(text, passes, ok=True):
            src.write_text(text)
            result = subprocess.run([str(TOOL), str(src), *passes,
                '--verify-each', '-o', str(out)], capture_output=True, text=True, timeout=120)
            assert (result.returncode == 0) == ok, result.stderr[-6000:]
            return (out.read_text() if ok else ''), result.stderr

        for typed in (False, True):
            for name, source, mode, uniform, fuse, reason in cases:
                options = (f'uniform-masks={str(uniform).lower()} '
                           f'fuse-mask-add={str(fuse).lower()}')
                passes = ['--convert-friskbase-to-' + ('fragment' if typed else 'thread') + '=' + options]
                text, log = run(source, passes)
                if typed:
                    text, log = run(text, ['--convert-fragment-to-thread'])
                assert 'frisk.mask_add_bypass' not in text, 'bypass must wait for ThreadIR pass'
                text, log = run(text, ['--finalize-thread-tiling', '--lower-frisk-fragments', '--ir-deep-optimize',
                    '--canonicalize', '--cse',
                    '--frisk-bypass-mask-adds=mode=' + mode, '--canonicalize', '--cse'])
                bypassed = reason in ('bypassed', 'select-bypass', 'unfused-bypass')
                assert ('frisk.mask_add_bypass' in text) == bypassed, (typed, name, log)
                if reason == 'unfused-bypass':
                    counts = re.search(r': (\d+) groups, (\d+) additions bypassed', log)
                    assert counts and int(counts[1]) == 1 and int(counts[2]) >= 2, (name, log)
                if reason == 'bypassed':
                    branches = list(uniform_branches(text))
                    assert len(branches) == 1, (typed, name)
                    clear, masked, attrs = branches[0]
                    assert 'frisk.mask_add_bypass' in attrs
                    assert 'arith.addf' not in clear, clear
                    assert 'arith.addf' in masked
                    assert 'fastmath' not in masked
                    # Masked additions retain operand order; score extracts
                    # may now be shared outside this branch after unrolling.
                    assert re.search(r'arith.addf %\w+, %\w+', masked)
                if name in ('off', 'nonzero') or reason == 'protected-fp':
                    branches = list(uniform_branches(text))
                    assert len(branches) == 1 and 'arith.addf' in branches[0][0], (typed, name)
                if name == 'rhs':
                    parsed, _ = run(text, [])
                    again, _ = run(text, ['--frisk-bypass-mask-adds=mode=' + mode])
                    assert parsed == again, 'not idempotent'
                print('PASS', 'typed' if typed else 'direct', name, flush=True)
            for invalid in ('legal', 'invalid'):
                _, error = run('module { func.func @f() { return } }',
                    ['--frisk-bypass-mask-adds=mode=' + invalid], ok=False)
                assert 'mask-add-bypass must be off or approx' in error



if __name__ == '__main__':
    main()
