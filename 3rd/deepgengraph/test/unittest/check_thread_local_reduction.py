#!/usr/bin/env python3
"""Tagged thread reductions: legality, boundaries, lowering and CPU execution."""
import ctypes
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-thread-local-reduction'


def run(command, source):
    p = subprocess.run(list(map(str, command)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr
    return p.stdout


def transform(source, options='approximate=true', *passes):
    return run([TOOL, '--verify-each', PASS + '=' + options, *passes], source)


def fixture(n=32, op='addf', flags='', tagged=True, gap=-1, escape=-1,
            attrs='', ty='f32', seed=None, right=False, split=-1, late=False):
    args = ', '.join(f'%x{i}: {ty}' for i in range(n))
    result_types = f'({ty}, {ty})' if escape >= 0 else ty
    lines = [f'func.func @reduce({args}) -> {result_types} {attrs} {{']
    if seed is not None:
        lines.append(f'%seed = arith.constant {seed} : {ty}')
    prev = '%seed' if seed is not None else '%x0'
    escaped = None
    for i in range(0 if seed is not None else 1, n):
        if i == split:
            lines += [f'cf.br ^next({prev} : {ty})', f'^next(%carry: {ty}):']
            prev = '%carry'
        value = f'%x{i}'
        if late:
            lines.append(f'%v{i} = arith.negf {value} : {ty}')
            value = f'%v{i}'
        mark = '{frisk.try_multichain}' if tagged and i != gap else ''
        lhs, rhs = (value, prev) if right and i % 2 else (prev, value)
        lines.append(f'%s{i} = arith.{op} {lhs}, {rhs} {flags} {mark} : {ty}')
        prev = f'%s{i}'
        if i == escape:
            escaped = prev
    lines.append(f'return {prev}, {escaped} : {ty}, {ty}' if escaped else f'return {prev} : {ty}')
    return '\n'.join(lines + ['}'])


def unchanged(source, options='approximate=true'):
    assert transform(source, options) == run([TOOL], source)


for source in (fixture(tagged=False), fixture(n=7), fixture(n=24, gap=12),
               fixture(n=24, escape=12), fixture(n=24, split=12),
               fixture(op='subf'), fixture(ty='vector<4xf32>'),
               fixture(ty='i32', op='addi')):
    unchanged(source)
for attrs in ('strictfp', 'llvm.optnone', 'passthrough = ["strictfp"]'):
    unchanged(fixture(attrs='attributes {' + attrs + '}'))
for flags in ('', 'fastmath<reassoc>', 'fastmath<nsz>'):
    unchanged(fixture(flags=flags), 'approximate=false')
unchanged(fixture(), 'approximate=true ops=max')
unchanged(fixture(n=15))


def depth(text):
    depths = {}
    for name, lhs, rhs in re.findall(r'(%\w+) = arith.\w+ (%\w+), (%\w+)', text):
        depths[name] = 1 + max(depths.get(lhs, 0), depths.get(rhs, 0))
    return max(depths.values())


for op, flags in [('addf', 'reassoc,nsz'), ('mulf', 'reassoc,nsz'),
                  ('minnumf', 'nnan,nsz'), ('maxnumf', 'nnan,nsz')]:
    for n, chains in [(16, 4), (32, 4), (37, 3), (41, 5), (47, 16)]:
        options = f'approximate=false chains={chains} min-inputs={max(16, 2 * chains)}'
        source = fixture(n=n, op=op, flags=f'fastmath<{flags}>', right=True, late=True)
        result = transform(source, options)
        assert 'frisk.local_reduction' in result and 'frisk.try_multichain' not in result
        assert result.count('arith.' + op) == n - 1
        assert 'fastmath<' not in result
        assert depth(result) < n - 1
        assert transform(result, options) == result
    unchanged(fixture(n=24, op=op, flags=f'fastmath<{flags}>', gap=12), 'approximate=false')

for op, seed in [('addf', '0.0'), ('mulf', '1.0'),
                  ('maxnumf', '0xFF800000'), ('minnumf', '0x7F800000')]:
    result = transform(fixture(op=op, seed=seed))
    assert result.count('arith.' + op) == 32
    assert result.count('frisk.local_reduction') == 32
    assert transform(result) == result

# A branching tree is not flattened. Shared values bound the spine, but a long
# tail can still be split without losing the escaped intermediate result.
tree = fixture(n=16).replace('return %s15 : f32',
    '%tree = arith.addf %s15, %s15 {frisk.try_multichain} : f32\nreturn %tree : f32')
assert transform(tree).count('arith.addf') == 16
assert transform(fixture(n=48, escape=16)).count('arith.addf') == 47

for options in ('chains=1', 'chains=17', 'chains=8 min-inputs=15', 'ops=bad'):
    p = subprocess.run([str(TOOL), PASS + '=' + options], input=fixture(),
                       text=True, capture_output=True)
    assert p.returncode != 0 and 'invalid thread local reduction options' in p.stderr

# Exercise the real producer: attributes survive fragment lowering/unrolling,
# while the cross-lane combines remain unmarked and outside the rewritten chain.
fragment = '''
!src = !frisk.fragment<vector<1x32xf32>, {}>
!dst = !frisk.fragment<vector<1xf32>, {}>
func.func @fragment(%v: vector<1x32xf32>) -> vector<1xf32> {
  %f = frisk.pack_frag %v : vector<1x32xf32> -> !src
  %r = frisk.reduce_frag "add" %f {dim = 1 : i64, lane_extent = 4 : i64,
      lane_stride = 1 : i64, warp_threads = 64 : i64, reduction_width = 4 : i64,
      output_fragment_shape = array<i64: 1>} : !src -> !dst
  %u = frisk.unpack_frag %r : !dst -> vector<1xf32>
  return %u : vector<1xf32>
}'''
lowered = run([TOOL, '--convert-fragment-to-thread', '--lower-frisk-fragments',
               '--ir-deep-optimize', '--canonicalize'], fragment)
assert 'frisk.try_multichain' in lowered and lowered.count('gpu.shuffle') == 2
result = transform(lowered)
assert 'frisk.local_reduction' in result and result.count('gpu.shuffle') == 2
assert sum('arith.addf' in line and 'frisk.local_reduction' not in line
           for line in result.splitlines()) == 2

# Compile actual transformed scalar functions and execute exact finite inputs.
functions = '\n'.join(transform(fixture(op=op)).replace('@reduce', '@' + op)
                       .removeprefix('module {').rsplit('}', 1)[0]
                       for op in ('addf', 'mulf', 'minnumf', 'maxnumf'))
lowered = run([LLVM / 'mlir-opt', '--convert-arith-to-llvm',
               '--convert-func-to-llvm', '--reconcile-unrealized-casts'],
              'module {\n' + functions + '\n}')
ll = run([LLVM / 'mlir-translate', '--mlir-to-llvmir'], lowered)
with tempfile.TemporaryDirectory(prefix='thread-local-reduction-') as tmp:
    tmp = Path(tmp)
    run([LLVM / 'llc', '-filetype=obj', '-relocation-model=pic', '-o', tmp / 'test.o'], ll)
    run(['cc', '-shared', tmp / 'test.o', '-o', tmp / 'test.so'], None)
    library = ctypes.CDLL(str(tmp / 'test.so'))
    rng = random.Random(17)
    for op in ('addf', 'mulf', 'minnumf', 'maxnumf'):
        fn = getattr(library, op)
        fn.argtypes = [ctypes.c_float] * 32
        fn.restype = ctypes.c_float
        for _ in range(100):
            values = [rng.choice([-1., 1., 2.]) if op == 'mulf'
                      else float(rng.randrange(-16, 17)) for _ in range(32)]
            expected = values[0]
            for value in values[1:]:
                expected = (expected + value if op == 'addf' else
                            expected * value if op == 'mulf' else
                            min(expected, value) if op == 'minnumf' else max(expected, value))
            assert fn(*values) == expected, (op, values)
print('PASS: attribute/legality gates, boundaries, dominance, chain counts/depth, '
      'idempotence, fragment producer/shuffles, 400 CPU cases')
