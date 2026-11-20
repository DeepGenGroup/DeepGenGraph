#!/usr/bin/env python3
"""Check thread reciprocal sharing, exclusions, scope, replay and CPU results."""
import ctypes
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

tool = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'
llvm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(command, source=None):
    result = subprocess.run(list(map(str, command)), input=source, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def transform(source):
    return run([tool, '--verify-each', '--frisk-approximate-thread-math'], source)


def fixture(ty='f32', attrs=''):
    return f'''module {{
      func.func @normalize(%x: {ty}, %y: {ty}, %d: {ty}) -> {ty} {attrs} {{
        %a = arith.divf %x, %d : {ty}
        %b = arith.divf %y, %d : {ty}
        %r = arith.addf %a, %b : {ty}
        return %r : {ty}
      }}
    }}'''


for ty in ('f32', 'vector<4xf32>', 'vector<2x4xf32>'):
    result = transform(fixture(ty))
    assert result.count('arith.divf') == 1 and result.count('arith.mulf') == 2, result
    assert result.count('fastmath<arcp>') == 1, result
    assert transform(result) == result, 'non-idempotent reciprocal transform'

for ty in ('f16', 'f64', 'vector<4xf16>', 'vector<2x4xf64>'):
    result = transform(fixture(ty))
    assert result.count('arith.divf') == 2 and 'arith.mulf' not in result, result

# An existing reciprocal can feed later divisions without nesting reciprocals.
source = '''module {
  func.func @existing(%x: f32, %d: f32) -> (f32, f32) {
    %one = arith.constant 1.0 : f32
    %r = arith.divf %one, %d fastmath<arcp> : f32
    %a = arith.divf %x, %d : f32
    return %r, %a : f32, f32
  }
}'''
result = transform(source)
assert result.count('arith.divf') == 1 and result.count('arith.mulf') == 1, result
assert transform(result) == result
result = transform(source.replace('fastmath<arcp>', 'fastmath<arcp,nnan,ninf>'))
assert 'nnan' not in result and 'ninf' not in result, result
assert result.count('arith.divf') == 1 and result.count('fastmath<arcp>') == 1, result
assert transform(result) == result

for attr in ('strictfp', 'llvm.strictfp', 'optnone', 'llvm.optnone',
             'optimize_none', 'llvm.optimize_none',
             'passthrough = ["strictfp"]', 'passthrough = [["optnone", ""]]'):
    result = transform(fixture(attrs='attributes {' + attr + '}'))
    assert result.count('arith.divf') == 2 and 'arith.mulf' not in result, result

source = '''module {
  func.func @constants(%x: vector<4xf32>) -> vector<4xf32> {
    %c = arith.constant dense<[2.0, 3.0, 4.0, 5.0]> : vector<4xf32>
    %r = arith.divf %x, %c : vector<4xf32>
    return %r : vector<4xf32>
  }
  func.func @branches(%cond: i1, %x: f32, %d: f32) -> f32 {
    %r = scf.if %cond -> f32 {
      %a = arith.divf %x, %d : f32
      scf.yield %a : f32
    } else {
      %b = arith.divf %x, %d : f32
      scf.yield %b : f32
    }
    return %r : f32
  }
  func.func @loop(%n: index, %x: f32, %d: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%v = %x) -> f32 {
      %a = arith.divf %v, %d : f32
      scf.yield %a : f32
    }
    return %r : f32
  }
}'''
result = transform(source)
assert result.count('arith.divf') == 4 and result.count('arith.mulf') == 3, result
branches = result.split('func.func @branches', 1)[1].split('func.func @loop', 1)[0]
assert 'arith.divf' not in branches.split('scf.if', 1)[0], branches
assert branches.split('} else {')[0].count('arith.divf') == 1, branches
assert branches.split('} else {')[1].count('arith.divf') == 1, branches
loop = result.split('func.func @loop', 1)[1]
assert 'arith.divf' not in loop.split('scf.for', 1)[0], loop
assert transform(result) == result

# Execute the actual transformed MLIR, with FP32 rounding after each operation.
lowered = run([llvm / 'mlir-opt', '--convert-arith-to-llvm',
               '--convert-func-to-llvm', '--reconcile-unrealized-casts'], transform(fixture()))
ll = run([llvm / 'mlir-translate', '--mlir-to-llvmir'], lowered)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    run([llvm / 'llc', '-filetype=obj', '-relocation-model=pic', '-o', tmp / 'math.o'], ll)
    run(['cc', '-shared', tmp / 'math.o', '-o', tmp / 'math.so'])
    lib = ctypes.CDLL(str(tmp / 'math.so'))
    normalize = lib.normalize
    normalize.argtypes = [ctypes.c_float] * 3
    normalize.restype = ctypes.c_float
    f32 = lambda value: ctypes.c_float(value).value
    rng = random.Random(41)
    cases = [(f32(rng.uniform(-100, 100)), f32(rng.uniform(-100, 100)),
              f32(rng.uniform(.125, 100))) for _ in range(1000)]
    cases += [(0., -0., 1.), (1., -1., -2.), (float('inf'), 1., 2.),
              (float('nan'), 1., 2.), (1., 2., float('inf'))]
    for x, y, d in cases:
        actual = normalize(x, y, d)
        reciprocal = f32(1. / d)
        expected = f32(f32(x * reciprocal) + f32(y * reciprocal))
        assert (math.isnan(actual) and math.isnan(expected)) or actual == expected, (x, y, d, actual, expected)
print('PASS: scalar/vector sharing, f16/f64/constants/protected functions, branch/loop scope, idempotence; 1005 CPU cases')
