#!/usr/bin/env python3
"""Structural and CPU numerical regressions for fragment optimization.

Usage: check_fragment_optimization.py [FragmentOptTest] [LLVM bin directory]
The CPU test checks the transformation's floating-point order, not MMAC hardware.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


def run(command, text=None):
    result = subprocess.run(list(map(str, command)), input=text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=120)
    assert result.returncode == 0, f"{command}\n{result.stderr[-6000:]}"
    return result.stdout


def fixture(reverse=False, offsets=((0, 0), (0, 4), (1, 0), (1, 4)),
            extra_use=False, late_accumulator=False):
    body = ['%zero = arith.constant dense<0.0> : vector<2x8xf32>']
    for i, (row, col) in enumerate(offsets):
        dest = '%zero' if i == 0 else f'%t{i-1}'
        body.append(f'%t{i} = vector.insert_strided_slice %d{i}, {dest} '
                    f'{{offsets = [{row}, {col}], strides = [1, 1], '
                    'frisk.mma_fragment} : vector<1x4xf32> into vector<2x8xf32>')
    last = f'%t{len(offsets)-1}'
    old = '%old'
    if late_accumulator:
        body.append('%late = arith.mulf %old, %old : vector<2x8xf32>')
        old = '%late'
    a, b = (last, old) if reverse else (old, last)
    body.append(f'%out = arith.addf {a}, {b} : vector<2x8xf32>')
    if extra_use:
        body.append(f'%extra = arith.addf %out, {last} : vector<2x8xf32>')
    body.append(f'return {"%extra" if extra_use else "%out"} : vector<2x8xf32>')
    args = ', '.join(['%old: vector<2x8xf32>'] +
                     [f'%d{i}: vector<1x4xf32>' for i in range(4)])
    return f'func.func @compute({args}) -> vector<2x8xf32> {{\n' + '\n'.join(body) + '\n}\n'


STATIC = '''
func.func @static_indices(%input: vector<2x8xf32>) -> vector<2x8xf32> {
  %zero = arith.constant dense<0.0> : vector<2x8xf32>
  %out = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> vector<2x8xf32> {
    %next = affine.for %j = 0 to 4 iter_args(%a = %acc) -> vector<2x8xf32> {
      %row = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
      %col = affine.apply affine_map<(d0, d1) -> ((d0 mod 2) * 4 + d1)>(%i, %j)
      %x = vector.extract %input[%row, %col] : f32 from vector<2x8xf32>
      %v = vector.insert %x, %a[%row, %col] : f32 into vector<2x8xf32>
      affine.yield %v : vector<2x8xf32>
    }
    affine.yield %next : vector<2x8xf32>
  }
  return %out : vector<2x8xf32>
}
func.func @dynamic_indices(%input: vector<8xf32>, %n: index) -> f32 {
  %zero = arith.constant 0.0 : f32
  %out = affine.for %i = 0 to %n iter_args(%a = %zero) -> f32 {
    %x = vector.extract %input[%i] : f32 from vector<8xf32>
    %s = arith.addf %a, %x : f32
    affine.yield %s : f32
  }
  return %out : f32
}
'''

# Cancellation is sensitive to incorrectly seeding the reduction with old O:
# (1e20 + -1e20) + 1 = 1, whereas (1 + 1e20) + -1e20 = 0 in f32.
MAIN = '''
func.func @main() -> i32 {
  %old = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]> : vector<2x8xf32>
  %zero = arith.constant dense<0.0> : vector<1x4xf32>
  %pos = arith.constant dense<1.0e20> : vector<1x4xf32>
  %neg = arith.constant dense<-1.0e20> : vector<1x4xf32>
  %c0 = arith.constant 0 : index
  %d0 = affine.for %k = 0 to 2 iter_args(%a = %zero) -> vector<1x4xf32> {
    %first = arith.cmpi eq, %k, %c0 : index
    %x = arith.select %first, %pos, %neg : vector<1x4xf32>
    %s = arith.addf %a, %x : vector<1x4xf32>
    affine.yield %s : vector<1x4xf32>
  }
  %d1 = arith.constant dense<[[1.0, -2.0, 3.0, -4.0]]> : vector<1x4xf32>
  %d2 = arith.constant dense<[[10.0, 20.0, 30.0, 40.0]]> : vector<1x4xf32>
  %d3 = arith.constant dense<[[-13.0, -14.0, -15.0, -16.0]]> : vector<1x4xf32>
  %out = func.call @compute(%old, %d0, %d1, %d2, %d3) :
    (vector<2x8xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>) -> vector<2x8xf32>
  %expected = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 6.0, 4.0, 10.0, 4.0],
                                    [19.0, 30.0, 41.0, 52.0, 0.0, 0.0, 0.0, 0.0]]> : vector<2x8xf32>
  %bits = arith.bitcast %out : vector<2x8xf32> to vector<2x8xi32>
  %ref = arith.bitcast %expected : vector<2x8xf32> to vector<2x8xi32>
  %equal = arith.cmpi eq, %bits, %ref : vector<2x8xi32>
  %flat = vector.shape_cast %equal : vector<2x8xi1> to vector<16xi1>
  %ok = vector.reduction <and>, %flat : vector<16xi1> into i1
  %success = arith.constant 0 : i32
  %failure = arith.constant 1 : i32
  %status = arith.select %ok, %success, %failure : i32
  return %status : i32
}
'''


def main():
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_DIR.parent / 'build/test/FragmentOptTest'
    llvm_bin = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')
    fuse = [binary, '--pass-pipeline=builtin.module(func.func(fuse-fragment-accumulator))']
    for reverse in (False, True):
        result = run(fuse, fixture(reverse=reverse))
        assert result.count('frisk.fragment_accumulate') == 4, result
        assert not re.search(r'arith.addf[^\n]*: vector<2x8xf32>', result), result
        print(f'PASS fusion operand order reverse={reverse}')
    cases = {
        'partial_coverage': {'offsets': ((0, 0), (0, 4))},
        'overlap': {'offsets': ((0, 0), (0, 0), (1, 0), (1, 4))},
        'multiple_users': {'extra_use': True},
        'late_accumulator': {'late_accumulator': True},
    }
    for name, args in cases.items():
        result = run(fuse, fixture(**args))
        assert 'frisk.fragment_accumulate' not in result, result
        print(f'PASS reject {name}')
    result = run([binary, '--pass-pipeline=builtin.module(func.func(ir-deep-optimize))'], STATIC)
    static, dynamic = result.split('func.func @dynamic_indices', 1)
    assert 'affine.for' not in static, static
    assert not re.search(r'vector\.(?:extract|insert)[^\n]*\[[^\]]*%', static), static
    assert 'affine.for' in dynamic and 'vector.extract' in dynamic, dynamic
    print('PASS derived indices staticized; dynamic outer loop retained')

    # Inline compute first so the K reduction and the accumulator are present in
    # the same function, as in the actual attention loop.
    with tempfile.TemporaryDirectory(prefix='fragment-numerics-') as temp:
        for reverse in (False, True):
            source = run([llvm_bin / 'mlir-opt', '--inline'], fixture(reverse=reverse) + MAIN)
            optimized = run(fuse, source)
            assert optimized.count('frisk.fragment_accumulate') >= 4
            for name, ir in [('baseline', source), ('fused', optimized)]:
                lowered = run([llvm_bin / 'mlir-opt', '--lower-affine', '--convert-scf-to-cf',
                               '--convert-vector-to-llvm', '--convert-arith-to-llvm',
                               '--convert-cf-to-llvm', '--convert-func-to-llvm',
                               '--convert-ub-to-llvm',
                               '--reconcile-unrealized-casts'], ir)
                llvm_ir = run([llvm_bin / 'mlir-translate', '--mlir-to-llvmir'], lowered)
                path = Path(temp) / f'{name}.ll'
                path.write_text(llvm_ir)
                run([llvm_bin / 'lli', path])
            print(f'PASS CPU bitwise result and zero-seeded K reduction reverse={reverse}')


if __name__ == '__main__':
    main()
