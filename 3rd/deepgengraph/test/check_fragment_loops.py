#!/usr/bin/env python3
"""Fragment tiling + SSA affine fusion regressions, including CPU execution."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEST = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TEST.parent / 'build/test/ThreadTilingTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, **kwargs):
    proc = subprocess.run([str(x) for x in args], text=True, capture_output=True, timeout=120, **kwargs)
    assert proc.returncode == 0, f'{args}\n{proc.stderr[-8000:]}\n{proc.stdout[-2000:]}'
    return proc.stdout


def vector_case(name, captured=True, reverse=False, whole=False):
    source = '%a' if captured else '%v'
    read = f'%x = vector.extract {source}[%i] : f32 from vector<4xf32>'
    if reverse:
        read = f'%j = affine.apply affine_map<(d0) -> (3 - d0)>(%i)\n    %x = vector.extract {source}[%j] : f32 from vector<4xf32>'
    if whole:
        read = f'%x = vector.reduction <add>, {source} : vector<4xf32> into f32'
    return f'''
func.func @{name}(%init: vector<4xf32>) -> vector<4xf32> {{
  %a = affine.for %i = 0 to 4 iter_args(%v = %init) -> vector<4xf32> {{
    %n = arith.index_cast %i : index to i32
    %x = arith.sitofp %n : i32 to f32
    %r = vector.insert %x, %v[%i] : f32 into vector<4xf32>
    affine.yield %r : vector<4xf32>
  }}
  %b = affine.for %i = 0 to 4 iter_args(%v = {'%init' if captured else '%a'}) -> vector<4xf32> {{
    {read}
    %y = arith.addf %x, %x : f32
    %r = vector.insert %y, %v[%i] : f32 into vector<4xf32>
    affine.yield %r : vector<4xf32>
  }}
  return %b : vector<4xf32>
}}
'''

MEMORY = '''
func.func @memory(%init: memref<4xf32>) -> memref<4xf32> {
  %a = affine.for %i = 0 to 4 iter_args(%m = %init) -> memref<4xf32> {
    %n = arith.index_cast %i : index to i32
    %x = arith.sitofp %n : i32 to f32
    affine.store %x, %m[%i] : memref<4xf32>
    affine.yield %m : memref<4xf32>
  }
  %one = arith.constant 1.0 : f32
  %b = affine.for %i = 0 to 4 iter_args(%m = %a) -> memref<4xf32> {
    %x = affine.load %m[%i] : memref<4xf32>
    %y = arith.addf %x, %one : f32
    affine.store %y, %m[%i] : memref<4xf32>
    affine.yield %m : memref<4xf32>
  }
  return %b : memref<4xf32>
}
'''
INDEPENDENT = '''
func.func @independent(%v: vector<4xf32>, %m: memref<4xf32>, %n: index) -> (vector<4xf32>, memref<4xf32>, vector<4xf32>) {
  %a:2 = affine.for %i = 0 to %n iter_args(%x = %v, %y = %m) -> (vector<4xf32>, memref<4xf32>) {
    %s = arith.addf %x, %x : vector<4xf32>
    affine.yield %s, %y : vector<4xf32>, memref<4xf32>
  }
  %b = affine.for %i = 0 to %n iter_args(%x = %v) -> vector<4xf32> {
    %s = arith.mulf %x, %x : vector<4xf32>
    affine.yield %s : vector<4xf32>
  }
  return %a#0, %a#1, %b : vector<4xf32>, memref<4xf32>, vector<4xf32>
}
'''
MAIN = '''
func.func @main() -> i32 {
  %z = arith.constant dense<0.0> : vector<4xf32>
  %x = func.call @pointwise(%z) : (vector<4xf32>) -> vector<4xf32>
  %y = func.call @dependent_init(%z) : (vector<4xf32>) -> vector<4xf32>
  %a = vector.extract %x[3] : f32 from vector<4xf32>
  %b = vector.extract %y[3] : f32 from vector<4xf32>
  %six = arith.constant 6.0 : f32
  %ok1 = arith.cmpf oeq, %a, %six : f32
  %ok2 = arith.cmpf oeq, %b, %six : f32
  %m = memref.alloca() : memref<4xf32>
  %r = func.call @memory(%m) : (memref<4xf32>) -> memref<4xf32>
  %c2 = arith.constant 2 : index
  %c = memref.load %r[%c2] : memref<4xf32>
  %three = arith.constant 3.0 : f32
  %ok3 = arith.cmpf oeq, %c, %three : f32
  %t = arith.andi %ok1, %ok2 : i1
  %ok = arith.andi %t, %ok3 : i1
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one : i32
  return %status : i32
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix='frisk-fragments-') as tmp:
        work = Path(tmp)
        cases = {
            'pointwise': (vector_case('pointwise'), 1),
            'dependent_init': (vector_case('dependent_init', captured=False), 1),
            'reverse': (vector_case('reverse', reverse=True), 2),
            'whole': (vector_case('whole', whole=True), 2),
            'memory': (MEMORY, 1),
            'independent': (INDEPENDENT, 1),
            'memory_reverse': (MEMORY.replace('@memory', '@memory_reverse').replace(
                '%x = affine.load %m[%i]', '%x = affine.load %m[3 - %i]'), 2),
            'alias': (MEMORY.replace('@memory', '@alias').replace(
                '%init: memref<4xf32>', '%init: memref<4xf32>, %other: memref<4xf32>').replace(
                'iter_args(%m = %a)', 'iter_args(%m = %other)'), 2),
            'barrier': (MEMORY.replace('@memory', '@barrier').replace(
                '  %one =', '  gpu.barrier\n  %one ='), 2),
            'nonzero_step': (vector_case('nonzero_step').replace(
                'affine.for %i = 0 to 4', 'affine.for %i = 1 to 8 step 2').replace(
                '-> vector<4xf32> {\n    ', '-> vector<4xf32> {\n    %index = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)\n    ').replace(
                '[%i]', '[%index]'), 1),
            'empty': (vector_case('empty').replace('0 to 4', '0 to 0'), 2),
            'different_bounds': (vector_case('different_bounds').replace(
                '%b = affine.for %i = 0 to 4', '%b = affine.for %i = 0 to 3'), 2),
        }
        source = work / 'fusion.mlir'
        source.write_text('module {\n' + ''.join(ir for ir, _ in cases.values()) + MAIN + '\n}')
        fused = work / 'fused.mlir'
        run([TOOL, source, '--frisk-frag-ir-reorder', '--verify-each', '-o', fused])
        text = fused.read_text()
        functions = re.split(r'  func.func @', text)[1:]
        for function in functions:
            name = function.split('(', 1)[0]
            if name not in cases:
                continue
            expected = cases[name][1]
            assert function.count('affine.for') == expected, (name, function)
            if expected == 1:
                assert 'frisk.fused' in function, name
            print('PASS fusion', name)
        twice = work / 'twice.mlir'
        run([TOOL, fused, '--frisk-frag-ir-reorder', '-o', twice])
        assert fused.read_text() == twice.read_text(), 'fusion must reach a fixed point'
        for name, path in [('baseline', source), ('fused', fused)]:
            lowered, ll = work / f'{name}.llvm.mlir', work / f'{name}.ll'
            run([LLVM / 'mlir-opt', path, '--lower-affine', '--convert-scf-to-cf',
                 '--convert-vector-to-llvm', '--finalize-memref-to-llvm',
                 '--convert-arith-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm',
                 '--reconcile-unrealized-casts', '-o', lowered])
            # Remove the unused barrier function for CPU execution.
            lowered.write_text(re.sub(r'  llvm.func @barrier\(.*?\n  }\n', '', lowered.read_text(), flags=re.S))
            run([LLVM / 'mlir-translate', '--mlir-to-llvmir', lowered, '-o', ll])
            run([LLVM / 'lli', ll])
            print('PASS CPU execution', name)
        casts = work / 'casts.mlir'
        casts.write_text("""
func.func @casts(%v: vector<2x4xf32>, %i: i32) -> (vector<2x4xf16>, f32)
    attributes {thread_num = 64 : i32} {
  %a = frisk.cast %v : vector<2x4xf32> -> vector<2x4xf16>
  %b = frisk.cast %i : i32 -> f32
  return %a, %b : vector<2x4xf16>, f32
}
""")
        cast_ir = work / 'casts-out.mlir'
        run([TOOL, casts, '--convert-friskbase-to-thread', '-o', cast_ir])
        for token in ['frisk.fragment_loop', 'arith.truncf', 'arith.sitofp']:
            assert token in cast_ir.read_text(), token
        assert 'frisk.cast' not in cast_ir.read_text()
        print('PASS scalar/vector cast tiling')
        # Actual attention lowering: check fragments, not whole-vector operators.
        tiled = work / 'attention.mlir'
        run([TOOL, TEST / 'test_friskBaseDebug.mlir', '--convert-friskbase-to-thread', '-o', tiled])
        ir = tiled.read_text()
        for token in ['frisk.fragment_loop', 'frisk.fragment_shape', 'mask_fragment',
                      'reduce_axis_fragment', 'arith.addf', 'math.exp2']:
            assert token in ir, token
        for line in ir.splitlines():
            if re.search(r'(?:arith\.(?:addf|subf|mulf|divf)|math.exp2) ', line):
                assert not re.search(r'vector<2x(?:8|32)xf32>', line), line
        finalized = work / 'finalized.mlir'
        run([TOOL, tiled, '--finalize-thread-tiling', '--frisk-frag-ir-reorder', '-o', finalized])
        assert not re.search(r'frisk\.(?:to_threadTile|from_threadTile|buffer_view)', finalized.read_text())
        assert 'frisk.fused' in finalized.read_text(), 'attention must expose fusible fragments'
        print('PASS attention fragment tiling and finalization')


if __name__ == '__main__':
    main()
