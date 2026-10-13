#!/usr/bin/env python3
"""Exact LDS read coalescing, including boundaries and CPU value semantics."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/FragmentOptTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-8000:]
    return p.stdout


def fixture(gap=2, between='', space=3):
    ty = f'memref<8xf32, {space}>'
    return f'''func.func @packet(%a: {ty}, %i: index) -> f32 {{
  %j = affine.apply affine_map<(d0) -> (d0 + {gap})>(%i)
  %lo = vector.load %a[%i] : {ty}, vector<2xf32>
  {between}
  %hi = vector.load %a[%j] : {ty}, vector<2xf32>
  %x = vector.reduction <add>, %lo : vector<2xf32> into f32
  %y = vector.reduction <add>, %hi : vector<2xf32> into f32
  %z = arith.addf %x, %y : f32
  return %z : f32
}}
'''


def thread_fixture(divisor):
    source = fixture().replace('memref<8xf32, 3>', 'memref<1024xf32, 3>')
    source = source.replace(', %i: index) -> f32 {',
        ') -> f32 attributes {thread_num = 128 : i32} {\n  %i = gpu.thread_id x')
    return source.replace('(d0 + 2)', f'(d0 + (d0 floordiv {divisor}) * 256 + 2)')


def main():
    from check_shared_packing import cpu
    with tempfile.TemporaryDirectory(prefix='lds-packets-') as tmp:
        work = Path(tmp)
        cases = {
            'adjacent': (fixture(), True),
            'bounded_lane': (fixture().replace('(d0 + 2)',
                '(d0 + (((d0 mod 64) floordiv 16 + 4) floordiv 8) * 256 + 2)'), True),
            'some_lanes_only': (fixture().replace('(d0 + 2)',
                '(d0 + (((d0 mod 64) floordiv 16 + 5) floordiv 8) * 256 + 2)'), False),
            'thread_domain': (thread_fixture(128), True),
            'thread_last_lane': (thread_fixture(127), False),
            'thread_domain_unknown': (thread_fixture(128).replace(
                ' attributes {thread_num = 128 : i32}', ''), False),
            'gap': (fixture(4), False),
            'overlap': (fixture(1), False),
            'duplicate': (fixture(0), False),
            'barrier': (fixture(between='gpu.barrier'), False),
            'write': (fixture(between='vector.store %lo, %a[%i] : memref<8xf32, 3>, vector<2xf32>'), False),
            'global': (fixture(space=1), False),
            'unknown_asm': (fixture(between='llvm.inline_asm has_side_effects "", "" : () -> ()'), False),
        }
        for name, (source, merged) in cases.items():
            path, out = work/f'{name}.mlir', work/f'{name}.out.mlir'
            path.write_text(source)
            run([TOOL, path, '--ir-deep-optimize=reorder-after-unroll=false', '--verify-each', '-o', out])
            text = out.read_text()
            wide = bool(re.search(r'vector.load[^\n]*vector<4xf32>', text))
            assert wide == merged, (name, text)
            print('PASS', name)
            if name != 'adjacent':
                continue
            main = '''func.func @main() -> i32 {
  %a = memref.alloc() : memref<8xf32, 3>
  affine.for %i = 0 to 8 {
    %n = arith.index_cast %i : index to i32
    %v = arith.sitofp %n : i32 to f32
    memref.store %v, %a[%i] : memref<8xf32, 3>
  }
  %i = arith.constant 2 : index
  %v = func.call @packet(%a, %i) : (memref<8xf32, 3>, index) -> f32
  %expected = arith.constant 14.0 : f32
  %ok = arith.cmpf oeq, %v, %expected : f32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one : i32
  memref.dealloc %a : memref<8xf32, 3>
  return %status : i32
}'''
            # mlir-opt prints a module wrapper; insert the executable driver
            # into that module, preserving the transformed function verbatim.
            cpu(source + main, work, 'before')
            cpu(text[:text.rfind('}')] + main + '\n}', work, 'after')
            print('PASS CPU both halves preserve all four loaded values')


if __name__ == '__main__':
    main()
