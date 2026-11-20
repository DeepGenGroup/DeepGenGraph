#!/usr/bin/env python3
"""Check loop-result layout conversion and conservative recurrence fallbacks.

Usage: check_loop_layout_recurrence.py [ThreadTilingTest]
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(command, source=None):
    with tempfile.TemporaryDirectory(prefix='loop-pass-') as tmp:
        output = Path(tmp) / 'result.mlir'
        p = subprocess.run(list(map(str, [*command, '-o', output])), input=source, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        assert p.returncode == 0, p.stderr[-8000:] + p.stdout[-4000:]
        return output.read_text()

def recurrence_fixture(nonsplat=False, extra_use=False):
    initial = ('dense<[' + ','.join('[' + ','.join(str(float(i * 64 + j))
               for j in range(64)) + ']' for i in range(64)) + ']>'
               if nonsplat else 'dense<0.0>')
    extra = ('%again = frisk.add %s, %g : memref<64x64xf32>, memref<64x64xf32> -> memref<64x64xf32>'
             if extra_use else '')
    value = '%again' if extra_use else '%s'
    return f'''func.func @recurrence(%a: memref<64x32xf16, 3>, %b: memref<32x64xf16, 3>, %out: memref<64x64xf32, 1>)
      attributes {{thread_num = 128 : i32}} {{
      %z = arith.constant {initial} : vector<64x64xf32>
      %r = affine.for %k = 0 to 5 iter_args(%acc = %z) -> vector<64x64xf32> {{
        %g_init = frisk.zero : memref<64x64xf32>
        %g = frisk.gemm(%a, %b, %g_init) {{transA = false, transB = false}} : memref<64x32xf16, 3>, memref<32x64xf16, 3>, memref<64x64xf32> -> memref<64x64xf32>
        %s = frisk.add %acc, %g : vector<64x64xf32>, memref<64x64xf32> -> memref<64x64xf32>
        {extra}
        frisk.copy {value} to %acc [affine_map<() -> (2)>] : memref<64x64xf32>, vector<64x64xf32>
        affine.yield %acc : vector<64x64xf32>
      }}
      frisk.copy %r to %out [affine_map<() -> (2)>] : vector<64x64xf32>, memref<64x64xf32, 1>
      return
    }}'''

def reduction_body(ir):
    start = re.search(r'affine.for [^\n]* to 5 [^\n]*\{', ir).end()
    depth = 1
    for i in range(start, len(ir)):
        depth += (ir[i] == '{') - (ir[i] == '}')
        if depth == 0:
            return ir[start:i]
    raise AssertionError('unclosed loop')


def main():
    tool = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'build/test/ThreadTilingTest'
    tiling = [tool, '--convert-friskbase-to-thread', '--finalize-thread-tiling', '--verify-each']
    recurrence = run(tiling, recurrence_fixture())
    assert 'gpu.shuffle' not in reduction_body(recurrence), recurrence
    assert 'gpu.shuffle' in recurrence, recurrence
    for options in [{'nonsplat': True}, {'extra_use': True}]:
        fallback = run(tiling, recurrence_fixture(**options))
        assert 'gpu.shuffle' in reduction_body(fallback), str(options) + fallback
    print('PASS loop-result layout conversion and nonuniform/multi-use fallbacks')


if __name__ == '__main__':
    main()
