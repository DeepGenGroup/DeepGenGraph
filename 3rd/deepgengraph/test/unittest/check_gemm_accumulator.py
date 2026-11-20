#!/usr/bin/env python3
"""Explicit GEMM seed, verifier and lowering regressions (no fusion passes)."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parent.parent / 'build/test/ThreadTilingTest').resolve()


def run(source, *passes, error=None):
    with tempfile.TemporaryDirectory(prefix='gemm-acc-') as tmp:
        output = Path(tmp) / 'output.mlir'
        p = subprocess.run([str(TOOL), '--verify-each', '-o', str(output), *passes],
                           input=source, text=True, capture_output=True, timeout=120)
        if error:
            assert p.returncode and error in p.stderr, p.stderr
            return ''
        assert p.returncode == 0, p.stderr[-8000:]
        return output.read_text()



def fixture(loop=False):
    a, b, c = 'memref<32x64xf16, 3>', 'memref<64x32xf16, 3>', 'memref<32x32xf32>'
    body = f'''%g = frisk.gemm(%a, %b, %seed) {{transA = false, transB = false}} : {a}, {b}, {c} -> {c}'''
    if loop:
        body = f'''%zero = frisk.zero : {c}
        %initial = frisk.add %acc, %zero : {c}, {c} -> {c}
        %g = affine.for %i = 0 to 3 iter_args(%seed = %initial) -> {c} {{
          {body}
          affine.yield %g : {c}
        }}'''
    else:
        body = body.replace('%seed', '%acc')
    return f'''func.func @gemm(%a: {a}, %b: {b}, %acc: {c}, %out: memref<32x32xf32, 1>)
      attributes {{thread_num = 64 : i32}} {{
      {body}
      frisk.copy %g to %out [affine_map<() -> (2)>] : {c}, memref<32x32xf32, 1>
      return
    }}'''


def main():
    source = fixture()
    normalized = run(source)
    assert re.search(r'frisk.gemm\(%arg0, %arg1, %arg2\)', normalized)
    assert run(normalized) == normalized
    # Type/shape diagnostics must be verifier failures, never null casts/asserts.
    bad = source.replace('%acc: memref<32x32xf32>', '%acc: memref<32x32xf16>').replace(
        ', memref<32x32xf32> ->', ', memref<32x32xf16> ->')
    run(bad, error='accumulator type must match result type')
    run(source.replace('32x64xf16', '32x48xf16'), error='A columns must equal B rows')
    run(source.replace('32x64xf16', '64xf16'), error='rank-two shaped values')
    run(source.replace('32x64xf16', '64x32xf16').replace('transA = false', 'transA = true'))
    run(source.replace('64x32xf16', '32x64xf16').replace('transB = false', 'transB = true'))
    run(source.replace('memref<32x64xf16, 3>', 'f16'), error='rank-two shaped values')
    run(source.replace('(%a, %b, %acc)', '(%a, %b)'), error="expected ','")
    print('PASS three-operand round trip, accumulator type, rank, dimensions and transpose verifier')

    for loop in (False, True):
        for layout in ('shared', 'shuffle', 'b-permute', 'inner-inst-b-permute'):
            for typed in (False, True):
                flag = '--convert-friskbase-to-fragment' if typed else '--convert-friskbase-to-thread'
                ir = run(fixture(loop), flag + '=matrix-loads=false wmma-layout=' + layout)
                # K's init must be a slice of acc, not a freshly synthesized zero.
                k_seeds = re.findall(r'affine.for [^\n]*iter_args\(%\w+ = (%\w+)\)', ir)
                assert k_seeds, ir
                if typed:
                    extracts = set(re.findall(r'(%\w+) = frisk.extract_frag ', ir))
                    assert any(seed in extracts for seed in k_seeds), ir
                    assert 'frisk.gemm_frag' in ir
                else:
                    extracts = set(re.findall(r'(%\w+) = vector.insert ', ir.split('frisk.warp_mma_rr')[0]))
                    assert any(seed in extracts for seed in k_seeds), ir
                    assert 'frisk.warp_mma_rr' in ir
                if not loop:
                    assert 'frisk.elementwise_frag "add"' not in ir and 'arith.addf' not in ir
                passes = ['--convert-fragment-to-thread'] if typed else []
                final = run(ir, *passes, '--finalize-thread-tiling', '--lower-frisk-fragments', '--canonicalize', '--cse')
                assert 'frisk.warp_mma_rr' in final
                assert 'frisk.gemm(' not in final and 'frisk.to_threadTile' not in final
                if not loop:
                    assert 'arith.addf' not in final
    print('PASS nonzero and loop-carried seeds: typed/vector lowering, four layouts, no accumulator fusion passes')


if __name__ == '__main__':
    main()
