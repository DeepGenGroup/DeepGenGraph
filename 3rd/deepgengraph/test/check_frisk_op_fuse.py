#!/usr/bin/env python3
"""Frisk block fusion legality, preservation, idempotence and lowering tests."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'build/test/ThreadTilingTest'
FUSE = '--frisk-op-fuse=allow-reassociation=true'
C = 'memref<32x32xf32>'
A = 'memref<32x64xf16, 3>'
B = 'memref<64x32xf16, 3>'


def run(source, *passes):
    with tempfile.TemporaryDirectory(prefix='frisk-op-fuse-') as tmp:
        output = Path(tmp) / 'out.mlir'
        p = subprocess.run([str(TOOL), '--verify-each', '-o', str(output), *passes],
                           input=source, text=True, capture_output=True, timeout=120)
        assert p.returncode == 0, p.stderr[-8000:]
        return output.read_text()


def fixture(seed=C, reverse=False, gap='', add_seed='%acc'):
    operands = f'{add_seed}, %g' if reverse else f'%g, {add_seed}'
    types = f'{seed}, {C}' if reverse else f'{C}, {seed}'
    return f'''func.func @f(%a: {A}, %b: {B}, %acc: {seed}, %alpha: f32,
                          %out: memref<32x32xf32, 1>) -> {C}
        attributes {{thread_num = 64 : i32}} {{
      %zero = frisk.zero : {C}
      %g = frisk.gemm(%a, %b, %zero) {{transA = false, transB = false, test_tag = "keep"}} : {A}, {B}, {C} -> {C}
      {gap}
      %r = frisk.add {operands} : {types} -> {C}
      return %r : {C}
    }}'''


def fused(source):
    ir = run(source, FUSE)
    assert ir.count('frisk.gemm(') == 1 and 'frisk.add ' not in ir, ir
    assert 'frisk.zero' not in ir and 'test_tag = "keep"' in ir, ir
    assert run(ir, FUSE) == ir, 'fusion is not idempotent'
    return ir


def rejected(name, source):
    before = run(source)
    after = run(source, FUSE)
    assert before == after, name + '\n' + after
    print('PASS fallback:', name)


def main():
    for reverse in (False, True):
        ir = fused(fixture(reverse=reverse))
        assert 'frisk.gemm(%arg0, %arg1, %arg2)' in ir, ir
        ir = fused(fixture(seed='vector<32x32xf32>', reverse=reverse))
        cast = re.search(r'(%\w+) = frisk.cast %arg2 : vector<32x32xf32> -> ' + C, ir)
        assert cast and f'frisk.gemm(%arg0, %arg1, {cast[1]})' in ir, ir
        late = fixture(reverse=reverse,
                       gap=f'%scaled = frisk.mul %acc, %alpha : {C}, f32 -> {C}',
                       add_seed='%scaled')
        ir = fused(late)
        assert ir.index('frisk.mul') < ir.index('frisk.gemm'), ir
    print('PASS both add orders, vector carriers, late seed and idempotence')

    transposed = fixture().replace(A, 'memref<64x32xf16, 3>').replace('transA = false', 'transA = true')
    assert 'transA = true' in fused(transposed)
    # A zero shared by another operation must remain live.
    source = fixture(gap=f'%scaled = frisk.mul %zero, %alpha : {C}, f32 -> {C}', add_seed='%scaled')
    ir = run(source, FUSE)
    assert ir.count('frisk.zero') == 1 and 'frisk.add ' not in ir, ir
    # Dense vector zero is also recognized without treating -0 as +0.
    source = fixture(seed='vector<32x32xf32>').replace(C, 'vector<32x32xf32>')
    source = source.replace('%zero = frisk.zero : vector<32x32xf32>',
                            '%zero = arith.constant dense<0.0> : vector<32x32xf32>')
    ir = run(source, FUSE)
    assert 'frisk.add ' not in ir, ir
    rejected('negative zero', source.replace('dense<0.0>', 'dense<-0.0>'))
    print('PASS transpose/attributes, shared zero and vector constant zero')

    assert run(fixture(), '--frisk-op-fuse') == run(fixture())
    assert run(fixture(), '--frisk-op-fuse=allow-reassociation=false') == run(fixture())
    rejected('strict function', fixture().replace('thread_num = 64 : i32', 'thread_num = 64 : i32, llvm.strictfp'))
    rejected('nonzero accumulator', fixture().replace('frisk.zero : ' + C, f'frisk.exp2 %acc : {C} -> {C}'))
    rejected('broadcast row', fixture(seed='memref<1x32xf32>'))
    rejected('scalar broadcast', fixture(seed='f32'))
    rejected('precision mismatch', fixture(seed='memref<32x32xf16>'))
    rejected('multiple uses', fixture().replace(f'return %r : {C}',
            f'frisk.copy %g to %out [affine_map<() -> (2)>] : {C}, memref<32x32xf32, 1>\n return %r : {C}'))
    rejected('same product twice', fixture().replace('%g, %acc', '%g, %g'))
    rejected('already tiled', fixture().replace('transA = false', 'tiled = true, transA = false'))
    rejected('authored schedule', fixture().replace('thread_num = 64 : i32', 'thread_num = 64 : i32, pipeline.stage = array<i64: 0>'))
    rejected('store to A', fixture(gap=f'''%i = arith.constant 0 : index
      %h = arith.constant 1.0 : f16
      memref.store %h, %a[%i, %i] : {A}'''))
    rejected('barrier', fixture(gap='gpu.barrier'))
    rejected('unknown call', 'func.func private @unknown()\n' + fixture(gap='func.call @unknown() : () -> ()'))
    rejected('intervening region', fixture(gap='affine.for %i = 0 to 2 { affine.yield }'))
    source = fixture().replace('%r = frisk.add', '%cond = arith.constant true\n %r = scf.if %cond -> (' + C + ') {\n %v = frisk.add')
    source = source.replace(f'return %r : {C}', f'scf.yield %v : {C}\n }} else {{ scf.yield %acc : {C} }}\n return %r : {C}')
    rejected('cross-region producer', source)

    # Execute the actual before/after block IR with nonuniform values. This
    # checks operand selection, late scaling and vector/memref representation.
    import numpy as np
    from check_attention_split_d import execute, parse_region
    rng = np.random.default_rng(42)
    a = rng.normal(size=(32, 64)).astype(np.float16)
    b = rng.normal(size=(64, 32)).astype(np.float16)
    acc = rng.normal(size=(32, 32)).astype(np.float32)
    alpha = np.float32(0.375)
    for reverse in (False, True):
        source = fixture(reverse=reverse,
            gap=f'%scaled = frisk.mul %acc, %alpha : {C}, f32 -> {C}', add_seed='%scaled')
        source = source.replace(f') -> {C}\n', ')\n').replace(f'return %r : {C}',
            f'frisk.copy %r to %out [affine_map<() -> (2)>] : {C}, memref<32x32xf32, 1>\n return')
        outputs = []
        for passes in ((), (FUSE,)):
            ir = run(source, *passes)
            body = ir.split('func.func @f', 1)[1].splitlines()[1:]
            nodes, _ = parse_region([line.strip() for line in body])
            out = np.full((32, 32), np.nan, np.float32)
            execute(nodes, dict(zip(('%arg0','%arg1','%arg2','%arg3','%arg4'),
                                    (a,b,acc,alpha,out))), {})
            outputs.append(out)
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(outputs[1], a.astype(np.float32) @ b.astype(np.float32) + alpha * acc,
                                   atol=1e-5, rtol=1e-5)
    print('PASS CPU execution of before/after block IR with nonzero scaled accumulators')

    # Complete both lowering paths, without the fragment or LLVM accumulator
    # fusion passes. Returning computed memrefs is replaced by explicit stores.
    for seed in (C, 'vector<32x32xf32>'):
        source = fixture(seed=seed).replace(f') -> {C}\n', ')\n').replace(
            f'return %r : {C}', f'frisk.copy %r to %out [affine_map<() -> (2)>] : {C}, memref<32x32xf32, 1>\n return')
        for typed in (False, True):
            for layout in ('shared', 'shuffle', 'b-permute', 'inner-inst-b-permute'):
                tiling = '--convert-friskbase-to-fragment' if typed else '--convert-friskbase-to-thread'
                passes = [FUSE, tiling + '=matrix-loads=false wmma-layout=' + layout]
                if typed:
                    passes += ['--convert-fragment-to-thread']
                ir = run(source, *passes, '--finalize-thread-tiling', '--lower-frisk-fragments', '--canonicalize', '--cse')
                assert 'frisk.warp_mma_rr' in ir and 'arith.addf' not in ir, ir
                assert 'frisk.gemm(' not in ir and 'frisk.to_threadTile' not in ir, ir
    print('PASS fused GEMM lowers directly in typed/vector paths and four layouts')

    # The fused recurrence stays native internally, but must restore the
    # downstream C contract at the loop exit, without per-iteration shuffles.
    source = f'''func.func @recurrence(%a: {A}, %b: {B}, %alpha: f32,
        %out: memref<32x32xf32, 1>) attributes {{thread_num = 64 : i32}} {{
      %init = arith.constant dense<0.0> : vector<32x32xf32>
      %result = affine.for %i = 0 to 3 iter_args(%acc = %init) -> (vector<32x32xf32>) {{
        %zero = frisk.zero : {C}
        %g = frisk.gemm(%a, %b, %zero) {{transA = false, transB = false}} : {A}, {B}, {C} -> {C}
        %scaled = frisk.mul %acc, %alpha : vector<32x32xf32>, f32 -> {C}
        %sum = frisk.add %g, %scaled : {C}, {C} -> {C}
        frisk.copy %sum to %acc [affine_map<() -> (2)>] : {C}, vector<32x32xf32>
        affine.yield %acc : vector<32x32xf32>
      }}
      frisk.copy %result to %out [affine_map<() -> (2)>] : vector<32x32xf32>, memref<32x32xf32, 1>
      return
    }}'''
    variants = {
        'scalar-scale': (source, True),
        'direct-vector-seed': (source.replace(
            f'%scaled = frisk.mul %acc, %alpha : vector<32x32xf32>, f32 -> {C}', '').replace(
            f'%sum = frisk.add %g, %scaled : {C}, {C} -> {C}',
            f'%sum = frisk.add %g, %acc : {C}, vector<32x32xf32> -> {C}'), True),
        'extra-carrier-read': (source.replace('        affine.yield %acc',
            f'        frisk.copy %acc to %out [affine_map<() -> (2)>] : vector<32x32xf32>, memref<32x32xf32, 1>\n        affine.yield %acc'), False),
        'nonuniform-init': (source.replace('%alpha: f32,', '%alpha: f32, %init: vector<32x32xf32>,').replace(
            '%init = arith.constant dense<0.0> : vector<32x32xf32>', ''), False),
    }
    for name, (loop_source, sink) in variants.items():
      for typed in (False, True):
        tiling = '--convert-friskbase-to-fragment' if typed else '--convert-friskbase-to-thread'
        passes = [FUSE, tiling + '=matrix-loads=false wmma-layout=shuffle']
        if typed:
            passes += ['--convert-fragment-to-thread']
        ir = run(loop_source, *passes, '--finalize-thread-tiling', '--lower-frisk-fragments', '--canonicalize', '--cse')
        assert 'frisk.warp_mma_rr' in ir, (name, ir)
        assert not re.search(r'memref.alloc\(\).*memref<[^>]*xf32, 3>', ir), (name, ir)
        assert 'gpu.shuffle' in ir, (name, ir)
        loop_start = re.search(r'affine.for %\w+ = 0 to 3 iter_args', ir).start()
        opening = ir.index('{', loop_start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (ir[end] == '{') - (ir[end] == '}')
            end += 1
        assert ('gpu.shuffle' not in ir[opening:end]) == sink, (name, ir)
        if sink:
            assert 'gpu.shuffle' in ir[end:], (name, ir)
    print('PASS fused loop retains public exit layout, with no in-loop shuffle or LDS scratch')


if __name__ == '__main__':
    main()
