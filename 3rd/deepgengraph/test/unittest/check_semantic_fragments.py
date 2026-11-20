#!/usr/bin/env python3
"""Semantic fragment boundary, serialized replay, strict reduction CPU checks."""
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

TEST = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TEST.parent / 'build/test/ThreadTilingTest'
LLVM = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))


def run(args, *, ok=True):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=240)
    assert (p.returncode == 0) == ok, f'{args}\n{p.stderr[-9000:]}'
    return p


def cpu_program():
    lines = ['module { func.func @main() -> i32 {', '%ok0 = arith.constant true']
    checks = []
    def check(vector, ty, expected):
        for i, row in enumerate(expected):
            for j, value in enumerate(row):
                n = len(checks)
                pos = f'{i}, {j}' if 'x' in ty.split('xf32')[0] else str(i)
                lines.extend([f'%v{n} = vector.extract {vector}[{pos}] : f32 from {ty}',
                              f'%e{n} = arith.constant {float(value):.9e} : f32',
                              f'%eq{n} = arith.cmpf oeq, %v{n}, %e{n} : f32'])
                checks.append(f'%eq{n}')
    for dim, result, expected in [(1, '2x1', [[3], [8]]), (0, '1x4', [[1e20, 3, -1e20, 5]])]:
        tag = f'add{dim}'
        lines += [f'%{tag}s = arith.constant dense<[[1.0e20, 1.0, -1.0e20, 3.0], [2.0, 2.0, 2.0, 2.0]]> : vector<2x4xf32>',
                  f'%{tag}f = frisk.pack_frag %{tag}s : vector<2x4xf32> -> !frisk.fragment<vector<2x4xf32>, {{}}>',
                  f'%{tag}r = frisk.reduce_frag "add" %{tag}f {{dim = {dim} : i64, lane_extent = 1 : i64, lane_stride = 1 : i64, warp_threads = 64 : i64, reduction_width = 2 : i64, output_fragment_shape = array<i64: {1 if dim == 0 else 2}, {4 if dim == 0 else 1}>}} : !frisk.fragment<vector<2x4xf32>, {{}}> -> !frisk.fragment<vector<{result}xf32>, {{}}>',
                  f'%{tag}v = frisk.unpack_frag %{tag}r : !frisk.fragment<vector<{result}xf32>, {{}}> -> vector<{result}xf32>']
        check(f'%{tag}v', f'vector<{result}xf32>', expected)
    for kind, expected in [('mul', [[24], [120]]), ('min', [[1], [2]]), ('max', [[4], [5]])]:
        lines += [f'%{kind}s = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]> : vector<2x4xf32>',
                  f'%{kind}f = frisk.pack_frag %{kind}s : vector<2x4xf32> -> !frisk.fragment<vector<2x4xf32>, {{}}>',
                  f'%{kind}r = frisk.reduce_frag "{kind}" %{kind}f {{dim = 1 : i64, lane_extent = 1 : i64, lane_stride = 1 : i64, warp_threads = 64 : i64, reduction_width = 2 : i64, output_fragment_shape = array<i64: 2>}} : !frisk.fragment<vector<2x4xf32>, {{}}> -> !frisk.fragment<vector<2xf32>, {{}}>',
                  f'%{kind}v = frisk.unpack_frag %{kind}r : !frisk.fragment<vector<2xf32>, {{}}> -> vector<2xf32>']
        check(f'%{kind}v', 'vector<2xf32>', expected)
    lines += ['%bs = arith.constant dense<[[3.0], [7.0]]> : vector<2x1xf32>',
              '%bf = frisk.pack_frag %bs : vector<2x1xf32> -> !frisk.fragment<vector<2x1xf32>, {}>',
              '%br = frisk.broadcast_frag %bf : !frisk.fragment<vector<2x1xf32>, {}> -> !frisk.fragment<vector<2x4xf32>, {}>',
              '%bv = frisk.unpack_frag %br : !frisk.fragment<vector<2x4xf32>, {}> -> vector<2x4xf32>']
    check('%bv', 'vector<2x4xf32>', [[3]*4, [7]*4])
    lines += ['%scalar = arith.constant -2.0 : f32',
              '%sr = frisk.broadcast_frag %scalar : f32 -> !frisk.fragment<vector<2x4xf32>, {}>',
              '%sv = frisk.unpack_frag %sr : !frisk.fragment<vector<2x4xf32>, {}> -> vector<2x4xf32>']
    check('%sv', 'vector<2x4xf32>', [[-2]*4, [-2]*4])
    for i, check_value in enumerate(checks):
        lines.append(f'%ok{i+1} = arith.andi %ok{i}, {check_value} : i1')
    lines += ['%zero = arith.constant 0 : i32', '%one = arith.constant 1 : i32',
              f'%status = arith.select %ok{len(checks)}, %zero, %one : i32', 'return %status : i32', '} }']
    return '\n'.join(lines)


def main():
    sys.argv = [sys.argv[0], str(TOOL)]
    fixtures = runpy.run_path(str(TEST/'check_plan2_thread_tiles.py'))
    with tempfile.TemporaryDirectory(prefix='semantic-fragments-') as tmp:
        d = Path(tmp)
        def opt(text, name, *passes, ok=True):
            src, out = d/(name+'.mlir'), d/(name+'.out.mlir')
            src.write_text(text)
            p = run([TOOL, src, *passes, '-o', out], ok=ok)
            return out.read_text() if ok else p.stderr
        source = fixtures['COMBINED']
        typed = opt(source, 'combined', '--convert-friskbase-to-fragment=matrix-loads=false')
        assert 'frisk.reduce_frag' in typed and 'frisk.broadcast_frag' in typed
        assert 'gpu.shuffle' not in typed
        assert 'reduce_axis_fragment' not in typed and 'iterLabel = "fill"' not in typed
        assert opt(typed, 'again', '--convert-friskbase-to-fragment=matrix-loads=false') == opt(typed, 'reparsed')
        lowered = opt(typed, 'thread', '--convert-fragment-to-thread')
        assert 'gpu.shuffle' in lowered and 'reduce_axis_fragment' in lowered
        assert not re.search(r'frisk\.(?:reduce|broadcast|convert_layout|mask)_frag\b', lowered)
        assert opt(lowered, 'thread-again', '--convert-fragment-to-thread') == opt(lowered, 'thread-reparsed')
        shuffle_tests = runpy.run_path(str(TEST/'check_wmma_layout_shuffle.py'))
        gemm = shuffle_tests['gemm'](16, 16, 16, 64)
        layout = opt(gemm, 'layout', '--convert-friskbase-to-fragment')
        assert 'frisk.convert_layout_frag' in layout and 'gpu.shuffle' not in layout
        threaded = opt(layout, 'layout-thread', '--convert-fragment-to-thread')
        shuffle_tests['execute_shuffle'](threaded)
        assert 'logical coordinates' in opt(layout.replace('xor_masks = array<i64: 16, 32>',
            'xor_masks = array<i64: 8, 32>'), 'bad-shuffle', '--convert-fragment-to-thread', ok=False)
        fallback = opt(gemm, 'layout-shared', '--plan-wmma-layout-conversions',
            '--convert-friskbase-to-fragment=wmma-shuffle=false')
        assert 'frisk.convert_layout_frag' in fallback and 'memref.alloc' not in fallback
        shared = opt(fallback, 'layout-shared-thread', '--convert-fragment-to-thread')
        assert 'memref.alloc' in shared and 'gpu.barrier' in shared
        mask_source = (TEST/'test_friskBase.mlir').read_text()
        mask_source = re.sub(r'%25 = scf.if %24 -> \(f32\) \{.*?\} else \{.*?\n        \}',
                            '%25 = arith.select %24, %cst_0, %cst : f32', mask_source, flags=re.S)
        for enabled, fused in [(True, False), (False, False), (True, True), (False, True)]:
            options = f'fuse-mask-add={str(fused).lower()} uniform-masks={str(enabled).lower()}'
            masks = opt(mask_source, 'masks', '--convert-friskbase-to-fragment=' + options)
            assert 'frisk.mask_frag' in masks and 'frisk.uniform_mask' not in masks
            assert 'gpu.shuffle' not in masks and 'reduce_axis_fragment' not in masks
            native_mask = opt(masks, 'masks-thread', '--convert-fragment-to-thread', '--finalize-thread-tiling')
            assert 'frisk.mask_frag' not in native_mask
            assert ('frisk.uniform_mask' in native_mask) == enabled
            assert ('frisk.fused_mask_add' in native_mask) == fused
            assert 'frisk.fused_mask_add' not in masks
        # The strict cancellation case distinguishes left-to-right reduction from
        # fragment-partial sums. Test keep/drop axes, all kinds, both broadcast forms.
        cpu = cpu_program()
        native = opt(cpu, 'cpu', '--convert-fragment-to-thread', '--lower-frisk-fragments', '--canonicalize', '--cse')
        src = d/'native.mlir'; src.write_text(native)
        unrolled, lowered, ll = d/'unrolled.mlir', d/'llvm.mlir', d/'cpu.ll'
        run([LLVM/'mlir-opt', src, '--pass-pipeline=builtin.module(func.func(affine-loop-unroll{unroll-factor=-1 unroll-num-reps=4}),canonicalize)', '-o', unrolled])
        run([LLVM/'mlir-opt', unrolled, '--lower-affine', '--convert-scf-to-cf', '--convert-vector-to-llvm',
             '--convert-math-to-llvm', '--convert-arith-to-llvm', '--finalize-memref-to-llvm',
             '--convert-func-to-llvm', '--convert-cf-to-llvm', '--convert-ub-to-llvm', '--reconcile-unrealized-casts', '-o', lowered])
        run([LLVM/'mlir-translate', lowered, '--mlir-to-llvmir', '-o', ll])
        run([LLVM/'lli', ll])
        bad_broadcast = '''func.func @bad(%v: vector<2x3xf32>) {
          %f = frisk.pack_frag %v : vector<2x3xf32> -> !frisk.fragment<vector<2x3xf32>, {}>
          %b = frisk.broadcast_frag %f : !frisk.fragment<vector<2x3xf32>, {}> -> !frisk.fragment<vector<2x4xf32>, {}>
          return
        }'''
        assert 'singleton' in opt(bad_broadcast, 'bad-broadcast', ok=False)
        for before, after, diagnostic in [
            ('reduction_width = 2', 'reduction_width = 3', 'reduction width'),
            ('lane_extent = 1', 'lane_extent = 3', 'lane topology'),
            ('"add" %add1f', '"bad" %add1f', 'unsupported reduction kind'),
            ('array<i64: 2, 1>', 'array<i64: 3, 1>', 'output fragment shape')]:
            error = opt(cpu.replace(before, after, 1), 'invalid', ok=False)
            assert diagnostic in error, error
    print('PASS: semantic boundary, fresh-process replay, idempotence, reduction order/axes/kinds, broadcast CPU values, layout exchange, mask variants and invalid recipes')


if __name__ == '__main__':
    main()
