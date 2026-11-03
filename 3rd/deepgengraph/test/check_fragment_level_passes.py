#!/usr/bin/env python3
"""Typed fragment scheduling/retention/accumulator regressions, including CPU values."""
import os
import re
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'build/test/ThreadTilingTest'
LLVM = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))


def run(args):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, f"{args}\n{p.stderr[-6000:]}"
    return p.stdout


FUSION = r"""
!f = !frisk.fragment<vector<2xf32>, {}>
func.func @compute(%x: vector<8xf32>, %old: vector<8xf32>) -> vector<8xf32> {
  %z = arith.constant dense<0.0> : vector<8xf32>
  %p = affine.for %i = 0 to 4 iter_args(%a = %z) -> vector<8xf32> {
    %v = frisk.extract_frag %x[%i] : vector<8xf32> -> !f
    %r = frisk.insert_frag %v, %a[%i] {frisk.mma_fragment} : !f into vector<8xf32>
    affine.yield %r : vector<8xf32>
  } {frisk.fragment_loop}
  %out = affine.for %j = 0 to 4 iter_args(%b = %z) -> vector<8xf32> {
    %index = affine.apply affine_map<(d0)->(d0)>(%j)
    %d = frisk.extract_frag %p[%index] : vector<8xf32> -> !f
    %o = frisk.extract_frag %old[%j] : vector<8xf32> -> !f
    %sum = frisk.elementwise_frag "add" %o, %d : (!f, !f) -> !f
    %r = frisk.insert_frag %sum, %b[%j] : !f into vector<8xf32>
    affine.yield %r : vector<8xf32>
  } {frisk.fragment_loop}
  return %out : vector<8xf32>
}
"""

PREFETCH = r"""
!f = !frisk.fragment<vector<2xf32>, {}>
func.func @compute(%src: memref<8xf32>) -> vector<8xf32> attributes {thread_num = 64 : i32} {
  %z = arith.constant dense<0.0> : vector<8xf32>
  %p = affine.for %i = 0 to 4 iter_args(%a = %z) -> vector<8xf32> {
    %v = frisk.load_frag %src[%i] {tile_type = vector<8xf32>} : memref<8xf32> -> !f
    %s = frisk.elementwise_frag "mul" %v, %v : (!f, !f) -> !f
    %r = frisk.insert_frag %s, %a[%i] : !f into vector<8xf32>
    affine.yield %r : vector<8xf32>
  } {frisk.fragment_loop}
  return %p : vector<8xf32>
}
"""

RETAIN = r"""
!f = !frisk.fragment<vector<2xf32>, {}>
func.func @retain(%src: memref<8xf32, 1>) -> !f attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %v0 = arith.constant dense<0.0> : vector<2xf32>
  %f0 = frisk.pack_frag %v0 : vector<2xf32> -> !f
  %out = affine.for %i = 0 to 4 iter_args(%acc = %f0) -> !f {
    %buf = memref.alloc() : memref<8xf32, 3>
    frisk.async_copy_gm_to_shm %src to %buf {alignment = 16 : i64} : memref<8xf32, 1>, memref<8xf32, 3>
    frisk.wait_buffer_ready %buf : memref<8xf32, 3>
    %v = frisk.load_frag %buf[%c0] {tile_type = vector<8xf32>} : memref<8xf32, 3> -> !f
    %s = frisk.elementwise_frag "mul" %v, %v : (!f, !f) -> !f
    %next = frisk.elementwise_frag "add" %acc, %s : (!f, !f) -> !f
    affine.yield %next : !f
  }
  return %out : !f
}
"""


MEMORY_DEPENDENCY = r"""
!f = !frisk.fragment<vector<2xf32>, {}>
func.func @alias(%src: memref<8xf32>, %dst: memref<8xf32>) {
  affine.for %i = 0 to 4 {
    %v = frisk.load_frag %src[%i] {tile_type = vector<8xf32>} : memref<8xf32> -> !f
    frisk.store_frag %v, %dst[%i] {tile_type = vector<8xf32>} : !f, memref<8xf32>
  }
  affine.for %j = 0 to 4 {
    %v = frisk.load_frag %dst[%j] {tile_type = vector<8xf32>} : memref<8xf32> -> !f
    frisk.store_frag %v, %src[%j] {tile_type = vector<8xf32>} : !f, memref<8xf32>
  }
  return
}
"""

# A four-element extract at grid point i covers twice as many elements as a
# two-element insert at that same point. It must not be treated as same-slice.
OVERLAP = r"""
!small = !frisk.fragment<vector<2xf32>, {}>
!large = !frisk.fragment<vector<4xf32>, {}>
func.func @overlap(%x: vector<16xf32>) -> vector<16xf32> {
  %z = arith.constant dense<0.0> : vector<16xf32>
  %a = affine.for %i = 0 to 4 iter_args(%v = %z) -> vector<16xf32> {
    %p = frisk.extract_frag %x[%i] : vector<16xf32> -> !small
    %r = frisk.insert_frag %p, %v[%i] : !small into vector<16xf32>
    affine.yield %r : vector<16xf32>
  }
  %b = affine.for %j = 0 to 4 iter_args(%v = %z) -> vector<16xf32> {
    %p = frisk.extract_frag %a[%j] : vector<16xf32> -> !large
    %r = frisk.insert_frag %p, %v[%j] : !large into vector<16xf32>
    affine.yield %r : vector<16xf32>
  }
  return %b : vector<16xf32>
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix='fragment-level-') as tmp:
        d = Path(tmp)
        def opt(text, name, *passes):
            src, out = d/(name+'.mlir'), d/(name+'.out.mlir')
            src.write_text(text)
            run([TOOL, src, *passes, '-o', out])
            return out.read_text()
        def idempotent(text, name, flag):
            once = opt(text, name, flag)
            twice = opt(once, name+'-twice', flag)
            assert once == twice, name+' is not idempotent'
            return once
        old_argv = sys.argv
        try:
            sys.argv = [sys.argv[0], str(TOOL)]
            fixtures = runpy.run_path(str(ROOT/'test/check_plan2_thread_tiles.py'))
        finally:
            sys.argv = old_argv
        for m,n,k,threads in [(16,16,16,64), (32,64,32,128), (64,64,64,256), (64,128,32,256)]:
            source = fixtures['gemm'](m,n,k,threads)
            tile, destination = f'memref<{m}x{n}xf32>', f'memref<{m}x{n}xf32, 1>'
            source = source.replace(f') -> {tile} attributes', f', %out: {destination}) attributes')
            source = source.replace(f'return %g : {tile}', f'frisk.copy %g to %out [affine_map<() -> (2)>] : {tile}, {destination}\n      return')
            typed = opt(source, f'gemm-{m}-{n}-{k}', '--convert-friskbase-to-fragment=matrix-loads=false')
            optimized = opt(typed, f'gemm-{m}-{n}-{k}-optimized', '--fuse-fragment-accumulator',
                '--frisk-frag-ir-reorder', '--frisk-retain-invariant-fragments=max-words=32')
            assert 'frisk.gemm_frag' in optimized and 'frisk.warp_mma_rr' not in optimized
            lowered = opt(optimized, f'gemm-{m}-{n}-{k}-lowered', '--convert-fragment-to-thread',
                '--finalize-thread-tiling', '--lower-frisk-fragments')
            assert '!frisk.fragment<' not in lowered
        fused = idempotent(FUSION, 'accumulator', '--fuse-fragment-accumulator')
        assert fused.count('affine.for') == 1 and 'frisk.fragment_accumulate' in fused
        assert 'frisk.elementwise_frag "add"' in fused
        reduction = FUSION.replace('%v = frisk.extract_frag %x[%i] : vector<8xf32> -> !f', r"""
    %zero = arith.constant dense<0.0> : vector<2xf32>
    %plus = arith.constant dense<1.0e20> : vector<2xf32>
    %minus = arith.constant dense<-1.0e20> : vector<2xf32>
    %c0 = arith.constant 0 : index
    %f0 = frisk.pack_frag %zero : vector<2xf32> -> !f
    %v = affine.for %k = 0 to 2 iter_args(%acc = %f0) -> !f {
      %first = arith.cmpi eq, %k, %c0 : index
      %term = arith.select %first, %plus, %minus : vector<2xf32>
      %f = frisk.pack_frag %term : vector<2xf32> -> !f
      %sum = frisk.elementwise_frag "add" %acc, %f : (!f, !f) -> !f
      affine.yield %sum : !f
    }
""")
        reduction_opt = idempotent(reduction, 'zero-seeded-k', '--fuse-fragment-accumulator')
        assert 'frisk.accumulator_fused' in reduction_opt and reduction_opt.count('affine.for') == 2
        negative = FUSION.replace('(d0)->(d0)', '(d0)->(3-d0)')
        assert 'frisk.accumulator_fused' not in opt(negative, 'reverse-accumulator', '--fuse-fragment-accumulator')
        negative = FUSION.replace('0 to 4', '0 to 3')
        assert 'frisk.accumulator_fused' not in opt(negative, 'incomplete-accumulator', '--fuse-fragment-accumulator')
        # Whole-vector extra users must preserve the original producer.
        negative = FUSION.replace('return %out', '%extra = arith.addf %out, %p : vector<8xf32>\n  return %extra')
        assert 'frisk.accumulator_fused' not in opt(negative, 'extra-user', '--fuse-fragment-accumulator')
        reordered = idempotent(FUSION, 'fusion', '--frisk-frag-ir-reorder')
        assert reordered.count('affine.for') == 1 and 'frisk.fused' in reordered
        assert 'frisk.fused' not in opt(FUSION.replace('(d0)->(d0)', '(d0)->(3-d0)'), 'reverse-fusion', '--frisk-frag-ir-reorder')
        carried = FUSION.replace('iter_args(%b = %z)', 'iter_args(%b = %p)').replace('frisk.extract_frag %p[%index]', 'frisk.extract_frag %b[%index]')
        carried_opt = idempotent(carried, 'carried-fusion', '--frisk-frag-ir-reorder')
        assert carried_opt.count('affine.for') == 1
        for name, source in [('alias-write', MEMORY_DEPENDENCY), ('overlap', OVERLAP)]:
            assert 'frisk.fused' not in opt(source, name, '--frisk-frag-ir-reorder')
        scheduled = idempotent(PREFETCH, 'prefetch', '--frisk-frag-ir-reorder')
        assert 'frisk.pipelined' in scheduled and '"prefetch"' in scheduled and '!frisk.fragment' in scheduled
        for name, extra in [('write', 'frisk.store_frag %v, %src[%i] {tile_type = vector<8xf32>} : !f, memref<8xf32>'), ('barrier', 'gpu.barrier')]:
            negative = PREFETCH.replace('%s = frisk.elementwise_frag', extra+'\n    %s = frisk.elementwise_frag')
            assert 'frisk.pipelined' not in opt(negative, name, '--frisk-frag-ir-reorder')
        retained = idempotent(RETAIN, 'retain', '--frisk-retain-invariant-fragments=max-words=2')
        assert 'frisk.retained_words = 2' in retained
        assert retained.index('frisk.load_frag') < retained.index('affine.for')
        assert retained.index('frisk.elementwise_frag "mul"') < retained.index('affine.for')
        for name, text, budget in [('budget', RETAIN, 1), ('variant', RETAIN.replace('%buf[%c0]', '%buf[%i]'), 2), ('empty', RETAIN.replace('0 to 4', '0 to 0'), 2), ('write-source', RETAIN.replace('    %next =', '    frisk.store_frag %v, %src[%c0] {tile_type = vector<8xf32>} : !f, memref<8xf32, 1>\n    %next ='), 2)]:
            assert 'frisk.retained_words' not in opt(text, name, f'--frisk-retain-invariant-fragments=max-words={budget}')
        # Execute original and optimized fragment graphs on the CPU. Values
        # exercise nonuniform positions and cancellation, not just zero tiles.
        harness = r"""
func.func @main() -> i32 {
  %x = arith.constant dense<[1.0, -2.0, 3.0, -4.0, 1.0e20, -1.0e20, 7.0, 8.0]> : vector<8xf32>
  %old = arith.constant dense<[9.0, 10.0, -11.0, 12.0, -1.0e20, 1.0e20, -13.0, 14.0]> : vector<8xf32>
  %a = func.call @baseline(%x, %old) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %b = func.call @optimized(%x, %old) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %eq = arith.cmpf oeq, %a, %b : vector<8xf32>
  %ok = vector.reduction <and>, %eq : vector<8xi1> into i1
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %r = arith.select %ok, %zero, %one : i32
  return %r : i32
}
"""
        memory_harness = r"""
func.func @main() -> i32 {
  %src = memref.alloca() : memref<8xf32>
  %zero = arith.constant 0 : index
  %x = arith.constant dense<[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0]> : vector<8xf32>
  vector.store %x, %src[%zero] : memref<8xf32>, vector<8xf32>
  %a = func.call @baseline(%src) : (memref<8xf32>) -> vector<8xf32>
  %b = func.call @optimized(%src) : (memref<8xf32>) -> vector<8xf32>
  %eq = arith.cmpf oeq, %a, %b : vector<8xf32>
  %ok = vector.reduction <and>, %eq : vector<8xi1> into i1
  %z = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %r = arith.select %ok, %z, %one : i32
  return %r : i32
}
"""
        def body(text, name):
            aliases = dict(re.findall(r'^(#\w+) = (.*)$', text, re.M))
            text = re.sub(r'#\w+', lambda match: aliases.get(match[0], match[0]), text)
            start = text.index('func.func')
            text = text[start:]
            if text.rstrip().endswith('}\n}'): text = text.rstrip()[:-1]
            return text.replace('@compute', '@'+name)
        for name, original, optimized in [('fuse', FUSION, fused), ('reorder', FUSION, reordered), ('carrier', carried, carried_opt), ('prefetch', PREFETCH, scheduled), ('k-order', reduction, reduction_opt)]:
            source = '!f = !frisk.fragment<vector<2xf32>, {}>\n'+body(original, 'baseline')+body(optimized, 'optimized')+(memory_harness if name == 'prefetch' else harness)
            native = opt(source, name+'-cpu', '--convert-fragment-to-thread', '--finalize-thread-tiling')
            src=d/(name+'.cpu.mlir'); src.write_text(native)
            unrolled=d/(name+'.unrolled.mlir')
            run([LLVM/'mlir-opt', src, '--pass-pipeline=builtin.module(func.func(affine-loop-unroll{unroll-factor=-1 unroll-num-reps=4}),canonicalize)', '-o', unrolled])
            lowered=d/(name+'.llvm.mlir')
            run([LLVM/'mlir-opt', unrolled, '--lower-affine', '--convert-scf-to-cf', '--convert-vector-to-llvm', '--convert-arith-to-llvm', '--finalize-memref-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm', '--convert-ub-to-llvm', '--reconcile-unrealized-casts', '-o', lowered])
            ll=d/(name+'.ll')
            run([LLVM/'mlir-translate', '--mlir-to-llvmir', lowered, '-o', ll])
            run([LLVM/'lli', ll])
    print('PASS: typed fragment fusion, prefetch, retention, rejection cases, idempotence and CPU values')


if __name__ == '__main__':
    main()
