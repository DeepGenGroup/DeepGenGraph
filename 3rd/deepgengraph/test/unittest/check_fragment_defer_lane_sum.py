#!/usr/bin/env python3
"""Fragment lane-sum graph rewrite, conservative proofs, and lowered execution."""
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from check_defer_lane_sum import Model

ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'build/test/ThreadTilingTest'
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-fragment-defer-lane-sum='


def run(command, source, ok=True):
    p = subprocess.run(list(map(str, command)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr[-12000:]
    return p.stdout if ok else p.stderr


def transform(source, options='approximate=true wave-size=64'):
    return run([TOOL, '--verify-each', PASS + options], source)


def fixture(trips=7, loop='affine', nested=False, running_max=False,
            factor='constant', initial='0.0', escape='', wave=64, stride=16,
            attrs='', typed=False):
    """Two rows; a four-lane reduction, including noncontiguous lane groups."""
    st = '!r' if typed else 'vector<2x1xf32>'
    lines = [
        '!x = !frisk.fragment<vector<2x4xf32>, {}>',
        '!r = !frisk.fragment<vector<2x1xf32>, {}>',
        '!s = !frisk.fragment<vector<1x1xf32>, {}>',
        f'func.func @statistics() -> f32 attributes {{thread_num = {wave * 2} : i32 {attrs}}} {{',
        '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
        f'%end = arith.constant {trips} : index',
        f'%z = arith.constant dense<{initial}> : vector<2x1xf32>',
        '%zf = frisk.pack_frag %z : vector<2x1xf32> -> !r',
        '%factor = arith.constant 0.5 : f32',
        '%tid = gpu.thread_id x', '%lane = arith.index_cast %tid : index to i32',
    ]
    bounds = f'0 to {trips}' if loop == 'affine' else '%c0 to %end step %c1'
    extra = ', %maximum = %zf' if running_max else ''
    types = st + (', !r' if running_max else '')
    lines += [f'%result:{2 if running_max else 1} = {loop}.for %iv = {bounds} '
              f'iter_args(%state = {"%zf" if typed else "%z"}{extra}) -> ({types}) {{',
              '%iter = arith.index_cast %iv : index to i32',
              '%seed = arith.xori %lane, %iter : i32',
              '%value = arith.uitofp %seed : i32 to f32',
              '%values = vector.broadcast %value : f32 to vector<2x4xf32>',
              '%x = frisk.pack_frag %values : vector<2x4xf32> -> !x']
    topology = (f'dim = 1 : i64, lane_extent = 4 : i64, lane_stride = {stride} : i64, '
                f'warp_threads = {wave} : i64, reduction_width = 2 : i64, '
                'output_fragment_shape = array<i64: 1, 1>')
    if running_max:
        lines += [f'%max = frisk.reduce_frag "max" %x {{{topology}}} : !x -> !r',
                  '%nextmax = frisk.elementwise_frag "max" %maximum, %max : (!r, !r) -> !r',
                  '%delta = frisk.elementwise_frag "sub" %maximum, %nextmax : (!r, !r) -> !r',
                  '%alpha = frisk.elementwise_frag "exp2" %delta : (!r) -> !r']
    elif factor == 'state':
        lines += (['%alpha = frisk.pack_frag %state : vector<2x1xf32> -> !r'] if not typed else [])
    else:
        scalar = '%value' if factor == 'lane' else '%factor'
        if factor == 'group':
            lines += [f'%mask = arith.constant {(wave - 1) ^ (3 * stride)} : i32',
                      '%group = arith.andi %lane, %mask : i32',
                      '%groupf = arith.uitofp %group : i32 to f32',
                      '%small = arith.constant 0.015625 : f32',
                      '%groupalpha = arith.mulf %groupf, %small : f32']
            scalar = '%groupalpha'
        lines += [f'%alpha = frisk.broadcast_frag {scalar} : f32 -> !r']
    alpha = '%state' if typed and factor == 'state' else '%alpha'
    lines += [f'%sum = frisk.reduce_frag "add" %x {{{topology}}} : !x -> !r']
    if nested:
        assert not typed
        lines += ['%sumv = frisk.unpack_frag %sum : !r -> vector<2x1xf32>',
                  '%av = frisk.unpack_frag %alpha : !r -> vector<2x1xf32>',
                  '%back = affine.for %row = 0 to 2 iter_args(%carrier = %z) -> vector<2x1xf32> {',
                  '%old = frisk.extract_frag %state[%row, %c0] : vector<2x1xf32> -> !s',
                  '%a = frisk.extract_frag %av[%row, %c0] : vector<2x1xf32> -> !s',
                  '%s = frisk.extract_frag %sumv[%row, %c0] : vector<2x1xf32> -> !s',
                  '%scaled = frisk.elementwise_frag "mul" %old, %a : (!s, !s) -> !s',
                  '%update = frisk.elementwise_frag "add" %scaled, %s : (!s, !s) -> !s',
                  '%inserted = frisk.insert_frag %update, %carrier[%row, %c0] : !s into vector<2x1xf32>',
                  'affine.yield %inserted : vector<2x1xf32>',
                  '} {frisk.fragment_loop}']
    else:
        if not typed:
            lines += ['%old = frisk.pack_frag %state : vector<2x1xf32> -> !r']
        lines += [f'%scaled = frisk.elementwise_frag "mul" {"%state" if typed else "%old"}, {alpha} : (!r, !r) -> !r',
                  '%update = frisk.elementwise_frag "add" %scaled, %sum : (!r, !r) -> !r']
        if not typed:
            lines += ['%back = frisk.unpack_frag %update : !r -> vector<2x1xf32>']
    if escape:
        assert not nested and not typed
        value = {'state': '%old', 'sum': '%sum', 'update': '%update'}[escape]
        lines += ['%mem = memref.alloca() : memref<2x1xf32>',
                  f'frisk.store_frag {value}, %mem[%c0, %c0] {{tile_type = vector<2x1xf32>}} : !r, memref<2x1xf32>']
    lines += [f'{loop}.yield {"%update" if typed else "%back"}'
              f'{", %nextmax" if running_max else ""} : {types}', '}']
    if typed:
        lines += ['%outv = frisk.unpack_frag %result#0 : !r -> vector<2x1xf32>']
    result = '%outv' if typed else '%result#0'
    lines += [f'%r0 = vector.extract {result}[0, 0] : f32 from vector<2x1xf32>',
              f'%r1 = vector.extract {result}[1, 0] : f32 from vector<2x1xf32>',
              '%out = arith.addf %r0, %r1 : f32', 'return %out : f32', '}']
    return '\n'.join(lines)


class Execution(Model):
    def memory(self, ptr, ty, write=None):
        assert write is not None
        self.result = write


def lower(source):
    source = source.replace('thread_num =', 'test.thread_num =')
    thread = run([TOOL, '--verify-each', '--convert-fragment-to-thread',
                  '--lower-frisk-fragments', '--ir-deep-optimize=reorder-after-unroll=false'], source)
    lowered = run([TOOL, '--test-async-to-llvm'], thread)
    ll = run([LLVM / 'mlir-translate', '--mlir-to-llvmir'], lowered)
    run([LLVM / 'opt', '-passes=default<O3>,verify', '-disable-output'], ll)
    ll = run([LLVM / 'opt', '-S', '-passes=instnamer'], ll)
    body = re.search(r'define[^\n]+@statistics\([^\n]*\n(.*?)\n}', ll, re.S)[1]
    entry = re.match(r'(\w+):', body)[1]
    body = re.sub(r'ret float ([^\n]+)', r'store float \1, ptr addrspace(1) %3\n  ret void', body)
    return ('@dummy = internal addrspace(3) global [1 x i8] zeroinitializer\n'
            'define amdgpu_kernel void @statistics(ptr addrspace(1) %0, '
            'ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) {\n'
            f'  br label %{entry}\n' + body + '\n}\n')


def main():
    for kw in [dict(initial='1.0'), dict(factor='lane'), dict(factor='state'),
               *[dict(escape=x) for x in ('state', 'sum', 'update')],
               *[dict(attrs=', ' + x) for x in ('strictfp', 'llvm.optnone',
                   'passthrough = ["strictfp"]', '"target-features" = "+wavefrontsize32"')]]:
        source = fixture(**kw)
        assert transform(source) == run([TOOL], source), kw
    for options in ('wave-size=64', 'approximate=true'):
        assert transform(fixture(), options) == run([TOOL], fixture())
    divergent = fixture(loop='scf').replace('to %end step', 'to %tid step')
    assert transform(divergent) == run([TOOL], divergent)
    partial = fixture().replace('thread_num = 128', 'thread_num = 96')
    assert transform(partial) == run([TOOL], partial)
    loaded = fixture().replace('%alpha = frisk.broadcast_frag %factor',
        '%mem = memref.alloca() : memref<1xf32>\n'
        '%unknown = memref.load %mem[%c0] : memref<1xf32>\n'
        '%alpha = frisk.broadcast_frag %unknown')
    assert transform(loaded) == run([TOOL], loaded)
    opaque = fixture().replace('%sum =', 'func.call @opaque() : () -> ()\n%sum =')
    opaque = 'func.func private @opaque()\n' + opaque
    assert transform(opaque) == run([TOOL], opaque)
    assert 'wave-size must be' in run([TOOL, PASS + 'wave-size=16'], fixture(), ok=False)

    # Unweighted sums and ordinary block barriers do not prevent the rewrite.
    unweighted = fixture().replace(
        '%scaled = frisk.elementwise_frag "mul" %old, %alpha : (!r, !r) -> !r', '').replace(
        '"add" %scaled, %sum', '"add" %old, %sum')
    for barrier in ('', 'frisk.sync_threads_in_block', 'gpu.barrier'):
        result = transform(unweighted.replace('%sum =', barrier + '\n%sum ='))
        assert 'frisk.lane_reduce_frag' in result
        assert 'frisk.defer_analysis_origin' not in result

    pairs = 0
    for kw in (dict(), dict(nested=True), dict(typed=True), dict(loop='scf'),
               dict(running_max=True), dict(nested=True, running_max=True), dict(factor='group')):
        source = fixture(**kw)
        result = transform(source)
        assert result.count('frisk.lane_reduce_frag') == 1, (kw, result)
        assert re.search(r'reduce_frag "add"[^\n]*lane_extent = 1 ', result)
        assert transform(result) == result
        # The proof copy's specialization must not change the scheduled loops.
        assert source.count('affine.for') == result.count('affine.for')
        for trips in (0, 1, 2, 7, 31):
            source = fixture(trips=trips, **kw)
            result = transform(source)
            values, counts = [], []
            for program in (source, result):
                model = Execution(lower(program), 64, False)
                model.run()
                values.append(model.result)
                counts.append(model.shuffles)
            np.testing.assert_allclose(*values, rtol=3e-6, atol=2e-4)
            extra = 4 * trips if kw.get('running_max') else 0
            assert counts == [4 * trips + extra, 4 + extra], (kw, trips, counts)
            pairs += 1

    for wave, stride in ((32, 1), (64, 16)):
        source = fixture(wave=wave, stride=stride)
        result = transform(source, f'approximate=true wave-size={wave}')
        assert 'frisk.lane_reduce_frag' in result
        lowered = run([TOOL, '--verify-each', '--convert-fragment-to-thread'], result)
        assert 'frisk.lane_reduce_frag' not in lowered
        assert lowered.count('gpu.shuffle') == 4
    result = run([TOOL, '--verify-each', '--pass-pipeline=builtin.module('
                  'frisk-fragment-scheduling{defer-lane-sum=true wave-size=64 '
                  'heuristic-scheduling=true})'], fixture(nested=True))
    assert 'frisk.lane_reduce_frag' in result
    result = transform(fixture())
    invalid = result.replace('lane_extent = 4', 'lane_extent = 3')
    assert 'invalid lane sum topology' in run([TOOL], invalid, ok=False)
    print(f'PASS: fragment recurrence proofs, rejection, idempotence, heuristic scheduling, '
          f'wave32/64, verifier/O3, {pairs} lowered numerical execution pairs')


if __name__ == '__main__':
    main()
