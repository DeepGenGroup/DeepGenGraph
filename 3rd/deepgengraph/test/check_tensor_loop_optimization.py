#!/usr/bin/env python3
"""Generic loop-jam legality, recurrence sharing, and split-D integration.

Usage: check_tensor_loop_optimization.py ThreadTilingTest MyTest
GPU timing and final MMAC numerical validation still require the DTK host.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_attention_split_d import check_numerics
sys.path.insert(0, str(HERE.parents[2]))
from generate_attention_ir import TEMPLATE, generate_ir


def fixture(alias=False, dependent_bound=False, factor=4):
    t = 'tensor<2x2xf32>'
    p = f'!deepgengraph_triton.ptr<{t}>'
    bp = f'!deepgengraph_triton<block_ptr{{{t}}}>'
    destination = '%inptr' if alias else '%out'
    bound = 'affine_map<(d0) -> (d0)>(%slice)' if dependent_bound else '%n'
    return f'''func.func @unrelated_name(%input: {t}, %n: index) -> {t} {{
      %inptr = deepgengraph_triton.ptr_of %input : ({t}) -> {p}
      %out = deepgengraph_triton.empty_ptr type = {t} : <{t}>
      %z = deepgengraph.zero shape = [2, 2], type = f32 : () -> {t}
      %c0 = arith.constant 0 : index
      affine.for %slice = 0 to {factor} {{
        %ip = deepgengraph_triton.block_ptr_of base = %inptr, base_offset = %c0, shape = [2, 2], stride = [2, 1], offset = [0, 0], block_shape = [2, 2], order = [1, 0] : ({p}, index) -> {bp}
        %iv = arith.index_cast %slice : index to i32
        %f = arith.sitofp %iv : i32 to f32
        %tile = tensor.splat %f : {t}
        %r:2 = affine.for %k = 0 to {bound} iter_args(%a = %z, %b = %z) -> ({t}, {t}) {{
          %x = deepgengraph_triton.block_load %ip : ({bp}) -> {t}
          %square = deepgengraph.mul %x, %x : ({t}, {t}) -> {t}
          %a1 = deepgengraph.add %a, %square : ({t}, {t}) -> {t}
          %b1 = deepgengraph.add %b, %tile : ({t}, {t}) -> {t}
          affine.yield %a1, %b1 : {t}, {t}
        }}
        %sum = deepgengraph.add %r#0, %r#1 : ({t}, {t}) -> {t}
        %op = deepgengraph_triton.block_ptr_of base = {destination}, base_offset = %c0, shape = [2, 2], stride = [2, 1], offset = [0, 0], block_shape = [2, 2], order = [1, 0] : ({p}, index) -> {bp}
        deepgengraph_triton.block_store %op, %sum : ({bp}, {t}) -> ()
      }}
      %result = deepgengraph_triton.tensor_from %out : ({p}) -> {t}
      return %result : {t}
    }}'''


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
        %g = frisk.gemm(%a, %b) {{transA = false, transB = false}} : memref<64x32xf16, 3>, memref<32x64xf16, 3> -> memref<64x64xf32>
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
    tool, mytest = (Path(p).resolve() for p in sys.argv[1:3])
    command = [tool, '--optimize-tensor-loops', '--verify-each']
    optimized = run(command, fixture())
    assert optimized.count('affine.for') == 1, optimized
    loop = next(line for line in optimized.splitlines() if 'affine.for' in line)
    assert len(re.findall(r'%\w+ = %\w+', loop.split('iter_args(', 1)[1])) == 5, loop
    assert optimized.count('deepgengraph.mul') == 1, optimized
    assert optimized.count('deepgengraph_triton.block_store') == 4, optimized
    assert run(command, optimized) == optimized
    print('PASS generic sharing: one recurrence shared, four distinct recurrences retained')
    for name, source in [('alias', fixture(alias=True)),
                         ('dependent-bound', fixture(dependent_bound=True)),
                         ('factor-limit', fixture(factor=11))]:
        rejected = run(command, source)
        assert rejected.count('affine.for') == 2, name + rejected
        assert rejected.count('deepgengraph_triton.block_store') == 1, name + rejected
        print('PASS conservative fallback:', name)
    partial = run([tool, '--optimize-tensor-loops=max-carrier-words=12', '--verify-each'], fixture())
    assert partial.count('affine.for') == 2, partial
    assert partial.count('deepgengraph_triton.block_store') == 2, partial
    bounded = run([tool, '--optimize-tensor-loops=max-carrier-words=8', '--verify-each'], fixture())
    assert bounded.count('deepgengraph_triton.block_store') == 1, bounded
    assert run([tool, '--optimize-tensor-loops=max-carrier-words=12'], partial) == partial
    print('PASS carrier budget: partial factor two, no jam, and idempotence')
    tiling = [tool, '--convert-friskbase-to-thread', '--finalize-thread-tiling', '--verify-each']
    recurrence = run(tiling, recurrence_fixture())
    assert 'gpu.shuffle' not in reduction_body(recurrence), recurrence
    assert 'gpu.shuffle' in recurrence, recurrence
    for options in [{'nonsplat': True}, {'extra_use': True}]:
        fallback = run(tiling, recurrence_fixture(**options))
        assert 'gpu.shuffle' in reduction_body(fallback), str(options) + fallback
    print('PASS loop-result layout conversion and nonuniform/multi-use fallbacks')
    # Extra uses of a recurrence and a non-splat initializer must remain legal;
    # integration below executes the actual normalized Frisk addresses/sums.
    source = TEMPLATE.read_text().replace('4096', '256').replace(
        '1x256x32x', '2x256x3x').replace('1x32x', '2x3x')
    with tempfile.TemporaryDirectory(prefix='tensor-loop-opt-') as tmp:
        for bm, bn, bk, budget in [(32, 64, 32, 24576), (64, 32, 16, 24576),
                                    (128, 128, 32, 24576), (128, 128, 32, 28672)]:
            path = Path(tmp) / f'{bm}-{bn}-{bk}.mlir'
            path.write_text(generate_ir(source, bm, bn, bk)[0])
            output = path.with_suffix('.ll')
            proc = subprocess.run([str(mytest), str(path), '0', str(output),
                                   '1', '1', '3.3', str(budget), 'strict'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
            assert proc.returncode == 0, proc.stdout[-10000:]
            normalized = proc.stdout.split('after createNormalizeAttentionKLayoutPass ---------', 1)[1].split('----- after', 1)[0]
            assert normalized.count('frisk.exp2') == 1, normalized
            expected = (3 if budget == 28672 else 2) if (bm, bn, bk) == (128, 128, 32) else 1 + 128 // bk
            assert normalized.count('frisk.gemm(') == expected, normalized
            error = check_numerics(proc.stdout, bm)
            llvm = output.read_text()
            pools = [int(n) for n in re.findall(r'addrspace\(3\) global \[(\d+) x i8\]', llvm)]
            assert pools and max(pools) <= 16384, pools
            print(f'PASS {bm}/{bn}/{bk}, budget={budget}: shared QK/exp2, lowered CPU numerics max_abs={error:.6g}')


if __name__ == '__main__':
    main()
