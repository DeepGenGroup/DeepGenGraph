#!/usr/bin/env python3
"""Execute real before/after Frisk block attention and verify independent staging."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
from check_attention_split_d import execute, parse_region
HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'build/test/ThreadTilingTest'
SOURCE = (HERE / 'attention_full_d_block.mlir').read_text()

def run(text, qk, vo, reassociate=True, ok=True, extra=(), before=()):
    with tempfile.TemporaryDirectory(prefix='block-tiles-') as tmp:
        out = Path(tmp) / 'out.mlir'
        cmd = [str(TOOL), '--verify-each', *before, f'--frisk-tile-attention=qk-tile={qk} vo-tile={vo} allow-reassociation={str(reassociate).lower()}', *extra, '-o', str(out)]
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=60)
        if not ok:
            assert r.returncode != 0, 'invalid tiling accepted'
            return r.stderr
        assert r.returncode == 0, r.stderr[-6000:]
        return out.read_text()

def evaluate(ir, arrays, block):
    aliases = dict(re.findall(r'^(#\w+) = (affine_map<.*>)$',ir,re.M))
    for name, text in sorted(aliases.items(),key=lambda item:-len(item[0])):
        ir = ir.replace(name,text)
    body = ir.split('func.func @Attn_p2',1)[1].splitlines()[1:]
    nodes,_ = parse_region([line.strip() for line in body])
    out = np.full(arrays[0].shape,np.nan,np.float16)
    env = dict(zip(('%arg0','%arg1','%arg2','%arg3'),(*arrays,out)))
    execute(nodes,env,dict(x=block,y=0,z=0))
    return out[0,0,block*32:(block+1)*32]

def main():
    rng=np.random.default_rng(4)
    q=rng.normal(0,.15,(1,1,128,128)).astype(np.float16)
    k=rng.normal(0,.15,(1,1,128,128)).astype(np.float16)
    v=rng.normal(0,.15,(1,1,128,128)).astype(np.float16)
    for qk,vo in ((64,32),(32,64),(16,16),(128,32),(128,128)):
        ir=run(SOURCE,qk,vo)
        assert ir.count('frisk.exp2 ') == 2, 'probability or row rescale duplicated'
        assert ir.count('frisk.gemm(') == 1+128//vo, ir
        assert f'memref<32x{qk}xf16, 3>' in ir and f'memref<{qk}x32xf16, 3>' in ir, ir
        assert f'memref<32x{vo}xf16, 3>' in ir, ir
        if qk<128 and vo<128:
            assert '-> memref<32x128xf16, 3>' not in ir, 'full Q/V staging survived'
            assert '-> memref<128x32xf16, 3>' not in ir, 'full K staging survived'
        assert run(ir,qk,vo)==ir, 'not idempotent'
        for block in (0,2):
            np.testing.assert_allclose(evaluate(ir,(q,k,v),block),evaluate(SOURCE,(q,k,v),block),atol=2e-4,rtol=2e-3)
        print('PASS independent tiling, staging, replay and numerics',qk,vo,flush=True)
    assert 'reassociation' in run(SOURCE,32,32,reassociate=False,ok=False)
    vector_score = SOURCE.replace('memref<32x32xf32>','vector<32x32xf32>')
    vector_score = vector_score.replace('%15 = frisk.zero : vector<32x32xf32>',
                                        '%15 = arith.constant dense<0.0> : vector<32x32xf32>')
    vector_tiled = run(vector_score,32,32)
    np.testing.assert_allclose(evaluate(vector_tiled,(q,k,v),2),evaluate(vector_score,(q,k,v),2),atol=2e-4,rtol=2e-3)
    ir=run(SOURCE,128,32,reassociate=False)
    assert 'frisk.exp2' in ir
    for qk,vo in ((24,32),(32,256),(-1,32)):
        assert 'tiles' in run(SOURCE,qk,vo,ok=False)
    # Re-reading an original source must never bypass a later staging write.
    # The fixture intentionally mutates Q after its full-tile load.
    mutation = SOURCE.replace('%2 = frisk.cast',
        'frisk.fill %0 {value = 0.0 : f16} : memref<32x128xf16, 3>\n    %2 = frisk.cast')
    assert 'producer graph' in run(mutation,32,32,ok=False)
    scaled_mutation = SOURCE.replace('%4 = frisk.zero',
        '%i = arith.constant 0 : index\n    %z = arith.constant 0.0 : f16\n'
        '    memref.store %z, %3[%i, %i] : memref<32x128xf16, 3>\n    %4 = frisk.zero')
    assert 'producer graph' in run(scaled_mutation,32,32,ok=False)
    gm = 'memref<1x1x128x128xf16, 1>'
    transposed = 'memref<1x1x128x128xf16, strided<[16384, 16384, 1, 128]>, 1>'
    alias = SOURCE.replace('%cst = arith.constant',
        f'%kt = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : {gm} to {transposed}\n    %cst = arith.constant',1)
    line = next(line for line in alias.splitlines() if '%13 = frisk.buffer_view' in line)
    alias = alias.replace(line,line.replace('%arg1','%kt').replace(gm,transposed))
    alias = alias.replace('%16 = frisk.gemm',
        f'%i = arith.constant 0 : index\n    %z = arith.constant 0.0 : f16\n'
        f'    memref.store %z, %arg1[%i, %i, %i, %i] : {gm}\n    %16 = frisk.gemm')
    assert 'producer graph' in run(alias,32,32,ok=False)
    partial = SOURCE.replace('frisk.copy %1 to %0 [affine_map<() -> (2)>]',
                            'frisk.copy %1 to %0 [affine_map<() -> (0, 1)>]')
    assert 'producer graph' in run(partial,32,32,ok=False)
    seed_store = SOURCE.replace('%30 = frisk.gemm',
        '%i = arith.constant 0 : index\n    %one = arith.constant 1.0 : f32\n'
        '    memref.store %one, %29[%i, %i] : memref<32x128xf32>\n    %30 = frisk.gemm')
    assert 'output columns' in run(seed_store,128,32,reassociate=False,ok=False)
    hint = SOURCE.replace('thread_num = 64 : i32',
                          'thread_num = 64 : i32, frisk.qk_pipeline_hint = 2 : i64')
    assert 'frisk.pipeline_hint = 2 : i64' in run(hint,32,32)
    exposed = run(SOURCE,32,32,extra=('--frisk-simplify-loop-control',))
    assert exposed.count('frisk.gemm(') == 8, 'bounded QK loop was not exposed for retention/stream planning'
    rolled = SOURCE.replace('thread_num = 64 : i32',
                            'thread_num = 64 : i32, frisk.qk_simplify_loop = false')
    assert run(rolled,32,32,extra=('--frisk-simplify-loop-control',)).count('frisk.gemm(') == 5
    print('PASS strict grouping and invalid tiles')

    # Fuse before V/O tiling: otherwise the first three GEMMs are separated
    # from their adds by later V staging writes and must not be moved across
    # them. Explicit full-D seeds must split into the correct output columns.
    fuse = '--frisk-op-fuse=allow-reassociation=true'
    late_only = run(SOURCE, 32, 32, extra=(fuse,))
    pv_adds = lambda ir, vo: re.findall(
        rf'frisk.add [^\n]*-> memref<32x{vo}xf32>', ir)
    assert len(pv_adds(late_only, 32)) == 3, late_only
    for qk, vo in ((32, 32), (64, 16), (32, 64), (128, 128)):
        early = run(SOURCE, qk, vo, before=(fuse,), extra=(fuse,))
        assert not pv_adds(early, vo), early
        seeds = set(re.findall(r'(%\w+) = frisk.mul [^\n]*-> memref<32x' + str(vo) + r'xf32>', early))
        gemm_seeds = re.findall(r'frisk.gemm\(%\w+, %\w+, (%\w+)\)', early)
        assert len(seeds) == 128 // vo and seeds.issubset(gemm_seeds), early
        assert early.count('frisk.exp2 ') == 2, 'softmax duplicated by seed slicing'
        assert run(early, qk, vo, before=(fuse,), extra=(fuse,)) == early
        for block in (0, 2):
            np.testing.assert_allclose(evaluate(early, (q, k, v), block),
                                       evaluate(SOURCE, (q, k, v), block),
                                       atol=2e-4, rtol=2e-3)
        print('PASS pre-tiling PV fusion, sliced seeds and CPU numerics', qk, vo, flush=True)
    no_fuse = '--frisk-op-fuse=allow-reassociation=false'
    strict = run(SOURCE, 128, 32, reassociate=False, before=(no_fuse,), extra=(no_fuse,))
    assert strict == run(SOURCE, 128, 32, reassociate=False)
    assert len(pv_adds(strict, 32)) == 4
    strict_function = SOURCE.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, llvm.strictfp')
    assert len(pv_adds(run(strict_function, 128, 32, before=(fuse,), extra=(fuse,)), 32)) == 4
    print('PASS strict mode/function preserve all unfused PV additions')

if __name__ == '__main__':main()
