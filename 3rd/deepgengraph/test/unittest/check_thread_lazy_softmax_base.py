#!/usr/bin/env python3
"""Structured lazy rebasing: legality, pure slow path, ABI and LLVM lowering."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'build/test/ThreadTilingTest'
PASS = '--frisk-thread-lazy-softmax-base'
LLVM = Path('/data2/xsl/install/bin')


def run(source, *passes, ok=True):
    p = subprocess.run([str(TOOL), '--verify-each', *passes], input=source,
                       text=True, capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def source(scf=False, typed=False):
    mma = ('frisk.warp_mma_rr %a, %b, %z4 {inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", '
           'inst_constraints = "=v,v,v,0"} : vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>'
           if typed else 'llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", '
           '"=v,v,v,0" %a, %b, %z4 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>')
    return '''module {
func.func private @observe(f32)
func.func @kernel(%a: vector<4xf16>, %b: vector<4xf16>) -> (vector<1x4xf32>, vector<1x1xf32>) attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z4 = arith.constant dense<0.0> : vector<4xf32>
  %oz = arith.constant dense<0.0> : vector<1x4xf32>
  %lz = arith.constant dense<0.0> : vector<1x1xf32>
  %inf = arith.constant dense<0xFF800000> : vector<1x1xf32>
  %loop:3 = LOOP iter_args(%o = %oz, %l = %lz, %m = %inf) -> (vector<1x4xf32>, vector<1x1xf32>, vector<1x1xf32>) {
    %score = MMA
    %x0 = vector.extract %score[0] : f32 from vector<4xf32>
    %x1 = vector.extract %score[1] : f32 from vector<4xf32>
    %x2 = vector.extract %score[2] : f32 from vector<4xf32>
    %x3 = vector.extract %score[3] : f32 from vector<4xf32>
    %p0 = arith.maxnumf %x0, %x1 : f32
    %p1 = arith.maxnumf %x2, %x3 : f32
    %peak = arith.maxnumf %p0, %p1 : f32
    %old = vector.extract %m[0, 0] : f32 from vector<1x1xf32>
    %new = arith.maxnumf %peak, %old : f32
    %base = vector.broadcast %new : f32 to vector<4xf32>
    %diff = arith.subf %old, %new : f32
    %alpha = math.exp2 %diff : f32
    %ov = vector.extract %o[0] : vector<4xf32> from vector<1x4xf32>
    %av = vector.broadcast %alpha : f32 to vector<4xf32>
    %os = arith.mulf %ov, %av : vector<4xf32>
    %lv = vector.extract %l[0, 0] : f32 from vector<1x1xf32>
    %ls = arith.mulf %lv, %alpha : f32
    %shift = arith.subf %score, %base : vector<4xf32>
    %w = math.exp2 %shift : vector<4xf32>
    %on = arith.addf %os, %w : vector<4xf32>
    %out = vector.shape_cast %on : vector<4xf32> to vector<1x4xf32>
    %w0 = vector.extract %w[0] : f32 from vector<4xf32>
    %ln = arith.addf %ls, %w0 : f32
    %sum = vector.from_elements %ln : vector<1x1xf32>
    %max = vector.from_elements %new : vector<1x1xf32>
    YIELD %out, %sum, %max : vector<1x4xf32>, vector<1x1xf32>, vector<1x1xf32>
  }
  return %loop#0, %loop#1 : vector<1x4xf32>, vector<1x1xf32>
}
}'''.replace('MMA', mma).replace('LOOP', 'scf.for %iv = %c0 to %c4 step %c1' if scf else 'affine.for %iv = 0 to 4').replace('YIELD', 'scf.yield' if scf else 'affine.yield')


def main():
    cases = 0
    for scf in (False, True):
        for typed in (False, True):
            original = source(scf, typed)
            result = run(original, PASS)
            assert 'frisk.lazy_softmax_base = "thread-v1"' in result
            assert 'rocdl.ballot' in result and 'scf.if' in result
            assert run(result, PASS) == result, 'pass is not idempotent'
            slow = result.split('scf.if', 1)[1].split('} else {')[0]
            assert all(s not in slow for s in ('mmac', 'warp_mma', 'barrier', 'load', 'store', 'ds_bpermute'))
            assert 'math.exp2' in slow and 'arith.select' in slow
            assert result.count('math.exp2') == 2
            # This small harness returns registers for inspection. The real
            # MyTest kernel is void; export the harness with the ordinary ABI.
            lowered = run(result, '--thread-ir-legalize', '--test-async-to-llvm')
            lowered = lowered.replace('rocdl.kernel = true, ', '')
            translation = subprocess.run([str(LLVM/'mlir-translate'), '--mlir-to-llvmir'], input=lowered,
                                         text=True, capture_output=True)
            assert translation.returncode == 0, translation.stderr
            ll = translation.stdout
            subprocess.run([str(LLVM/'opt'), '-passes=default<O3>,verify', '-disable-output'],
                           input=ll, text=True, capture_output=True, check=True)
            assert 'llvm.amdgcn.ballot.i64' in ll
            cases += 1
    original = source()
    for t in ('0', '-1', '16', 'nan', 'inf'):
        run(original, PASS+'=threshold='+t, ok=False)
    run(original, PASS+'=wave-size=32', ok=False)
    for t in ('4', '15'):
        assert 'rocdl.ballot' in run(original, PASS+'=threshold='+t)
    rejects = [
        original.replace('thread_num = 64', 'thread_num = 32'),
        original.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, strictfp'),
        original.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, llvm.target_features = "+wavefrontsize32"'),
        original.replace('%oz = arith.constant dense<0.0>', '%oz = arith.constant dense<1.0>'),
        original.replace('%ls = arith.mulf %lv, %alpha', '%ls = arith.addf %lv, %alpha'),
        original.replace('%w = math.exp2', 'func.call @observe(%old) : (f32) -> ()\n    %w = math.exp2'),
        original.replace('%w = math.exp2', 'func.call @observe(%x0) : (f32) -> ()\n    %w = math.exp2'),
        original.replace('%shift = arith.subf %score, %base', '%biased = arith.addf %score, %z4 : vector<4xf32>\n    %shift = arith.subf %biased, %base'),
        original.replace('return %loop#0, %loop#1', 'return %loop#0, %loop#2'),
    ]
    for rejected in rejects:
        assert run(rejected, PASS) == run(rejected), 'unsupported input was changed'
        cases += 1
    print(f'PASS {cases} structured lazy-base cases; affine/scf, MMA forms, legality, idempotence, LLVM/O3 verify')


if __name__ == '__main__':
    main()
