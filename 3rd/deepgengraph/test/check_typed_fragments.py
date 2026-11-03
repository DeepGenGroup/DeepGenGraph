#!/usr/bin/env python3
"""Typed fragment IR: round trips, legacy equivalence, verifier and CPU values."""
import os
import difflib
import hashlib
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

TEST = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TEST.parent / 'build/test/ThreadTilingTest'
LLVM = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))


def run(args, **kwargs):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=180, **kwargs)
    assert p.returncode == 0, f'{args}\n{p.stderr[-8000:]}\n{p.stdout[-1000:]}'
    return p.stdout


CPU = r'''
!frag = !frisk.fragment<vector<2x2xf32>, {}>
module {
func.func @main() -> i32 attributes {thread_num = 64 : i32} {
  %src = memref.alloca() : memref<4x4xf32>
  %dst = memref.alloca() : memref<4x4xf32>
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %n = affine.apply affine_map<(d0,d1)->(d0*4+d1)>(%i,%j)
      %ni = arith.index_cast %n : index to i32
      %f = arith.sitofp %ni : i32 to f32
      affine.store %f, %src[%i,%j] : memref<4x4xf32>
    }
  }
  %z = arith.constant dense<0.0> : vector<4x4xf32>
  %result = affine.for %i = 0 to 2 iter_args(%row = %z) -> vector<4x4xf32> {
    %next = affine.for %j = 0 to 2 iter_args(%tile = %row) -> vector<4x4xf32> {
      %f = frisk.load_frag %src[%i,%j] {tile_type = vector<4x4xf32>} : memref<4x4xf32> -> !frag
      %acc = affine.for %k = 0 to 2 iter_args(%a = %f) -> !frag {
        %b = frisk.elementwise_frag "add" %a, %f : (!frag, !frag) -> !frag
        affine.yield %b : !frag
      }
      frisk.store_frag %acc, %dst[%i,%j] {tile_type = vector<4x4xf32>} : !frag, memref<4x4xf32>
      %updated = frisk.insert_frag %acc, %tile[%i,%j] : !frag into vector<4x4xf32>
      affine.yield %updated : vector<4x4xf32>
    }
    affine.yield %next : vector<4x4xf32>
  }
  %true = arith.constant true
  %three = arith.constant 3.0 : f32
  %ok = affine.for %i = 0 to 4 iter_args(%rowOK = %true) -> i1 {
    %inner = affine.for %j = 0 to 4 iter_args(%prev = %rowOK) -> i1 {
      %x = affine.load %src[%i,%j] : memref<4x4xf32>
      %expected = arith.mulf %x, %three : f32
      %y = affine.load %dst[%i,%j] : memref<4x4xf32>
      %v = vector.extract %result[%i,%j] : f32 from vector<4x4xf32>
      %eq1 = arith.cmpf oeq, %y, %expected : f32
      %eq2 = arith.cmpf oeq, %v, %expected : f32
      %both = arith.andi %eq1, %eq2 : i1
      %all = arith.andi %prev, %both : i1
      affine.yield %all : i1
    }
    affine.yield %inner : i1
  }
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one : i32
  return %status : i32
}
}
'''



def normalized_graph(text):
    """Ignore SSA names and placement of pure constants/indices/register extracts."""
    aliases = dict(re.findall(r'^(#\w+) = (.*)$', text, re.M))
    names, lines = {}, []
    value_id = 0
    def refs(line):
        line = re.sub(r'#\w+', lambda m: aliases.get(m[0], m[0]), line)
        return re.sub(r'%[\w.]+', lambda m: names.get(m[0], m[0]), line)
    for line in text.splitlines():
        if line.startswith('#'): continue
        pure = re.match(r'\s*(%[\w.]+) = (affine.apply .*|arith.constant .*|vector.extract .*)$', line)
        if pure:
            expr = refs(pure[2])
            names[pure[1]] = 'expr_' + hashlib.sha256(expr.encode()).hexdigest()[:16]
            continue
        definition = re.match(r'\s*((?:%[\w.]+(?::\d+)?)(?:, %[\w.]+)*) = ', line)
        if definition:
            for name in re.findall(r'%[\w.]+', definition[1]):
                names[name] = f'value_{value_id}'
                value_id += 1
        lines.append(refs(line))
    return '\n'.join(lines)


def main():
    sys.argv = [sys.argv[0], str(TOOL)]
    fixtures = runpy.run_path(str(TEST / 'check_plan2_thread_tiles.py'))
    with tempfile.TemporaryDirectory(prefix='typed-fragments-') as tmp:
        d = Path(tmp)
        def opt(source, name, *passes):
            src, out = d / (name + '.mlir'), d / (name + '.out.mlir')
            src.write_text(source)
            run([TOOL, src, *passes, '-o', out])
            text = out.read_text()
            assert not re.search(r'(?<!!)frisk\.fragment(?:\s|_yield\b)', text), name
            return text
        cases = [(f'gemm-{m}-{n}-{k}-{t}', fixtures['gemm'](m,n,k,t))
                 for m,n,k,t in [(16,16,16,64), (32,64,32,128), (64,64,64,256), (64,128,32,256)]] + [('combined', fixtures['COMBINED'])]
        for name, source in cases:
            typed = opt(source, name, '--convert-friskbase-to-fragment=matrix-loads=false')
            assert 'frisk.gemm_frag' in typed and '!frisk.fragment<' in typed
            assert 'frisk.load_frag' in typed and 'frisk.insert_frag' in typed
            assert 'reuse_axes' in typed and 'calc_axes' in typed
            assert 'frisk.warp_mma_rr' not in typed
            if name == 'combined':
                assert 'frisk.elementwise_frag' in typed and 'frisk.store_frag' in typed
            reparsed = opt(typed, name + '-reparse')
            twice = opt(typed, name + '-twice', '--convert-friskbase-to-fragment=matrix-loads=false')
            assert reparsed == twice, 'fragment tiling must be idempotent'
            lowered = opt(typed, name + '-lower', '--convert-fragment-to-thread', '--lower-frisk-fragments', '--canonicalize', '--cse')
            legacy = opt(source, name + '-legacy', '--convert-friskbase-to-thread=matrix-loads=false', '--lower-frisk-fragments', '--canonicalize', '--cse')
            assert '!frisk.fragment<' not in lowered
            assert normalized_graph(lowered) == normalized_graph(legacy), f'{name}: typed lowering differs from legacy thread IR\n' + '\n'.join(difflib.unified_diff(normalized_graph(legacy).splitlines(), normalized_graph(lowered).splitlines()))[-7000:]
            assert opt(lowered, name + '-lower-twice', '--convert-fragment-to-thread') == opt(lowered, name + '-normalized')
        bad = d / 'invalid.mlir'
        bad.write_text(typed.replace('role = "A"', 'role = "B"'))
        p = subprocess.run([str(TOOL), str(bad)], capture_output=True, text=True)
        assert p.returncode != 0 and 'role/instruction layout mismatch' in p.stderr
        bad.write_text(CPU.replace('tile_type = vector<4x4xf32>', 'tile_type = vector<8x4xf32>', 1))
        p = subprocess.run([str(TOOL), str(bad)], capture_output=True, text=True)
        assert p.returncode != 0 and 'requires a serialized tile_layout' in p.stderr
        bad.write_text('''func.func @bad(%v: vector<4x4xf32>) {
          %i = arith.constant 2 : index
          %f = frisk.extract_frag %v[%i,%i] : vector<4x4xf32> -> !frisk.fragment<vector<2x2xf32>, {}>
          return
        }''')
        p = subprocess.run([str(TOOL), str(bad)], capture_output=True, text=True)
        assert p.returncode != 0 and 'out of bounds' in p.stderr
        native = opt(CPU, 'cpu', '--convert-fragment-to-thread', '--finalize-thread-tiling', '--lower-frisk-fragments', '--canonicalize')
        assert '!frisk.fragment<' not in native
        src = d / 'cpu.native.mlir'; src.write_text(native)
        lowered = d / 'cpu.llvm.mlir'
        unrolled = d / 'cpu.unrolled.mlir'
        run([LLVM / 'mlir-opt', src, '--pass-pipeline=builtin.module(func.func(affine-loop-unroll{unroll-factor=-1 unroll-num-reps=4}),canonicalize)', '-o', unrolled])
        run([LLVM / 'mlir-opt', unrolled, '--lower-affine', '--convert-scf-to-cf',
             '--convert-vector-to-llvm', '--convert-math-to-llvm', '--convert-arith-to-llvm',
             '--finalize-memref-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm', '--convert-ub-to-llvm',
             '--reconcile-unrealized-casts', '-o', lowered])
        ll = d / 'cpu.ll'
        run([LLVM / 'mlir-translate', '--mlir-to-llvmir', lowered, '-o', ll])
        run([LLVM / 'lli', ll])
    print('PASS: typed fragment round trip, idempotence, legacy equivalence, verifier, CPU load/store and loop-carried arithmetic')


if __name__ == '__main__':
    main()
