#!/usr/bin/env python3
"""Post-unroll scheduling: dependence boundaries, alternation and CPU semantics.

Usage: check_post_unroll_reorder.py [FragmentOptTest] [LLVM bin] [latest log]
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/FragmentOptTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, source=None):
    p = subprocess.run(list(map(str, args)), input=source, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-8000:]
    return p.stdout


def optimize(source, reorder=True):
    return run([TOOL, '--ir-deep-optimize=reorder-after-unroll='+str(reorder).lower(),
                '--verify-each'], source)


UNROLL = '''func.func @unroll(%a: memref<16xf32>) -> vector<16xf32> {
  %zero = arith.constant dense<0.0> : vector<16xf32>
  %v = affine.for %i = 0 to 16 iter_args(%v = %zero) -> vector<16xf32> {
    %x = memref.load %a[%i] : memref<16xf32>
    %r = vector.insert %x, %v[%i] : f32 into vector<16xf32>
    affine.yield %r : vector<16xf32>
  }
  %r = affine.for %i = 0 to 16 iter_args(%out = %zero) -> vector<16xf32> {
    %x = vector.extract %v[%i] : f32 from vector<16xf32>
    %y = arith.mulf %x, %x : f32
    %r = vector.insert %y, %out[%i] : f32 into vector<16xf32>
    affine.yield %r : vector<16xf32>
  }
  return %r : vector<16xf32>
}
'''
ALIAS = '''func.func @alias(%a: memref<1xf32>, %b: memref<1xf32>, %x: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %old = memref.load %b[%c0] {test.id = "read_old"} : memref<1xf32>
  %v = arith.addf %x, %one : f32
  memref.store %v, %a[%c0] {test.id = "write_one"} : memref<1xf32>
  %now = memref.load %b[%c0] {test.id = "read_new"} : memref<1xf32>
  %twice = arith.mulf %now, %now : f32
  memref.store %twice, %a[%c0] {test.id = "write_two"} : memref<1xf32>
  %last = memref.load %b[%c0] {test.id = "read_last"} : memref<1xf32>
  %sum = arith.addf %old, %last : f32
  return %sum : f32
}
'''


def activity(ir):
    return ''.join('M' if 'memref.load ' in line else 'C'
                   for line in ir.splitlines()
                   if 'memref.load ' in line or 'arith.mulf ' in line or 'llvm.inline_asm ' in line)


def transitions(sequence):
    return sum(a != b for a, b in zip(sequence, sequence[1:]))


def ids(ir):
    return re.findall(r'test.id = "([^"]+)"', ir)


def main():
    before, after = optimize(UNROLL, False), optimize(UNROLL)
    assert 'affine.for' not in after
    assert all('%' not in x for x in re.findall(r'vector\.(?:insert|extract).*?\[([^]]*)\]', after))
    assert activity(before).count('M') == activity(after).count('M') == 16
    assert activity(before).count('C') == activity(after).count('C') == 16
    assert transitions(activity(after)) > transitions(activity(before)), (activity(before), activity(after))
    assert all(len(group) % 4 == 0 for group in re.findall(r'M+', activity(after))), activity(after)
    print('PASS unroll, static indices and four-load packets:', activity(before), '->', activity(after))
    ordered = ['read_old', 'write_one', 'read_new', 'write_two', 'read_last']
    assert ids(optimize(ALIAS)) == ordered
    # Different SSA handles can still alias. A subview must not weaken ordering.
    subview = ALIAS.replace('  %old =', '  %view = memref.subview %b[0] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1]>>\n  %old =')
    subview = subview.replace('memref.load %b[%c0]', 'memref.load %view[%c0]').replace('} : memref<1xf32>\n', '} : memref<1xf32, strided<[1]>>\n')
    # Only the loads use the strided view type.
    subview = re.sub(r'(memref.store[^\n]*) : memref<1xf32, strided<\[1\]>>', r'\1 : memref<1xf32>', subview)
    assert ids(optimize(subview)) == ordered
    print('PASS RAW/WAR/WAW with aliased arguments and subviews')

    for name, boundary in [
        ('barrier', 'gpu.barrier {test.id = "boundary"}'),
        ('opaque_asm', 'llvm.inline_asm has_side_effects {test.id = "boundary"} "s_nop 0", "" : () -> ()'),
        ('atomic', '%atom = memref.atomic_rmw addf %x, %a[%c0] {test.id = "boundary"} : (f32, memref<1xf32>) -> f32'),
        ('call', 'func.call @opaque() {test.id = "boundary"} : () -> ()'),
        ('conditional', 'scf.if %cond {\n memref.store %x, %a[%c0] : memref<1xf32>\n} {test.id = "boundary"}'),
    ]:
        source = '''func.func private @opaque()
func.func @boundary(%a: memref<1xf32>, %x: f32, %cond: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0] {test.id = "before"} : memref<1xf32>
  BOUNDARY
  %w = memref.load %a[%c0] {test.id = "after"} : memref<1xf32>
  %r = arith.mulf %v, %w : f32
  return %r : f32
}'''.replace('BOUNDARY', boundary)
        assert ids(optimize(source)) == ['before', 'boundary', 'after'], name
        print('PASS boundary', name)

    # The recognised MMA is opaque as a whole: retain its padding and relative
    # order, and reject a near-match containing a memory instruction or clobber.
    padding = 's_nop 7\\0A\\09' * 8
    asm = padding + 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\\0A\\09' + padding
    mma_source = '''func.func @mma(%a: memref<4xf16>, %b: vector<4xf16>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %x = vector.load %a[%c0] : memref<4xf16>, vector<4xf16>
  %r = llvm.inline_asm has_side_effects {test.id = "mma0"} "ASM", "=v,v,v,0" %x, %b, %c : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  %y = vector.load %a[%c0] : memref<4xf16>, vector<4xf16>
  %s = llvm.inline_asm has_side_effects {test.id = "mma1"} "ASM", "=v,v,v,0" %y, %b, %c : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  return %r, %s : vector<4xf32>, vector<4xf32>
}'''.replace('ASM', asm)
    scheduled = optimize(mma_source)
    assert ids(scheduled) == ['mma0', 'mma1']
    assert scheduled.count('s_nop 7') == 32
    print('PASS MMA order and intact hardware padding')
    tagged = mma_source.replace('%x = vector.load %a[%c0]', '%x = vector.load %a[%c0] {test.id = "load0"}').replace(
        '%y = vector.load %a[%c0]', '%y = vector.load %a[%c0] {test.id = "load1"}')
    assert ids(optimize(tagged)) == ['load0', 'load1', 'mma0', 'mma1']
    tuned = 's_nop 1\\0A\\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\\0A\\09s_nop 1\\0A\\09'
    tuned_result = optimize(tagged.replace(asm, tuned))
    assert ids(tuned_result) == ['load0', 'load1', 'mma0', 'mma1']
    assert tuned_result.count('s_nop 1') == 4
    print('PASS tuned MMA padding remains schedulable and unchanged')
    for name, source in [
        ('extra_instruction', tagged.replace(asm, asm+'s_waitcnt lgkmcnt(0)')),
        ('memory_clobber', tagged.replace('"=v,v,v,0"', '"=v,v,v,0,~{memory}"')),
    ]:
        assert ids(optimize(source)) == ['load0', 'mma0', 'load1', 'mma1'], name
        print('PASS conservative asm recognition', name)

    with tempfile.TemporaryDirectory(prefix='post-unroll-') as tmp:
        work = Path(tmp)
        main_ir = '''func.func @main() -> i32 {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<1xf32>
  %c0 = arith.constant 0 : index
  %data = arith.constant dense<[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]> : vector<16xf32>
  vector.store %data, %a[%c0] : memref<16xf32>, vector<16xf32>
  %r = func.call @unroll(%a) : (memref<16xf32>) -> vector<16xf32>
  %sum = vector.reduction <add>, %r : vector<16xf32> into f32
  %expected = arith.constant 1496.0 : f32
  %ok0 = arith.cmpf oeq, %sum, %expected : f32
  %two = arith.constant 2.0 : f32
  memref.store %two, %b[%c0] : memref<1xf32>
  %r2 = func.call @alias(%b, %b, %two) : (memref<1xf32>, memref<1xf32>, f32) -> f32
  %eleven = arith.constant 11.0 : f32
  %ok1 = arith.cmpf oeq, %r2, %eleven : f32
  %ok = arith.andi %ok0, %ok1 : i1
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one : i32
  return %status : i32
}'''
        for mode in [False, True]:
            src, low, ll = [work/('cpu'+ext) for ext in ['.mlir','.llvm.mlir','.ll']]
            src.write_text(optimize(UNROLL+ALIAS+main_ir, mode))
            run([LLVM/'mlir-opt', src, '--lower-affine', '--convert-scf-to-cf',
                 '--convert-vector-to-llvm', '--finalize-memref-to-llvm', '--convert-arith-to-llvm',
                 '--convert-func-to-llvm', '--convert-cf-to-llvm', '--reconcile-unrealized-casts', '-o', low])
            run([LLVM/'mlir-translate', '--mlir-to-llvmir', low, '-o', ll])
            run([LLVM/'lli', ll])
            print('PASS CPU', 'scheduled' if mode else 'baseline')

    if len(sys.argv) > 3:
        text = Path(sys.argv[3]).read_text().split('---- after createIRDeepOptimizePass -----', 1)[1]
        text = text.split('---- after affine-scalrep -----', 1)[0]
        before, after = optimize(text, False), optimize(text)
        assert after.count('llvm.inline_asm ') == before.count('llvm.inline_asm ')
        assert after.count('gpu.barrier') == before.count('gpu.barrier')
        old, new = activity(before), activity(after)
        assert old.count('M') == new.count('M') and old.count('C') == new.count('C')
        # Alternation count alone is not a performance metric: preserving a
        # load packet intentionally reduces scalar M/C switches.
        assert not re.search(r'vector\.(?:extract|insert)\s.*\[[^\]]*%', after)
        Path('/tmp/post-unroll-attention.mlir').write_text(after)
        print('PASS latest attention: M/C transitions', transitions(old), '->', transitions(new))


if __name__ == '__main__':
    main()
