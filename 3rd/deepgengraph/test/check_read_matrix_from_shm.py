#!/usr/bin/env python3
"""Check fixed-shape HCU matrix reads through Frisk, LLVM dialect and LLVM IR.

The upstream LLVM verifier checks the exported ABI, not HCU instruction
selection or the hardware's per-lane matrix layout.
"""
import re

from check_frisk_async_copy import LLVM, command, lower, llvm_ir


VARIANTS = [
    (32, 64, 4, False, 0, 'm32x64.b4'),
    (32, 32, 8, False, 0, 'm32x32.b8'),
    (32, 16, 16, False, 0, 'm32x16.b16'),
    (32, 16, 16, False, 1, 'm32x16.b16.alt'),
    (32, 8, 32, False, 0, 'm32x8.b32'),
    (32, 32, 8, False, 2, 'm32x32.b8.alt2'),
    (64, 16, 8, False, 4, 'm64x16.b8.alt4'),
    (16, 16, 32, False, 0, 'm16x16.b32'),
    (16, 16, 32, True, 0, 'mt16x16.b32'),
    (16, 32, 16, True, 2, 'mt16x32.b16.alt2'),
]


def fixture(variant=VARIANTS[2], result='vector<4xi32>', storage='memref<4096xi8, 3>'):
    m, n, bits, transpose, alt, _ = variant
    element = result.split('x')[-1][:-1]
    return f'''func.func @read_matrix(%src: {storage}, %out: memref<64x{element}, 1>)
        attributes {{thread_num = 64 : i32}} {{
      %z = arith.constant 0 : index
      %r = frisk.read_matrix_from_shm %src[%z] {{
        m = {m} : i64, n = {n} : i64, element_bit_width = {bits} : i64,
        transpose = {str(transpose).lower()}, alt = {alt} : i64, offset = 16 : i32
      }} : {storage} -> {result}
      vector.store %r, %out[%z] : memref<64x{element}, 1>, {result}
      return
    }}'''


def verified_ir(source):
    ir = llvm_ir(source)
    command([LLVM / 'opt', '-passes=verify', '-disable-output'], ir)
    return ir


def main():
    for variant in VARIANTS:
        source = fixture(variant)
        parsed = lower(source)
        assert lower(parsed) == parsed
        ir = verified_ir(source)
        name = 'llvm.hcu.ds.read.' + variant[-1] + '.v4i32.p3'
        assert f'call <4 x i32> @{name}(ptr addrspace(3)' in ir, ir
        assert 'i32 16)' in ir
        assert 'convergent' in ir and 'willreturn' in ir and 'memory(read)' in ir
        assert len(re.findall(r'declare .*@' + re.escape(name) + r'\(', ir)) == 1
    print('PASS all 10 variants: round-trip, vector ABI, immediates and LLVM verification')

    # The register vector ABI is explicit, independent of packed storage type.
    for result, suffix, llvm_type in [('vector<8xf16>', 'v8f16', '<8 x half>'),
                                      ('vector<4xf32>', 'v4f32', '<4 x float>')]:
        ir = verified_ir(fixture(result=result))
        assert f'call {llvm_type} @llvm.hcu.ds.read.m32x16.b16.{suffix}.p3' in ir
    print('PASS explicit floating-point result overloads')

    # Indices are in storage elements: 5 + 2*128 + 3*2 = 267 i16 elements.
    source = fixture(storage='memref<32x32xi16, strided<[128, 2], offset: 5>, 3>')
    source = source.replace('%r =', '%row = arith.constant 2 : index\n'
                           '      %col = arith.constant 3 : index\n      %r =')
    source = source.replace('%src[%z]', '%src[%row, %col]')
    ir = verified_ir(source)
    # Bare-pointer ABI reconstruction applies the view offset first; the read
    # then applies the indexed strides. Check that the two GEPs form a chain.
    base = re.search(r'(%\w+) = getelementptr i16, ptr addrspace\(3\) %\w+, i\d+ 5\b', ir)
    assert base, ir
    address = re.search(r'(%\w+) = getelementptr i16, ptr addrspace\(3\) ' +
                        re.escape(base[1]) + r', i\d+ 262\b', ir)
    assert address, ir
    assert f'(ptr addrspace(3) {address[1]}, i32 16)' in ir
    assert 'i32 16)' in ir  # The instruction immediate is separate from the GEP.
    print('PASS strided source with nonzero offset and element-scaled addressing')

    source = fixture()
    defaults = source.replace(',\n        transpose = false, alt = 0 : i64, offset = 16 : i32', '')
    assert 'i32 0)' in verified_ir(defaults)
    # A write between identical reads prevents CSE; both calls share one symbol.
    second = source[source.index('      %r ='):source.index('      return')]
    second = second.replace('%r', '%r2')
    twice = source.replace('      return', '''      %v = arith.constant 1 : i8
      memref.store %v, %src[%z] : memref<4096xi8, 3>
''' + second + '      return')
    ir = verified_ir(twice)
    name = 'llvm.hcu.ds.read.m32x16.b16.v4i32.p3'
    assert ir.count('call <4 x i32> @' + name) == 2, ir
    assert len(re.findall(r'declare .*@' + re.escape(name) + r'\(', ir)) == 1
    print('PASS default attributes, declaration reuse and reads across a write')

    cases = [
        (source.replace('m = 32', 'm = 8'), 'unsupported HCU'),
        (source.replace('n = 16', 'n = 32'), 'unsupported HCU'),
        (source.replace('element_bit_width = 16', 'element_bit_width = 8'), 'unsupported HCU'),
        (source.replace('alt = 0', 'alt = 2'), 'unsupported HCU'),
        (source.replace('transpose = false', 'transpose = true'), 'unsupported HCU'),
        (source.replace('4096xi8, 3', '4096xi8, 1'), 'requires shared memory'),
        (source.replace('%src[%z]', '%src[]'), 'one source index'),
        (source.replace('4096xi8', '4096xi4'), 'byte-addressable'),
        (source.replace('vector<4xi32>', 'vector<2x2xi32>'), 'one-dimensional'),
        (source.replace('vector<4xi32>', 'vector<[4]xi32>'), 'one-dimensional'),
    ]
    for invalid, diagnostic in cases:
        assert diagnostic in lower(invalid, ok=False)
    print('PASS invalid instruction tuples, address spaces, storage and vector shapes')


if __name__ == '__main__':
    main()
