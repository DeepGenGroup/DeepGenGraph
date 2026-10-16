#!/usr/bin/env python3
"""GEMM matrix-read selection, fragment lowering, packing and LLVM ABI.

Run with build/test/ThreadTilingTest and optionally an LLVM bin directory.
Hardware lane semantics come from HWSpecification; device execution is separate.
"""
import re
import ctypes as C
import tempfile
from pathlib import Path

from check_frisk_async_copy import lower, command, LLVM


def export(source):
    mlir = lower(source, '--thread-ir-legalize', '--test-async-to-llvm')
    assert 'unrealized_conversion_cast' not in mlir
    return command([LLVM / 'mlir-translate', '--mlir-to-llvmir'], mlir)


def fixture(k=32, n=64, element='f16', threads=128, owned=False):
    a = f'memref<32x{k}x{element}, 3>'
    b = f'memref<{k}x{n}x{element}, 3>'
    c = f'memref<32x{n}xf32>'
    output = f'memref<32x{n}xf32, 1>'
    arg = f'memref<{k}x{n}x{element}, 1>' if owned else b
    setup = f'''%b = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {b}
      frisk.async_copy_gm_to_shm %input to %b {{alignment = 16 : i64}} : {arg}, {b}
      frisk.wait_buffer_ready %b : {b}''' if owned else ''
    return f'''func.func @gemm(%a: {a}, %{'input' if owned else 'b'}: {arg}, %out: {output})
        attributes {{thread_num = {threads} : i32}} {{
      {setup}
      %g = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
      frisk.copy %g to %out [affine_map<() -> (2)>] : {c}, {output}
      return
    }}'''


def finalize(source, mode='legacy'):
    return lower(source,
        '--convert-friskbase-to-thread=dma-pack-mode=' + mode,
        '--finalize-thread-tiling', '--lower-frisk-fragments',
        '--canonicalize', '--cse', '--pack-shared-memory')


def execute_loaded_b(element, threads, packed):
    """Execute emitted addresses/register extraction, replacing only the ISA read.

    The intrinsic shim uses the documented wave64 lane exchange equations.
    Distinct bit patterns detect wrong K/N tiles, lanes, wave placement and
    fragment-half selection without relying on floating-point arithmetic.
    """
    k, n = 32, 64
    btype = f'memref<{k}x{n}x{element}, 3>'
    outtype = f'memref<{k}x{n}x{element}, 1>'
    source = fixture(k=k, n=n, element=element, threads=threads, owned=packed)
    source = re.sub(r'      frisk.copy .*\n', '', source)
    source = source.replace(f'%out: memref<32x{n}xf32, 1>', '%out: ' + outtype)
    selected = lower(source, '--frisk-promote-gemm-matrix-loads')
    read = re.search(r'(%\w+) = frisk.read_matrix_from_shm', selected)[1]
    # Parser/printer canonicalizes function arguments to arg0/arg1/arg2.
    selected = selected.replace('    return',
        f'    frisk.copy {read} to %arg2 [affine_map<() -> (2)>] : '
        f'memref<{k}x{n}x{element}>, {outtype}\n    return')
    finished = finalize(selected, 'useKPack' if packed else 'legacy')
    assert 'warp_mma' not in finished, finished
    rows_attr = re.search(r'frisk.packed_packet_rows = (\d+)', finished)
    packet_rows = int(rows_attr[1]) if rows_attr else 0
    if packed and element == 'f16':
        assert packet_rows == 2, 'matrix packet bank plan lost during allocation lowering'
    if packed:
        # Observe reads from the packed allocation with externally initialized
        # storage. DMA itself is covered by the async-copy address regressions.
        allocation = re.search(r'(%\w+) = memref.alloc\(\).* : (memref<\d+x\w+, 3>)', finished)
        assert allocation, finished
        name, storage = allocation[1], allocation[2]
        finished = re.sub(r'%arg1: memref<[^>]+>', '%arg1: ' + storage, finished, count=1)
        finished = re.sub(r'^.*(?:memref.alloc\(|async_copy_gm_to_shm|wait_buffer_ready).*\n',
                          '', finished, flags=re.M)
        finished = re.sub(re.escape(name) + r'\b', '%arg1', finished)
    ir = export(finished)
    bits = 16 if element == 'f16' else 32
    ir = ir.replace('llvm.amdgcn.workitem.id.x', 'test_tid')
    ir = ir.replace('llvm.amdgcn.s.barrier', 'test_barrier')
    ir = re.sub(r'llvm.hcu.ds.read.m32x\d+.b\d+.v4i32.p3', 'test_matrix', ir)
    ir = re.sub(r'^target (triple|datalayout).*$', '', ir, flags=re.M)
    ir = ir.replace('amdgpu_kernel ', '').replace(' immarg', '')
    ir = re.sub(r' addrspace\([13]\)', '', ir)
    assert '@llvm.hcu.' not in ir and '@llvm.amdgcn.' not in ir, ir
    packet, creg = 128 // bits, 128 // bits // 2
    physical_expr = f'r * {n} + c'
    if packed:
        physical_expr = f'(c / {packet} * {k} + r) * {packet} + c % {packet}'
        if packet_rows:
            physical_expr = (f'((r / {packet_rows} * {n // packet} + c / {packet}) * '
                             f'{packet_rows} + r % {packet_rows}) * {packet} + c % {packet}')
    shim = f'''
      #include <stdint.h>
      #include <string.h>
      static unsigned tid;
      void set_tid(unsigned value) {{ tid = value; }}
      unsigned test_tid(void) {{ return tid; }}
      void test_barrier(void) {{}}
      typedef uint32_t words __attribute__((vector_size(16)));
      static int physical(int r, int c) {{
        return {physical_expr};
      }}
      words test_matrix(void *ptr, int offset) {{
        uint{bits}_t *p = (uint{bits}_t *)((char *)ptr + offset);
        uint{bits}_t result[{128 // bits}];
        unsigned lane = tid % 64;
        int origin = physical(lane / {32 // packet}, lane % {32 // packet} * {packet});
        for (int i = 0; i < {128 // bits}; ++i)
          result[i] = p[physical(lane / 16 * {creg} + i % {creg},
                                 lane % 16 + i / {creg} * 16) - origin];
        words output;
        memcpy(&output, result, 16);
        return output;
      }}
    '''
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'read.ll').write_text(ir)
        (root / 'shim.c').write_text(shim)
        command([LLVM / 'llc', '-mtriple=x86_64-unknown-linux-gnu',
                 '-relocation-model=pic', '-filetype=obj', root / 'read.ll', '-o', root / 'read.o'])
        command(['gcc', '-shared', '-fPIC', root / 'read.o', root / 'shim.c', '-o', root / 'read.so'])
        lib = C.CDLL(str(root / 'read.so'))
        scalar = C.c_uint16 if bits == 16 else C.c_uint32
        values, output, unused = (scalar * (k * n))(), (scalar * (k * n))(), (scalar * (32 * k))()
        expected = []
        for r in range(k):
            for c in range(n):
                value = (r * n + c) * 19 % 65535 + 1
                index = (c // packet * k + r) * packet + c % packet if packed else r * n + c
                if packed and packet_rows:
                    index = ((r // packet_rows * (n // packet) + c // packet) * packet_rows + r % packet_rows) * packet + c % packet
                values[index] = value
                expected.append(value)
        lib.gemm.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        for tid in range(threads):
            lib.set_tid(tid)
            lib.gemm(unused, values, output)
        assert list(output) == expected, (element, threads, packed,
            [(i, got, want) for i, (got, want) in enumerate(zip(output, expected))
             if got != want][:12])
    print(f'PASS emitted matrix-read addresses/fragments: {element}, {threads} threads, packed={packed}')


def main():
    disabled = lower(fixture(), '--convert-friskbase-to-thread=matrix-loads=false')
    assert 'frisk.read_matrix_from_shm' not in disabled
    for element, k, variant in [('f16', 32, 'm32x16.b16'),
                                 ('f32', 16, 'm32x8.b32')]:
        for threads in (64, 128, 256):
            source = fixture(k=k, element=element, threads=threads)
            selected = lower(source, '--frisk-promote-gemm-matrix-loads')
            assert 'frisk.read_matrix_from_shm' in selected
            assert re.search(r'read_matrix_from_shm .*\[\].*-> memref', selected)
            assert lower(selected, '--frisk-promote-gemm-matrix-loads') == selected
            tiled = lower(selected, '--convert-friskbase-to-thread')
            assert not re.search(r'read_matrix_from_shm .*-> memref', tiled)
            assert 'frisk.read_matrix_from_shm' in tiled and 'frisk.warp_mma' in tiled
            assert lower(tiled, '--convert-friskbase-to-thread') == lower(tiled)
            finished = finalize(selected)
            assert 'frisk.to_threadTile' not in finished
            assert 'frisk.from_threadTile' not in finished
            ir = export(finished)
            assert f'@llvm.hcu.ds.read.{variant}.v4i32.p3' in ir
            command([LLVM / 'opt', '-passes=verify', '-disable-output'], ir)
    print('PASS f16/f32, multi-tile B, 1/2/4 waves, idempotence and LLVM export')

    for source in (fixture(n=16, threads=64), fixture(k=4, element='f32', threads=64),
                   fixture().replace('transB = false', 'transB = true')):
        selected = lower(source, '--frisk-promote-gemm-matrix-loads')
        assert 'frisk.read_matrix_from_shm' not in selected
    print('PASS unsupported shape, instruction K and transposed-B fallbacks')

    # Selection follows the complete consumer layout, not attention roles.
    # B-permute breaks the producer B's column packets; the consumer B only
    # permutes rows of those packets and can still use matrix reads. With
    # shuffle both B layouts match, including the operand playing K's role.
    from check_wmma_b_permute import gemm
    for threads in (64, 256):
        source = gemm(128, 128, 64, threads).replace('@gemm', '@unrelated_chain')
        # Close the computed tiles with global stores. Returning memref-typed
        # computed values is intentionally unsupported by finalization.
        source = source.replace(') -> (memref<128x128xf32>, memref<128x64xf32>)',
            ', %out0: memref<128x128xf32, 1>, %out1: memref<128x64xf32, 1>)')
        source = source.replace('return %c, %o : memref<128x128xf32>, memref<128x64xf32>',
            'frisk.copy %c to %out0 [affine_map<() -> (2)>] : memref<128x128xf32>, memref<128x128xf32, 1>\n'
            '      frisk.copy %o to %out1 [affine_map<() -> (2)>] : memref<128x64xf32>, memref<128x64xf32, 1>\n'
            '      return')
        source = source.replace(f'thread_num = {threads} : i32',
            f'thread_num = {threads} : i32, frisk.block_warps = array<i64: {threads // 64}, 1>')
        for mode, expected in (('b-permute', [False, True]), ('shuffle', [True, True])):
            selected = lower(source, '--frisk-promote-gemm-matrix-loads=wmma-layout=' + mode)
            reads = set(re.findall(r'(%\w+) = frisk.read_matrix_from_shm', selected))
            operands = re.findall(r'frisk.gemm\(%\w+, (%\w+)\)', selected)
            assert [operand in reads for operand in operands] == expected, selected
            assert lower(selected, '--frisk-promote-gemm-matrix-loads=wmma-layout=' + mode) == selected
            tiled = lower(selected, '--convert-friskbase-to-thread=wmma-layout=' + mode)
            assert 'frisk.read_matrix_from_shm' in tiled
            if mode == 'b-permute':
                finished = lower(tiled, '--finalize-thread-tiling', '--lower-frisk-fragments',
                                 '--canonicalize', '--cse')
                assert 'memref.alloc' not in finished, 'GEMM chain unexpectedly stages register tiles through LDS'
    print('PASS layout-based selection: row-permuted B retained, incompatible column packets rejected, compatible producer B retained')

    selected = lower(fixture(), '--frisk-promote-gemm-matrix-loads')
    # The alternate instruction has the same input shape/type but different
    # per-lane N ownership. Revalidation must reject it, never silently load it.
    invalid = selected.replace('element_bit_width = 16', 'alt = 1 : i64, element_bit_width = 16')
    if 'alt = 0 : i64' in invalid:
        invalid = invalid.replace('alt = 0 : i64, ', '')
    assert 'WMMA B layout' in lower(invalid, '--convert-friskbase-to-thread', ok=False)
    print('PASS same-shaped ALT output-layout mismatch is rejected')

    # A block read follows the rotated shared slot and remains after its wait
    # when the software pipeline is constructed after matrix-read selection.
    pipeline = (Path(__file__).parent / 'test_frisk_async_pipeline.mlir').read_text()
    selected = lower(pipeline, '--frisk-promote-gemm-matrix-loads')
    assert 'pipeline.stage = array<i64: 0, 0, 1, 1, 1, 1, 1>' in selected
    scheduled = lower(selected, '--frisk-pipeline-schedule')
    assert 'pipeline.slots = 2' in scheduled
    assert 'frisk.read_matrix_from_shm' in scheduled
    lower(scheduled, '--convert-friskbase-to-thread', '--finalize-thread-tiling')
    print('PASS matrix reads preserve software-pipeline stages and rotated slots')

    for mode in ('legacy', 'useKPack', 'noKPack'):
        selected = lower(fixture(owned=True), '--frisk-promote-gemm-matrix-loads')
        assert selected.index('wait_buffer_ready') < selected.index('read_matrix_from_shm')
        tiled = lower(selected, '--convert-friskbase-to-thread=dma-pack-mode=' + mode)
        if mode != 'legacy':
            assert 'frisk.kpack = 2 : i64' in tiled
            assert 'frisk.dma_bytes = 16 : i64' in tiled
        if mode == 'noKPack':
            assert 'frisk.copy_threads = 64 : i64' in tiled
        finished = finalize(selected, mode)
        assert 'frisk.source_map' in finished, finished
        assert re.search(r'read_matrix_from_shm .* : memref<2048xf16, 3> -> vector<4xi32>', finished)
        ir = export(finished)
        assert '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3' in ir
        command([LLVM / 'opt', '-passes=verify', '-disable-output'], ir)
    print('PASS legacy/useKPack/noKPack physical packing, readiness and matrix intrinsic')
    # Column-major GM must stage full-width DMA before repacking the row
    # packets consumed by matrix reads. Exercise the real exported addresses.
    from check_async_layout_integration import packed_case
    packed_case(32, 64, axis=1, pack=8, alignment=16,
                source_transposed=True, dma_bytes=16)
    for mode in ('useKPack', 'noKPack'):
        source = fixture(owned=True).replace('memref<32x64xf16, 1>',
            'memref<32x64xf16, strided<[1, 32]>, 1>')
        source = source.replace('%input: memref<32x64xf16, strided<[1, 32]>, 1>',
                                '%input: memref<64x32xf16, 1>')
        source = source.replace('frisk.async_copy_gm_to_shm %input',
            '%transposed = memref.transpose %input (d0, d1) -> (d1, d0) : '
            'memref<64x32xf16, 1> to memref<32x64xf16, strided<[1, 32]>, 1>\n'
            '      frisk.async_copy_gm_to_shm %transposed')
        ir = export(finalize(source, mode))
        assert '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3' in ir
        assert '@async_lds_staging_' in ir
        command([LLVM / 'opt', '-passes=verify', '-disable-output'], ir)
    print('PASS column-major GM, full-width staged DMA and matrix-read packets')
    hybrid = lower(pipeline, '--convert-friskbase-to-thread=async-direct-only=true')
    assert hybrid.count('frisk.async_copy_gm_to_shm') == 2
    assert hybrid.count('frisk.wait_buffer_ready') == 2
    assert 'frisk.read_matrix_from_shm' in hybrid
    for element in ('f16', 'f32'):
        for threads in (64, 256):
            for packed in (False, True):
                execute_loaded_b(element, threads, packed)


if __name__ == '__main__':
    main()
