#!/usr/bin/env python3
"""Generic planning boundary and CPU semantics of composed physical layouts."""
from pathlib import Path
import subprocess
import tempfile
import re
import check_shared_packing as packing
from check_gemm_matrix_loads import fixture as matrix_fixture

TOOL = Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'


def run(source, args, out=None, ok=True):
    command = [str(TOOL), *args, '--verify-each']
    if out is not None:
        command += ['-o', str(out)]
    p = subprocess.run(command, input=source, text=True, capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr[-10000:]
    return p


def async_gemm(rows=32, cols=64, reduction=32):
    source = packing.gemm(rows, cols, reduction, 64)
    src = f'memref<{reduction}x{cols}xf16, 1>'
    dst = f'memref<{reduction}x{cols}xf16, 3>'
    source = source.replace(
        f'frisk.copy %global to %b [affine_map<() -> (2)>] : {src}, {dst}',
        f'%task = frisk.async_copy_gm_to_shm %global to %b {{alignment = 16 : i64}} : {src}, {dst} -> !frisk.async_task\n'
        '  frisk.wait_buffer_ready %task : !frisk.async_task')
    return source


def main():
    with tempfile.TemporaryDirectory(prefix='shared-memory-planning-') as path:
        work = Path(path)
        source = async_gemm()
        analysis_file = work / 'analysis.mlir'
        analyzed = run(source, ['--convert-friskbase-to-thread=shared-memory-planning=analyze matrix-loads=false'], analysis_file)
        assert 'Shared group 0' in analyzed.stderr and ' selected' in analyzed.stderr
        assert 'bytes=8' in analyzed.stderr and 'exportable=0' in analyzed.stderr
        assert 'frisk.shared_plan' not in analysis_file.read_text()
        assert 'frisk.shared_pack' not in analysis_file.read_text()
        renamed = source.replace('@matmul', '@unrelated_kernel').replace('%global', '%input_tensor').replace('%b', '%scratch')
        other = run(renamed, ['--convert-friskbase-to-thread=shared-memory-planning=analyze matrix-loads=false'], work/'renamed.mlir')
        assert other.stderr == analyzed.stderr, 'planning must not depend on names or roles'
        print('PASS computation boundary, read-only planning, generic names, exporter capabilities')

        for matrix in ('false', 'true'):
            result = work/f'auto-{matrix}.mlir'
            run(source, [f'--convert-friskbase-to-thread=shared-memory-planning=auto matrix-loads={matrix}',
                         '--finalize-thread-tiling', '--pack-shared-memory'], result)
            text = result.read_text()
            assert 'frisk.packed_width' in text and 'frisk.source_map' in text
            assert 'frisk.dma_bytes = 8' not in text
            run(text, ['--frisk-lower-async-copy'], work/f'lowered-{matrix}.mlir')
        print('PASS ordinary GEMM and matrix readers consume published plans through DMA lowering')

        # A matrix read's collective bank cost remains unknown, but its packet
        # placement must still participate in selection. In particular, XOR
        # "off" must not suppress rows=1 (the 128x32 regression) or rows=2.
        for threads in (64, 128):
            for rows, cols, expected in ((128, 32, 1), (32, 64, 2)):
                matrix_source = matrix_fixture(k=rows, n=cols, threads=threads, owned=True)
                matrix_source = matrix_source.replace('@gemm', '@packet_reader')
                for xor_option in ('', ' swizzle-v=off swizzle-v-arg=1'):
                    options = 'dma-pack-mode=useKPack' + xor_option
                    analysis = run(matrix_source, [
                        '--convert-friskbase-to-thread=shared-memory-planning=analyze ' + options],
                        work/'matrix-analysis.mlir')
                    selected = next(line for line in analysis.stderr.splitlines() if ' selected' in line)
                    assert f'rows={expected} ' in selected and 'banks=unknown' in selected, selected
                    assert 'packet-cost=' in analysis.stderr
                    result = work/'matrix-packets.mlir'
                    run(matrix_source, ['--convert-friskbase-to-thread=' + options,
                        '--finalize-thread-tiling', '--pack-shared-memory'], result)
                    text = result.read_text()
                    assert f'frisk.packed_packet_rows = {expected}' in text
                    assert 'frisk.dma_bytes = 16' in text and 'frisk.source_map' in text
                    if xor_option:
                        assert not re.search(r'frisk\.(?:shared_swizzle|packed_swizzle)', text)
        print('PASS matrix packet cost, row-major candidate, explicit XOR off and 16B DMA')

        # Existing scalar execution oracle verifies actual stores and reloads.
        # Test dependent XORs too: their inverse must reverse operation order.
        for bits in ('8, 6, 9, 3', '8, 6, 6, 3'):
            scalar = packing.fixture(32, 64, 4, 1).replace(
                'frisk.shared_pack = 4 : i64',
                f'frisk.shared_pack = 4 : i64, frisk.shared_plan, frisk.shared_swizzles = array<i64: {bits}>')
            packed_file = work/'packed.mlir'
            run(scalar, ['--pack-shared-memory'], packed_file)
            packing.cpu(packed_file.read_text(), work, 'composed-' + bits.replace(', ', '-'))
        print('PASS CPU numerical semantics for independent and dependent composed XOR layouts')

        scalar_kernel = """func.func @scalar_kernel() -> f32 attributes {thread_num = 64 : i32} {
          %data = memref.alloc() alignment = 16 : memref<2x64xf32, 3>
          %tid = gpu.thread_id x
          %zero = arith.constant 0.0 : f32
          affine.for %row = 0 to 2 {
            affine.store %zero, %data[%row, %tid] : memref<2x64xf32, 3>
          }
          %loaded = affine.load %data[0, %tid] : memref<2x64xf32, 3>
          return %loaded : f32
        }"""
        scalar_analysis = run(scalar_kernel, ['--convert-friskbase-to-thread=shared-memory-planning=analyze'], work/'scalar-analysis.mlir')
        assert 'complete=1' in scalar_analysis.stderr and ' selected' in scalar_analysis.stderr
        run(scalar_kernel, ['--convert-friskbase-to-thread=shared-memory-planning=auto', '--pack-shared-memory'], work/'scalar-auto.mlir')
        assert 'memref<128xf32, 3>' in (work/'scalar-auto.mlir').read_text()
        print('PASS explicit scalar read/write kernels need no GEMM layout anchor')

        forced = source.replace('alignment = 16 : i64}', 'alignment = 16 : i64, frisk.dma_bytes = 8 : i64}')
        failure = run(forced, ['--convert-friskbase-to-thread=shared-memory-planning=auto matrix-loads=false'], ok=False)
        assert 'no exportable candidate' in failure.stderr
        print('PASS unsupported explicit DMA export fails instead of silently narrowing')

        # The same generic binding works after renaming the kernel and buffer.
        # It carries both inverse placement and thread/round ownership to IR.
        reordered = source.replace('%a: memref<32x32xf16, 3>, %global: memref<32x64xf16, 1>',
                                   '%global: memref<32x64xf16, 1>, %a: memref<32x32xf16, 3>')
        for fixture, argument in ((source, 1), (renamed, 1), (reordered, 0)):
            planned = work/'planned8.mlir'
            run(fixture, [f'--convert-friskbase-to-thread=matrix-loads=false dma-export-widths=4,8,16 shared-copy-constraints={argument}:8:9:3',
                          '--finalize-thread-tiling', '--pack-shared-memory'], planned)
            text = planned.read_text()
            assert 'frisk.dma_bytes = 8' in text and 'frisk.dma_destination_map' in text
            assert 'frisk.dma_source_map' in text and 'frisk.source_map' in text
            lowered = work/'emitted8.mlir'
            run(text, ['--frisk-lower-async-copy'], lowered)
            assert 'llvm.call @__frisk_async_load_lds' in lowered.read_text()
            assert 'llvm.mlir.constant(8 : i32)' in lowered.read_text()
            invalid = text.replace('frisk.dma_total_elements = 2048', 'frisk.dma_total_elements = 1024')
            failure = run(invalid, ['--frisk-lower-async-copy'], ok=False)
            assert 'plan invalidated' in failure.stderr
        print('PASS planned 8B DMA emits operations and rejects stale plans without fallback')


if __name__ == '__main__':
    main()
