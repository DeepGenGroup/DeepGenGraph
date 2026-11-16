#!/usr/bin/env python3
"""Execute the planned source/destination maps with the CPU DMA address shim."""
import re
import check_frisk_async_copy as dma


def main():
    for threads in (64, 128, 256):
        for rows, cols in ((32, 64), (24, 64)):
            src = f'memref<{rows}x{cols}xi16, 1>'
            dst = f'memref<{rows}x{cols}xi16, 3>'
            source = f'''module {{
              func.func @copy_kernel(%src: {src}) attributes {{thread_num = {threads} : i32}} {{
                %dst = memref.alloc() alignment = 16 {{frisk.shared_pack_axis = 1 : i64,
                    frisk.shared_pack = 8 : i64, frisk.shared_swizzles = array<i64: 8, 6, 9, 3>}} : {dst}
                %task = frisk.async_copy_gm_to_shm %src to %dst {{alignment = 16 : i64,
                    frisk.dma_bytes = 8 : i64}} : {src}, {dst} -> !frisk.async_task
                frisk.wait_buffer_ready %task : !frisk.async_task
                return
              }}
            }}'''
            planned = dma.lower(source,
                '--convert-friskbase-to-thread=dma-export-widths=4,8,16', '--pack-shared-memory')
            assert 'frisk.dma_destination_map' in planned and 'frisk.dma_source_map' in planned
            # Expose the owned allocation to the existing CPU harness after
            # planning/materialization, without changing any generated maps.
            allocation = re.search(r'^\s*(%\w+) = memref.alloc\(\)[^\n]* : (memref<[^\n]+>)$', planned, re.M)
            assert allocation, planned
            planned = planned.replace(allocation[0], '')
            planned = re.sub(r'(func.func @copy_kernel\([^\n]+)(\) attributes)',
                lambda m: m[1] + ', ' + allocation[1] + ': ' + allocation[2] + m[2], planned, count=1)
            def physical(r, c):
                byte = ((c // 8 * rows + r) * 8 + c % 8) * 2
                byte ^= ((byte >> 8) & 1) << 6
                byte ^= ((byte >> 9) & 1) << 3
                return byte // 2

            # Keep the published copy ownership, but also run it in a larger
            # block. Inactive waves must remain masked; full-block plans need
            # no participant branch. Both paths must copy the same bytes.
            for block_threads in sorted({threads, 256}):
                kernel = planned.replace(f'thread_num = {threads} : i32',
                                         f'thread_num = {block_threads} : i32')
                ir = dma.llvm_ir(kernel)
                assert 'i32 8, i32 0, i32 0' in ir
                dma.execute_copy(ir, rows, cols, 16, threads=block_threads,
                                 destination_index=physical)
                print(f'PASS planned 8B DMA {rows}x{cols}, {threads}/{block_threads} '
                      'copy threads, composed XOR and guard bytes')


if __name__ == '__main__':
    main()
