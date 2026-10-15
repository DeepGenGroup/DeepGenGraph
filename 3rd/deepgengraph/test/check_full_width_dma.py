#!/usr/bin/env python3
"""Execute real lowered 128-bit packets on CPU, including inactive waves.

Usage: check_full_width_dma.py [ThreadTilingTest] [LLVM bin directory]
The shim checks bytes and guards; hardware async timing is not simulated.
"""
import re
from check_frisk_async_copy import fixture, lower, llvm_ir, execute_copy


for shape, participants in ((4096, 128), (8192, 256), (264, 64)):
    source = fixture(str(shape), 'f16', alignment=16, threads=256)
    source = source.replace('alignment = 16 : i64',
                            f'alignment = 16 : i64, frisk.dma_bytes = 16 : i64, '
                            f'frisk.copy_threads = {participants} : i64')
    ir = llvm_ir(source)
    calls = re.findall(r'call void @llvm.amdgcn.global.load.async.lds\((.*)\)', ir)
    assert calls and all('i32 16, i32 0, i32 0' in c for c in calls)
    execute_copy(ir, 1, shape, 16, threads=256)
    print(f'PASS {shape} FP16 elements, 16B DMA, {participants}/256 copy threads; guards intact')

    for malformed in (source.replace('alignment = 16', 'alignment = 4'),
                      source.replace('frisk.dma_bytes = 16', 'frisk.dma_bytes = 8'),
                      source.replace(f'frisk.copy_threads = {participants}', 'frisk.copy_threads = 96')):
        lower(malformed, '--frisk-lower-async-copy', ok=False)
print('PASS rejects unaligned/unsupported DMA widths and partial copy waves')

# A transpose in GM must still produce the agreed physical packet order.
source = '''module {
  func.func @copy_kernel(%src: memref<32x128xf16, strided<[1,128]>, 1>,
                         %dst: memref<4096xf16, 3>) attributes {thread_num = 256 : i32} {
    frisk.async_copy_gm_to_shm %src to %dst {
      alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.copy_threads = 128 : i64,
      frisk.source_map = affine_map<(d0) -> ((d0 floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8) mod 128)>
    } : memref<32x128xf16, strided<[1,128]>, 1>, memref<4096xf16, 3>
    frisk.wait_buffer_ready %dst : memref<4096xf16, 3>
    return
  }
}'''
ir = llvm_ir(source)
execute_copy(ir, 32, 128, 16, strides=(1, 128), threads=256,
             destination_index=lambda r, c: (r // 8 * 128 + c) * 8 + r % 8)
print('PASS transposed global source / packed LDS inverse mapping')
