#!/usr/bin/env python3
"""Physical async packets, permutation semantics, scheduling and storage reuse.

Executes exported addresses with the delayed CPU DMA shim. GPU bank throughput
and multi-wave execution still require hardware validation.
"""
import re
from check_frisk_async_copy import lower, llvm_ir, execute_copy


def packed_case(rows, cols, axis, pack, pair=0, store=0, store_pair=0,
                threads=128, alignment=4, source_transposed=None, dma_bytes=0):
    transposed = axis == 0 if source_transposed is None else source_transposed
    padding = max(4, dma_bytes // 2)
    strides = (1, rows + padding) if transposed else (cols + padding, 1)
    src = f'memref<{rows}x{cols}xi16, strided<[{strides[0]}, {strides[1]}], offset: {padding}>, 1>'
    dst = f'memref<{rows}x{cols}xi16, 3>'
    attrs = f'frisk.shared_pack = {pack} : i64, frisk.shared_pack_axis = {axis} : i64'
    if pair:
        attrs += f', frisk.shared_pair_stride = {pair} : i64'
    if store:
        attrs += f', frisk.shared_store_stride = {store} : i64, frisk.shared_store_pair = {store_pair} : i64'
    dma_attrs = f', frisk.dma_bytes = {dma_bytes} : i64' if dma_bytes else ''
    source = f'''func.func @copy_kernel(%src: {src}) attributes {{thread_num = {threads} : i32}} {{
      %tile = memref.alloc() {{{attrs}}} : {dst}
      frisk.async_copy_gm_to_shm %src to %tile {{alignment = {alignment} : i64{dma_attrs}}} : {src}, {dst}
      frisk.wait_buffer_ready %tile : {dst}
      %c0 = arith.constant 0 : index
      %v = memref.load %tile[%c0, %c0] : {dst}
      return
    }}'''
    packed = lower(source, '--pack-shared-memory')
    assert 'frisk.source_map' in packed, packed
    assert lower(packed, '--pack-shared-memory') == packed
    # Replace only the owned allocation by caller storage to observe ALL copied
    # elements and guards through the ordinary bare-pointer test ABI.
    allocation = re.search(r'(%\w+) = memref.alloc\(\).* : (memref<[^>]+>)', packed)
    assert allocation, packed
    packed = packed.replace(') attributes {thread_num',
                            f', %external: {allocation[2]}) attributes {{thread_num', 1)
    packed = packed[:packed.index(allocation[0])] + packed[packed.index(allocation[0]):].replace(allocation[0], '', 1)
    packed = re.sub(r'(?<![\w])' + re.escape(allocation[1]) + r'\b', '%external', packed)

    def physical(r, c):
        if store:
            reg = c // store
            group = store_pair // store
            paired = reg // (2 * group) * (2 * group) + reg % group * 2 + reg // group % 2
            return (c % store * rows + r) * (cols // store) + paired
        if pair:
            frag, element = divmod(r, pack)
            groups = pair // pack
            group = frag // (2 * groups) * groups + frag % groups
            if pack == 4 and cols % 8 == 0:
                cg = c // 8
                c = cg * 8 + ((c // 4 + cg // 4 + group // (1 if groups == 1 else 2)) % 2) * 4 + (c + cg) % 4
            return (group * cols + c) * (2 * pack) + frag // groups % 2 * pack + element
        if axis == 0:
            return (r // pack * cols + c) * pack + r % pack
        return (c // pack * rows + r) * pack + c % pack

    # Independently prove which physical dwords have contiguous source data.
    inverse = {physical(r, c): r * strides[0] + c * strides[1]
               for r in range(rows) for c in range(cols)}
    assert len(inverse) == rows * cols
    dma = alignment >= 4 and all(inverse[i] % 2 == 0 and inverse[i+1] == inverse[i]+1
                                for i in range(0, rows * cols, 2))
    for asynchronous in ((True,) if dma_bytes else (True, False)):
        ir = llvm_ir(packed, asynchronous)
        # Noncontiguous packed dwords use an aligned source-order staging tile.
        assert ('call void @llvm.amdgcn.global.load.async.lds' in ir) == (asynchronous and alignment >= 4)
        assert ('addrspace(3) global' in ir) == (asynchronous and alignment >= 4 and not dma)
        execute_copy(ir, rows, cols, 16, strides, padding, threads, physical)
    if axis == 0 and not pair:
        selected = lower(packed, '--frisk-lower-async-to-sync')
        ir = llvm_ir(selected)
        assert 'call void @llvm.amdgcn.global.load.async.lds' not in ir
        execute_copy(ir, rows, cols, 16, strides, padding, threads, physical)
    print(f'PASS physical copy {rows}x{cols}, axis={axis}, pair={pair}, store={store}, direct_DMA={dma}')


def schedule_checks():
    source = '''func.func @schedule(%src: memref<64xf32, 1>, %out: memref<1xf32, 1>, %x: f32) attributes {thread_num = 64 : i32} {
      %a = memref.alloc() : memref<64xf32, 3>
      %z = arith.constant 0 : index
      %v = math.exp2 %x : f32
      frisk.async_copy_gm_to_shm %src to %a : memref<64xf32, 1>, memref<64xf32, 3>
      frisk.wait_buffer_ready %a : memref<64xf32, 3>
      %w = arith.mulf %v, %x : f32
      %r = memref.load %a[%z] : memref<64xf32, 3>
      %s = arith.addf %r, %w : f32
      memref.store %s, %out[%z] : memref<1xf32, 1>
      return
    }'''
    result = lower(source, '--frisk-schedule-async-copies')
    assert result.index('async_copy_gm_to_shm') < result.index('math.exp2')
    assert result.index('arith.mulf') < result.index('wait_buffer_ready') < result.index('memref.load')
    assert lower(result, '--frisk-schedule-async-copies') == result
    late_alloc = source.replace('      %a = memref.alloc() : memref<64xf32, 3>\n', '')
    late_alloc = late_alloc.replace('      frisk.async_copy',
                                  '      %a = memref.alloc() : memref<64xf32, 3>\n      frisk.async_copy')
    result = lower(late_alloc, '--frisk-schedule-async-copies')
    assert result.index('memref.alloc') < result.index('async_copy_gm_to_shm') < result.index('math.exp2')
    for barrier in ('gpu.barrier', 'memref.store %x, %src[%z] : memref<64xf32, 1>',
                    'memref.store %x, %a[%z] : memref<64xf32, 3>',
                    'affine.for %i = 0 to 64 { memref.store %x, %a[%i] : memref<64xf32, 3> }'):
        blocked = source.replace('      frisk.async_copy', '      ' + barrier + '\n      frisk.async_copy')
        result = lower(blocked, '--frisk-schedule-async-copies')
        assert result.index('math.exp2') < result.index('async_copy_gm_to_shm')
    # Descriptor setup must move before the issue, never after it.
    view = source.replace('%src: memref<64xf32, 1>', '%base: memref<128xf32, 1>')
    view = view.replace('      frisk.async_copy', '''      %src = memref.subview %base[0] [64] [1] : memref<128xf32, 1> to memref<64xf32, strided<[1]>, 1>
      frisk.async_copy''').replace(': memref<64xf32, 1>,', ': memref<64xf32, strided<[1]>, 1>,')
    result = lower(view, '--frisk-schedule-async-copies')
    assert result.index('memref.subview') < result.index('async_copy_gm_to_shm') < result.index('math.exp2')
    print('PASS issue hoisting, late wait, descriptor dependencies, alias/write/barrier boundaries, idempotence')


def reuse_checks():
    source = '''func.func @reuse(%src: memref<64xf32, 1>) -> f32 attributes {thread_num = 64 : i32} {
      %a = memref.alloc() : memref<64xf32, 3>
      %b = memref.alloc() : memref<64xf32, 3>
      %z = arith.constant 0 : index
      frisk.async_copy_gm_to_shm %src to %a : memref<64xf32, 1>, memref<64xf32, 3>
      frisk.wait_buffer_ready %a : memref<64xf32, 3>
      %x = memref.load %a[%z] : memref<64xf32, 3>
      frisk.async_copy_gm_to_shm %src to %b : memref<64xf32, 1>, memref<64xf32, 3>
      frisk.wait_buffer_ready %b : memref<64xf32, 3>
      %y = memref.load %b[%z] : memref<64xf32, 3>
      %s = arith.addf %x, %y : f32
      return %s : f32
    }'''
    result = lower(source, '--test-shm-reuse')
    assert 'memref<256xi8, 3>' in result and 'frisk.shm_pool' in result
    assert result.index('memref.load') < result.index('gpu.barrier') < result.rindex('async_copy_gm_to_shm')
    assert lower(result, '--test-shm-reuse') == result
    incomplete = source.replace('frisk.wait_buffer_ready %a : memref<64xf32, 3>', '')
    assert 'frisk.shm_pool' not in lower(incomplete, '--test-shm-reuse')
    pending = source.replace('      %x = memref.load', '''      frisk.async_copy_gm_to_shm %src to %b : memref<64xf32, 1>, memref<64xf32, 3>
      %x = memref.load''')
    assert 'frisk.shm_pool' not in lower(pending, '--test-shm-reuse')
    print('PASS reuse after completion, release barrier, incomplete/overlapping DMA rejection')


def separate_completion_checks():
    # Regression: alias independence must not move completion across another
    # issue. This is a conservative policy, not a proven fix for target errors.
    for count, threads in [(64, 64), (384, 128), (1024, 256)]:
        ty = f'memref<{count}xf32, 3>'
        src = f'memref<{count}xf32, 1>'
        source = f'''func.func @group(%src: {src}, %out: memref<1xf32, 1>) attributes {{thread_num = {threads} : i32}} {{
          %a = memref.alloc() : {ty}
          %b = memref.alloc() : {ty}
          %z = arith.constant 0 : index
          frisk.async_copy_gm_to_shm %src to %a : {src}, {ty}
          frisk.wait_buffer_ready %a : {ty}
          frisk.async_copy_gm_to_shm %src to %b : {src}, {ty}
          %x = memref.load %a[%z] : {ty}
          frisk.wait_buffer_ready %b : {ty}
          %y = memref.load %b[%z] : {ty}
          %r = arith.addf %x, %y : f32
          memref.store %r, %out[%z] : memref<1xf32, 1>
          return
        }}'''
        result = lower(source, '--frisk-schedule-async-copies')
        assert result.index('async_copy_gm_to_shm') < result.index('wait_buffer_ready') < result.rindex('async_copy_gm_to_shm'), result
        assert lower(result, '--frisk-schedule-async-copies') == result
        # Both tiles remain live at the first read and cannot share storage.
        reused = lower(result, '--test-shm-reuse')
        assert reused.count('memref.alloc()') == 2 or f'memref<{2 * count * 4}xi8, 3>' in reused, reused
        ir = llvm_ir(result)
        assert ir.count('call void @llvm.amdgcn.wait.asyncmark') == 2, ir
        assert ir.count('call void @llvm.amdgcn.s.barrier') == 2, ir
        # Explicit synchronization and potential destination aliases forbid
        # motion. Keep writes and control flow as completion boundaries.
        for boundary in ['gpu.barrier',
                         f'affine.for %i = 0 to {count} {{ gpu.barrier }}']:
            blocked = source.replace(f'frisk.async_copy_gm_to_shm %src to %b',
                                     boundary + '\nfrisk.async_copy_gm_to_shm %src to %b')
            scheduled = lower(blocked, '--frisk-schedule-async-copies')
            assert scheduled.index('wait_buffer_ready') < scheduled.rindex('async_copy_gm_to_shm')
        # Insert a write after the first read, hence after the first drain.
        written = re.sub(r'(%\w+) = memref.load (%\w+)\[(%\w+)\] : (' + re.escape(ty) + ')',
                         r'\1 = memref.load \2[\3] : \4\nmemref.store \1, \2[\3] : \4', result, count=1)
        assert llvm_ir(written).count('call void @llvm.amdgcn.wait.asyncmark') == 2
        alias = source.replace(f'%b = memref.alloc() : {ty}', '').replace('%b', '%a')
        alias = alias.replace(f'          %x = memref.load %a[%z] : {ty}\n', '')
        alias = alias.replace(f'          %y = memref.load',
                              f'          %x = memref.load %a[%z] : {ty}\n          %y = memref.load')
        scheduled = lower(alias, '--frisk-schedule-async-copies')
        assert scheduled.index('wait_buffer_ready') < scheduled.rindex('async_copy_gm_to_shm')
        # Unknown same-address-space arguments may alias even when SSA names
        # differ. Do not treat different names as an independence proof.
        unknown = alias.replace(f'%src: {src},', f'%a: {ty}, %b: {ty}, %src: {src},')
        unknown = unknown.replace(f'%a = memref.alloc() : {ty}', '')
        second = unknown.rindex('frisk.async_copy_gm_to_shm')
        unknown = unknown[:second] + unknown[second:].replace('to %a', 'to %b', 1)
        scheduled = lower(unknown, '--frisk-schedule-async-copies')
        assert scheduled.index('wait_buffer_ready') < scheduled.rindex('async_copy_gm_to_shm')
    print('PASS separate DMA completions, varying sizes, no storage overlap, alias/write/barrier guards')

    # Explicitly authored overlapping issues retain BOTH named completions.
    # The CPU model checks data and guards, not hardware synchronization safety.
    for count, threads in [(63, 64), (128, 128), (260, 256)]:
        src = f'memref<{2 * count}xi32, 1>'
        dst = f'memref<{2 * count}xi32, 3>'
        source = f'''func.func @copy_kernel(%src: {src}, %dst: {dst}) attributes {{thread_num = {threads} : i32}} {{
          %s0 = memref.subview %src[0] [{count}] [1] : {src} to memref<{count}xi32, strided<[1]>, 1>
          %s1 = memref.subview %src[{count}] [{count}] [1] : {src} to memref<{count}xi32, strided<[1], offset: {count}>, 1>
          %d0 = memref.subview %dst[0] [{count}] [1] : {dst} to memref<{count}xi32, strided<[1]>, 3>
          %d1 = memref.subview %dst[{count}] [{count}] [1] : {dst} to memref<{count}xi32, strided<[1], offset: {count}>, 3>
          frisk.async_copy_gm_to_shm %s0 to %d0 : memref<{count}xi32, strided<[1]>, 1>, memref<{count}xi32, strided<[1]>, 3>
          frisk.async_copy_gm_to_shm %s1 to %d1 : memref<{count}xi32, strided<[1], offset: {count}>, 1>, memref<{count}xi32, strided<[1], offset: {count}>, 3>
          frisk.wait_buffer_ready %d0 : memref<{count}xi32, strided<[1]>, 3>
          frisk.wait_buffer_ready %d1 : memref<{count}xi32, strided<[1], offset: {count}>, 3>
          return
        }}'''
        ir = llvm_ir(source)
        assert ir.count('call void @llvm.amdgcn.wait.asyncmark') == 2
        assert ir.count('call void @llvm.amdgcn.s.barrier') == 2
        execute_copy(ir, 1, 2 * count, 32, threads=threads)
    print('PASS explicit named waits preserved; CPU DMA data, tails and memory guards')


def permutation_guards():
    template = '''func.func @invalid(%src: memref<8x8xi16, 1>, %dst: memref<64xi16, 3>) {
      frisk.async_copy_gm_to_shm %src to %dst {frisk.source_map = affine_map<(d0) -> (EXPR)>}
        : memref<8x8xi16, 1>, memref<64xi16, 3>
      return
    }'''
    for expression, error in [('0, 0', 'bijective'), ('d0, 0', 'outside source tile'),
                              ('d0', 'flat destination elements')]:
        assert error in lower(template.replace('EXPR', expression), ok=False)
    print('PASS malformed, out-of-bounds and non-bijective permutation rejection')


if __name__ == '__main__':
    for args in [(16, 24, 0, 4), (24, 16, 1, 4), (32, 24, 0, 4, 4),
                 (64, 40, 0, 4, 16), (32, 64, 1, 4, 0, 4, 16)]:
        packed_case(*args)
    packed_case(16, 24, 0, 4, threads=256, alignment=2)
    packed_case(16, 24, 0, 4, source_transposed=False)
    schedule_checks()
    separate_completion_checks()
    reuse_checks()
    permutation_guards()
