#!/usr/bin/env python3
"""Consumer-driven readiness scheduling before and after physical tiling."""
from check_frisk_async_copy import lower

PASS = '--frisk-sink-wait-buffer-ready'
TY = 'memref<64xf32, 3>'
SRC = 'memref<64xf32, 1>'
ISSUE = f'frisk.async_copy_gm_to_shm %src to %a : {SRC}, {TY}'
WAIT = f'frisk.wait_buffer_ready %a : {TY}'


def fixture(body, prefix='', setup=''):
    return f'''module {{
      {prefix}
      func.func @test(%src: {SRC}, %x: f32, %cond: i1) -> f32 {{
        %a = memref.alloc() : {TY}
        %b = memref.alloc() : {TY}
        %z = arith.constant 0 : index
        {setup}
        {ISSUE}
        {WAIT}
        {body}
        %r = memref.load %a[%z] : {TY}
        return %r : f32
      }}
    }}'''


def run(source):
    result = lower(source, PASS, '--verify-each')
    assert lower(result, PASS, '--verify-each') == result
    return result


def main():
    # Independent scalar computation, source reads, fresh allocations and
    # physical writes to unrelated owned storage can overlap the DMA.
    result = run(fixture(f'''
      %v = arith.mulf %x, %x : f32
      %s = memref.load %src[%z] : {SRC}
      %tmp = memref.alloc() : {TY}
      memref.store %s, %b[%z] : {TY}
    '''))
    assert result.index('memref.store') < result.index('wait_buffer_ready')
    assert result.index('wait_buffer_ready') < result.rindex('memref.load')

    # Source/destination mutations, release, unknown effects, synchronization,
    # and regions are boundaries even when no direct destination use is seen.
    for blocker, marker, prefix in [
        (f'memref.store %x, %src[%z] : {SRC}', 'memref.store', ''),
        (f'memref.store %x, %a[%z] : {TY}', 'memref.store', ''),
        (f'memref.dealloc %src : {SRC}', 'memref.dealloc', ''),
        ('gpu.barrier', 'gpu.barrier', ''),
        ('frisk.sync_threads_in_block', 'frisk.sync_threads_in_block', ''),
        ('func.call @unknown() : () -> ()', 'call @unknown',
         'func.func private @unknown()'),
        (f'scf.if %cond {{ %v = memref.load %a[%z] : {TY} }}', 'scf.if', ''),
        ('affine.for %i = 0 to 4 {}', 'affine.for', ''),
        (f'%tile = frisk.to_threadTile %a : {TY} -> vector<64xf32>',
         'frisk.to_threadTile', ''),
        (f'frisk.async_copy_gm_to_shm %src to %b : {SRC}, {TY}',
         'async_copy_gm_to_shm', ''),
        (f'frisk.wait_buffer_ready %b : {TY}', 'wait_buffer_ready', ''),
    ]:
        result = run(fixture(blocker + '\n %v = arith.mulf %x, %x : f32', prefix))
        assert result.index('wait_buffer_ready') < result.rindex(marker), result

    # Logical views and physical casts/subviews/selects are descriptors, not
    # consumption. Sink past descriptor construction but stop at alias reads.
    for descriptor, alias_type in [
        (f'%alias = frisk.buffer_view %a[0], ranges = [64] : {TY} -> {TY}', TY),
        (f'%alias = memref.cast %a : {TY} to memref<?xf32, 3>', 'memref<?xf32, 3>'),
        (f'%alias = memref.memory_space_cast %a : {TY} to memref<64xf32>',
         'memref<64xf32>'),
        (f'%alias = memref.subview %a[0] [64] [1] : {TY} to memref<64xf32, strided<[1]>, 3>',
         'memref<64xf32, strided<[1]>, 3>'),
        (f'%alias = arith.select %cond, %a, %b : {TY}', TY),
    ]:
        result = run(fixture(f'''{descriptor}
          %v = memref.load %alias[%z] : {alias_type}
          %w = arith.mulf %v, %x : f32'''))
        assert result.index('wait_buffer_ready') < result.index('memref.load'), result
        assert result.index('wait_buffer_ready') > result.index(' = ', result.index('async_copy_gm_to_shm')), result

    # An alias of the global source must also remain unchanged until completion.
    result = run(fixture(f'''
      %alias = memref.cast %src : {SRC} to memref<?xf32, 1>
      memref.store %x, %alias[%z] : memref<?xf32, 1>
    '''))
    assert result.index('memref.cast') < result.index('wait_buffer_ready') < result.index('memref.store')

    # The motivating high-level case: wait K only when the GEMM reads K.
    source = '''func.func @tiles(%src: memref<16x16xf16, 1>, %x: f16) {
      %q = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<16x16xf16, 3>
      %k = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<16x16xf16, 3>
      frisk.async_copy_gm_to_shm %src to %q : memref<16x16xf16, 1>, memref<16x16xf16, 3>
      frisk.wait_buffer_ready %q : memref<16x16xf16, 3>
      frisk.async_copy_gm_to_shm %src to %k : memref<16x16xf16, 1>, memref<16x16xf16, 3>
      frisk.wait_buffer_ready %k : memref<16x16xf16, 3>
      %scaled = frisk.mul %q, %x : memref<16x16xf16, 3>, f16 -> memref<16x16xf16, 3>
      %r = frisk.gemm(%scaled, %k) {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<16x16xf16, 3>, memref<16x16xf16, 3> -> memref<16x16xf32>
      return
    }'''
    result = run(source)
    assert result.index('frisk.mul') < result.rindex('wait_buffer_ready') < result.index('frisk.gemm')
    assert result.index('wait_buffer_ready') < result.rindex('async_copy_gm_to_shm')

    # Do not repair malformed readiness, move unpaired waits, or invalidate a
    # pre-existing per-operation software-pipeline schedule.
    source = fixture('%v = arith.mulf %x, %x : f32')
    for untouched in [
        source.replace(ISSUE, ''),
        source.replace(WAIT, f'%early = memref.load %a[%z] : {TY}\n' + WAIT),
        source.replace(WAIT, f'frisk.wait_buffer_ready %a {{pipeline.stage = 1 : i64}} : {TY}'),
        source.replace('-> f32 {', '-> f32 attributes {pipeline.order = array<i64: 0>} {'),
        source.replace('-> f32 {', '-> f32 attributes {pipeline.scheduled} {'),
    ]:
        assert run(untouched) == lower(untouched), untouched

    # A wait with no consumer still completes before returning; never delete it.
    no_read = fixture('%v = arith.mulf %x, %x : f32').replace(
        f'%r = memref.load %a[%z] : {TY}', '').replace('return %r', 'return %x')
    result = run(no_read)
    assert result.index('arith.mulf') < result.index('wait_buffer_ready') < result.index('return')
    print('PASS wait sinking: tile consumers, aliases, source lifetime, DMA/barrier/control-flow boundaries, schedules and idempotence')


if __name__ == '__main__':
    main()
