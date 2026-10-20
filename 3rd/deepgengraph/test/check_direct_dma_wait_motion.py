#!/usr/bin/env python3
"""Direct-only DMA crossing, barrier lifetime proof and retained completions."""
import re

from check_frisk_async_copy import lower

PASS = '--frisk-schedule-async-copies'
SRC = 'memref<128xi32, 1>'
DST = 'memref<128xi32, 3>'
ISSUE_A = f'frisk.async_copy_gm_to_shm %src to %a : {SRC}, {DST}'
ISSUE_B = f'frisk.async_copy_gm_to_shm %src to %b : {SRC}, {DST}'
WAIT_A = f'frisk.wait_buffer_ready %a : {DST}'
WAIT_B = f'frisk.wait_buffer_ready %b : {DST}'


def fixture(prefix='', body_prefix='', body_suffix='', attributes=''):
    return f'''module {{
      func.func @test(%src: {SRC}, %cond: i1) -> i32
          attributes {{thread_num = 64 : i32{attributes}}} {{
        %a = memref.alloc() : {DST}
        %b = memref.alloc() : {DST}
        %z = arith.constant 0 : index
        {prefix}
        {body_prefix}
        {ISSUE_A}
        {WAIT_A}
        {ISSUE_B}
        {WAIT_B}
        %x = memref.load %a[%z] : {DST}
        %y = memref.load %b[%z] : {DST}
        {body_suffix}
        return %z32 : i32
      }}
    }}'''.replace('%z = arith.constant 0 : index',
                  '%z = arith.constant 0 : index\n %z32 = arith.constant 0 : i32')


def sequence(ir):
    return re.findall(r'frisk\.(async_copy_gm_to_shm|wait_buffer_ready)\b', ir)


def check(source, crosses, passes=(PASS,)):
    result = lower(source, *passes, '--verify-each')
    expected = ['async_copy_gm_to_shm', 'wait_buffer_ready'] * 2
    if crosses:
        expected = ['async_copy_gm_to_shm'] * 2 + ['wait_buffer_ready'] * 2
    assert sequence(result) == expected, result
    assert lower(result, PASS, '--verify-each') == result, 'not idempotent'
    return result


def main():
    result = check(fixture(), True)
    lowered = lower(result, '--frisk-lower-async-copy', '--verify-each')
    assert lowered.count('rocdl.asyncmark') == 2
    assert lowered.count('rocdl.wait.asyncmark 0') == 2
    assert lowered.count('gpu.barrier') == 2
    assert 'lds-repack' not in lowered

    # Independence is physical, not a different SSA name. Unknown/external or
    # pooled destinations are not evidence that early issue is safe.
    check(fixture().replace(ISSUE_B, ISSUE_B.replace('to %b', 'to %a')), False)
    alias = fixture().replace('%b = memref.alloc() : ' + DST,
                              f'%b = memref.cast %a : {DST} to {DST}')
    check(alias, False)
    check(fixture().replace('%b = memref.alloc()', '%b = memref.alloc() {frisk.shm_pool}'), False)
    check(fixture().replace('%b = memref.alloc()',
                            '%b = memref.alloc() {frisk.shared_pack = 4 : i64}'), False)
    external = fixture().replace(f'%cond: i1)', f'%cond: i1, %b: {DST})')
    check(external.replace(f'%b = memref.alloc() : {DST}', ''), False)
    selected = fixture().replace('%b = memref.alloc()', '%storage = memref.alloc()')
    selected = selected.replace('%z = arith.constant', f'%b = arith.select %cond, %a, %storage : {DST}\n %z = arith.constant')
    check(selected, False)

    # Do not cross fallback/staging transfers in either direction.
    for issue in (ISSUE_A, ISSUE_B):
        for attr in ('frisk.force_sync', 'alignment = 2 : i64'):
            check(fixture().replace(issue, issue.replace(' : ', f' {{{attr}}} : ', 1)), False)
    staged_src = 'memref<16x16xf16, strided<[1, 16]>, 1>'
    staged_dst = 'memref<256xf16, 3>'
    staged_issue = f'''frisk.async_copy_gm_to_shm %staged_src to %staged_dst
      {{alignment = 4 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 16, d0 mod 16)>}}
      : {staged_src}, {staged_dst}'''
    for first in (False, True):
        text = fixture().replace('%cond: i1)', f'%cond: i1, %staged_src: {staged_src})')
        text = text.replace('%z = arith.constant',
                            f'%staged_dst = memref.alloc() : {staged_dst}\n %z = arith.constant')
        text = text.replace(ISSUE_A if first else ISSUE_B, staged_issue)
        text = text.replace(WAIT_A if first else WAIT_B,
                            f'frisk.wait_buffer_ready %staged_dst : {staged_dst}')
        out = check(text, False)
        assert 'lds-repack' in lower(out, '--frisk-lower-async-copy', '--verify-each')

    # Old readers/writers of B may rely on wait A's block barrier, even though
    # the two destination allocations are disjoint. An earlier explicit
    # barrier releases those readers independently.
    old_read = f'%old = memref.load %b[%z] : {DST}'
    check(fixture(prefix=old_read), False)
    check(fixture(prefix=old_read + '\n gpu.barrier'), True)
    check(fixture(prefix=f'affine.for %i = 0 to 4 {{ {old_read} }}'), False)
    check(fixture(prefix=f'affine.for %i = 0 to 4 {{ %old = memref.load %src[%i] : {SRC} }}'), True)
    check(fixture(prefix=f'memref.store %z32, %src[%z] : {SRC}'), False)
    check(fixture(prefix=f'affine.for %i = 0 to 4 {{ memref.store %z32, %src[%i] : {SRC} }}'), False)
    check(fixture(prefix='func.call @unknown() : () -> ()').replace(
        'module {', 'module {\n func.func private @unknown()'), False)

    # Leave manually scheduled and divergent code alone.
    for attr in ('pipeline.scheduled', 'pipeline.stage = 0 : i64',
                 'frisk.pipelined'):
        check(fixture(attributes=', ' + attr), False)
    check(fixture(body_prefix='scf.if %cond {', body_suffix='}'), False)
    check(fixture(body_prefix='affine.for %i = 0 to 4 {', body_suffix='}'), False)
    check(fixture(body_prefix='affine.for %i = 0 to 4 {',
                  body_suffix='gpu.barrier\n }'), True)
    check(fixture(body_prefix='affine.for %i = 0 to 4 {', body_suffix='}'),
          True, ('--lower-frisk-fragments',))
    # A block-varying loop remains forbidden even with a tail barrier.
    check(fixture(prefix='%tid = gpu.thread_id x',
                  body_prefix='affine.for %i = 0 to %tid {',
                  body_suffix='gpu.barrier\n }'), False)
    print('PASS direct DMA wait motion: direct/staged/fallback, aliases, old readers, '
          'source writes, control flow, loop backedges, retained zero waits/barriers and idempotence')


if __name__ == '__main__':
    main()
