#!/usr/bin/env python3
"""Fragment SHM semantics, conservative DMA motion and carried-task completion."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'build/test/ThreadTilingTest'


def run(source, *passes):
    p = subprocess.run([str(TOOL), '-', '--allow-unregistered-dialect',
                        *passes, '--verify-each'], input=source, text=True,
                       capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout, p.stderr


def intervals(report):
    return {int(n): (int(first), int(last)) for n, first, last in
            re.findall(r'shm\[(\d+)\].*live \[(\d+), (\d+)\]', report)}


FRAGMENTS = '''
!f = !frisk.fragment<vector<2xf32>, {}>
func.func @fragments(%src: memref<8xf32, 1>, %x: f32) -> !f
    attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<8xf32, 3>
  %b = memref.alloc() : memref<8xf32, 3>
  %v = arith.constant dense<1.0> : vector<2xf32>
  %f = frisk.pack_frag %v : vector<2xf32> -> !f
  %e = math.exp2 %x : f32
  frisk.store_frag %f, %b[%c0] {tile_type = vector<8xf32>} : !f, memref<8xf32, 3>
  %other = frisk.load_frag %b[%c0] {tile_type = vector<8xf32>} : memref<8xf32, 3> -> !f
  %t = frisk.async_copy_gm_to_shm %src to %a : memref<8xf32, 1>, memref<8xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %t : !frisk.async_task
  %s = frisk.elementwise_frag "add" %other, %f : (!f, !f) -> !f
  %data = frisk.load_frag %a[%c0] {tile_type = vector<8xf32>} : memref<8xf32, 3> -> !f
  %r = frisk.elementwise_frag "add" %data, %s : (!f, !f) -> !f
  return %r : !f
}
'''


def fragment_checks():
    parsed, _ = run(FRAGMENTS)
    analyzed, report = run(FRAGMENTS, '--frisk-analyze-shm-liveness')
    assert analyzed == parsed  # Analysis never changes the IR or plans storage.
    live = intervals(report)
    assert len(live) == 2 and live[1][1] < live[0][0], report
    # Register fragment users after the load must not extend B's SHM lifetime.
    deferred, _ = run(FRAGMENTS, '--test-shm-reuse')
    assert 'frisk.shm_pool' not in deferred and deferred == parsed
    out, report = run(FRAGMENTS, '--frisk-schedule-async-copies',
                      '--frisk-analyze-shm-liveness')
    assert out.index('async_copy_gm_to_shm') < out.index('math.exp2')
    assert out.index('async_copy_gm_to_shm') < out.index('store_frag')
    assert out.index('elementwise_frag') < out.index('wait_buffer_ready') < out.rindex('load_frag')
    live = intervals(report)
    assert live[0][0] < live[1][0] < live[1][1] < live[0][1], report
    assert run(out, '--frisk-schedule-async-copies')[0] == out
    # A destination read/write, a source mutation, a barrier, an unknown effect
    # or a deferred bridge must prevent issue hoisting across that operation.
    for boundary in (
        'frisk.store_frag %f, %a[%c0] {tile_type = vector<8xf32>} : !f, memref<8xf32, 3>',
        '%old = frisk.load_frag %a[%c0] {tile_type = vector<8xf32>} : memref<8xf32, 3> -> !f',
        'frisk.store_frag %f, %src[%c0] {tile_type = vector<8xf32>} : !f, memref<8xf32, 1>',
        'gpu.barrier',
        '"test.unknown"() : () -> ()',
        '%bridge = frisk.to_threadTile %a : memref<8xf32, 3> -> vector<2xf32>',
    ):
        blocked = FRAGMENTS.replace('  %t =', f'  {boundary}\n  %t =')
        result, _ = run(blocked, '--frisk-schedule-async-copies')
        assert result.index('math.exp2') < result.index('async_copy_gm_to_shm'), result
    # Waits cannot cross even independent launches at the logical boundary.
    two = FRAGMENTS.replace('  %s =', '''
  %tb = frisk.async_copy_gm_to_shm %src to %b : memref<8xf32, 1>, memref<8xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %tb : !frisk.async_task
  %s =''')
    out, _ = run(two, '--frisk-schedule-async-copies')
    assert out.index('wait_buffer_ready') < out.rindex('async_copy_gm_to_shm')
    scheduled = FRAGMENTS.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, pipeline.scheduled')
    assert run(scheduled, '--frisk-schedule-async-copies')[0] == run(scheduled)[0]
    # A later physical pass must not move a launch against a stale reuse plan.
    pooled = FRAGMENTS.replace('%a = memref.alloc()', '%a = memref.alloc() {frisk.shm_pool}')
    assert run(pooled, '--frisk-schedule-async-copies')[0] == run(pooled)[0]
    print('PASS fragment reads/writes, register-only uses, logical motion and deferred pooling')


def carried(loop, conditional=False, slots=2):
    header = ('affine.for %i = 0 to 4' if loop == 'affine' else
              'scf.for %i = %c0 to %n step %c1')
    issue = '''%task = frisk.async_copy_gm_to_shm %src to %a : memref<16xf32, 1>, memref<16xf32, 3> -> !frisk.async_task'''
    if conditional:
        issue = f'''%task = scf.if %cond -> (!frisk.async_task) {{
          %active = frisk.async_copy_gm_to_shm %src to %a : memref<16xf32, 1>, memref<16xf32, 3> -> !frisk.async_task
          scf.yield %active : !frisk.async_task
        }} else {{
          scf.yield %done : !frisk.async_task
        }}'''
    task_types = ', '.join(['!frisk.async_task'] * slots)
    inits = ', '.join(f'%p{i} = %done' for i in range(slots))
    shifted = ', '.join([f'%p{i}' for i in range(1, slots)] + ['%task'])
    drains = '\n'.join(f'frisk.wait_buffer_ready %tail#{i} : !frisk.async_task'
                       for i in range(slots))
    return f'''func.func @carried(%src: memref<16xf32, 1>, %n: index, %cond: i1)
        attributes {{thread_num = 64 : i32}} {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %x = arith.constant 1.0 : f32
      %a = memref.alloc() : memref<16xf32, 3>
      %b = memref.alloc() : memref<16xf32, 3>
      %c = memref.alloc() : memref<16xf32, 3>
      %done = frisk.async_task_completed : !frisk.async_task
      %tail:{slots} = {header} iter_args({inits}) -> ({task_types}) {{
        frisk.wait_buffer_ready %p0 : !frisk.async_task
        {issue}
        memref.store %x, %b[%c0] : memref<16xf32, 3>
        {loop}.yield {shifted} : {task_types}
      }}
      {drains}
      memref.store %x, %c[%c0] : memref<16xf32, 3>
      return
    }}'''


def token_checks():
    for slots in (3, 4):
        for loop in ('affine', 'scf'):
            _, report = run(carried(loop, True, slots), '--test-shm-reuse')
            assert 'shm bytes 192 -> 128' in report and 'skipped:' not in report, report
    rotating = carried('scf').replace('      %done =', '''
      %d = memref.alloc() : memref<16xf32, 3>
      %done =''').replace('        %task =', '''
        %odd = arith.cmpi eq, %i, %c1 : index
        %slot = arith.select %odd, %a, %d : memref<16xf32, 3>
        %task =''').replace('%src to %a', '%src to %slot')
    _, report = run(rotating, '--test-shm-reuse')
    assert 'shm bytes 256 -> 192' in report and 'skipped:' not in report, report
    assert len(re.findall('async across iterations', report)) == 2, report
    for loop in ('affine', 'scf'):
        for conditional in (False, True):
            source = carried(loop, conditional)
            out, report = run(source, '--test-shm-reuse')
            assert 'async across iterations' in report and 'skipped:' not in report, report
            assert 'shm bytes 192 -> 128' in report, report
            assert 'placement 0 @ 0' in report and 'placement 1 @ 64' in report, report
            # A and B interfere throughout the loop; C can reuse A only AFTER
            # the exit drains, even though A has no ordinary data reader.
            assert 'placement 2 @ 0' in report, report
            assert run(out, '--test-shm-reuse')[0] == out
            for tail in (0, 1):
                missing = source.replace(f'frisk.wait_buffer_ready %tail#{tail} : !frisk.async_task', '')
                _, report = run(missing, '--frisk-analyze-shm-liveness')
                assert re.search(r'shm\[0\].*skipped: async task', report), report
            conditional_wait = source.replace('frisk.wait_buffer_ready %p0 : !frisk.async_task',
                'scf.if %cond { frisk.wait_buffer_ready %p0 : !frisk.async_task }')
            _, report = run(conditional_wait, '--frisk-analyze-shm-liveness')
            assert re.search(r'shm\[0\].*skipped: async task', report), report
            # Cycle with no in-loop completion cannot prove itself complete.
            no_head = source.replace('frisk.wait_buffer_ready %p0 : !frisk.async_task', '')
            _, report = run(no_head, '--frisk-analyze-shm-liveness')
            assert re.search(r'shm\[0\].*skipped: async task', report), report
        seeded = carried(loop).replace('      %tail:2 =', '''
      %seed = frisk.async_copy_gm_to_shm %src to %a : memref<16xf32, 1>, memref<16xf32, 3> -> !frisk.async_task
      %tail:2 =''').replace('%p0 = %done', '%p0 = %seed')
        _, report = run(seeded, '--frisk-analyze-shm-liveness')
        assert len(intervals(report)) == 3 and 'skipped:' not in report, report
        _, report = run(seeded.replace('frisk.wait_buffer_ready %tail#0 : !frisk.async_task', ''),
                        '--frisk-analyze-shm-liveness')
        assert re.search(r'shm\[0\].*skipped: async task', report), report
    # A select may discard a task even if its selected result is awaited.
    discarded = FRAGMENTS.replace('%x: f32', '%x: f32, %cond: i1').replace(
        '  frisk.wait_buffer_ready %t : !frisk.async_task', '''
  %done = frisk.async_task_completed : !frisk.async_task
  %selected = arith.select %cond, %t, %done : !frisk.async_task
  frisk.wait_buffer_ready %selected : !frisk.async_task''')
    _, report = run(discarded, '--frisk-analyze-shm-liveness')
    assert re.search(r'shm\[0\].*skipped: async task', report), report
    conditional_forward = FRAGMENTS.replace('%x: f32', '%x: f32, %cond: i1').replace(
        '  frisk.wait_buffer_ready %t : !frisk.async_task', '''
  %done = frisk.async_task_completed : !frisk.async_task
  %selected = scf.if %cond -> (!frisk.async_task) {
    scf.yield %t : !frisk.async_task
  } else {
    scf.yield %done : !frisk.async_task
  }
  frisk.wait_buffer_ready %selected : !frisk.async_task''')
    _, report = run(conditional_forward, '--frisk-analyze-shm-liveness')
    assert re.search(r'shm\[0\].*skipped: async task', report), report
    # Yielding an outer task from a loop without passing it as an init loses
    # the task on the zero-trip path, despite an unconditional exit wait.
    zero_trip = FRAGMENTS.replace('%x: f32', '%x: f32, %n: index').replace(
        '  frisk.wait_buffer_ready %t : !frisk.async_task', '''
  %done = frisk.async_task_completed : !frisk.async_task
  %selected = affine.for %i = 0 to %n iter_args(%pending = %done) -> (!frisk.async_task) {
    frisk.wait_buffer_ready %pending : !frisk.async_task
    affine.yield %t : !frisk.async_task
  }
  frisk.wait_buffer_ready %selected : !frisk.async_task''')
    _, report = run(zero_trip, '--frisk-analyze-shm-liveness')
    assert re.search(r'shm\[0\].*skipped: async task', report), report
    print('PASS affine/scf FIFO task completion, branch carriers, exit drains and unsafe paths')


if __name__ == '__main__':
    fragment_checks()
    token_checks()
