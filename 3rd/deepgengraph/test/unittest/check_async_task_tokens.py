#!/usr/bin/env python3
"""Task identity, loop/branch carriers, lowering, and async storage lifetime."""
import re
from check_frisk_async_copy import lower, fixture, llvm_ir
from check_auto_pipeline import fixture as pipeline_fixture


def main():
    source = fixture('128')
    parsed = lower(source, '--verify-each')
    issue = re.search(r'(%\w+) = frisk.async_copy_gm_to_shm', parsed)[1]
    assert f'frisk.wait_buffer_ready {issue} : !frisk.async_task' in parsed
    assert lower(parsed) == parsed
    bad = source.replace('frisk.wait_buffer_ready %task_dst : !frisk.async_task',
                         'frisk.wait_buffer_ready %dst : memref<128xi32, 3>')
    assert 'async' in lower(bad, ok=False)
    bad = source.replace('-> !frisk.async_task', '-> i32')
    assert 'async' in lower(bad, ok=False)
    print('PASS token type verification and text round trip')

    # The second issue writes exactly the same buffer. Its wait must retain
    # the first issue's identity, rather than being rebound by destination.
    duplicate = source.replace('    frisk.wait_buffer_ready', '''
    %second = frisk.async_copy_gm_to_shm %src to %dst : memref<128xi32, 1>, memref<128xi32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready''')
    out = lower(duplicate, '--frisk-schedule-async-copies', '--verify-each')
    tokens = re.findall(r'(%\w+) = frisk.async_copy_gm_to_shm', out)
    assert len(tokens) == 2
    assert f'frisk.wait_buffer_ready {tokens[0]} :' in out
    assert f'frisk.wait_buffer_ready {tokens[1]} :' not in out
    print('PASS two issues to one buffer keep distinct task identities')

    for loop in ('affine', 'scf'):
        header = ('affine.for %i = 0 to 4' if loop == 'affine' else
                  'scf.for %i = %c0 to %c4 step %c1')
        carried = f'''module {{
          func.func @copy_kernel(%src: memref<128xi32, 1>, %dst: memref<128xi32, 3>, %cond: i1)
              attributes {{thread_num = 128 : i32}} {{
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            %done = frisk.async_task_completed : !frisk.async_task
            %last = {header} iter_args(%pending = %done) -> (!frisk.async_task) {{
              frisk.wait_buffer_ready %pending : !frisk.async_task
              %next = scf.if %cond -> (!frisk.async_task) {{
                %task = frisk.async_copy_gm_to_shm %src to %dst : memref<128xi32, 1>, memref<128xi32, 3> -> !frisk.async_task
                scf.yield %task : !frisk.async_task
              }} else {{
                scf.yield %done : !frisk.async_task
              }}
              {loop}.yield %next : !frisk.async_task
            }}
            frisk.wait_buffer_ready %last : !frisk.async_task
            return
          }}
        }}'''
        lower(carried, '--verify-each')
        sync = lower(carried, '--frisk-lower-async-to-sync', '--verify-each')
        assert '!frisk.async_task' not in sync
        assert sync.count('frisk.sync_threads_in_block') == 2
        for enabled in ('true', 'false'):
            out = lower(carried, f'--frisk-lower-async-copy=use-async-copy={enabled}', '--verify-each')
            assert '!frisk.async_task' not in out and 'frisk.async_task_completed' not in out
            assert out.count('gpu.barrier') == 2
            if enabled == 'true':
                assert out.count('rocdl.wait.asyncmark') == 2
        assert 'frisk' not in llvm_ir(carried)
    print('PASS affine/scf loop tokens, branch yields, tail waits, sync and LLVM lowering')

    for stages in (2, 3, 4):
        source = pipeline_fixture(f'{stages} : i64').replace('-> (f32, f32) {',
            '-> (f32, f32) attributes {thread_num = 128 : i32} {', 1)
        out = lower(source, '--frisk-auto-pipeline', '--verify-each')
        assert 'pipeline.scheduled' in out
        steady = out[out.index('affine.for'):]
        assert len(re.findall(r'frisk.wait_buffer_ready %arg\d+ : !frisk.async_task', steady)) == 2
        assert '!frisk.async_task' in steady.splitlines()[0]
        lower(out, '--frisk-lower-async-to-sync', '--verify-each')
        lower(out, '--frisk-lower-async-copy', '--verify-each')
    print('PASS automatic 2/3/4-stage pipelines wait on carried tasks')

    # A completion-only use still occupies a rotating destination. Stage 0
    # issues two iterations ahead, stage 1 waits, and stage 2 closes the loop.
    explicit = '''func.func @scheduled(%src: memref<128xi32, 1>, %ub: index) attributes {thread_num = 128 : i32} {
  %tile = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128xi32, 3>
  %zero = arith.constant dense<0.0> : vector<1xf32>
  %r = affine.for %i = 0 to %ub iter_args(%acc = %zero) -> (vector<1xf32>) {
    %task = frisk.async_copy_gm_to_shm %src to %tile : memref<128xi32, 1>, memref<128xi32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    frisk.copy %zero to %acc [affine_map<() -> (1)>] : vector<1xf32>, vector<1xf32>
    affine.yield %acc : vector<1xf32>
  } {pipeline.stage = array<i64: 0, 1, 2>, pipeline.order = array<i64: 0, 1, 2>}
  return
}
'''
    out = lower(explicit, '--frisk-pipeline-schedule', '--verify-each')
    assert out.count('frisk.alloc_buffer') == 2, out
    assert re.search(r'scf.if %\w+ -> \(!frisk.async_task\)', out)
    assert re.search(r'frisk.wait_buffer_ready %arg\d+ : !frisk.async_task', out)
    assert lower(out, '--frisk-pipeline-schedule') == out
    lower(out, '--frisk-lower-async-to-sync', '--verify-each')
    lower(out, '--frisk-lower-async-copy', '--verify-each')
    # Allocation and issue may belong to different statements in the same stage.
    allocation = '  %tile = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128xi32, 3>\n'
    inside = explicit.replace(allocation, '').replace('    %task =', allocation + '    %task =')
    inside = inside.replace('pipeline.stage = array<i64: 0, 1, 2>',
                            'pipeline.stage = array<i64: 0, 0, 1, 2>')
    inside = inside.replace('pipeline.order = array<i64: 0, 1, 2>',
                            'pipeline.order = array<i64: 0, 1, 2, 3>')
    assert lower(inside, '--frisk-pipeline-schedule', '--verify-each').count('frisk.alloc_buffer') == 2
    print('PASS explicit guarded task yields, iteration carriers and completion-only storage rotation')

    # No data reader keeps A live: its unfinished task alone must prevent
    # A/B from sharing storage while B's write is issued.
    lifetimes = '''func.func @lifetime(%src: memref<128xi32, 1>) {
      %a = memref.alloc() : memref<128xi32, 3>
      %b = memref.alloc() : memref<128xi32, 3>
      %ta = frisk.async_copy_gm_to_shm %src to %a : memref<128xi32, 1>, memref<128xi32, 3> -> !frisk.async_task
      %tb = frisk.async_copy_gm_to_shm %src to %b : memref<128xi32, 1>, memref<128xi32, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %ta : !frisk.async_task
      frisk.wait_buffer_ready %tb : !frisk.async_task
      return
    }'''
    out = lower(lifetimes, '--test-shm-reuse', '--verify-each')
    assert 'memref<512xi8, 3>' not in out, out
    assert out.count('memref.alloc()') == 2 or 'memref<1024xi8, 3>' in out
    print('PASS task completion extends destination storage lifetime without readers')


if __name__ == '__main__':
    main()
