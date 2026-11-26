#!/usr/bin/env python3
"""The unanchored stream head overlaps prior compute; waits/interior stay fixed."""
from pathlib import Path
import subprocess

TOOL = Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest'
PASS = '--frisk-schedule-async-copies'


def fixture(fragment=False, boundary='', schedule=''):
    def read(name, buffer):
        if fragment:
            return (f'%{name} = frisk.load_frag %{buffer}[%zero] {{tile_type = vector<64xf32>}} : '
                    'memref<64xf32, 3> -> '
                    '!frisk.fragment<vector<2xf32>, {}>')
        return f'%{name} = memref.load %{buffer}[%zero] : memref<64xf32, 3>'
    return f'''module {{
func.func @test(%src: memref<64xf32, 1>, %x: f32) -> f32
    attributes {{thread_num = 64 : i32{schedule}}} {{
  %zero = arith.constant 0 : index
  %a = memref.alloc() : memref<64xf32, 3>
  %b = memref.alloc() : memref<64xf32, 3>
  %e = math.exp2 %x : f32
  {boundary}
  %t0 = frisk.async_copy_gm_to_shm %src to %a
    {{frisk.shared_pipeline_plan = array<i64: 0, 2, 0, -1>}} :
    memref<64xf32, 1>, memref<64xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %t0 : !frisk.async_task
  {read('r0', 'a')}
  %sum = arith.addf %e, %x : f32
  %t1 = frisk.async_copy_gm_to_shm %src to %b
    {{frisk.shared_pipeline_plan = array<i64: 1, 2, 1, 0>}} :
    memref<64xf32, 1>, memref<64xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %t1 : !frisk.async_task
  {read('r1', 'b')}
  return %sum : f32
}}
}}'''


def run(source, scheduled=True):
    args = [str(TOOL), '-', '--allow-unregistered-dialect', '--verify-each']
    if scheduled:
        args.append(PASS)
    result = subprocess.run(args, input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    cases = 0
    for fragment in (False, True):
        source = fixture(fragment)
        result = run(source)
        first = result.index('async_copy_gm_to_shm')
        compute = result.index('math.exp2')
        ready = result.index('wait_buffer_ready')
        reader = result.index('frisk.load_frag' if fragment else 'memref.load')
        second = result.rindex('async_copy_gm_to_shm')
        assert first < compute < ready < reader < result.index('arith.addf') < second, result
        assert run(result) == result, 'head hoisting must be idempotent'
        cases += 1
        for boundary in ('gpu.barrier', '"test.unknown"() : () -> ()',
                         'memref.store %x, %src[%zero] : memref<64xf32, 1>',
                         'memref.store %x, %a[%zero] : memref<64xf32, 3>',
                         '%old = memref.load %a[%zero] : memref<64xf32, 3>'):
            out = run(fixture(fragment, boundary))
            assert out.index('math.exp2') < out.index('async_copy_gm_to_shm'), out
            cases += 1
        for schedule in (', pipeline.scheduled', ', pipeline.stage = 0 : i64'):
            source = fixture(fragment, schedule=schedule)
            assert run(source) == run(source, False)
            cases += 1
        for invalid in ('0, 2, 1, -1', '0, 2, 0, 0', '0, 0, 0, -1', '0, 2, 0'):
            source = fixture(fragment).replace('0, 2, 0, -1', invalid)
            assert run(source) == run(source, False), 'malformed plan must not grant head motion'
            cases += 1
    print(f'PASS {cases} stream-head cases: compute overlap, fixed waits/interior, aliases, effects and explicit schedules')


if __name__ == '__main__':
    main()
