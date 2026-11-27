#!/usr/bin/env python3
"""Check the two ordered ThreadIR reductions and their independent options."""
from pathlib import Path
import subprocess
import sys


TOOL = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
        Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest')
# Tag the update as well as the local chain. Before lane deferral the shuffle
# separates them. Reversing the stages would reassociate the loop carry too.
FUNCTION = '''
func.func @reduce(%x: f32, %y: f32) -> f32 attributes {thread_num = 64 : i32} {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1 : i32
  %wave = arith.constant 64 : i32
  %result = affine.for %i = 0 to 8 iter_args(%acc = %zero) -> f32 {
    %a = arith.addf %x, %y {frisk.try_multichain} : f32
    %b = arith.addf %a, %y {frisk.try_multichain} : f32
    %c = arith.addf %b, %y {frisk.try_multichain} : f32
    %d = arith.addf %c, %y {frisk.try_multichain} : f32
    %peer, %valid = gpu.shuffle xor %d, %one, %wave : f32
    %sum = arith.addf %d, %peer : f32
    %next = arith.addf %acc, %sum {frisk.try_multichain} : f32
    affine.yield %next : f32
  }
  return %result : f32
}
'''
# Allow MLIR to clone and run nested function passes over multiple functions.
SOURCE = FUNCTION + FUNCTION.replace('@reduce', '@second')


def run(*passes, source=SOURCE):
    result = subprocess.run([str(TOOL), '--verify-each', *passes],
                            input=source, text=True, capture_output=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def check(local, lane, ops='add', wave=64, source=SOURCE, *, fastmath=False, integer=False):
    local_options = f'chains=2 min-inputs=4 ops={ops}'
    stages = []
    if local != 'off':
        stages.append('--frisk-thread-local-reduction=' +
                      f'approximate={str(local == "approx").lower()} {local_options}')
    if lane != 'off':
        stages.append('--frisk-thread-defer-lane-sum=' +
                      f'approximate={str(lane == "approx").lower()} wave-size={wave}')
    expected = run(*stages, source=source)
    marked = not integer and ops == 'add' and (local == 'approx' or local == 'legal' and fastmath)
    assert expected.count('frisk.local_reduction') == (8 if marked else 0), expected
    deferred = wave == 64 and (lane == 'approx' or lane == 'legal' and integer)
    assert expected.count('gpu.shuffle') == 2
    for body in expected.split('func.func ')[1:]:
        assert ('gpu.shuffle' in body.split('affine.yield')[0]) != deferred, body
    if marked and deferred:
        assert expected != run(*reversed(stages), source=source), 'fixture must distinguish stage order'
    combined = ('--frisk-thread-reductions=' +
                f'local-reduce={local} defer-lane-sum={lane} {local_options} wave-size={wave}')
    assert run(combined, source=source) == expected, combined
    return combined


if __name__ == '__main__':
    assert run('--frisk-thread-reductions') == run()
    for local in ('off', 'legal', 'approx'):
        for lane in ('off', 'legal', 'approx'):
            check(local, lane)
    check('approx', 'approx', ops='max')
    check('approx', 'approx', wave=32)
    check('legal', 'legal', fastmath=True,
          source=SOURCE.replace('{frisk.try_multichain}', 'fastmath<reassoc,nsz> {frisk.try_multichain}'))
    check('legal', 'legal', integer=True,
          source=SOURCE.replace('f32', 'i32').replace('0.0', '0').replace('arith.addf', 'arith.addi'))
    enabled = check('approx', 'approx')
    protected = SOURCE.replace('thread_num =', 'llvm.strictfp, thread_num =')
    assert run(enabled, source=protected) == run(source=protected)
    print('PASS: ordered local/lane reductions, independent modes, thresholds, operation filters, wave size and strict FP')
