#!/usr/bin/env python3
"""Execution contracts, precise effects, CSE barriers and immediate LLVM export."""
from pathlib import Path
import subprocess

from check_pv_interleave import fixture, ids

BIN = Path(__file__).resolve().parents[2] / 'build/test'


def run(tool, source, *args, ok=True):
    p = subprocess.run([str(BIN / tool), *args], input=source, text=True,
                       capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def main():
    source = '''module {
func.func @semantics(%a: vector<4xf16>, %b: vector<4xf16>, %c: vector<4xf32>,
                    %sm: memref<4xf32, 3>, %gm: memref<4xf32, 1>, %i: index) {
  %v = frisk.add %c, %c {test.id = "vector"} : vector<4xf32>, vector<4xf32> -> vector<4xf32>
  %m = frisk.add %sm, %c {test.id = "mixed"} : memref<4xf32, 3>, vector<4xf32> -> vector<4xf32>
  %mma = frisk.warp_mma_rr %a, %b, %c {test.id = "mma",
    inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", inst_constraints = "=v,v,v,0"}
    : vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>
  %f = frisk.pack_frag %c {test.id = "pack"} : vector<4xf32> -> !frisk.fragment<vector<4xf32>, {}>
  %local = frisk.reduce_frag "add" %f {test.id = "local_reduce", dim = 0 : i64,
    lane_extent = 1 : i64, lane_stride = 1 : i64, warp_threads = 64 : i64,
    reduction_width = 4 : i64, output_fragment_shape = array<i64: 1>}
    : !frisk.fragment<vector<4xf32>, {}> -> !frisk.fragment<vector<1xf32>, {}>
  %wave = frisk.reduce_frag "add" %f {test.id = "wave_reduce", dim = 0 : i64,
    lane_extent = 64 : i64, lane_stride = 1 : i64, warp_threads = 64 : i64,
    reduction_width = 4 : i64, output_fragment_shape = array<i64: 1>}
    : !frisk.fragment<vector<4xf32>, {}> -> !frisk.fragment<vector<1xf32>, {}>
  %unknown = llvm.inline_asm has_side_effects {test.id = "unknown", frisk.register_only = true}
    "", "=v,0" %c : (vector<4xf32>) -> vector<4xf32>
  %l = frisk.load_frag %sm[%i] {test.id = "load", tile_type = vector<4xf32>}
    : memref<4xf32, 3> -> !frisk.fragment<vector<4xf32>, {}>
  frisk.store_frag %f, %sm[%i] {test.id = "store", tile_type = vector<4xf32>}
    : !frisk.fragment<vector<4xf32>, {}>, memref<4xf32, 3>
  %task = frisk.async_copy_gm_to_shm %gm to %sm {test.id = "issue"}
    : memref<4xf32, 1>, memref<4xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %task {test.id = "wait"} : !frisk.async_task
  frisk.sync_threads_in_block {test.id = "barrier"}
  frisk.sched_barrier {mask = 6 : i32, test.id = "sched_barrier"}
  return
}}
'''
    report = run('FriskSemanticsTest', source, '-')
    rows = {line.split()[0]: line for line in report.splitlines()}
    assert 'registers=1' in rows['vector'] and 'memory_free=1' in rows['vector']
    assert 'read@0' in rows['mixed'] and 'read@unknown' not in rows['mixed']
    assert 'registers=0' in rows['mixed'] and 'memory_free=0' in rows['mixed']
    for key in ('mma', 'issue', 'wait', 'barrier'):
        assert 'collective=1' in rows[key] and 'boundary=1' in rows[key]
        assert 'speculatable=0' in rows[key]
    assert 'scope=2' in rows['mma'] and 'memory_free=1' in rows['mma']
    assert 'speculatable=1' in rows['pack'] and 'boundary=0' in rows['pack']
    assert 'scope=1' in rows['local_reduce'] and 'collective=0' in rows['local_reduce']
    assert 'scope=2' in rows['wave_reduce'] and 'collective=1' in rows['wave_reduce']
    assert 'boundary=1' in rows['wave_reduce'] and 'speculatable=0' in rows['wave_reduce']
    assert 'registers=0' in rows['unknown'] and 'boundary=1' in rows['unknown']
    assert 'read@0' in rows['load'] and 'write@1' in rows['store']
    for contract in ('scope=1', 'collective=0', 'boundary=1', 'memory_free=0', 'speculatable=0'):
        assert contract in rows['sched_barrier']

    # A write to the operand buffer invalidates reuse of the computed read.
    cse = '''module { func.func @reuse(%v: vector<4xf32>, %sm: memref<4xf32, 3>, %x: f32, %i: index)
        -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
      %a = frisk.add %v, %v : vector<4xf32>, vector<4xf32> -> vector<4xf32>
      %b = frisk.add %v, %v : vector<4xf32>, vector<4xf32> -> vector<4xf32>
      %c = frisk.add %sm, %v : memref<4xf32, 3>, vector<4xf32> -> vector<4xf32>
      memref.store %x, %sm[%i] : memref<4xf32, 3>
      %d = frisk.add %sm, %v : memref<4xf32, 3>, vector<4xf32> -> vector<4xf32>
      return %a, %b, %c, %d : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
    }}'''
    out = run('ThreadTilingTest', cse, '--cse', '--verify-each')
    assert out.count('frisk.add') == 3 and 'memref.store' in out, out

    # Both forms have identical MMA ordering and retain the real barrier's ordinal.
    for sideeffects in (True, False):
        src = fixture()
        if not sideeffects:
            src = src.replace('has_side_effects ', '')
        out = run('ThreadTilingTest', src, '--frisk-thread-interleave-pv=width=4', '--verify-each')
        assert ids(out) == [f'{c}:{k}' for k in range(3) for c in range(4)]
        assert len(ids(out.split('gpu.barrier')[0])) == 1
        assert out.count('frisk.sched_barrier') == 24
    shaped = fixture(warp=True).replace('vector<4xf32>', 'vector<1x4xf32>')
    shaped_out = run('ThreadTilingTest', shaped, '--frisk-thread-interleave-pv=width=4')
    assert ids(shaped_out) == [f'{c}:{k}' for k in range(3) for c in range(4)]

    # Export must supply attributes before cleanupRegisterMMA or LLVM optimization.
    llvm_mlir = '''module { llvm.func @mma(%a: vector<4xf16>, %b: vector<4xf16>, %c: vector<4xf32>) -> vector<4xf32> {
      %r = llvm.inline_asm has_side_effects convergent {frisk.register_mma}
        "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0" %a, %b, %c
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      llvm.return %r : vector<4xf32>
    }}'''
    ir = run('FriskSemanticsTest', llvm_mlir, '-', '--llvm')
    assert 'asm sideeffect' not in ir
    for attr in ('convergent', 'nounwind', 'willreturn', 'memory(none)'):
        assert attr in ir, ir
    assert 'speculatable' not in ir
    bad = llvm_mlir.replace('v_mmac_f32_16x16x16_f16 $0, $1, $2, $3', 's_barrier')
    assert 'invalid frisk.register_mma contract' in run('FriskSemanticsTest', bad, '-', '--llvm', ok=False)

    # Test the real producer as well as the hook, retaining the MLIR anchor.
    from check_hcu_mmac import fixture as mma_fixture
    lowered = run('ThreadTilingTest', mma_fixture(enabled='false'),
                  '--thread-ir-legalize', '--test-async-to-llvm')
    assert lowered.count('frisk.register_mma') == 2
    assert lowered.count('llvm.inline_asm has_side_effects') == 2
    exported = run('FriskSemanticsTest', lowered, '-', '--llvm')
    assert exported.count('asm ') == 2 and 'asm sideeffect' not in exported
    print('PASS execution scopes/effects, collective boundaries, CSE memory dependence, '
          'MMA scheduling with/without side effects, immediate LLVM attributes and invalid marker rejection')


if __name__ == '__main__':
    main()
