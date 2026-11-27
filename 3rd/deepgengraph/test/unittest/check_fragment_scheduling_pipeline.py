#!/usr/bin/env python3
"""Check the unified pipeline against the original scheduling sequence."""
from itertools import product
from pathlib import Path
import subprocess
import sys

from check_fragment_level_passes import FUSION, PREFETCH, RETAIN, MEMORY_DEPENDENCY
from check_fragment_async_liveness import FRAGMENTS


TOOL = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
        Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest')
CLEANUP = 'canonicalize{disable-patterns=FromElementsToShapeCast},cse'
# Keep the independent compute live through canonicalization so its order
# relative to DMA still distinguishes scheduling from CP-SAT preparation.
DMA = FRAGMENTS.replace(' -> !f\n    attributes', ' -> (!f, f32)\n    attributes').replace(
    'return %r : !f', 'return %r, %e : !f, f32')

# The mask consumes a loop-carried score; its branch must stay inside the
# loop while the immutable operand's DMA/wait/load can still be retained.
MASK = r'''
!f = !frisk.fragment<vector<1x2xf32>, {}>
#layout = {tile_layout = affine_map<(t, b0, b1, w0, w1, r0, r1) -> (t, r1)>,
  warp_threads = 64 : i64, ignore_dim = -1 : i64,
  thread_widths = array<i64: 1, 2>, warp_repeat = array<i64: 1, 1>,
  warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 64, 1>,
  block_layout = array<i64: 1, 1>, block_repeat = array<i64: 1, 1>,
  thread_creg_order = array<i64: 0, 1>, warp_repeat_order = array<i64: 0, 1>,
  warp_layout_order = array<i64: 0, 1>, block_layout_order = array<i64: 0, 1>}
func.func @masked_recurrence(%src: memref<64x2xf32, 1>, %row: index) -> !f
    attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %inf = arith.constant 0xFF800000 : f32
  %z = arith.constant dense<0.0> : vector<1x2xf32>
  %initial = frisk.pack_frag %z : vector<1x2xf32> -> !f
  %out = affine.for %i = 0 to 4 iter_args(%acc = %initial) -> !f {
    %buf = memref.alloc() : memref<64x2xf32, 3>
    %task = frisk.async_copy_gm_to_shm %src to %buf {alignment = 16 : i64}
      : memref<64x2xf32, 1>, memref<64x2xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %v = frisk.load_frag %buf[%c0, %c0] {tile_type = vector<1x2xf32>,
      tile_layout = affine_map<(t, b0, b1, w0, w1, r0, r1) -> (t, r1)>,
      thread_widths = array<i64: 1, 2>, warp_repeat = array<i64: 1, 1>,
      warp_inst_unroll = array<i64: 1, 1>, ignore_dim = -1 : i64,
      thread_creg_order = array<i64: 0, 1>, warp_repeat_order = array<i64: 0, 1>}
      : memref<64x2xf32, 3> -> !f
    %score = frisk.elementwise_frag "add" %acc, %v : (!f, !f) -> !f
    %masked = "frisk.mask_frag"(%row, %i, %score) <{
      block_type = memref<64x2xf32>, fragment_shape = array<i64: 1, 2>,
      layout = #layout, uniform = true, mask_on_lhs = false,
      operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%r: index, %c: index):
      %next = arith.addi %r, %c1 : index
      %future = arith.cmpi ule, %next, %c : index
      %value = arith.select %future, %inf, %zero : f32
      frisk.mask_yield %value : f32
    }) : (index, index, !f) -> !f
    affine.yield %masked : !f
  }
  return %out : !f
}
'''


def run(source, pipeline):
    result = subprocess.run(
        [str(TOOL), '-', '--verify-each',
         f'--pass-pipeline=builtin.module({pipeline})'],
        input=source, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    cases = 0
    for scheduling, simplify, words in product((False, True), (False, True), (0, 32)):
        passes = []
        if scheduling:
            passes.append('func.func(frisk-frag-ir-reorder)')
        passes.append(CLEANUP)
        if simplify:
            passes.extend(('func.func(frisk-simplify-loop-control)', 'cse'))
        if words:
            passes.append(f'func.func(frisk-retain-invariant-fragments{{max-words={words}}})')
        if scheduling:
            passes.append('func.func(frisk-schedule-async-copies)')
        options = (f'heuristic-scheduling={str(scheduling).lower()} '
                   f'simplify-loop-control={str(simplify).lower()} '
                   f'retain-fragment-words={words}')
        unified = f'frisk-fragment-scheduling{{{options}}}'
        for source in (FUSION, PREFETCH, RETAIN, MEMORY_DEPENDENCY, DMA):
            assert run(source, unified) == run(source, ','.join(passes)), options
            cases += 1

    # CP-SAT preparation must keep retention but skip lookahead and DMA motion.
    cpsat = ('frisk-fragment-scheduling{heuristic-scheduling=false '
             'retain-fragment-words=32}')
    assert 'frisk.pipelined' not in run(PREFETCH, cpsat)
    retained = run(RETAIN, cpsat)
    assert 'frisk.retained_words' in retained
    assert retained.index('async_copy_gm_to_shm') < retained.index('affine.for')
    dma = run(DMA, cpsat)
    assert dma.index('math.exp2') < dma.index('async_copy_gm_to_shm')
    assert 'frisk.pipelined' in run(PREFETCH, 'frisk-fragment-scheduling')
    dma = run(DMA, 'frisk-fragment-scheduling')
    assert dma.index('async_copy_gm_to_shm') < dma.index('math.exp2')
    print(f'PASS: {cases} pipeline equivalence cases, defaults and CP-SAT preparation')

    for scheduling in (False, True):
        options = (f'heuristic-scheduling={str(scheduling).lower()} '
                   'retain-fragment-words=2')
        base = f'frisk-fragment-scheduling{{{options}}}'
        early = f'frisk-fragment-scheduling{{{options} bypass-mask-adds=true}}'
        result = run(MASK, early)
        assert result == run(MASK, 'func.func(frisk-bypass-fragment-mask-adds),' + base)
        assert 'frisk.mask_add_bypass' in result
        assert 'frisk.mask_add_bypass' not in run(MASK, base)
        before, loop = result.split('affine.for', 1)
        assert 'async_copy_gm_to_shm' in before and 'wait_buffer_ready' in before
        assert 'load_frag' in before and 'load_frag' not in loop
        assert 'scf.if' in loop and 'frisk.retained_words = 2 : i32' in loop
        assert loop.index('elementwise_frag "add"') < loop.index('scf.if')
        assert 'frisk.mask_add_bypass' not in run(
            MASK.replace('attributes {thread_num', 'attributes {strictfp, thread_num'), early)
        # Enabling bypass must add no cleanup when there is no eligible mask.
        # In particular, do not CSE shared operand/broadcast carriers before
        # the scheduler has selected its fusion and retention boundaries.
        for source in (FUSION, PREFETCH, RETAIN, MEMORY_DEPENDENCY, DMA):
            assert run(source, early) == run(source, base)
    print('PASS: early mask bypass, score dependencies, DMA retention and FP guards')


if __name__ == '__main__':
    main()
