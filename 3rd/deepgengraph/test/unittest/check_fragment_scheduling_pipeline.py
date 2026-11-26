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


if __name__ == '__main__':
    main()
