#!/usr/bin/env python3
"""Early V stream scheduling contracts, ownership, rejection and idempotence."""
from pathlib import Path
import re
import subprocess

TOOL = Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest'
PLAN = 'frisk.shared_pipeline_plan'


def fixture(stages=4):
    lines = ['module { func.func @kernel(%gm: memref<128x32xf16, 1>, '
             '%other: memref<128x32xf16, 1>, %a: memref<32x32xf16, 3>, '
             '%acc: memref<32x32xf32>) attributes {thread_num = 64 : i32} {',
             '%zero = arith.constant 0 : index', '%tid = gpu.thread_id x',
             'affine.for %bn = 0 to 4 {']
    for i in range(stages):
        lines += [
            f'%src{i} = frisk.buffer_view %gm[{i * 32}, 0], ranges = [32, 32] : '
            'memref<128x32xf16, 1> -> memref<32x32xf16, 1>',
            f'%buf{i} = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> memref<32x32xf16, 3>',
            f'%task{i} = frisk.async_copy_gm_to_shm %src{i} to %buf{i} {{alignment = 16 : i64}} : '
            'memref<32x32xf16, 1>, memref<32x32xf16, 3> -> !frisk.async_task',
            f'frisk.wait_buffer_ready %task{i} : !frisk.async_task',
            f'%pv{i} = frisk.gemm(%a, %buf{i}, %acc) {{transA = false, transB = false}} : '
            'memref<32x32xf16, 3>, memref<32x32xf16, 3>, memref<32x32xf32> -> memref<32x32xf32>',
        ]
    lines += ['}', 'return', '} }']
    return '\n'.join(lines)


def run(source, enabled=True):
    result = subprocess.run([str(TOOL), '-', '--allow-unregistered-dialect',
                             '--frisk-auto-pipeline=double-buffer=' + str(enabled).lower(),
                             '--verify-each'], input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    count = 0
    def check(source, accepted):
        nonlocal count
        out = run(source)
        contracts = re.findall(PLAN + r' = array<i64: ([^>]+)>', out)
        assert len(contracts) == accepted, out
        assert run(out) == out, 'planning is not idempotent'
        serial = re.sub(', ' + PLAN + r' = array<i64: [^>]+>', '', out)
        assert serial == run(source, False), 'provisional plan changed serial fallback'
        for i, contract in enumerate(contracts):
            assert list(map(int, contract.split(', '))) == [i, accepted, i % 2, i - 1]
        count += 1

    for stages in (2, 3, 4):
        check(fixture(stages), stages)
    base = fixture()
    check(fixture(1), 0)
    check(base.replace('to 4 {', 'to %tid {'), 0)
    check(base.replace('}\nreturn', '} {pipeline.scheduled}\nreturn'), 0)
    check(base.replace('%gm[64, 0]', '%other[64, 0]'), 0)
    check(base.replace('%src1 =', '"test.unknown"() : () -> ()\n%src1 ='), 0)
    check(base.replace('%src1 =', 'frisk.sync_threads_in_block\n%src1 ='), 0)
    # An extra reader makes that slice ineligible. The unknown access between
    # the remaining stages must also prevent a partial plan.
    check(base.replace('%src1 =', '%extra = memref.load %buf0[%zero, %zero] : '
                       'memref<32x32xf16, 3>\n%src1 ='), 0)
    check(base.replace('frisk.wait_buffer_ready %task1 : !frisk.async_task', ''), 0)
    check(base.replace('%src1 =', 'frisk.wait_buffer_ready %task0 : !frisk.async_task\n%src1 ='), 0)
    print(f'PASS {count} AutoPipeline stream cases: early contracts, ownership, fallback and idempotence')


if __name__ == '__main__':
    main()
