#!/usr/bin/env python3
"""Joint execution planning: projected lifetimes, slots, task order and fallback."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def fixture(threads=64, stages=3, rounds=1, reserve=True):
    size = threads * 16 * rounds
    elements = size // 2
    ty = f'memref<{elements}xf16, 3>'
    gm = f'memref<{stages}x{elements}xf16, 1>'
    lines = [f'module {{ func.func @kernel(%gm: {gm}, %out: memref<4xi32, 1>) '
             f'attributes {{thread_num = {threads} : i32}} {{',
             '%c0 = arith.constant 0 : index', '%tid = gpu.thread_id x',
             '%idx = affine.apply affine_map<(d0) -> (d0 * 8)>(%tid)']
    # Earlier overlapping users establish the original 2-slot pool budget.
    # Their storage dies before the stream, as Q/K storage does in attention.
    if reserve:
        lines += [f'%init = memref.subview %gm[0, 0] [1, {elements}] [1, 1] : {gm} '
                  f'to memref<{elements}xf16, strided<[1], offset: 0>, 1>']
        for n in (0, 1):
            lines += [f'%warm{n} = memref.alloc() alignment = 16 : {ty}',
                      f'%warmtask{n} = frisk.async_copy_gm_to_shm %init to %warm{n} '
                      f'{{alignment = 16 : i64, frisk.dma_bytes = 16 : i64}} : '
                      f'memref<{elements}xf16, strided<[1], offset: 0>, 1>, {ty} -> !frisk.async_task',
                      f'frisk.wait_buffer_ready %warmtask{n} : !frisk.async_task']
        for n in (0, 1):
            lines += [f'%old{n} = memref.load %warm{n}[%idx] : {ty}']
        lines += ['gpu.barrier']
    lines += ['affine.for %iv = 0 to 4 {']
    for i in range(stages):
        source = f'memref<{elements}xf16, strided<[1], offset: {i * elements}>, 1>'
        lines += [f'%slot{i} = memref.alloc() alignment = 16 : {ty}',
                  f'%src{i} = memref.subview %gm[{i}, 0] [1, {elements}] [1, 1] : {gm} to {source}',
                  f'%task{i} = frisk.async_copy_gm_to_shm %src{i} to %slot{i} '
                      f'{{alignment = 16 : i64, frisk.dma_bytes = 16 : i64, '
                      f'frisk.shared_pipeline_plan = array<i64: {i}, {stages}, {i % 2}, {i - 1}>}} : {source}, {ty} -> !frisk.async_task',
                  f'frisk.wait_buffer_ready %task{i} : !frisk.async_task',
                  f'%read{i} = frisk.read_matrix_from_shm %slot{i}[%idx] '
                  f'{{m = 32 : i64, n = 16 : i64, element_bit_width = 16 : i64}} : {ty} -> vector<4xi32>']
    lines += ['gpu.barrier', f'vector.store %read{stages-1}, %out[%c0] : memref<4xi32, 1>, vector<4xi32>',
              '}', 'return', '} }']
    return '\n'.join(lines) + '\n'


def simulate(text, stages, size):
    """Conservative async/block oracle using the emitted views and task tokens.

    An issue reserves its bytes until wait; a matrix read protects its entire
    slice until a collective wait/barrier. Repeat the stream across a backedge.
    Distinct source labels detect a wrong slot even if addresses stay in bounds.
    """
    constants, views, sources = {}, {}, {}
    before, body = text.split('affine.for', 1)
    for line in text.splitlines():
        if m := re.search(r'(%\w+) = arith.constant (\d+) : index', line):
            constants[m[1]] = int(m[2])
        if m := re.search(r'(%\w+) = memref.view %\w+\[(%\w+)\]\[\]', line):
            views[m[1]] = constants[m[2]]
        if m := re.search(r'(%\w+) = memref.subview %\w+\[(\d+), 0\]', line):
            sources[m[1]] = int(m[2])
    capacity = int(re.search(r'memref<(\d+)xi8, 3>', text)[1])
    memory = [None] * capacity
    pending, readers = {}, []
    def overlaps(a, b):
        return a < b + size and b < a + size
    def execute(lines, expected_reads):
        reads = 0
        for line in lines.splitlines():
            if m := re.search(r'(%\w+) = frisk.async_copy_gm_to_shm (%\w+) to (%\w+)', line):
                offset = views[m[3]]
                assert 0 <= offset <= capacity - size
                assert not any(overlaps(offset, r) for r in readers), 'overwrite before reader handoff'
                assert not any(overlaps(offset, p[0]) for p in pending.values()), 'overlapping in-flight writes'
                pending[m[1]] = (offset, sources[m[2]])
            elif m := re.search(r'frisk.wait_buffer_ready (%\w+)', line):
                offset, value = pending.pop(m[1])
                memory[offset:offset+size] = [value] * size
                readers.clear()
            elif m := re.search(r'frisk.read_matrix_from_shm (%\w+)\[', line):
                offset = views[m[1]]
                assert not any(overlaps(offset, p[0]) for p in pending.values()), 'read before DMA completion'
                assert memory[offset:offset+size] == [reads] * size, 'wrong source/slot contents'
                readers.append(offset)
                reads += 1
            elif 'gpu.barrier' in line:
                readers.clear()
        assert reads == expected_reads and not pending
    execute(before, 0)
    for _ in range(3):
        execute(body, stages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=Path, default=Path(__file__).resolve().parents[2]/'build/test/ThreadTilingTest')
    args = parser.parse_args()
    count = 0
    with tempfile.TemporaryDirectory(prefix='shared-execution-') as tmp:
        path = Path(tmp)/'input.mlir'
        def run(text, enabled=True):
            path.write_text(text)
            command = [str(args.optimizer), str(path),
                       '--lower-frisk-fragments=double-buffer=' + str(enabled).lower(), '--verify-each']
            result = subprocess.run(command, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            return result.stdout, result.stderr
        def check(text, accepted=True, stages=3, size=1024):
            nonlocal count
            result, log = run(text)
            assert ('Shared memory pipeline: stages=' in log) == accepted, log
            if accepted:
                assert 'frisk.shared_pipeline' in result
                assert f'slot-bytes={size}' in log and f'LDS={size*2}' in log, log
                simulate(result, stages, size)
                again, _ = run(result)
                assert again == result, 'not idempotent'
            else:
                baseline, _ = run(text, False)
                assert result == baseline, 'rejected schedule changed ordinary allocation/lowering'
            count += 1
        for threads, stages, rounds in [(64,2,1),(128,3,1),(256,4,2),(512,5,1),(1024,2,1)]:
            check(fixture(threads, stages, rounds), stages=stages, size=threads*16*rounds)
        base = fixture()
        unit = args.optimizer.parent/'SharedExecutionPlanTest'
        path.write_text(base)
        result = subprocess.run([str(unit), str(path)], text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout.strip())
        for width in (4, 8):
            check(base.replace('frisk.dma_bytes = 16', f'frisk.dma_bytes = {width}'))
        check(fixture(reserve=False), False)
        check(fixture(stages=1), False)
        check(re.sub(r', frisk.shared_pipeline_plan = array<i64: [^>]*>', '', base), False)
        check(base.replace('array<i64: 1, 3, 1, 0>', 'array<i64: 1, 3, 0, 0>'), False)
        check(base.replace('array<i64: 2, 3, 0, 1>', 'array<i64: 2, 3, 0, 0>'), False)
        check(base.replace('array<i64: 1, 3, 1, 0>', 'array<i64: 1, 4, 1, 0>'), False)
        check(base.replace(', frisk.shared_pipeline_plan = array<i64: 1, 3, 1, 0>', ''), False)
        check(base.replace('d0 * 8', 'd0 * 8 + 512'), False)
        check(base.replace('frisk.dma_bytes = 16', 'frisk.dma_bytes = 3'), False)
        check(base.replace('alignment = 16 : i64', 'alignment = 4 : i64'), False)
        check(base.replace('frisk.dma_bytes = 16 : i64', 'frisk.dma_bytes = 16 : i64, frisk.force_sync'), False)
        check(base.replace('affine.for %iv = 0 to 4', 'affine.for %iv = 0 to %tid'), False)
        check(base.replace('%slot1 =', 'gpu.barrier\n%slot1 ='), False)
        check(base.replace('func.func @kernel', 'func.func private @unknown()\nfunc.func @kernel').replace(
            '%read0 =', 'func.call @unknown() : () -> ()\n%read0 ='), False)
        check(base.replace('%read0 =', '%extra = memref.load %slot0[%idx] : memref<512xf16, 3>\n%read0 ='), False)
        check(base.replace('}\nreturn', '} {pipeline.scheduled}\nreturn'), False)
        check(base.replace('%src1 =', '%indirect = memref.load %out[%c0] : memref<4xi32, 1>\n'
                           '%address = arith.index_cast %indirect : i32 to index\n%src1 =').replace(
                           '%gm[1, 0]', '%gm[%address, 0]').replace('offset: 512>', 'offset: ?>'), False)
    print(f'PASS {count} joint execution cases: slots, async ownership, backedges, budget, rejection and idempotence')


if __name__ == '__main__':
    main()
