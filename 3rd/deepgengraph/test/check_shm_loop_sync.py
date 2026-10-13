#!/usr/bin/env python3
"""Check shared-buffer backedges, including the exported split-D Q race.

Usage: check_shm_loop_sync.py build/test/ThreadTilingTest [attention.ll]
"""
import re
import sys
from pathlib import Path

from check_shm_liveness import kernel, run
from check_direct_shared_copy import Trace


def fixtures():
    alloc = '%a = memref.alloc() : memref<128xf32, 3>'
    write = 'memref.store %x, %a[%c0] : memref<128xf32, 3>'
    read = '%v = memref.load %a[%c1] : memref<128xf32, 3>'
    for loop in ('scf.for %i = %c0 to %c4 step %c1',
                 'affine.for %i = 0 to 4'):
        for name, body, expected in [
            ('publication_only', f'{write} gpu.barrier {read}', 2),
            ('existing_tail', f'{write} gpu.barrier {read} gpu.barrier', 2),
            ('existing_prefix', f'gpu.barrier {write} gpu.barrier {read}', 2),
            ('read_only', read, 0), ('write_only', write, 0),
            ('nested', f'affine.for %j = 0 to 4 {{ {write} gpu.barrier {read} }}', 3),
            ('alias', f'''%alias = memref.subview %a[0] [128] [1]
                : memref<128xf32, 3> to memref<128xf32, strided<[1]>, 3>
                {write} gpu.barrier
                %v = memref.load %alias[%c1] : memref<128xf32, strided<[1]>, 3>''', 2),
        ]:
            source = kernel(f'{alloc} {loop} {{ {body} }} return %x : f32')
            ir, _ = run(source, 'test-shm-loop-sync')
            assert ir.count('gpu.barrier') == expected, (name, ir)
            again, _ = run(ir, 'test-shm-loop-sync')
            assert again == ir, name
            print('PASS', loop.split()[0], name)

    # Never introduce a collective barrier under a divergent loop/conditional.
    for scope in ('%tid = gpu.thread_id x\nscf.for %i = %c0 to %tid step %c1',
                  'scf.if %cond'):
        source = kernel(f'{alloc} {scope} {{ {write} {read} }} return %x : f32')
        ir, _ = run(source, 'test-shm-loop-sync')
        assert 'gpu.barrier' not in ir
    # Exercise the real integration even when no storage can be pooled.
    source = kernel(f'{alloc} affine.for %i = 0 to 4 {{ '
                    f'{write} gpu.barrier {read} }} return %x : f32')
    ir, _ = run(source, 'lower-frisk-fragments')
    assert ir.count('gpu.barrier') == 2, ir
    print('PASS divergent scopes and single-allocation lowering integration')


def exported(text):
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    common = {n: 0 for n, e in defs.items() if e.startswith('phi i64 ') or
              '@llvm.amdgcn.workgroup.id.' in e}
    # The QK loop carries the full 128x128 FP32 score tile.
    qk = next(e for e in defs.values() if e.startswith('phi [4 x <32 x float>]'))
    backedge = re.search(r'\[ %\w+, (%\w+) \]', qk)[1][1:]
    body = re.search(r'^' + backedge + r':.*?\n(.*?)(?=^\w+:)',
                     text, re.M | re.S)[1]
    write = re.search(r'store <32 x half> %\w+, ptr addrspace\(3\) (%\w+)', body)
    reads = list(re.finditer(r'load half, ptr addrspace\(3\) (%\w+)', body))
    assert write and reads
    writers = {}
    for thread in range(128):
        lane = Trace(defs, {**common, tid: thread})
        base, offset = lane.value(write[1])
        for j in range(32):
            writers[base, offset + 2*j] = thread
    cross_wave = 0
    for thread in range(128):
        lane = Trace(defs, {**common, tid: thread})
        for read in reads:
            address = lane.value(read[1])
            if address in writers and writers[address] // 64 != thread // 64:
                cross_wave += 1
    assert cross_wave == 2048, cross_wave
    barriers = [m.start() for m in re.finditer('@llvm.amdgcn.s.barrier()', body)]
    assert any(p < write.start() or p > reads[-1].end() for p in barriers), \
        'QK backedge can overwrite Q while another wave still reads it'
    print('PASS exported QK backedge: 2048 cross-wave Q accesses protected')


if __name__ == '__main__':
    fixtures()
    if len(sys.argv) > 2:
        exported(Path(sys.argv[2]).read_text())
