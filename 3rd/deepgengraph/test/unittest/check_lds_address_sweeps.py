#!/usr/bin/env python3
"""Exercise the production driver when the LDS anchor budget needs a second sweep."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile


TOOL = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
        Path(__file__).resolve().parents[2] / 'build/test/MyTest')

# Each independent index consumes an anchor on the aligned global. Only after
# those loads become frisk.lds_load can the two trailing loads share an anchor.
indices = ', '.join(f'%i{i}: index' for i in range(128))
body = ['%pool = memref.get_global @pool : memref<4096xf32, 3>',
        '%sum0 = arith.constant 0.0 : f32']
for i in range(128):
    body += [f'%v{i} = memref.load %pool[%i{i}] : memref<4096xf32, 3>',
             f'%sum{i + 1} = arith.addf %sum{i}, %v{i} : f32']
aligned_reads = '\n    '.join(body)
source = f'''module {{
  memref.global "private" @pool : memref<4096xf32, 3> = uninitialized alignment = 16
  func.func @residual(%s: memref<4096xf32, 3>, %out: memref<f32, 1>, {indices})
      attributes {{thread_num = 64 : i32}} {{
    {aligned_reads}
    gpu.barrier
    %one = arith.constant 1 : index
    %next = arith.addi %i0, %one : index
    %a = memref.load %s[%i0] : memref<4096xf32, 3>
    %b = memref.load %s[%next] : memref<4096xf32, 3>
    %first = arith.addf %sum128, %a : f32
    %sum = arith.addf %first, %b : f32
    memref.store %sum, %out[] : memref<f32, 1>
    return
  }}
}}'''

prefetch = '--k-lds-prefetch=1'
qk = '--qk-interleave=2'
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'probe.mlir'
    path.write_text(source)
    for options in ([prefetch], [qk], [prefetch, qk]):
        run = subprocess.run(
            [str(TOOL), str(path), '--reorder=0', '--llvm-opt=0',
             f'--output={directory}/probe.ll', *options],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
            env={**os.environ, 'FRISK_GM_TO_SHM_MODE': 'legacy'})
        assert run.returncode == 0, run.stdout[-3000:]
        stage = 'frisk-thread-interleave-qk' if qk in options else 'thread finalizeFragmentLDSPrefetch'
        snapshot = run.stdout.split(f'---- after {stage} -----\n', 1)[1].split('\n}', 1)[0]
        assert snapshot.count('frisk.lds_load') >= 128, options
        if prefetch in options and qk in options:
            assert 'byte_offset = 4 : i64' in snapshot, 'QK lost the remaining address proof'

print('PASS: prefetch and QK each get their LDS proof, including a residual second sweep')
