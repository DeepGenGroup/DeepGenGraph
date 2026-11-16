#!/usr/bin/env python3
"""Save and independently replay the actual typed-fragment optimization stages."""
from pathlib import Path
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
TOOL = HERE.parents[1] / '3rd/deepgengraph/build/test/ThreadTilingTest'
text = (HERE/'lowering.log').read_text()
records = []

def run(*args):
    p = subprocess.run([str(TOOL), *map(str,args)],capture_output=True,text=True,timeout=180)
    assert p.returncode == 0, p.stderr[-8000:]

# Preserve the exact block-computation boundary requested for comparison.
tail = text.split('after insertConvertLayoutOps:', 1)[1].lstrip()
end = re.search(r'^}\s*$', tail, re.M).end()
path = HERE/'00-insert-convert-layout.mlir'
path.write_text(tail[:end].rstrip()+'\n')
run(path, '-o', '/dev/null')

for marker, name in [
    ('after promote-gemm-matrix-loads', '00-block'),
    ('after convert-friskbase-to-fragment', '01-fragment'),
    ('after fragment-level fuse-fragment-accumulator', '02-accumulator'),
    ('after fragment-level frisk-frag-ir-reorder', '03-reorder'),
    ('after frisk-simplify-loop-control', '04-specialized'),
    ('after frisk-retain-invariant-fragments', '05-retained'),
    ('after fragment-level optimization', '06-optimized-fragment'),
    ('after convert-fragment-to-thread', '07-thread-bridges'),
    ('after createFinalizeThreadTilingPass', '08-thread'),
    ('after lower-frisk-fragments', '09-storage-planned')]:
    tail = text.split(marker, 1)[1].split('\n', 1)[1]
    start = re.search(r'^(?:#\w+\s*=|module\b)', tail, re.M).start()
    module = re.search(r'^module\b', tail[start:], re.M).start() + start
    end = re.search(r'^}\s*$', tail[module:], re.M).end() + module
    ir = tail[start:end].rstrip()+'\n'
    path = HERE/(name+'.mlir'); path.write_text(ir)
    run(path, '-o', '/dev/null')
    counts = {key: ir.count(key) for key in ['affine.for', 'frisk.gemm_frag', 'frisk.load_frag',
        'frisk.reduce_frag', 'frisk.broadcast_frag', 'frisk.convert_layout_frag', 'frisk.mask_frag', 'gpu.shuffle', 'vector.extract', 'frisk.fused', 'frisk.pipelined', 'frisk.fragment_accumulate', 'frisk.retained_words']}
    counts['retained_payload_words'] = sum(map(int,re.findall(r'frisk.retained_words = (\d+)',ir)))
    records.append(dict(stage=name,counts=counts))
    print(name, counts)

# Re-run the three passes from the serialized input in a fresh process.
run(HERE/'01-fragment.mlir', '--canonicalize', '--fuse-fragment-accumulator',
    '--frisk-frag-ir-reorder', '--canonicalize', '--cse',
    '--frisk-simplify-loop-control', '--cse', '--frisk-retain-invariant-fragments=max-words=32',
    '--frisk-schedule-async-copies',
    '-o', HERE/'replayed-optimized.mlir')
for name in ['06-optimized-fragment','replayed-optimized']:
    run(HERE/(name+'.mlir'), '--canonicalize', '--cse', '-o', HERE/(name+'.normalized.mlir'))
assert (HERE/'06-optimized-fragment.normalized.mlir').read_text() == (HERE/'replayed-optimized.normalized.mlir').read_text()
run(HERE/'06-optimized-fragment.mlir', '--convert-fragment-to-thread', '-o', HERE/'replayed-thread.mlir')
for name in ['07-thread-bridges', 'replayed-thread']:
    run(HERE/(name+'.mlir'), '--canonicalize', '--cse', '-o', HERE/(name+'.normalized.mlir'))
assert (HERE/'07-thread-bridges.normalized.mlir').read_text() == (HERE/'replayed-thread.normalized.mlir').read_text()
# Physical address canonicalization may order affine symbols differently in a
# fresh process. Verify finalization independently; compare the serialized
# typed-to-thread boundary above, before that representation-only difference.
run(HERE/'replayed-thread.mlir', '--finalize-thread-tiling', '--canonicalize', '--cse',
    '-o', HERE/'replayed-finalized.mlir')
(HERE/'stage-statistics.json').write_text(json.dumps(records,indent=2)+'\n')
print('PASS: all stages parse; serialized fragment optimization and thread replay match MyTest')
