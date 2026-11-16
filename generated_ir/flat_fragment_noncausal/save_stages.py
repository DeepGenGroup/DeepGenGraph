#!/usr/bin/env python3
"""Extract complete, independently parseable MyTest stage dumps."""
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
TOOL = HERE.parents[1] / '3rd/deepgengraph/build/test/ThreadTilingTest'
text = (HERE/'lowering.log').read_text()
assert not re.search(r'(?<!!)frisk\.fragment(?:\s|_yield\b)', text), 'region op in lowering log'
for marker, name in [
    ('after promote-gemm-matrix-loads', '00-block.mlir'),
    ('after convert-friskbase-to-fragment', '01-fragment.mlir'),
    ('after convert-fragment-to-thread', '02-thread-bridges.mlir'),
    ('after createFinalizeThreadTilingPass', '03-thread.mlir')]:
    tail = text.split(marker, 1)[1].split('\n', 1)[1]
    start = re.search(r'^(?:#\w+\s*=|module\b)', tail, re.M).start()
    module = re.search(r'^module\b', tail[start:], re.M).start() + start
    end = re.search(r'^}\s*$', tail[module:], re.M).end() + module
    path = HERE/name
    path.write_text(tail[start:end].rstrip()+'\n')
    p = subprocess.run([str(TOOL), str(path), '-o', '/dev/null'],capture_output=True,text=True)
    assert p.returncode == 0, p.stderr[-8000:]
    print('PARSED', name)
# A second process lowers the serialized fragment stage, without LowerInfo.
p = subprocess.run([str(TOOL), str(HERE/'01-fragment.mlir'), '--convert-fragment-to-thread',
                    '-o', str(HERE/'replayed-thread.mlir')],capture_output=True,text=True)
assert p.returncode == 0,p.stderr[-8000:]
for name in ['02-thread-bridges', 'replayed-thread']:
    subprocess.run([str(TOOL), str(HERE/(name+'.mlir')), '--canonicalize', '--cse', '-o', str(HERE/(name+'.normalized.mlir'))],check=True)
assert (HERE/'02-thread-bridges.normalized.mlir').read_text() == (HERE/'replayed-thread.normalized.mlir').read_text()
print('PASS: serialized fragment replay exactly matches the in-process thread stage')
