#!/usr/bin/env python3
"""Check the split-D best-config export; no GPU performance claims.

Usage: check_async_attention_export.py output.ll output.log [LLVM bin directory] [staged|hybrid|hybrid-grouped|staged-grouped]
Generic packet and concurrency semantics are tested by check_async_layout_integration.py.
"""
import re
import subprocess
import sys
from pathlib import Path

path, log_path = map(Path, sys.argv[1:3])
llvm = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/data2/xsl/install/bin')
mode = sys.argv[4] if len(sys.argv) > 4 else 'staged'
assert mode in ('staged', 'hybrid', 'hybrid-grouped', 'staged-grouped')
hybrid = mode.startswith('hybrid')
grouped = mode.endswith('-grouped')
text, log = path.read_text(), log_path.read_text()
assert not re.search(r'error:|LLVM ERROR|Assertion .*failed', log)
assert ('2 direct, 0 staged, 0 fallback; staging bytes 0' if hybrid else
        '2 direct, 4 staged, 0 fallback; staging bytes 8192') in log
assert re.search(r'^@shm_\w+ = .*global \[16384 x i8\]', text, re.M)
assert len(re.findall(r'^@async_lds_staging_\w+ = .*global \[4096 x half\]', text, re.M)) == (0 if hybrid else 1)
assert len(re.findall(r'^@.*addrspace\(3\).*global', text, re.M)) == (1 if hybrid else 2)
assert '@malloc' not in text

def check_overlap(ir):
    dma = list(re.finditer(r'call void @llvm.amdgcn.global.load.async.lds\(', ir))
    marks = list(re.finditer(r'call void @llvm.amdgcn.asyncmark\(', ir))
    waits = list(re.finditer(r'call void @llvm.amdgcn.wait.asyncmark\(', ir))
    assert len(marks) == (2 if hybrid else 6)
    assert len(waits) == len(marks) - int(grouped)
    completion = [waits[0], waits[0], *waits[1:]] if grouped else waits
    # One mark commits a tile, including all statically expanded DMA rounds.
    counts = []
    previous = 0
    for mark, wait in zip(marks, completion):
        counts.append(sum(previous < d.start() < mark.start() for d in dma))
        assert mark.start() < wait.start()
        previous = mark.end()
    assert counts == ([8] * 2 if hybrid else [8] * 6) or (not hybrid and counts == [1] * 6), counts
    if not grouped:
        # Preserve the conservative separate-completion policy: the first
        # tile's wait precedes the next issue, rather than merely counting waits.
        next_issue = next(d for d in dma if d.start() > marks[0].end())
        assert waits[0].start() < next_issue.start()
    assert dma[-1].start() < marks[-1].start()
    # First V tile overlaps softmax; the next tiles overlap the previous PV.
    for i in range(2, len(marks)):
        between = ir[marks[i].end():completion[i].start()]
        assert '@llvm.amdgcn.exp2.f32(' in between or 'v_mmac_' in between, i

check_overlap(text)
defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))

def provenance(value):
    pending, visited, flags = [value], set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        expression = defs.get(name, '')
        if '@llvm.amdgcn.exp2.f32(' in expression:
            flags.add('exp2')
            continue  # The score computation before softmax may read Q/K.
        if expression.startswith('load '):
            flags.add('load')
            continue
        pending.extend(re.findall(r'%[\w.]+', expression))
    return flags

mma = [expr for expr in defs.values() if 'asm sideeffect ' in expr and 'v_mmac_' in expr]
assert len(mma) == 160
for expression in mma[32:]:
    operand = re.search(r'"\(<4 x half> (%[\w.]+),', expression)[1]
    assert provenance(operand) == {'exp2'}, 'P must reach PV in registers'
for value in re.findall(r'^  store <\d+ x half> (%[\w.]+), ptr addrspace\(3\)', text, re.M):
    assert provenance(value) == {'load'}, 'Only load-derived V data may be written to LDS'

subprocess.run([llvm/'llvm-as', path, '-o', '/dev/null'], check=True)
subprocess.run([llvm/'opt', '-passes=verify', '-disable-output', path], check=True)
optimized = subprocess.run([llvm/'opt', '-passes=default<O3>', '-S', path, '-o', '-'],
                           check=True, capture_output=True, text=True).stdout
check_overlap(optimized)
if hybrid:
    from check_attention_lds_banks import check
    check(text, minimum_locations=3)
    global_loads = re.findall(r' = load (half|<\d+ x half>), ptr addrspace\(1\)', text)
    assert global_loads == ['<4 x half>'] * 16, global_loads
    print('PASS split-D hybrid: Q/K direct DMA, V 64-bit global loads, 16 KiB LDS, no staging')
else:
    print('PASS split-D export: Q/K direct DMA, V staged DMA, one reused staging tile, 24 KiB LDS')
print('PASS all 128 PV A operands retain exp2-to-register provenance; no P LDS round trip')
print('PASS tile issue/mark/wait groups, LLVM parse/verify and O3 scheduling checks')
