#!/usr/bin/env python3
"""Interleave independent PV accumulator chains; preserve each chain's SSA edges."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'noncausal_v_handoff_sweep_group32/dma_after4.ll'
SOURCE_SHA = '57a8f633c911ce4579b0e8a076d9ce08c05053d6071b025a7f0db39c3a1baecd'
LLVM = Path('/data2/xsl/install/bin')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


h = module('vhelpers', HERE.parent/'noncausal_v_lds_overlap_group32/generate.py')


def order(width):
    # width=1: A0..A7,B0..B7,C0..C7,D0..D7 (matched serial control).
    # width=2: A0,B0,A1,B1,...,C0,D0,C1,D1,...
    # width=4: A0,B0,C0,D0,A1,B1,C1,D1,...
    return [chain*8+k for group in range(0, 4, width)
            for k in range(8) for chain in range(group, group+width)]


def transform(source, width):
    lines = source.splitlines()
    original_mmas = [h.DEF.match(l)[1] for l in lines if h.MMA in l]
    reads = [h.DEF.match(l)[1] for l in lines if h.DEF.match(l) and h.VREAD in l]
    for stage in range(4):
        d = h.defs(lines)
        names = original_mmas[128+stage*32:160+stage*32]
        # Verify four separate eight-MMAC chains before reordering.
        for chain in range(4):
            for k in range(8):
                acc = re.search(r'<4 x float> (%[\w.]+)\)', d[names[chain*8+k]][1])[1]
                assert acc == names[chain*8+k-1] if k else acc not in names
        start = d[reads[stage*8+7]][0]+1
        end = d[names[-1]][0]+1
        handoff = []
        excluded = set()
        if stage < 3:
            a = next(i for i in range(start, end) if 'fence syncscope("workgroup") release' in lines[i])
            b = max(i for i in range(a, end) if 'buffer_load_dwordx4 ' in lines[i])+1
            handoff = lines[a:b]
            excluded = set(range(a, b))
        emitted = set()
        scheduled = []

        def emit(name, mma=False):
            if name not in d or d[name][0] < start or name in emitted:
                return
            i, expr = d[name]
            assert i < end and i not in excluded, (stage, name, expr)
            assert h.MMA in expr if mma else h.PURE.match(expr), (name, expr)
            for dep in h.TOKEN.findall(expr):
                emit(dep)
            if mma:
                scheduled.append(h.HINT)
            scheduled.append(lines[i])
            if mma:
                scheduled.append(h.HINT)
            emitted.add(name)

        for position, ordinal in enumerate(order(width), 1):
            emit(names[ordinal], mma=True)
            if position == 4:
                # Original release/barrier/acquire and next-V DMA, unchanged.
                for line in handoff:
                    m = h.DEF.match(line)
                    for dep in h.TOKEN.findall(m[2] if m else line):
                        if dep not in emitted and dep in d and d[dep][0] >= start:
                            assert d[dep][0] in excluded, ('handoff dependency', dep)
                    scheduled.append(line)
                    if m:
                        emitted.add(m[1])
        for i in range(start, end):
            if i in excluded or 'sched.barrier(' in lines[i]:
                continue
            m = h.DEF.match(lines[i])
            if m and m[1] in emitted:
                continue
            assert not lines[i].strip() or (m and h.PURE.match(m[2])), lines[i]
            scheduled.append(lines[i])
        lines[start:end] = scheduled
    old = source.splitlines()
    assert Counter(l for l in lines if 'sched.barrier(' not in l) == Counter(l for l in old if 'sched.barrier(' not in l)
    assert h.operations(lines) == h.operations(old)
    assert [l for l in lines if h.MMA in l][:128] == [l for l in old if h.MMA in l][:128]
    return '\n'.join(lines)+'\n'


def audit(text, source, width):
    original = [h.DEF.match(l)[1] for l in source.splitlines() if h.MMA in l]
    actual = [h.DEF.match(l)[1] for l in text.splitlines() if h.MMA in l]
    expected = original[:128]+[original[128+s*32+i] for s in range(4) for i in order(width)]
    assert actual == expected
    lines = text.splitlines()
    d = h.defs(lines)
    for name in actual[128:]:
        i = d[name][0]
        assert 'sched.barrier(i32 0)' in lines[i-1] and 'sched.barrier(i32 0)' in lines[i+1]
    count = 0
    reads, dma, barriers = [], [], []
    for line in lines:
        if h.DEF.match(line) and h.VREAD in line:
            reads.append(count)
        if 'buffer_load_dwordx4 ' in line and '%dma.buffer.rsrc1349' in line:
            dma.append(count)
        if 'call void @llvm.amdgcn.s.barrier()' in line:
            barriers.append(count)
        if h.MMA in line:
            count += 1
    assert reads == [128+s*32 for s in range(4) for _ in range(8)]
    assert dma == [96,96,132,132,164,164,196,196]
    assert [n for n in barriers if 128 < n < 256] == [132,160,164,192,196,224]
    assert text.count('call void @llvm.amdgcn.sched.barrier(') == 292
    return dict(stage_mmac_ordinals=order(width), v_reads=reads, v_dma=dma,
                hardware_barriers=barriers, scheduling_hints=292)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    source = SOURCE.read_text()
    kdir = HERE.parent/'noncausal_grouped_lds_prefetch'
    kgen = module('khelpers', kdir/'generate.py')
    km = json.loads((kdir/'manifest_protected.json').read_text())
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, group_size=32,
        reverse=0, BM=128, BN=128, BK=32, threads=256, lds_bytes=16384,
        reader_handoff_after=4, gpu_tested=False, variants={})
    for name, width in [('baseline_after4',0), ('pv_serial_guarded',1), ('pv_interleave2',2), ('pv_interleave4',4)]:
        text = transform(source, width) if width else source
        path = HERE/(name+'.ll')
        path.write_text(text)
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S',path,'-o',optimized],check=True)
        opt_text = optimized.read_text()
        if not width:
            baseline_operations = h.operations(opt_text.splitlines())
        assert h.operations(opt_text.splitlines()) == baseline_operations
        kgen.assert_groups_unbroken(opt_text.splitlines(),km['groups'])
        d = h.defs(opt_text.splitlines())
        for group in km['moved_groups']:
            assert max(d[n][0] for n in group['loads']) < d[group['anchor']][0]
            assert 'sched.barrier(i32 0)' in opt_text.splitlines()[d[group['anchor']][0]-1]
        schedule = audit(text,source,width) if width else None
        if width:
            assert audit(opt_text,source,width) == schedule
        manifest['variants'][name] = dict(width=width,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            llvm_and_o3_verified=True,k_prefetch_preserved=True,memory_sync_order_preserved=True,schedule=schedule)
        print(f'PASS {name}: LLVM/O3, memory/sync order, K prefetch, PV schedule',flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
