#!/usr/bin/env python3
"""Fingerprint-scoped V LDS/PV scheduling experiment on the group32 baseline.

Keep all memory operations, hardware synchronization and MMACs in their
original respective orders. Move register-only PV work between V reads,
before the existing reader handoff and overwrite DMA. No LDS resize.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'noncausal_group_sweep_forward/group32.ll'
SOURCE_SHA = '5331d9048af733894568fcf0135dce9409cceab1f8029da8698a3abe72e41af7'
LLVM = Path('/data2/xsl/install/bin')
DEF = re.compile(r'^  (%[\w.]+) = (.*)$')
TOKEN = re.compile(r'%[\w.]+')
VREAD = '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3('
MMA = 'call <4 x float> asm "v_mmac_f32_16x16x16_f16 '
HINT = '  call void @llvm.amdgcn.sched.barrier(i32 0) #99'
PURE = re.compile(r'(getelementptr|add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|ptrtoint|inttoptr|bitcast|shufflevector|extractelement|insertelement|extractvalue|insertvalue|fmul) ')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def defs(lines):
    return {m[1]: (i, m[2]) for i, line in enumerate(lines) if (m := DEF.match(line))}


def operations(lines):
    # Include every call except register-only MMAC and scheduling pseudo-ops.
    # In particular V reads, global->LDS DMA, all other calls, loads/stores,
    # fences and control flow must keep their exact relative order.
    return [line for line in lines if MMA not in line and 'sched.barrier' not in line
            and (re.search(r'\b(call|load|store|fence|br)\b', line))]


def transform(source, distance, stages=(0, 1, 2, 3)):
    # distance=2/4 means V[i] is issued before MMAC[max(0,i-distance)].
    # distance=None is the late-DMA control: all eight reads precede MMAC0.
    lines = source.splitlines()
    mma_names = [DEF.match(line)[1] for line in lines if MMA in line]
    read_names = [DEF.match(line)[1] for line in lines if DEF.match(line) and VREAD in line]
    assert len(mma_names) == 256 and len(read_names) == 32
    records = []
    for stage in stages:
        d = defs(lines)
        reads = read_names[stage*8:(stage+1)*8]
        prefix = mma_names[128+stage*32:136+stage*32]
        first_read = d[reads[0]][0]
        start = max(i for i in range(first_read)
                    if 'fence syncscope("workgroup") acquire' in lines[i]) + 1
        end = d[prefix[-1]][0]
        region = lines[start:end+1]
        assert sum(VREAD in line for line in region) == 8
        assert sum(MMA in line for line in region) == 8
        assert sum('buffer_load_dwordx4 ' in line for line in region) == (2 if stage < 3 else 0)
        assert sum('@llvm.amdgcn.s.barrier(' in line for line in region) == (1 if stage < 3 else 0)
        emitted, scheduled = set(), []

        def dependencies(name, kind='pure'):
            if name not in d or d[name][0] < start or name in emitted:
                return
            index, expr = d[name]
            assert index <= end, ('forward dependency', name)
            if kind == 'read':
                assert name in reads and VREAD in expr
            elif kind == 'mma':
                assert name in prefix and MMA in expr
            else:
                assert PURE.match(expr), ('non-pure or unscheduled dependency', name, expr)
            for value in TOKEN.findall(expr):
                dependencies(value)
            if kind == 'mma':
                scheduled.append(HINT)
            scheduled.append(lines[index])
            if kind == 'mma':
                scheduled.append(HINT)
            emitted.add(name)

        issued = 0
        for j, mma in enumerate(prefix):
            target = 8 if distance is None else min(8, j+distance+1)
            while issued < target:
                dependencies(reads[issued], 'read')
                issued += 1
            dependencies(mma, 'mma')
        for line in region:
            match = DEF.match(line)
            if not match or match[1] not in emitted:
                scheduled.append(line)
        lines[start:end+1] = scheduled
        records.append(dict(stage=stage, reads=reads, early_mmacs=prefix,
                            distance=distance, next_v_dma_overlap_mmac=24 if stage < 3 else None))
    old = source.splitlines()
    assert Counter(line for line in lines if line != HINT) == Counter(line for line in old if line != HINT)
    assert operations(lines) == operations(old), 'Memory/synchronization order changed'
    assert [line for line in lines if MMA in line] == [line for line in old if MMA in line]
    return '\n'.join(lines)+'\n', records


def validate_schedule(text, records):
    lines = text.splitlines()
    d = defs(lines)
    for record in records:
        reads, mmas, distance = record['reads'], record['early_mmacs'], record['distance']
        for name in mmas:
            assert 'sched.barrier(i32 0)' in lines[d[name][0]-1]
            assert 'sched.barrier(i32 0)' in lines[d[name][0]+1]
        for i, read in enumerate(reads):
            target = 0 if distance is None else max(0, i-distance)
            assert d[read][0] < d[mmas[target]][0]
            if target:
                assert d[mmas[target-1]][0] < d[read][0]
        # The reader barrier and overwrite remain AFTER the final V read;
        # the first eight MMACs now precede that barrier too.
        if record['stage'] < 3:
            handoff = next(i for i in range(d[reads[0]][0], len(lines))
                           if '@llvm.amdgcn.s.barrier(' in lines[i])
            assert d[mmas[-1]][0] < handoff
            assert max(d[name][0] for name in reads) < handoff


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Baseline changed; re-audit V ownership'
    source = SOURCE.read_text()
    kbase = HERE.parent/'noncausal_grouped_lds_prefetch'
    spec = importlib.util.spec_from_file_location('kgen', kbase/'generate.py')
    kgen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kgen)
    kmanifest = json.loads((kbase/'manifest_protected.json').read_text())
    kgroups = kmanifest['groups']
    variants = {'baseline': (source, [])}
    for name, distance, stages in [('v_late_dma_control', None, (0,1,2,3)),
                                    ('v_prefetch2', 2, (0,1,2,3)),
                                    ('v_prefetch4', 4, (0,1,2,3)),
                                    ('v_tail_prefetch2', 2, (3,))]:
        variants[name] = transform(source, distance, stages)
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, group_size=32,
                    reverse=0, threads=256, BM=128, BN=128, BK=32,
                    lds_bytes=16384, gpu_tested=False, variants={})
    # The whole QK/softmax prefix remains byte-identical in every variant.
    unchanged_prefix = source[:source.index('  %address.local929 =')]
    for name, (text, records) in variants.items():
        assert text.startswith(unchanged_prefix)
        validate_schedule(text, records)
        path = HERE/(name+'.ll')
        path.write_text(text)
        (HERE/(name+'.launch.json')).write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path, '-o', optimized], check=True)
        opt_text = optimized.read_text()
        validate_schedule(opt_text, records)
        kgen.assert_groups_unbroken(opt_text.splitlines(), kgroups)
        opt_lines = opt_text.splitlines()
        od = defs(opt_lines)
        for record in kmanifest['moved_groups']:
            assert max(od[v][0] for v in record['loads']) < od[record['anchor']][0]
            assert 'sched.barrier(i32 0)' in opt_lines[od[record['anchor']][0]-1]
        if name == 'baseline':
            baseline_o3_operations = operations(opt_lines)
        assert operations(opt_lines) == baseline_o3_operations, 'O3 changed memory/sync order'
        assert sum(VREAD in line for line in opt_text.splitlines() if DEF.match(line)) == 32
        assert sum(MMA in line for line in opt_text.splitlines()) == 256
        assert text.count('@llvm.amdgcn.s.barrier()') == source.count('@llvm.amdgcn.s.barrier()')
        manifest['variants'][name] = dict(sha256=sha(path), stages=records,
            llvm_and_o3_verified=True, all_memory_and_sync_order_preserved=True,
            all_mmac_order_preserved=True, o3_v_schedule_preserved=True,
            o3_all_k_pairs_protected=True, o3_all_k_prefetch_anchors_preserved=True,
            o3_memory_and_sync_order_preserved=True, sched_hints=text.count(HINT))
        print(f'PASS {name}: LLVM/O3, memory/sync and MMAC order, V schedule, K pairs', flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
