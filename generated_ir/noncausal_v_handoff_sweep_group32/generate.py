#!/usr/bin/env python3
"""只扫描 V reader barrier/DMA 的位置，保持8次读取尽早发出。

输入为已测试的 v_late_dma_control.ll；after8逐字节复制该输入。
after2/4只移动前三个V阶段的reader fence/barrier和下一块V DMA区段，
全部MMAC、读取、调度提示均保留。第四阶段没有后续V DMA，完全不改。
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'noncausal_v_lds_overlap_group32'
SOURCE = PREVIOUS/'v_late_dma_control.ll'
SOURCE_SHA = '8f0bfda5ed515550aba74efe59f1aa804591082fbbb8fd0f46e41a0366151f53'
LLVM = Path('/data2/xsl/install/bin')
spec = importlib.util.spec_from_file_location('v_overlap_generator', PREVIOUS/'generate.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source, after):
    assert after in (2, 4, 8)
    lines = source.splitlines()
    mmas = [helper.DEF.match(line)[1] for line in lines if helper.MMA in line]
    moved = []
    for stage in range(3):
        d = helper.defs(lines)
        first = 128+stage*32
        last_early = d[mmas[first+7]][0]
        start = next(i for i in range(last_early+1, len(lines))
                     if 'fence syncscope("workgroup") release' in lines[i])
        ninth = d[mmas[first+8]][0]
        dma_indices = [i for i in range(start, ninth) if 'buffer_load_dwordx4 ' in lines[i]]
        assert len(dma_indices) == 2
        end = dma_indices[-1]+1
        block = lines[start:end]
        assert sum('@llvm.amdgcn.s.barrier(' in line for line in block) == 1
        assert sum(line.strip().startswith('fence ') for line in block) == 2
        assert all(helper.MMA not in line and helper.VREAD not in line for line in block)
        target = d[mmas[first+after-1]][0]+2  # 保留该MMAC之后的原有sched.barrier。
        assert lines[target-1] == helper.HINT
        local_defs = set()
        for line in block:
            match = helper.DEF.match(line)
            rhs = match[2] if match else line
            for value in helper.TOKEN.findall(rhs):
                assert value in local_defs or value not in d or d[value][0] < target, (stage, value)
            if match:
                local_defs.add(match[1])
        if after != 8:
            lines = lines[:target] + block + lines[target:start] + lines[end:]
        moved.append(dict(stage=stage, after_mmac_count=after,
                          next_v_dma_overlap_mmac=32-after,
                          dma_results=[helper.DEF.match(line)[1] for line in block
                                       if 'buffer_load_dwordx4 ' in line],
                          block_lines=len(block)))
    result = '\n'.join(lines)+'\n'
    assert Counter(lines) == Counter(source.splitlines())
    assert helper.operations(lines) == helper.operations(source.splitlines())
    assert [l for l in lines if helper.MMA in l] == [l for l in source.splitlines() if helper.MMA in l]
    assert result[result.index('  %grp.n1054 ='):] == source[source.index('  %grp.n1054 ='):]
    if after == 8:
        assert result == source
    return result, moved


def audit_schedule(text, after):
    mmac = 0
    reads, dma, barriers = [], [], []
    for line in text.splitlines():
        if helper.DEF.match(line) and helper.VREAD in line:
            reads.append(mmac)
        if 'buffer_load_dwordx4 ' in line and '%dma.buffer.rsrc1349' in line:
            dma.append(mmac)
        if 'call void @llvm.amdgcn.s.barrier()' in line:
            barriers.append(mmac)
        if helper.MMA in line:
            mmac += 1
    assert mmac == 256 and len(reads) == 32
    assert reads == [128+stage*32 for stage in range(4) for _ in range(8)]
    assert dma == [96, 96] + [128+stage*32+after for stage in range(3) for _ in range(2)], dma
    assert [v for v in barriers if 128 < v < 256] == [128+after, 160, 160+after, 192, 192+after, 224]
    assert barriers[-1] == 256
    # 各版本保留相同100个调度提示，避免同时改变调度约束的数量。
    assert text.count('call void @llvm.amdgcn.sched.barrier(') == 100
    return dict(v_read_mmac_positions=reads, v_dma_mmac_positions=dma,
                hardware_barrier_mmac_positions=barriers)


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Source changed; re-audit handoff'
    source = SOURCE.read_text()
    kdir = HERE.parent/'noncausal_grouped_lds_prefetch'
    kspec = importlib.util.spec_from_file_location('kgen', kdir/'generate.py')
    kgen = importlib.util.module_from_spec(kspec)
    kspec.loader.exec_module(kgen)
    km = json.loads((kdir/'manifest_protected.json').read_text())
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                    group_size=32, reverse=0, threads=256, BM=128, BN=128, BK=32,
                    lds_bytes=16384, gpu_tested=False, variants={})
    for after in (8, 2, 4):
        name = f'dma_after{after}'
        text, records = transform(source, after)
        audit_schedule(text, after)
        path = HERE/(name+'.ll')
        path.write_text(text)
        (HERE/(name+'.launch.json')).write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path, '-o', optimized], check=True)
        opt_text = optimized.read_text()
        positions = audit_schedule(opt_text, after)
        if after == 8:
            old_o3_operations = helper.operations(opt_text.splitlines())
            old_o3_mmas = [l for l in opt_text.splitlines() if helper.MMA in l]
        assert helper.operations(opt_text.splitlines()) == old_o3_operations
        assert [l for l in opt_text.splitlines() if helper.MMA in l] == old_o3_mmas
        kgen.assert_groups_unbroken(opt_text.splitlines(), km['groups'])
        d = helper.defs(opt_text.splitlines())
        for r in km['moved_groups']:
            assert max(d[v][0] for v in r['loads']) < d[r['anchor']][0]
            assert 'sched.barrier(i32 0)' in opt_text.splitlines()[d[r['anchor']][0]-1]
        manifest['variants'][name] = dict(sha256=sha(path), after_mmac_count=after,
            next_v_dma_overlap_mmac=32-after, changed_stages=records,
            llvm_and_o3_verified=True, memory_sync_and_mmac_orders_preserved=True,
            k_pairs_and_prefetch_preserved=True, o3_schedule=positions)
        print(f'PASS {name}: 8 early reads, handoff after {after} MMACs, {32-after} MMACs remain; LLVM/O3 and K protection', flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
