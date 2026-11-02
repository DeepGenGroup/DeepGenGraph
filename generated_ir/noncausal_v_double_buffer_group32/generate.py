#!/usr/bin/env python3
"""Fingerprint-scoped V ping-pong experiment on the user-confirmed best case."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'noncausal_pv_interleave_pass_group32/pv_interleave4.ll'
SOURCE_SHA = '2bef36b60ab7309928d68bea12a033ee0615d907f577333871fae41f74156c51'
LLVM = Path('/data2/xsl/install/bin')
MMA = 'call <4 x float> asm "v_mmac_f32_16x16x16_f16 '
VREAD = '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3('
DMA = 'buffer_load_dwordx4 '
VRESOURCE = '%dma.buffer.rsrc1349'
DEF = re.compile(r'^  (%[\w.]+) = (.*)$')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source):
    lines = source.splitlines()
    # Move V1/V2/V3 issue before the *current* V reads, after its ready barrier.
    # This barrier also releases the previous owner of the alternate buffer:
    # K3 for V1, V0 for V2, and V1 for V3. The former after4 reader barrier
    # becomes redundant because the DMA no longer overwrites the current V.
    for stage in reversed(range(3)):
        reads = [i for i, line in enumerate(lines) if DEF.match(line) and VREAD in line]
        start = reads[stage * 8]
        insert = max(i for i in range(start) if 'fence syncscope("workgroup") acquire' in lines[i]) + 1
        end_read = reads[stage * 8 + 7]
        barrier = next(i for i in range(end_read, len(lines)) if '@llvm.amdgcn.s.barrier(' in lines[i])
        assert 'fence syncscope("workgroup") release' in lines[barrier-1]
        assert 'fence syncscope("workgroup") acquire' in lines[barrier+1]
        ready_wait = next(i for i in range(barrier+2, len(lines)) if 's_waitcnt vmcnt(0)' in lines[i])
        issues = [i for i in range(barrier+2, ready_wait)
                  if DMA in lines[i] and VRESOURCE in lines[i]]
        assert len(issues) == 2
        block = lines[barrier+2:issues[-1]+1]
        assert all(DEF.match(line) for line in block)
        assert sum(MMA in line for line in lines[start:barrier]) == 4
        if (stage + 1) % 2:
            block = [line.replace('i32 %dma.buffer.lds1374,', 'i32 %dma.buffer.lds1368,')
                     .replace('i32 %dma.buffer.lds1371,', 'i32 %dma.buffer.lds1365,')
                     for line in block]
        del lines[barrier-1:issues[-1]+1]
        lines[insert:insert] = [f'  ; V double buffer: issue V{stage+1} to slot {(stage+1)%2} before consuming V{stage}.'] + block

    result, read_index = [], 0
    for line in lines:
        if DEF.match(line) and VREAD in line:
            stage, packet = divmod(read_index, 8)
            if stage % 2:
                ptr = re.search(r'\(ptr addrspace\(3\) (%[\w.]+), i32 0\)', line)[1]
                shifted = f'%vdb.read.s{stage}.p{packet}'
                result.append(f'  {shifted} = getelementptr i8, ptr addrspace(3) {ptr}, i64 8192')
                line = line.replace(f'ptr addrspace(3) {ptr},', f'ptr addrspace(3) {shifted},')
            read_index += 1
        result.append(line)
    assert read_index == 32
    output = '\n'.join(result) + '\n'
    # All arithmetic, including all four PV accumulator chains, is identical.
    assert [l for l in result if MMA in l] == [l for l in source.splitlines() if MMA in l]
    prefix = source[:source.index('  %address.local929 =')]
    assert output.startswith(prefix), 'QK / softmax / V0 issue changed'
    return output


def audit(text, baseline, double):
    mmas, reads, issues, waits, barriers = 0, [], [], 0, 0
    for line in text.splitlines():
        if MMA in line:
            mmas += 1
        if DEF.match(line) and VREAD in line:
            reads.append(mmas)
        if DMA in line and VRESOURCE in line:
            issues.append(mmas)
        waits += 's_waitcnt vmcnt(0)' in line
        barriers += 'call void @llvm.amdgcn.s.barrier(' in line
    assert mmas == 256 and reads == [128+s*32 for s in range(4) for _ in range(8)]
    assert issues == [n for n in ([96,128,160,192] if double else [96,132,164,196]) for _ in range(2)]
    assert '@shm_0 = addrspace(3) global [16384 x i8]' in text
    assert barriers == baseline.count('call void @llvm.amdgcn.s.barrier(') - (3 if double else 0)
    assert waits == baseline.count('s_waitcnt vmcnt(0)')
    norm = lambda l: re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l))
    assert [norm(l) for l in text.splitlines() if MMA in l] == [norm(l) for l in baseline.splitlines() if MMA in l]
    return dict(v_read_mmac_ordinals=reads, v_dma_mmac_ordinals=issues,
                static_mmac=mmas, static_waits=waits, static_barriers=barriers)


def main():
    assert digest(SOURCE) == SOURCE_SHA, 'Best baseline changed; re-audit LDS ownership'
    source = SOURCE.read_text()
    kdir = HERE.parent/'noncausal_grouped_lds_prefetch'
    spec = importlib.util.spec_from_file_location('kgen', kdir/'generate.py')
    kgen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kgen)
    km = json.loads((kdir/'manifest_protected.json').read_text())
    config = json.loads((HERE.parent/'noncausal_frag_prefetch_forward/control/config.json').read_text())
    params = dict(config['parameters'], **{'workgroup-group-size': '32', 'pv-interleave': '4'})
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                    baseline_selection='User confirmed pv_interleave4 as latest measured best',
                    parameters=params, BM=128, BN=128, BK=32, threads=256,
                    waves=[4,1], lds_bytes=16384, slot_bytes=8192, v_slots=[0,1,0,1],
                    gpu_tested=False, variants={})
    for name, text in [('baseline_pv_interleave4', source), ('v_double_buffer', transform(source))]:
        path = HERE/(name+'.ll')
        path.write_text(text)
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path, '-o', optimized], check=True)
        schedule = audit(text, source, name == 'v_double_buffer')
        opt_text = optimized.read_text()
        assert audit(opt_text, source, name == 'v_double_buffer') == schedule
        kgen.assert_groups_unbroken(opt_text.splitlines(), km['groups'])
        defs = {m[1]: i for i, l in enumerate(opt_text.splitlines()) if (m := DEF.match(l))}
        for group in km['moved_groups']:
            assert max(defs[n] for n in group['loads']) < defs[group['anchor']]
            assert 'sched.barrier(i32 0)' in opt_text.splitlines()[defs[group['anchor']]-1]
        manifest['variants'][name] = dict(sha256=digest(path), llvm_and_o3_verified=True,
            k_prefetch_preserved=True, all_mmac_operands_and_order_preserved=True, schedule=schedule)
        print('PASS', name, 'LLVM/O3, PV chains, K prefetch, V schedule, 16 KiB LDS', flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
