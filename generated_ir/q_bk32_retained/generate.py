#!/usr/bin/env python3
"""Controlled BK0 Q-retention experiment on the measured best attention IR.

This is a fingerprinted experiment, not a general loop-hoisting compiler pass.
The retained fragment's address/DMA/read expressions are cloned by dependency,
so the original DMA packing and FP16 scaling semantics are preserved exactly.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT/'generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll'
SOURCE_SHA256 = 'c19c67ba6ecf235946229e89bf647f8e10640f015d3b1da7ce54fd91d0045f2f'
LLVM = Path('/data2/xsl/install/bin')


def transform(source):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256
    # Name numeric locals so moving/removing definitions cannot break LLVM's
    # implicit SSA numbering. Keep ABI arguments 0..3 and entry block 4 intact.
    source = re.sub(r'%(\d+)\b', lambda m: m[0] if int(m[1]) < 5
                    else '%qr.n'+m[1], source)
    source = re.sub(r'^(\d+):', lambda m: 'qr.n'+m[1]+':', source, flags=re.M)
    lines = source.splitlines()
    definitions = {}
    positions = {}
    for i, line in enumerate(lines):
        m = re.match(r'  (%[\w.]+) = (.*)', line)
        if m:
            definitions[m[1]] = line
            positions[m[1]] = i
    entry_end = next(i for i,l in enumerate(lines) if l.startswith('  br label %qr.n40,'))
    dma_roots = ['%qr.n62','%qr.n76']
    kept = ['%qr.n120','%qr.n122','%qr.n123','%qr.n124']
    loads = ['%qr.n117','%qr.n118','%qr.n119','%qr.n121']
    assert all('buffer_load_dwordx4' in definitions[x] for x in dma_roots)
    assert all('= fmul <4 x half>' in definitions[x] for x in kept)
    assert all('= load <4 x half>' in definitions[x] for x in loads)
    # No DMA return value is consumed. All four LDS loads feed only scaling.
    for name in dma_roots + loads:
        uses = len(re.findall(re.escape(name)+r'(?![\w.])',source))
        assert uses == (1 if name in dma_roots else 2), (name,uses)
    clones, renamed, visiting = [], {}, set()

    def clone(name):
        if name not in definitions or positions[name] < entry_end:
            return name
        if name in renamed:
            return renamed[name]
        assert name not in visiting, 'loop-dependent fragment'
        visiting.add(name)
        line = definitions[name]
        assert ' = phi ' not in line and name != '%qr.n41', line
        body = line.split(' = ',1)[1]
        for dependency in re.findall(r'%[\w.]+',body):
            clone(dependency)
        new_name = '%qkeep.'+name[1:]
        renamed[name] = new_name
        body = re.sub(r'%[\w.]+',lambda m: renamed.get(m[0],m[0]),body)
        clones.append('  '+new_name+' = '+body)
        visiting.remove(name)
        return new_name

    for name in dma_roots:
        clone(name)
    dma_prologue = list(clones)
    for name in kept:
        clone(name)
    read_prologue = clones[len(dma_prologue):]
    assert set(loads).issubset(renamed)
    assert '%qr.n41' not in renamed
    sync = [
        '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10',
        '  fence syncscope("workgroup") release',
        '  call void @llvm.amdgcn.s.barrier()',
        '  fence syncscope("workgroup") acquire',
    ]
    remove = set(dma_roots+loads+kept)
    output = []
    for i,line in enumerate(lines):
        m = re.match(r'  (%[\w.]+) = ',line)
        if m and m[1] in remove:
            continue
        if i == entry_end:
            output += ['  ; Load and scale Q[:, 0:32] once before the BN loop.',
                       *dma_prologue, *sync, *read_prologue,
                       '  ; Four <4 x half> fragments remain live across BN iterations.']
        line = re.sub(r'%[\w.]+', lambda m: renamed[m[0]] if m[0] in kept else m[0],line)
        output.append(line)
    candidate = '\n'.join(output)+'\n'
    entry, loop = candidate.split('qr.n40:',1)
    assert entry.count('buffer_load_dwordx4') == 2
    assert loop.count('buffer_load_dwordx4') == 22
    assert entry.count('call void @llvm.amdgcn.s.barrier(') == 1
    assert loop.count('call void @llvm.amdgcn.s.barrier(') == 16
    assert candidate.count('v_mmac_f32_16x16x16_f16 ') == 256
    return candidate


def main():
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA256
    (HERE/'baseline.ll').write_bytes(data)
    (HERE/'q_bk32_retained.ll').write_text(transform(data.decode()))
    config = json.loads((ROOT/'generated_ir/bm_bn_sweep/bm128_bn128.launch.json').read_text())
    for name in ('baseline','q_bk32_retained'):
        path = HERE/f'{name}.ll'
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify',path,'-disable-output'],check=True)
        (HERE/f'{name}.launch.json').write_text(json.dumps(config,indent=2)+'\n')
    result = dict(source=str(SOURCE.relative_to(ROOT)), baseline_sha256=SOURCE_SHA256,
        candidate_sha256=hashlib.sha256((HERE/'q_bk32_retained.ll').read_bytes()).hexdigest(),
        source_program_median_ms=1.7790470123291016,
        retained_q_columns=[0,32], retained_fragment_type='4 x <4 x half> per thread',
        packed_state_bytes_per_thread=32, packed_32bit_words_per_thread=8,
        static_lds_bytes=16384, launch=config,
        dma_sites=dict(baseline_loop=24,candidate_prologue=2,candidate_loop=22),
        q_dma_calls_per_block=dict(baseline='8*N',candidate='2+6*N'),
        kv_dma_calls_per_block='16*N in both',
        barriers_per_block=dict(baseline='16*N',candidate='1+16*N'),
        waits_per_block=dict(baseline='12*N',candidate='1+12*N'),
        N='query_tile+1', llvm_assemble_and_O3_verify=True,
        gpu_tested=False,target_gfx936_codegen_tested=False,
        limitations='Fixed-shape fingerprinted experiment; final VGPR/scratch and GPU speed require target compilation and timing.')
    (HERE/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
