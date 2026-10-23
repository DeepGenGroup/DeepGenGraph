#!/usr/bin/env python3
"""Small, fingerprinted LLVM scheduling experiments on the measured winner.

This is an experiment generator, not a general compiler pass. No partial
vmcnt/lgkmcnt values are guessed. All workgroup barriers and fences survive.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'generated_ir/invariant_fragment_pass/retained_8words.ll'
SOURCE_SHA256 = '8923ce4d910b6c2e6ce91fa3345c5f2e702b57d2bef5ef37cd12c726b0efea19'
LLVM = Path('/data2/xsl/install/bin')
WAIT = 's_waitcnt vmcnt(0)'
BARRIER = 'call void @llvm.amdgcn.s.barrier()'
HINT = '  call void @llvm.amdgcn.sched.barrier(i32 0) #11'
DMA = 'buffer_load_dwordx4 '
MMA = 'v_mmac_f32_16x16x16_f16 '
READ = '= load <4 x half>, ptr addrspace(3)'


def normalize(text):
    text = re.sub(r'%(\d+)\b', lambda m: m[0] if int(m[1]) < 5 else '%sched.n'+m[1], text)
    return re.sub(r'^(\d+):', lambda m: 'sched.n'+m[1]+':', text, flags=re.M)


def check_sync(lines, index):
    assert WAIT in lines[index]
    assert 'fence syncscope("workgroup") release' in lines[index+1]
    assert BARRIER in lines[index+2]
    assert 'fence syncscope("workgroup") acquire' in lines[index+3]


def overlap_qk_tails(text):
    lines = text.splitlines()
    waits = [i for i,l in enumerate(lines) if WAIT in l]
    delete, before, records = set(), {}, []
    for first, second in zip(waits, waits[1:]):
        window = lines[first+4:second]
        if sum(READ in l for l in window) != 4 or sum(DMA in l for l in window) != 2:
            continue
        if any(MMA in l or BARRIER in l or re.match(r'^\w[\w.]*:',l) or l.lstrip().startswith('br ') for l in window):
            continue
        check_sync(lines,first)
        check_sync(lines,second)
        previous_wait = max(i for i in waits if i < first)
        q_copies = [i for i in range(previous_wait+4,first) if DMA in lines[i]]
        assert len(q_copies) == 2
        tail = [i for i in range(previous_wait+4,q_copies[0]) if MMA in lines[i]][-2:]
        assert len(tail) == 2
        left, right = tail
        # The existing one-packet hint surrounds the penultimate MMAC. Move
        # that hint pair with both final MMACs, bracketing the whole tail at its
        # new location. The LDS reads stay before their original barriers.
        assert 'call void @llvm.amdgcn.sched.barrier(' in lines[left-1]
        assert 'call void @llvm.amdgcn.sched.barrier(' in lines[left+1]
        assert right == left+2
        first_value = re.match(r'  (%[\w.]+) =',lines[left])[1]
        second_value = re.match(r'  (%[\w.]+) =',lines[right])[1]
        for line in lines[right+1:first]:
            assert not ({first_value,second_value} & set(re.findall(r'%[\w.]+',line)))
            assert READ not in line and MMA not in line
        moved = [lines[left-1],lines[left],lines[right],lines[left+1]]
        delete.update(range(left-1,right+1))
        before[first] = moved
        records.append(dict(original_mmac_lines=[left+1,right+1],
                            q_dma_lines=[i+1 for i in q_copies],
                            before_wait_line=first+1, mmac_count=2))
    assert len(records) == 3, records
    output = []
    for i,line in enumerate(lines):
        output.extend(before.get(i,[]))
        if i not in delete: output.append(line)
    return '\n'.join(output)+'\n', records


def prefetch_read_pairs(text, surviving_reads):
    lines = text.splitlines()
    mmas = [i for i,l in enumerate(lines) if MMA in l]
    delete, before, after, records = set(), {}, {}, []
    for current, following in zip(mmas,mmas[1:]):
        window = lines[current+1:following]
        # Advance the complete adjacent b64 read group, preserving load order
        # and the opportunity for DS combining. Do not cross any real sync,
        # existing scheduling hint, DMA, store, branch, matrix read, or call.
        if sum(READ in l for l in window) != 2: continue
        if not all(re.match(r'  %[\w.]+ = (getelementptr|load|add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|ptrtoint|inttoptr) ',l) for l in window):
            continue
        if any('= load ' in l and READ not in l for l in window): continue
        loads = [re.match(r'  (%[\w.]+)',l)[1] for l in window if READ in l]
        # Do not add hard scheduling boundaries around duplicate loads that O3
        # already eliminates. Keep only complete, surviving two-read packets.
        if not all(n in surviving_reads for n in loads): continue
        current_value = re.match(r'  (%[\w.]+) =', lines[current])[1]
        assert not any(current_value in re.findall(r'%[\w.]+',l.split(' = ',1)[1]) for l in window)
        delete.update(range(current+1,following))
        before[current] = window+[HINT]
        after[current] = [HINT]
        records.append(dict(mmac_original_line=current+1, raw_read_words=4,
                            loads=loads))
    assert len(records) == 27, len(records)
    output = []
    for i,line in enumerate(lines):
        output.extend(before.get(i,[]))
        if i not in delete: output.append(line)
        output.extend(after.get(i,[]))
    return '\n'.join(output)+'\n', records


def instructions(text, token):
    return [l for l in text.splitlines() if token in l]


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    source = normalize(raw.decode())
    analysis = subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S','-o','-','-'],
                              input=source,text=True,capture_output=True,check=True).stdout
    surviving_reads = set(re.findall(r'^  (%[\w.]+) = load <4 x half>',analysis,re.M))
    wait, wait_changes = overlap_qk_tails(source)
    fragment, fragment_changes = prefetch_read_pairs(source,surviving_reads)
    combined, combined_changes = prefetch_read_pairs(wait,surviving_reads)
    variants = {'baseline':raw.decode(), 'wait_tail2':wait,
                'fragment_pair':fragment, 'combined':combined}
    config = json.loads(SOURCE.with_suffix('.launch.json').read_text())
    records = {}
    for name,text in variants.items():
        canonical = normalize(text) if name == 'baseline' else text
        for token in (DMA,MMA,READ,'= fmul <4 x half>',WAIT,BARRIER,'fence syncscope',
                      'llvm.hcu.ds.read.m32x16.b16.v4i32.p3('):
            assert instructions(canonical,token) == instructions(source,token), (name,token)
        # Only instruction ordering and compiler scheduling hints change.
        filtered = lambda s: Counter(l for l in s.splitlines() if 'call void @llvm.amdgcn.sched.barrier(' not in l)
        assert filtered(canonical) == filtered(source)
        output = HERE / (name+'.ll')
        output.write_text(text)
        (HERE/(name+'.launch.json')).write_text(json.dumps(config,indent=2)+'\n')
        subprocess.run([LLVM/'llvm-as',output,'-o','/dev/null'],check=True)
        # Preserve the optimized text as review evidence, not the benchmark input.
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify',output,'-S','-o',HERE/(name+'.opt.ll')],check=True)
        records[name] = dict(sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                            waits_static=text.count(WAIT), barriers_static=text.count(BARRIER),
                            sched_hints_static=text.count('call void @llvm.amdgcn.sched.barrier('),
                            LDS_bytes=16384, DMA_sites=text.count(DMA), MMAC_sites=text.count(MMA),
                            llvm_assemble_and_O3_verify=True)
        print(name, json.dumps(records[name]), flush=True)
    assert (HERE/'baseline.ll').read_bytes() == raw
    manifest = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=SOURCE_SHA256,
                    baseline_byte_identical=True, source_measured_median_ms=1.6460869908332825,
                    wait_changes=wait_changes, fragment_changes=fragment_changes,
                    variants=records, launch=config, gpu_tested=False,
                    limitations='Fingerprint-restricted scheduling experiment; 4 raw words per read pair is not a bound on final VGPR delta. Device codegen/performance must be measured.')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
