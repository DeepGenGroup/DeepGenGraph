#!/usr/bin/env python3
"""Fingerprint-restricted LDS address-pair scheduling experiment, not a Pass."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / 'generated_ir/retained_local_schedule'
LLVM = Path('/data2/xsl/install/bin')
spec = importlib.util.spec_from_file_location('previous_schedule_generator', OLD/'generate.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
READ = previous.READ
MMA = previous.MMA
HINT = previous.HINT
DEF = re.compile(r'^  (%[\w.]+) = (.*)$')
PURE = re.compile(r'(getelementptr|add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|ptrtoint|inttoptr) ')


def definitions(lines):
    return {m[1]: (i,m[2]) for i,l in enumerate(lines) if (m := DEF.match(l))}


def address_pairs(source, surviving):
    lines = source.splitlines()
    defs = definitions(lines)
    def address(ptr):
        expr = defs[ptr][1]
        m = re.match(r'getelementptr i8, ptr addrspace\(3\) (%[\w.]+), i64 (-?\d+)(?:,|$)',expr)
        if m:
            base, offset = address(m[1])
            return base, offset+int(m[2])
        return ptr, 0
    reads = []
    for name,(index,expr) in defs.items():
        if READ in lines[index] and name in surviving:
            ptr = re.search(r'ptr addrspace\(3\) (%[\w.]+)',expr)[1]
            reads.append((index,name,*address(ptr)))
    pairs = []
    for low,low_name,base,offset in reads:
        matches = [(i,n) for i,n,b,o in reads if i>low and b==base and o==offset+4096]
        if not matches:
            continue
        high,high_name = matches[0]
        # Exactly two MMACs separate the original loads: the first consumes
        # the previous pair, the second consumes this pair's low half.
        # No other memory operation, hint, opaque call, or control flow may
        # be crossed. This guard also confines each pair to its LDS epoch.
        middle = lines[low+1:high]
        if not all(MMA in l or ((m:=DEF.match(l)) and PURE.match(m[2])) for l in middle):
            continue
        mmas = [i for i in range(low+1,high) if MMA in lines[i]]
        if len(mmas) != 2:
            continue
        anchor = mmas[0]
        if low_name in re.findall(r'%[\w.]+',lines[anchor]):
            continue
        assert low_name in re.findall(r'%[\w.]+',lines[mmas[1]])
        moved = {high}
        def visit(name):
            if name not in defs or defs[name][0] < anchor:
                return
            i,expr = defs[name]
            assert anchor < i < high and PURE.match(expr), (name,expr)
            if i in moved:
                return
            moved.add(i)
            for dep in re.findall(r'%[\w.]+',expr):
                visit(dep)
        for dep in re.findall(r'%[\w.]+',defs[high_name][1]):
            visit(dep)
        pairs.append(dict(loads=[low_name,high_name],base=base,
                          byte_offsets=[offset,offset+4096],
                          anchor=DEF.match(lines[anchor])[1],
                          anchor_index=anchor,moved_indices=sorted(moved)))
    assert len(pairs) == 28, len(pairs)
    assert len({p['anchor_index'] for p in pairs}) == 28
    return pairs


def transform(source, pairs, hints):
    lines = source.splitlines()
    removed = {i for p in pairs for i in p['moved_indices']}
    assert len(removed) == sum(len(p['moved_indices']) for p in pairs)
    anchors = {p['anchor_index']:p for p in pairs}
    output = []
    for i,line in enumerate(lines):
        if i in anchors:
            output.extend(lines[j] for j in anchors[i]['moved_indices'])
            if hints: output.append(HINT)
        if i not in removed: output.append(line)
        if i in anchors and hints: output.append(HINT)
    result = '\n'.join(output)+'\n'
    filt = lambda s: Counter(l for l in s.splitlines() if 'call void @llvm.amdgcn.sched.barrier(' not in l)
    assert filt(source) == filt(result)
    # Even the memory access ORDER is unchanged: only pure address operations
    # and the high read move over register MMACs. No FP reassociation occurs.
    for token in (READ,MMA,previous.DMA,previous.WAIT,previous.BARRIER,
                  'fence syncscope','llvm.hcu.ds.read.m32x16.b16.v4i32.p3(',
                  '= fmul <4 x half>'):
        assert previous.instructions(result,token) == previous.instructions(source,token)
    return result


def main():
    raw = previous.SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == previous.SOURCE_SHA256
    assert raw == (OLD/'baseline.ll').read_bytes()
    source = previous.normalize(raw.decode())
    opt = subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S','-o','-','-'],
                         input=source,text=True,capture_output=True,check=True).stdout
    surviving = set(re.findall(r'^  (%[\w.]+) = load <4 x half>',opt,re.M))
    pairs = address_pairs(source,surviving)
    variants = dict(baseline=raw.decode(),fragment_pair=(OLD/'fragment_pair.ll').read_text(),
                    ds_pair_prefetch=transform(source,pairs,True),
                    ds_pair_prefetch_soft=transform(source,pairs,False))
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    assert hashlib.sha256(variants['fragment_pair'].encode()).hexdigest() == old_manifest['variants']['fragment_pair']['sha256']
    records = {}
    for name,text in variants.items():
        path = HERE/(name+'.ll')
        path.write_text(text)
        (HERE/(name+'.launch.json')).write_bytes(previous.SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify',path,'-S','-o',HERE/(name+'.opt.ll')],check=True)
        optimized = (HERE/(name+'.opt.ll')).read_text()
        ods = definitions(optimized.splitlines())
        record = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
                      hints_static=text.count('call void @llvm.amdgcn.sched.barrier('),
                      vm_waits_static=text.count(previous.WAIT),
                      hardware_barriers_static=text.count(previous.BARRIER),
                      o3_ordinary_lds_loads=optimized.count(READ),
                      llvm_assemble_and_O3_verify=True)
        assert record['o3_ordinary_lds_loads'] == 80
        assert record['vm_waits_static'] == 12 and record['hardware_barriers_static'] == 18
        if name.startswith('ds_pair_'):
            olines = optimized.splitlines()
            for pair in pairs:
                a,b = [ods[v][0] for v in pair['loads']]
                anchor = ods[pair['anchor']][0]
                assert a<b<anchor, (name,pair)
                assert not any('@llvm.amdgcn.sched.barrier(' in l for l in olines[a:b+1])
                if name=='ds_pair_prefetch':
                    assert 'sched.barrier(i32 0)' in olines[anchor-1]
                    assert 'sched.barrier(i32 0)' in olines[anchor+1]
            record['o3_address_pairs_before_independent_mmac'] = len(pairs)
            record['o3_pair_internal_sched_barriers'] = 0
        records[name] = record
        print(name,json.dumps(record),flush=True)
    manifest = dict(source=str(previous.SOURCE.relative_to(ROOT)),source_sha256=previous.SOURCE_SHA256,
                    pairs=pairs,variants=records,baseline_byte_identical=True,
                    previous_fragment_pair_byte_identical=True,gpu_tested=False,
                    limitations='Address pairing and O3 checks do not prove gfx936 DS combining. Actual assembly, VGPR/scratch and performance require target DTK compilation.')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
