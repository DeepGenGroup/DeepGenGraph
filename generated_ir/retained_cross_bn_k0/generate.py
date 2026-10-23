#!/usr/bin/env python3
"""Fingerprint-scoped cross-BN K0 experiment; not a general compiler pass."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'generated_ir/q_bk_retention_sweep/retained_32words.ll'
SHA = 'c5627d7e797e5ee93c73ba04d7112a238ba4dbcb52a888267a4c7b30520f9915'
TOKEN = re.compile(r'%[\w.]+')


def transform(raw, early):
    # Name numeric locals so moving definitions cannot invalidate LLVM numbering.
    text = re.sub(r'%(\d+)', lambda m: '%bnk.n'+m[1] if int(m[1]) >= 5 else m[0], raw)
    text = re.sub(r'^(\d+):', lambda m: 'bnk.n'+m[1]+':', text, flags=re.M)
    lines = text.splitlines()
    definitions = {m[1]: (i,line) for i,line in enumerate(lines)
                   if (m := re.match(r'  (%[\w.]+) = ',line))}
    entry_end = lines.index('  br label %bnk.n204, !dbg !13')
    roots = ['%bnk.n239', '%bnk.n253']

    def copy_dma(prefix, bn):
        mapping = {'%bnk.n205': bn}
        copied = []
        def clone(name):
            if name in mapping:
                return mapping[name]
            if name not in definitions or definitions[name][0] < entry_end:
                return name
            _, line = definitions[name]
            assert ' = phi ' not in line, line
            # These roots are DMA, all other dependencies are address arithmetic,
            # the original thread lifetime identity, or readfirstlane.
            if ' call ' in line and name not in roots:
                assert 'readfirstlane' in line or 'asm sideeffect "", "=v,0"' in line, line
            rhs = line.split(' = ',1)[1]
            for dep in TOKEN.findall(rhs):
                clone(dep)
            mapped = '%' + prefix + '.' + name[1:]
            mapping[name] = mapped
            copied.append('  '+mapped+' = '+TOKEN.sub(lambda m: mapping.get(m[0],m[0]),rhs))
            return mapped
        for root in roots:
            clone(root)
        return '\n'.join(copied)+'\n'

    prime = copy_dma('k0.prime', '0')
    future = copy_dma('k0.next', '%bnk.n1580')
    for root in roots:
        line = definitions[root][1]+'\n'
        assert text.count(line) == 1
        text = text.replace(line,'')
    entry_branch = '  br label %bnk.n204, !dbg !13\n'
    # N is 1..32 for this fixed valid launch, so initial K0 is always needed.
    text = text.replace(entry_branch, '; Prime K0 for BN=0 after the Q handoff.\n'+prime+entry_branch, 1)
    handoff = ('  fence syncscope("workgroup") release, !dbg !13\n'
               '  call void @llvm.amdgcn.s.barrier(), !dbg !13\n'
               '  fence syncscope("workgroup") acquire, !dbg !13\n')
    advance = '  %bnk.n1580 = add nuw nsw i64 %bnk.n205, 1, !dbg !13\n'
    old_tail = handoff+advance+entry_branch
    assert text.count(old_tail) == 1
    text = text.replace(old_tail,entry_branch)
    anchor = definitions['%bnk.n1510'][1]+'\n'
    # Transfer ownership only once every wave has completed all final V reads.
    text = text.replace(anchor,'; V3 readers complete before low LDS is reused.\n'+handoff+advance+anchor)
    guarded = ('  %k0.has.next = icmp slt i64 %bnk.n1580, %bnk.n191\n'
               '  br i1 %k0.has.next, label %k0.issue, label %k0.join\n\n'
               'k0.issue:\n'+future+'  br label %k0.join\n\n'
               'k0.join:\n')
    if early:
        text = text.replace(anchor,guarded+anchor)
    else:
        # Same prologue, address construction, guard and reader barrier; no PV
        # arithmetic between future K0 issue and the next BN's existing wait.
        tail = definitions['%bnk.n1579'][1]+'\n'
        assert text.count(tail) == 1
        text = text.replace(tail,tail+guarded)
    # Loop-carried values now come from the newly split final block.
    text = re.sub(r'^(.* = phi .*), %bnk\.n847(.*)$', r'\1, %k0.join\2', text, flags=re.M)
    return text


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, 'baseline changed; re-audit anchors and LDS ownership'
    variants = {'baseline': raw.decode(), 'k0_late_control': transform(raw.decode(),False),
                'k0_prefetch_pv_tail': transform(raw.decode(),True)}
    info = {}
    for name,text in variants.items():
        (HERE/(name+'.ll')).write_text(text)
        (HERE/(name+'.launch.json')).write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        info[name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
                          static_lds_bytes=16384, retained_words=32,
                          overlap_pv_mmac_per_nonfinal_bn=32 if name=='k0_prefetch_pv_tail' else 0)
    (HERE/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)),
        source_sha256=SHA, variants=info, target='gfx936',
        assumptions='Fixed BHSD=(1,32,4096,128), BM=BN=128, BK=32, 256 threads; valid launch N=1..32.',
        dynamic_counts='For N BN iterations: K DMA=8N, V DMA=8N, Q DMA=8, waits=4+8N, barriers=5+12N; unchanged.',
        algorithm='Prime K0(0); guarded K0(n+1) in low LDS after V3 reader handoff; retain next-header wait/publication.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
