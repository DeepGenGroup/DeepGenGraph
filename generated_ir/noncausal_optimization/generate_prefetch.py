#!/usr/bin/env python3
"""Fingerprint-scoped non-causal K0 prefetch experiment, not a compiler pass.

Adapted from ../retained_cross_bn_k0/generate.py. Preserve all waits and
barriers; move the last V-reader handoff before the final 32 PV MMACs and
issue the next K0 after that handoff. Never issue a 33rd key tile.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'generated_ir/test_input_spidD_noncasual.ll'
SHA = 'a3509e3fbef30fe726e8e6bf83a99351c35a6fbae0f34e11825019ff54049fbd'
TOKEN = re.compile(r'%[\w.]+')
LLVM = Path('/data2/xsl/install/bin')


def transform(raw, early):
    text = re.sub(r'%(\d+)', lambda m: '%nc.n'+m[1] if int(m[1]) >= 5 else m[0], raw)
    text = re.sub(r'^(\d+):', lambda m: 'nc.n'+m[1]+':', text, flags=re.M)
    lines = text.splitlines()
    definitions = {m[1]: (i, line) for i, line in enumerate(lines)
                   if (m := re.match(r'  (%[\w.]+) = ', line))}
    branch = '  br label %nc.n125, !dbg !7\n'
    entry_end = lines.index(branch.rstrip())
    roots = ['%nc.n143', '%nc.n148']

    def copy_dma(prefix, bn):
        mapping = {'%nc.n126': bn}
        copied = []

        def clone(name):
            if name in mapping:
                return mapping[name]
            if name not in definitions or definitions[name][0] < entry_end:
                return name
            _, line = definitions[name]
            assert ' = phi ' not in line, line
            if ' call ' in line and name not in roots:
                assert 'readfirstlane' in line or 'asm sideeffect "", "=v,0"' in line, line
            rhs = line.split(' = ', 1)[1]
            for dep in TOKEN.findall(rhs):
                clone(dep)
            mapped = '%' + prefix + '.' + name[1:]
            mapping[name] = mapped
            copied.append('  '+mapped+' = '+TOKEN.sub(lambda m: mapping.get(m[0], m[0]), rhs))
            return mapped

        for root in roots:
            clone(root)
        return '\n'.join(copied)+'\n'

    prime = copy_dma('k0.prime', '0')
    future = copy_dma('k0.next', '%nc.n1152')
    for root in roots:
        line = definitions[root][1]+'\n'
        assert text.count(line) == 1
        text = text.replace(line, '')
    text = text.replace(branch, '; Prime K0 after the Q reader handoff.\n'+prime+branch, 1)
    handoff = ('  fence syncscope("workgroup") release, !dbg !7\n'
               '  call void @llvm.amdgcn.s.barrier(), !dbg !7\n'
               '  fence syncscope("workgroup") acquire, !dbg !7\n')
    advance = '  %nc.n1152 = add nuw nsw i64 %nc.n126, 1, !dbg !7\n'
    assert text.count(handoff+advance+branch) == 1
    text = text.replace(handoff+advance+branch, branch)
    anchor = definitions['%nc.n1076'][1]+'\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '; V3 readers complete before reusing low LDS.\n'+handoff+advance+anchor)
    guarded = ('  %k0.has.next = icmp ult i64 %nc.n1152, 32\n'
               '  br i1 %k0.has.next, label %k0.issue, label %k0.join\n\n'
               'k0.issue:\n'+future+'  br label %k0.join\n\n'
               'k0.join:\n')
    if early:
        text = text.replace(anchor, guarded+anchor)
    else:
        tail = definitions['%nc.n1151'][1]+'\n'
        text = text.replace(tail, tail+guarded)
    text = re.sub(r'^(.* = phi .*), %nc\.n134(.*)$', r'\1, %k0.join\2', text, flags=re.M)
    return text


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, 'input changed: re-audit LDS ownership and anchors'
    variants = {'baseline': raw.decode(), 'k0_late_control': transform(raw.decode(), False),
                'k0_prefetch': transform(raw.decode(), True)}
    manifest = {}
    for name, text in variants.items():
        path = HERE / (name+'.ll')
        path.write_text(text)
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path,
                        '-o', HERE/(name+'.o3.ll')], check=True)
        if name != 'baseline':
            optimized = (HERE/(name+'.o3.ll')).read_text()
            issue = optimized.index('%k0.next.nc.n148 =')
            first = optimized.index('%nc.n1080 =')
            last = optimized.index('%nc.n1133 =')
            assert (issue < first < last) if name == 'k0_prefetch' else (first < last < issue)
            if name == 'k0_prefetch':
                assert optimized[issue:last].count('v_mmac_') == 31
        manifest[name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
                              llvm_as_and_o3_verified=True, static_lds_bytes=16384)
    (HERE/'prefetch-manifest.json').write_text(json.dumps(dict(source=str(SOURCE),
        source_sha256=SHA, variants=manifest, gpu_tested=False,
        next_k0_overlap_mmac=32*31, key_tiles=32), indent=2)+'\n')


if __name__ == '__main__':
    main()
