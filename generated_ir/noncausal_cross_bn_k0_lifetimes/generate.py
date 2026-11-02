#!/usr/bin/env python3
"""Scope-limited candidates for the measured cross-tile K0 spill regression."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'noncausal_cross_bn_k0'
LLVM = Path('/data2/xsl/install/bin')
TOKEN = re.compile(r'%[\w.]+')
ROOTS = ('%i104', '%i105', '%i107', '%i108')


def anchor_sums(text):
    # Force each exact FP32 add to finish at its fragment, using the same
    # tied-register identity idiom as the existing register-lifetime pass.
    # Merely placing fadd early in IR did not prevent the backend from moving
    # the complete reduction to k0.join in the measured gfx936 assembly.
    result = []
    count = 0
    for line in text.splitlines():
        m = re.match(r'  (%[\w.]+) = fadd float ', line)
        if m and (m[1].startswith('%local.reduce') or m[1] in ('%i588', '%i589', '%i696', '%i697')):
            name = m[1]
            result.append(line.replace(name + ' =', name + '.before.anchor =', 1))
            result.append(f'  {name} = call float asm sideeffect "", "=v,0"(float {name}.before.anchor) #9, !dbg !96, !frisk.register_lifetimes !58')
            count += 1
        else:
            result.append(line)
    assert count == 64
    return '\n'.join(result) + '\n'


def remat_next(text, baseline):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*)$', baseline, re.M))
    mapping = {'%i87': '%i1020', '%kdma.tid': '%k0.remat.tid',
               '%address.thread498': '%k0.remat.tid', '%i97': '%i97',
               '%dma.buffer.rsrc1373': '%dma.buffer.rsrc1373'}
    copied = ['  %k0.remat.tid = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !66, !range !62, !frisk.register_lifetimes !58']

    def clone(name):
        if name in mapping:
            return mapping[name]
        assert name in definitions, name
        rhs = definitions[name]
        assert 'phi ' not in rhs and 'load ' not in rhs, rhs
        if 'call ' in rhs and name not in ROOTS:
            assert 'readfirstlane' in rhs, rhs
        for dep in TOKEN.findall(rhs):
            clone(dep)
        mapped = '%k0.remat.' + name[1:]
        mapping[name] = mapped
        copied.append('  ' + mapped + ' = ' + TOKEN.sub(lambda m: mapping[m[0]], rhs))
        return mapped

    for root in ROOTS:
        clone(root)
    start = text.index('k0.issue:\n') + len('k0.issue:\n')
    end = text.index('  br label %k0.join', start)
    return text[:start] + '\n'.join(copied) + '\n' + text[end:]


def after16(text):
    # Relocate the whole uniform guard + issue + join; leave all PV arithmetic,
    # LDS reads and hardware synchronization exactly in their original order.
    start = text.index('  %i1020 =')
    end = text.index('k0.join:\n', start) + len('k0.join:\n')
    guard = text[start:end]
    text = text[:start] + text[end:]
    start_pv = text.index('  %i928 =')
    mma = list(re.finditer(r'^  %i\d+ = .*v_mmac_[^\n]*\n', text[start_pv:], re.M))
    assert len(mma) == 32
    pos = start_pv + mma[15].end()
    return text[:pos] + guard + text[pos:]


def audit(text, overlap, remat):
    current = None
    anchored = []
    for line in text.splitlines():
        if re.match(r'^[\w.]+:', line):
            current = line.split(':')[0]
        if re.search(r'call float asm sideeffect "", "=v,0"', line):
            anchored.append(current)
    assert len(anchored) == 64 and set(anchored) == {'bb95'}, anchored
    issue = text.index('k0.issue:')
    join = text.index('k0.join:')
    end = text.index('bb1021:')
    assert text[join:end].count('v_mmac_') == overlap
    if remat:
        block = text[issue:join]
        defs = set(re.findall(r'^  (%[\w.]+) =', block, re.M))
        external = set(TOKEN.findall(block)) - defs - {'%k0.join', '%bb95'}
        assert external <= {'%kdma.tid', '%i1020', '%i97', '%dma.buffer.rsrc1373',
                            '%k0.prime.i97', '%k0.prime.dma.buffer.rsrc1373'}, external
        assert '%k0.remat.tid = ' in block
    return dict(anchored_FP32_adds=64, anchors_before_K0_guard=True,
                remaining_PV_MMAC=overlap, next_K0_addresses_rematerialized=remat)


def main():
    manifest = json.loads((OLD / 'manifest.json').read_text())
    baseline = (OLD / 'control.ll').read_text()
    original = (OLD / 'k0_prefetch_pv_tail.ll').read_text()
    for name, text in [('control', baseline), ('k0_prefetch_pv_tail', original)]:
        assert hashlib.sha256(text.encode()).hexdigest() == manifest['variants'][name]['sha256']
    anchored = anchor_sums(original)
    remat = remat_next(anchored, baseline)
    variants = {'control': (baseline, 0, False), 'k0_sum_anchor': (anchored, 32, False),
                'k0_sum_anchor_remat': (remat, 32, True),
                'k0_sum_anchor_remat_after16': (after16(remat), 16, True)}
    records = {}
    for name, (text, overlap, rebuilt) in variants.items():
        path = HERE / (name + '.ll')
        path.write_text(text)
        path.with_suffix('.launch.json').write_bytes((OLD / 'control.launch.json').read_bytes())
        subprocess.run([LLVM / 'llvm-as', path, '-o', '/dev/null'], check=True)
        opt = path.with_suffix('.o3.ll')
        subprocess.run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', path, '-o', opt], check=True)
        checks = {}
        if name != 'control':
            checks = audit(text, overlap, rebuilt)
            assert audit(opt.read_text(), overlap, rebuilt) == checks
        # Check arithmetic opcodes, operands and order ignoring the new tied
        # identity and metadata (full values are also checked by verify.py).
        def fp_ops(s):
            return [re.sub(r', !.*', '', l.replace('.before.anchor', '')) for l in s.splitlines()
                    if re.search(r' = (fadd|fmul|fsub|fptrunc|fpext) |v_mmac_', l)]
        assert fp_ops(text) == fp_ops(baseline)
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            o3_sha256=hashlib.sha256(opt.read_bytes()).hexdigest(), overlap=overlap,
            LLVM_and_O3_verified=True, checks=checks)
        print('PASS generated', name, checks, flush=True)
    (HERE / 'manifest.json').write_text(json.dumps(dict(variants=records,
        source=str(OLD / 'k0_prefetch_pv_tail.ll'), source_sha256=manifest['variants']['k0_prefetch_pv_tail']['sha256'],
        target='gfx936', shape=[1, 32, 4096, 128], static_LDS_bytes=16384,
        gpu_tested=False, target_codegen_tested=False,
        note='Candidates only: absence of spill must be verified with DTK assembly.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
