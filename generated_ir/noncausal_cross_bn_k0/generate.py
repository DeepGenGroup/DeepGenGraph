#!/usr/bin/env python3
"""Fingerprint-scoped noncausal K0 prefetch experiment, not a compiler pass."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'noncausal_qk_tail_prefetch_pass/k_tail_prefetch4.ll'
SHA = '314b4dc079c3be8b90c93b1154e300b9092ed4757972598004aa60a081fc885a'
TOKEN = re.compile(r'%[\w.]+')
ROOTS = ('%i104', '%i105', '%i107', '%i108')
LLVM = Path('/data2/xsl/install/bin')


def transform(text, early):
    lines = text.splitlines()
    definitions = {m[1]: (i, line) for i, line in enumerate(lines)
                   if (m := re.match(r'  (%[\w.]+) = ', line))}
    branch = '  br label %bb86, !dbg !64\n'
    entry_end = lines.index(branch.rstrip())
    dependent = {'%i87': True}

    def depends(name):
        if name not in dependent:
            dependent[name] = (name in definitions and
                definitions[name][0] > entry_end and
                any(depends(dep) for dep in TOKEN.findall(definitions[name][1].split(' = ', 1)[1])))
        return dependent[name]

    def copy_dma(prefix, bn, prime):
        mapping = {'%i87': bn}
        copied = []

        def clone(name):
            if name in mapping:
                return mapping[name]
            if name not in definitions or definitions[name][0] < entry_end:
                return name
            if not prime and name not in ROOTS and not depends(name):
                return name
            line = definitions[name][1]
            assert ' = phi ' not in line, line
            if ' call ' in line and name not in ROOTS:
                assert 'readfirstlane' in line or 'asm sideeffect "", "=v,0"' in line, line
            rhs = line.split(' = ', 1)[1]
            for dep in TOKEN.findall(rhs):
                clone(dep)
            mapped = '%' + prefix + '.' + name[1:]
            mapping[name] = mapped
            copied.append('  ' + mapped + ' = ' + TOKEN.sub(lambda m: mapping.get(m[0], m[0]), rhs))
            return mapped

        for root in ROOTS:
            clone(root)
        return '\n'.join(copied) + '\n'

    prime = copy_dma('k0.prime', '0', True)
    future = copy_dma('k0.next', '%i1020', False)
    for root in ROOTS:
        text = text.replace(definitions[root][1] + '\n', '', 1)
    text = text.replace(branch, '; Prime K0(0) after the retained Q reader handoff.\n' + prime + branch, 1)
    advance = definitions['%i1020'][1] + '\n'
    text = text.replace(advance, '', 1)
    guarded = (advance + '  %k0.has.next = icmp ult i64 %i1020, 32\n'
               '  br i1 %k0.has.next, label %k0.issue, label %k0.join\n\n'
               'k0.issue:\n' + future + '  br label %k0.join\n\n'
               'k0.join:\n')
    if early:
        anchor = definitions['%i928'][1] + '\n'
        text = text.replace(anchor, '; V2 readers released low LDS; V3 reads only high LDS.\n' + guarded + anchor, 1)
    else:
        # Same prologue/guard/address work, issued after all PV arithmetic and
        # the existing final reader handoff. No extra hardware barriers.
        assert text.count(branch) == 2
        pos = text.rindex(branch)
        text = text[:pos] + guarded + text[pos:]
    text = re.sub(r'^(.* = phi .*), %bb95(.*)$', r'\1, %k0.join\2', text, flags=re.M)
    text = text.replace('bb86:                                             ; preds = %bb95, %bb', 'bb86:')
    # Every original instruction except the four relocated K0 DMA and moved
    # loop advance/PHI predecessor remains byte-for-byte in its original order.
    original = [line for line in lines if line.startswith('  ') and
                line not in [definitions[r][1] for r in ROOTS] and
                line != advance.rstrip() and ' = phi ' not in line]
    retained = [line for line in text.splitlines() if line in set(original)]
    assert retained == original
    return text


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, 'baseline changed: re-audit anchors and LDS ownership'
    variants = {'control': raw.decode(), 'k0_late_control': transform(raw.decode(), False),
                'k0_prefetch_pv_tail': transform(raw.decode(), True)}
    info = {}
    for name, text in variants.items():
        path = HERE / (name + '.ll')
        path.write_text(text)
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM / 'llvm-as', path, '-o', '/dev/null'], check=True)
        opt = path.with_suffix('.o3.ll')
        subprocess.run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', path, '-o', opt], check=True)
        if name != 'control':
            optimized = opt.read_text()
            dma = optimized.index('%k0.next.i108 =')
            first, last = [optimized.index('%i' + str(n) + ' =') for n in (947, 1001)]
            assert (dma < first < last) if name == 'k0_prefetch_pv_tail' else (first < last < dma)
        info[name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
            o3_sha256=hashlib.sha256(opt.read_bytes()).hexdigest(), LLVM_and_O3_verified=True,
            overlap_PV_MMAC_per_nonfinal_tile=32 if name == 'k0_prefetch_pv_tail' else 0)
        print('PASS generated and LLVM/O3 verified', name, flush=True)
    (HERE / 'manifest.json').write_text(json.dumps(dict(source=str(SOURCE), source_sha256=SHA,
        variants=info, target='gfx936', shape=[1, 32, 4096, 128], threads=256,
        BM=128, BN=128, BK=32, static_LDS_bytes=16384, inherited_k_tail_prefetch=4,
        algorithm='Prime K0(0); guard next tile <32; issue K0 into low LDS after V3-ready barrier; retain final reader barrier and next K0 wait/publication.',
        expected_dynamic_counts=dict(K_DMA64=512, V_DMA128=256, Q_DMA128=8,
            vm_waits=260, hardware_barriers=293, MMAC=8192), gpu_tested=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
