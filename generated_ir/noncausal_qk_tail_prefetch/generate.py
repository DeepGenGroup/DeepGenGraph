#!/usr/bin/env python3
"""Scoped IR experiment: complete K3 paired LDS prefetch on the measured IR.

This is an artifact-specific scheduling experiment, not a compiler pass.
The input hash and structural assertions deliberately reject other kernels.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'noncausal_softmax_fragments/softmax_pv.ll'
EXPECTED_SHA = '5183329f5984b2d841ccb1a1b1561cde44708c17123f1a23a192ecbf20c4dc4a'
LLVM = Path(os.environ.get('IR_CHECK_LLVM_DIR', '/data2/xsl/install/bin'))
PAIR_TAG = 'frisk.k_tail_pair'
HINT_TAG = 'frisk.k_tail_prefetch'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(text):
    lines = text.splitlines()
    pairs = defaultdict(list)
    for i, line in enumerate(lines):
        if m := re.search(r'!frisk.k_tail_pair !(\d+)', line):
            pairs[m[1]].append(i)
    hints = [i for i, l in enumerate(lines) if '@llvm.amdgcn.sched.barrier(' in l]
    result = []
    for positions in pairs.values():
        assert len(positions) == 2
        a, b = positions
        assert not any(a < h < b for h in hints), 'hint split an LDS pair'
        names = [re.match(r'  (%[\w.]+) =', lines[p])[1] for p in positions]
        uses = [i for i, l in enumerate(lines) if 'v_mmac_' in l
                and any(re.search(re.escape(n) + r'(?=[, )])', l) for n in names)]
        assert uses and b < min(uses)
        lead = sum('v_mmac_' in l for l in lines[b + 1:min(uses)])
        result.append(dict(loads=names, independent_mmacs_before_first_use=lead))
    return dict(pairs=result, pair_count=len(pairs),
                added_schedule_hints=sum(HINT_TAG in l and 'call void' in l for l in lines))


def transform(source, distance):
    assert distance in (2, 4, 8)
    lines = source.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('  %i300 = load '))
    end = next(i for i, l in enumerate(lines) if l.startswith('  %i414 = call '))
    region = lines[start:end]
    # Transparent FP max and extract operations remain in their original order.
    # No DMA, wave operation, store, wait, fence or hardware barrier is crossed.
    for l in region:
        assert re.match(r'  %[\w.]+ = (?:load <4 x half>, ptr addrspace\(3\) .*align 8,|'
                        r'call <4 x float> asm "v_mmac_|extractelement |'
                        r'call float @llvm.maxnum.f32\()', l), l
    mmas = [l for l in region if 'v_mmac_' in l]
    loads = [l for l in region if ' = load ' in l]
    assert len(loads) == 16 and len(mmas) == 32
    pointers = {}
    for l in lines[:start]:
        if m := re.match(r'  (%[\w.]+) = getelementptr i8, ptr addrspace\(3\) (%[\w.]+), i64 (\d+),', l):
            pointers[m[1]] = (m[2], int(m[3]))
    records = []
    for l in loads:
        m = re.match(r'  (%[\w.]+) = load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 8,', l)
        name, pointer = m.groups()
        base, offset = pointers[pointer]
        assert offset >= 8192 and offset + 8 <= 16384, 'K3 must use upper slot'
        first = next(i for i, mma in enumerate(mmas)
                     if re.search(re.escape(name) + r'(?=[, )])', mma))
        records.append(dict(name=name, pointer=pointer, base=base, offset=offset, first=first, line=l))
    # Match the existing read2 pairs used for the preceding upper-slot K tile.
    # Their offsets differ by 128 bytes; no LDS layout/address changes.
    pairs, consumed = [], set()
    for a in records:
        if a['name'] in consumed:
            continue
        bs = [b for b in records if b['name'] not in consumed and b['base'] == a['base']
              and b['offset'] == a['offset'] + 128]
        assert len(bs) == 1
        b = bs[0]
        assert a['first'] < b['first']
        pairs.append((a, b))
        consumed.update((a['name'], b['name']))
    assert len(pairs) == 8 and len(consumed) == 16
    md_base = 1 + max(map(int, re.findall(r'^!(\d+) =', source, re.M)))
    batches = defaultdict(list)
    for p, (a, b) in enumerate(pairs):
        target = max(0, a['first'] - distance)
        for item in (a, b):
            # Every pointer definition already dominates the whole region.
            assert any(l.startswith('  ' + item['pointer'] + ' =') for l in lines[:start])
            batches[target].append(item['line'] + f', !{PAIR_TAG} !{md_base + p}')
    hint = f'  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !{HINT_TAG} !58'
    out, mma_index = [], 0
    for l in region:
        if l in loads:
            continue
        if 'v_mmac_' in l:
            if mma_index in batches:
                out.extend(batches[mma_index])
                out.append(hint)
            out.append(l)
            if mma_index in batches:
                out.append(hint)
            mma_index += 1
        else:
            out.append(l)
    result = '\n'.join(lines[:start] + out + lines[end:] +
                       [f'!{md_base + p} = !{{i32 {p}}}' for p in range(8)]) + '\n'
    # Strong textual proof: only these loads move; all old operands, flags,
    # arithmetic order, memory effects and pre-existing scheduling hints survive.
    def original_instructions(text):
        return [re.sub(r', !frisk.k_tail_pair !\d+', '', l)
                for l in text.splitlines() if l.startswith('  ') and HINT_TAG not in l]
    before, after = original_instructions(source), original_instructions(result)
    assert Counter(before) == Counter(after)
    assert [l for l in before if l not in loads] == [l for l in after if l not in loads]
    original_fixed = [l for l in region if l not in loads]
    assert [l for l in out if HINT_TAG not in l and PAIR_TAG not in l] == original_fixed
    report = inspect(result)
    assert report['pair_count'] == 8
    assert [p['independent_mmacs_before_first_use'] for p in report['pairs']] == [
        min(distance, a['first']) for a, b in pairs]
    return result, report


def main():
    assert digest(SOURCE) == EXPECTED_SHA, 'baseline changed; review before regenerating'
    source = SOURCE.read_text()
    variants = {}
    for name, d in [('control', 0)] + [(f'k_tail_prefetch{d}', d) for d in (2, 4, 8)]:
        path = HERE / (name + '.ll')
        if d:
            text, report = transform(source, d)
            path.write_text(text)
        else:
            path.write_bytes(SOURCE.read_bytes())
            report = inspect(source)
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM / 'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = path.with_suffix('.o3.ll')
        subprocess.run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', path, '-o', optimized], check=True)
        o3 = inspect(optimized.read_text())
        if d:
            assert o3 == report, (name, report, o3)
        variants[name] = dict(distance=d, sha256=digest(path), schedule=report, O3_schedule=o3,
                              llvm_verified=True, O3_verified=True,
                              identical_original_instruction_multiset=True,
                              identical_order_except_16_tail_K_loads=True)
        print('PASS', name, report, flush=True)
    (HERE / 'manifest.json').write_text(json.dumps(dict(
        source=str(SOURCE), source_sha256=EXPECTED_SHA, generator_sha256=digest(Path(__file__)),
        transformation='Artifact-specific Python IR schedule experiment; not a compiler pass',
        variants=variants, gpu_tested=False, gfx936_codegen_tested=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
