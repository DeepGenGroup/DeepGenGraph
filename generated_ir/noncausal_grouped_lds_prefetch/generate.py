#!/usr/bin/env python3
"""Assembly-guided, fingerprinted K read-group scheduling experiment.

Not a general compiler pass. Map the 32 actual ds_read2 groups to their
surviving LLVM loads, then move complete pairs before independent QK MMACs.
Only ordinary LDS reads and pure address dependencies move. Never cross a
DMA, hardware wait/barrier, store, branch or opaque call. No guessed waitcnt.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "generated_ir/noncausal_frag_prefetch_forward/control/kernel.ll"
ASSEMBLY = ROOT / "ktest.hsaco.amdgcn"
LLVM = Path("/data2/xsl/install/bin")
SOURCE_SHA = "70d1db321015d2c90f189e4c3ab802e2a66d3bdf90f8593e6157035909adc7ff"
ASM_SHA = "77c7e4c0a886e3bd4fc63441100186ff022772ebb19b07a381767f759e490aea"
TOKEN = re.compile(r'%[\w.]+')
DEF = re.compile(r'^  (%[\w.]+) = (.*)$')
READ = 'load <4 x half>, ptr addrspace(3)'
MMA = 'call <4 x float> asm "v_mmac_f32_16x16x16_f16 '
PURE = re.compile(r'(getelementptr|add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|ptrtoint|inttoptr) ')
HINT = '  call void @llvm.amdgcn.sched.barrier(i32 0) #99'
VARIANT = 'group_prefetch2_protected'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(text):
    text = re.sub(r'%(\d+)\b', lambda m: m[0] if int(m[1]) < 5 else '%grp.n'+m[1], text)
    return re.sub(r'^(\d+):', lambda m: 'grp.n'+m[1]+':', text, flags=re.M)


def definitions(lines):
    return {m[1]: (i, m[2]) for i, line in enumerate(lines) if (m := DEF.match(line))}


def qk_mmas(text):
    # Full retained-Q baseline: first 128 MMACs are QK, following 128 are PV.
    return [line for line in text.splitlines() if MMA in line][:128]


def asm_groups(text, assembly):
    mmas = qk_mmas(text)
    defs = definitions(text.splitlines())
    groups, registers, accumulator_defs = [], {}, {}
    llvm_results = {}
    ordinal = 0
    for line in assembly.split('.LBB0_1:', 1)[1].splitlines():
        if ordinal == 128:
            break
        if match := re.match(r'\s*(ds_read2\w+) v\[(\d+):(\d+)\]', line):
            lo, hi = map(int, match.groups()[1:])
            assert hi-lo == 3
            group = dict(assembly=line.strip(), loads=[], uses=[], halves={})
            groups.append(group)
            for reg in range(lo, hi+1):
                registers[reg] = (len(groups)-1, (reg-lo)//2)
        if 'v_mmac_f32_16x16x16_f16' not in line:
            continue
        reg_ranges = [tuple(map(int, pair)) for pair in re.findall(r'v\[(\d+):(\d+)\]', line)]
        result, _, k_regs, acc = reg_ranges
        assert result == acc
        args = re.search(r'"\(<4 x half> (%[\w.]+), <4 x half> (%[\w.]+), <4 x float> (%[\w.]+|zeroinitializer)\)', mmas[ordinal])
        assert args, mmas[ordinal]
        # Check MMAC chain correspondence instead of relying only on ordinal.
        if args[3] in llvm_results:
            assert accumulator_defs[result] == llvm_results[args[3]]
        llvm_results[DEF.match(mmas[ordinal])[1]] = ordinal
        accumulator_defs[result] = ordinal
        gid, half = registers[k_regs[0]]
        assert registers[k_regs[1]] == (gid, half)
        value = args[2]
        assert defs[value][1].startswith(READ), defs[value]
        group = groups[gid]
        assert group['halves'].get(half, value) == value
        group['halves'][half] = value
        group['uses'].append(ordinal)
        ordinal += 1
    assert ordinal == 128 and len(groups) == 32
    for group in groups:
        assert len(group['halves']) == 2
        group['loads'] = sorted(group.pop('halves').values(), key=lambda v: defs[v][0])
    assert len({v for g in groups for v in g['loads']}) == 64
    return groups


def protect_read_groups(lines, groups, anchors):
    """Close every original read pair before inserting any scheduling hints.

    An unprefetched pair can straddle another group's anchor. Hoist its later
    half and pure address dependencies next to its earlier half, within the
    same immutable LDS epoch. Fail closed if this cannot be done safely.
    """
    repairs = []
    for gid, group in enumerate(groups):
        defs = definitions(lines)
        first, last = sorted(group['loads'], key=lambda value: defs[value][0])
        lo, hi = defs[first][0], defs[last][0]
        crossed = [name for name in anchors if lo < defs[name][0] < hi]
        if not crossed:
            continue
        insertion = lo + 1
        moving = {hi}

        def address(value):
            if value not in defs or defs[value][0] < insertion:
                return True
            index, expr = defs[value]
            if index in moving:
                return True
            assert PURE.match(expr), (gid, 'non-pure dependency', value, expr)
            moving.add(index)
            return all(address(dep) for dep in TOKEN.findall(expr))

        assert all(address(dep) for dep in TOKEN.findall(defs[last][1]))
        assert max(moving) == hi, (gid, 'dependency after load')
        for line in lines[insertion:hi+1]:
            if not line.strip() or line.lstrip().startswith(';'):
                continue
            match = DEF.match(line)
            assert match and (PURE.match(match[2]) or match[2].startswith(READ)
                              or MMA in line), (gid, 'unsafe movement boundary', line)
        block = [lines[i] for i in sorted(moving)]
        output = []
        for i, line in enumerate(lines):
            if i == insertion:
                output.extend(block)
            if i not in moving:
                output.append(line)
        lines = output
        repairs.append(dict(group=gid, loads=group['loads'],
                            crossed_anchors=sorted(crossed, key=lambda name: defs[name][0]),
                            moved_definitions=[DEF.match(line)[1] for line in block]))
    # Check ALL groups, including those skipped by the prefetch transformation.
    defs = definitions(lines)
    for gid, group in enumerate(groups):
        lo, hi = sorted(defs[value][0] for value in group['loads'])
        assert not any(lo < defs[name][0] < hi for name in anchors), gid
    return lines, repairs


def assert_groups_unbroken(lines, groups):
    defs = definitions(lines)
    for gid, group in enumerate(groups):
        lo, hi = sorted(defs[value][0] for value in group['loads'])
        assert not any('@llvm.amdgcn.sched.barrier(' in line
                       for line in lines[lo:hi+1]), (gid, group['loads'])


def transform(source, groups, lookahead=2):
    lines = source.splitlines()
    mma_names = [DEF.match(line)[1] for line in qk_mmas(source)]
    records, skipped = [], []
    anchors = set()
    for gid, group in enumerate(groups):
        first = min(group['uses'])
        phase_start = first//32*32
        # Start-of-BK reads have no earlier independent compute in this epoch.
        if first == phase_start:
            skipped.append(dict(group=gid, reason='first MMAC in BK'))
            continue
        success = False
        for ordinal in range(max(phase_start, first-lookahead), first):
            defs = definitions(lines)
            anchor = mma_names[ordinal]
            anchor_index = defs[anchor][0]
            positions = [defs[v][0] for v in group['loads']]
            moving = set(positions)
            def address(value):
                if value not in defs or defs[value][0] < anchor_index:
                    return True
                index, expr = defs[value]
                if index in moving:
                    return True
                if not PURE.match(expr):
                    return False
                moving.add(index)
                return all(address(dep) for dep in TOKEN.findall(expr))
            if not all(address(dep) for v in group['loads'] for dep in TOKEN.findall(defs[v][1])):
                continue
            lo, hi = min(anchor_index, *moving), max(anchor_index, *moving)
            # Complete movement interval contains only immutable LDS reads,
            # register MMACs and pure address operations, all in one epoch.
            allowed = True
            for line in lines[lo:hi+1]:
                if not line.strip() or line.lstrip().startswith(';'):
                    continue
                match = DEF.match(line)
                if not match or not (PURE.match(match[2]) or match[2].startswith(READ) or MMA in line):
                    allowed = False
                    break
            if not allowed:
                continue
            # No value can be delayed past a user. Hoisted dependencies must
            # dominate each moved instruction (also verified by LLVM below).
            if any(defs[v][0] < anchor_index and any(v in TOKEN.findall(line.split(' = ', 1)[-1])
                   for line in lines[defs[v][0]+1:anchor_index]) for v in group['loads']):
                continue
            block = [lines[i] for i in sorted(moving)]
            output = []
            for i, line in enumerate(lines):
                if i == anchor_index:
                    output.extend(block)
                if i not in moving:
                    output.append(line)
            lines = output
            anchors.add(anchor)
            records.append(dict(group=gid, loads=group['loads'], anchor=anchor,
                                first_consumer=mma_names[first], lookahead_mmac=first-ordinal,
                                moved_definitions=[DEF.match(l)[1] for l in block]))
            success = True
            break
        if not success:
            skipped.append(dict(group=gid, reason='address dependency or epoch boundary'))
    lines, repairs = protect_read_groups(lines, groups, anchors)
    # Bracket the independent compute only after protecting all original pairs.
    result = []
    for line in lines:
        match = DEF.match(line)
        if match and match[1] in anchors:
            result.extend([HINT, line, HINT])
        else:
            result.append(line)
    text = '\n'.join(result)+'\n'
    assert_groups_unbroken(result, groups)
    assert Counter(source.splitlines()) == Counter(l for l in text.splitlines() if l != HINT)
    assert [line for line in source.splitlines() if MMA in line] == [line for line in result if MMA in line]
    text += '\ndeclare void @llvm.amdgcn.sched.barrier(i32 immarg) #99\n'
    text += 'attributes #99 = { convergent nounwind memory(inaccessiblemem: readwrite) }\n'
    assert len(records) > 4, records
    return text, records, skipped, repairs


def main():
    assert digest(SOURCE) == SOURCE_SHA and digest(ASSEMBLY) == ASM_SHA
    raw = named(SOURCE.read_text())
    source = subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', '-o', '-', '-'],
                            input=raw, text=True, capture_output=True, check=True).stdout
    (HERE/'control.ll').write_text(source)
    groups = asm_groups(source, ASSEMBLY.read_text())
    candidate, records, skipped, repairs = transform(source, groups)
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, assembly_sha256=ASM_SHA,
                    group_size=4, reverse=0, lookahead_mmac=2, groups=groups,
                    moved_groups=records, skipped_groups=skipped, protected_groups=repairs,
                    all_original_groups_protected=True, gpu_tested=False, variants={})
    for name, text in [('control', source), (VARIANT, candidate)]:
        path = HERE/(name+'.ll')
        path.write_text(text)
        (HERE/(name+'.launch.json')).write_bytes((SOURCE.parent/'launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path, '-o', optimized], check=True)
        opt_lines = optimized.read_text().splitlines()
        od = definitions(opt_lines)
        assert_groups_unbroken(opt_lines, groups)
        if name != 'control':
            for record in records:
                indices = [od[v][0] for v in record['loads']]
                assert max(indices) < od[record['anchor']][0]
                assert 'sched.barrier(i32 0)' in opt_lines[od[record['anchor']][0]-1]
        manifest['variants'][name] = dict(sha256=digest(path), llvm_verified=True,
                                         o3_lds_loads=optimized.read_text().count(READ))
    assert manifest['variants']['control']['o3_lds_loads'] == manifest['variants'][VARIANT]['o3_lds_loads'] == 80
    (HERE/'manifest_protected.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'PASS: {len(records)}/32 complete K read groups advanced; all 32 pairs protected before/after O3')
    print('Repaired groups:', [r['group'] for r in repairs])
    print('Lookahead:', dict(Counter(r['lookahead_mmac'] for r in records)))


if __name__ == '__main__':
    main()
