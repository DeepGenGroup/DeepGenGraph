#!/usr/bin/env python3
"""Fingerprint-scoped Q/K subpacket swizzle experiment; V schedule is preserved."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'noncausal_v_double_buffer_group32/v_double_buffer.ll'
LLVM = Path('/data2/xsl/install/bin')
RESOURCES = {'Q': '%dma.buffer.rsrc1397', 'K': '%dma.buffer.rsrc1373'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(text, k_swizzle, q_swizzle=False):
    """Map new DWORD destination back to its original 16B packet and source.

    Original per-thread source displacement H(t):
      Q: 256*(t&127) + 16*(t>>7)
      K: 256*((t&127) XOR ((t>>2)&4)) + 16*(t>>7).
    New owner = wave_base + quarter*16 + lane//4; intra = 4*(lane%4).
    K bit9->3 toggles intra by 8 for quarters 2,3;
    Q bit7->3 toggles intra by 8 when lane bit5 is set.
    m0 remains wave-uniform, with a 256-byte displacement per quarter.
    """
    kinds = ['K'] + (['Q'] if q_swizzle else [])
    header = ['  %bank.lane = and i32 %grp.n25, 63',
              '  %bank.wave = and i32 %grp.n25, 192',
              '  %bank.owner.low = lshr i32 %bank.lane, 2',
              '  %bank.owner.base = or i32 %bank.wave, %bank.owner.low',
              '  %bank.intra.word = and i32 %bank.lane, 3',
              '  %bank.intra = shl i32 %bank.intra.word, 2']

    def displacement(prefix, thread, kind):
        code = [f'  {prefix}.row = and i32 {thread}, 127']
        row = prefix+'.row'
        if kind == 'K':
            code += [f'  {prefix}.shift = lshr i32 {thread}, 2',
                     f'  {prefix}.mask = and i32 {prefix}.shift, 4',
                     f'  {prefix}.xor = xor i32 {row}, {prefix}.mask']
            row = prefix+'.xor'
        code += [f'  {prefix}.row.bytes = shl i32 {row}, 8',
                 f'  {prefix}.half = lshr i32 {thread}, 7',
                 f'  {prefix}.half.bytes = shl i32 {prefix}.half, 4',
                 f'  {prefix}.bytes = add i32 {prefix}.row.bytes, {prefix}.half.bytes']
        return code

    for kind in kinds:
        prefix = '%bank.'+kind
        header += displacement(prefix+'.old', '%grp.n25', kind)
        for j in range(4):
            p = prefix+f'.q{j}'
            header += [f'  {p}.owner = add i32 %bank.owner.base, {16*j}']
            header += displacement(p, p+'.owner', kind)
            header += [f'  {p}.delta = sub i32 {p}.bytes, {prefix}.old.bytes']
            if kind == 'Q':
                header += [f'  {p}.shift = lshr i32 %bank.lane, 2',
                           f'  {p}.mask = and i32 {p}.shift, 8',
                           f'  {p}.intra = xor i32 %bank.intra, {p}.mask']
            else:
                header += [f'  {p}.intra = xor i32 %bank.intra, {8 if k_swizzle and j>=2 else 0}']
            header += [f'  {p}.source.delta = add i32 {p}.delta, {p}.intra']

    # Identify readers by debug location associated with the current Q/K loads.
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M))
    dbg = {kind: re.search(r'!dbg !(\d+)', definitions[name])[1]
           for kind, name in [('Q', '%grp.n51'), ('K', '%grp.n159')]}
    output, sites, pointers = [], {'Q': 0, 'K': 0}, {}
    for line in text.splitlines():
        kind = next((k for k in kinds if 'buffer_load_dwordx4 ' in line and RESOURCES[k] in line), None)
        if kind:
            dst, off, resource = re.search(r'\(i32 (%[\w.]+), i32 (%[\w.]+), <4 x i32> (%[\w.]+)\)', line).groups()
            name = re.match(r'  (%[\w.]+) =', line)[1]
            assert len(re.findall(re.escape(name)+r'(?![\w.])', text)) == 1
            for j in range(4):
                p = f'%bank.{kind}.dma{sites[kind]}.q{j}'
                output += [f'  {p}.off = add i32 {off}, %bank.{kind}.q{j}.source.delta',
                           f'  {p}.dst = add i32 {dst}, {256*j}']
                changed = line.replace('buffer_load_dwordx4 ', 'buffer_load_dword ')
                changed = changed.replace(f'i32 {dst},', f'i32 {p}.dst,').replace(f'i32 {off},', f'i32 {p}.off,')
                if j != 3:
                    changed = changed.replace(name+' =', p+'.call =', 1)
                output.append(changed)
            sites[kind] += 1
            continue
        m = re.search(r'load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 8, !dbg !(\d+)', line)
        if m:
            kind = next(k for k in dbg if dbg[k] == m[2])
            enabled = k_swizzle if kind == 'K' else q_swizzle
            if enabled:
                key = (kind, m[1])
                if key not in pointers:
                    p = f'%bank.read{len(pointers)}'
                    src = 9 if kind == 'K' else 7
                    output += [f'  {p}.addr = ptrtoint ptr addrspace(3) {m[1]} to i32',
                               f'  {p}.shift = lshr i32 {p}.addr, {src-3}',
                               f'  {p}.mask = and i32 {p}.shift, 8',
                               f'  {p}.xor = xor i32 {p}.addr, {p}.mask',
                               f'  {p}.ptr = inttoptr i32 {p}.xor to ptr addrspace(3)']
                    pointers[key] = p+'.ptr'
                line = line.replace(f'ptr addrspace(3) {m[1]},', f'ptr addrspace(3) {pointers[key]},')
        output.append(line)
        if '@llvm.amdgcn.workitem.id.x()' in line and line.lstrip().startswith('%'):
            output += header
    assert sites == {'Q': 8 if q_swizzle else 0, 'K': 8}
    candidate = '\n'.join(output)+'\n'
    for token in ('v_mmac_', 's_waitcnt', 'call void @llvm.amdgcn.s.barrier',
                  'call <4 x i32> @llvm.hcu.ds.read', RESOURCES['Q'] if not q_swizzle else 'NO_MATCH',
                  '%dma.buffer.rsrc1349'):
        assert [l for l in text.splitlines() if token in l] == [l for l in candidate.splitlines() if token in l]
    return candidate


def main():
    source = SOURCE.read_text()
    expected = json.loads((SOURCE.parent/'validation.json').read_text())['sha256']['v_double_buffer']
    assert digest(SOURCE) == expected
    variants = {'baseline': source,
                'k_split32_control': transform(source, False),
                'k_swizzle32': transform(source, True),
                'qk_swizzle32': transform(source, True, True)}
    manifest = dict(source=str(SOURCE), source_sha256=digest(SOURCE),
                    gpu_tested=False, gfx936_codegen_tested=False, variants={})
    for name, text in variants.items():
        path = HERE/(name+'.ll')
        path.write_text(text)
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path,
                        '-o', path.with_suffix('.o3.ll')], check=True)
        manifest['variants'][name] = dict(sha256=digest(path), llvm_and_o3_verified=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Generated baseline, split32 control, K swizzle and QK swizzle; LLVM/O3 verified.')


if __name__ == '__main__':
    main()
