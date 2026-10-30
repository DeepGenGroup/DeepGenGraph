#!/usr/bin/env python3
"""Functional regression tests for the LLVM PV scheduling pass (no GPU needed)."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def fixture():
    lines = ['declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32)',
             'declare void @llvm.amdgcn.s.barrier()',
             'declare void @llvm.amdgcn.sched.barrier(i32 immarg)',
             'declare void @unknown()',
             'declare i32 @llvm.amdgcn.readfirstlane.i32(i32)',
             'define void @kernel(ptr addrspace(3) %p, ptr %out, <4 x half> %a, <4 x float> %seed) {',
             'entry:']
    for k in range(2):
        lines += [f'  %r{k} = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %p, i32 {k*256})',
                  f'  %v{k} = bitcast <4 x i32> %r{k} to <8 x half>']
    for chain in 'ABCD':
        for k in range(2):
            half = 0 if chain in 'AC' else 4
            lines.append(f'  %{chain}b{k} = shufflevector <8 x half> %v{k}, <8 x half> poison, <4 x i32> <i32 {half}, i32 {half+1}, i32 {half+2}, i32 {half+3}>')
            acc = '%seed' if k == 0 else f'%{chain}0'
            lines.append(f'  %{chain}{k} = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a, <4 x half> %{chain}b{k}, <4 x float> {acc}) #0')
            if chain == 'A' and k == 0:
                lines += ['  fence syncscope("workgroup") release',
                          '  call void @llvm.amdgcn.s.barrier()',
                          '  fence syncscope("workgroup") acquire']
    # Live results; intentionally shared destination is after the scheduling window.
    lines += [f'  store <4 x float> %{c}1, ptr %out' for c in 'ABCD']
    lines += ['  ret void', '}', 'attributes #0 = { convergent nounwind willreturn memory(none) }']
    return '\n'.join(lines)+'\n'


def mmas(text):
    return re.findall(r'(%\w+) = (?:tail )?call <4 x float> asm ', text)


def instructions(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exporter', type=Path, default=Path(__file__).resolve().parents[1]/'build/test/LegacyLLVMExportTest')
    args = parser.parse_args()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='pv-interleave-') as tmp:
        source = Path(tmp)/'input.ll'
        def run(text, width=4):
            source.write_text(text)
            result = subprocess.run([str(args.exporter), f'--pv-interleave={width}', str(source)], text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            return result.stdout

        base = fixture()
        for width, expected in [(0,'A0 A1 B0 B1 C0 C1 D0 D1'),
                                (1,'A0 A1 B0 B1 C0 C1 D0 D1'),
                                (2,'A0 B0 A1 B1 C0 D0 C1 D1'),
                                (4,'A0 B0 C0 D0 A1 B1 C1 D1')]:
            out = run(base, width)
            assert mmas(out) == ['%'+n for n in expected.split()]
            assert out.count('call void @llvm.amdgcn.sched.barrier(') == (16 if width else 0)
            count = 0
            for line in out.splitlines():
                if 'call <4 x float> asm ' in line: count += 1
                if 'call void @llvm.amdgcn.s.barrier()' in line: assert count == 1
            assert instructions(run(out,width)) == instructions(out), 'not idempotent'
            cases += 1
        renamed = re.sub(r'%([A-Za-z]\w*)',r'%renamed_\1',base)
        assert mmas(run(renamed)) == ['%renamed_'+n for n in 'A0 B0 C0 D0 A1 B1 C1 D1'.split()]
        cases += 1
        barrier = '  call void @llvm.amdgcn.s.barrier()'
        wave_address = base.replace(barrier, barrier+'\n  %uniform = call i32 @llvm.amdgcn.readfirstlane.i32(i32 128)')
        out = run(wave_address)
        assert mmas(out) == ['%'+n for n in 'A0 B0 C0 D0 A1 B1 C1 D1'.split()]
        assert out.index('%A0 =') < out.index('%uniform =') < out.index('%B0 =')
        cases += 1
        rejected = {
            'unknown_call': base.replace(barrier, barrier+'\n  call void @unknown()'),
            'nonzero_hint': base.replace(barrier, barrier+'\n  call void @llvm.amdgcn.sched.barrier(i32 1)'),
            'volatile_store': base.replace(barrier, barrier+'\n  store volatile <4 x float> %seed, ptr %out'),
            'intermediate_fanout': base.replace(barrier,barrier+'\n  store <4 x float> %A0, ptr %out'),
            'cross_chain_dependency': base.replace('<4 x float> %seed) #0\n  %Bb1', '<4 x float> %A1) #0\n  %Bb1'),
            'sideeffect_mma': base.replace('float> asm "v_mmac','float> asm sideeffect "v_mmac'),
            'unknown_packet': base.replace('llvm.hcu.ds.read.m32x16.b16.v4i32.p3','unknown_read'),
        }
        late_lines = base.splitlines()
        moving = [l for l in late_lines if l.startswith(('  %r1 =','  %v1 ='))]
        late_lines = [l for l in late_lines if l not in moving]
        index = next(i for i,l in enumerate(late_lines) if l.startswith('  %Ab1 ='))
        late_lines[index:index] = moving
        rejected['late_read'] = '\n'.join(late_lines)+'\n'
        for name, text in rejected.items():
            out = run(text)
            assert '!frisk.pv_interleave' not in out, name
            assert instructions(out) == instructions(run(text,0)), name
            cases += 1
        for value in ('3','-1','8','true','4294967296',''):
            source.write_text(base)
            result = subprocess.run([str(args.exporter),f'--pv-interleave={value}',str(source)],capture_output=True)
            assert result.returncode != 0, value
            cases += 1
    print(f'PASS {cases} PV interleave regression cases (order, handoff, idempotence, renaming, rejection, CLI)')


if __name__ == '__main__':
    main()
