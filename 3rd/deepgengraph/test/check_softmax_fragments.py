#!/usr/bin/env python3
"""Regression tests for dependency-driven softmax fragment scheduling."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import tempfile

TOOL = Path(__file__).resolve().parents[1]/'build/test/LegacyLLVMExportTest'
MMA = 'call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0"'


def fixture():
    lines = ['declare float @llvm.amdgcn.exp2.f32(float)',
             'declare void @llvm.amdgcn.s.barrier()',
             'declare void @unknown()',
             'define void @kernel(<4 x float> %score, <4 x float> %max, <4 x half> %v, ptr %out) {',
             'entry:']
    for f in range(4): lines.append(f'  %sub{f} = fsub <4 x float> %score, %max')
    for f in range(4):
        for lane in range(4):
            lines += [f'  %x{f}_{lane} = extractelement <4 x float> %sub{f}, i32 {lane}',
                      f'  %e{f}_{lane} = call float @llvm.amdgcn.exp2.f32(float %x{f}_{lane})',
                      f'  %i{f}_{lane} = insertelement <4 x float> '+
                      ('poison' if lane == 0 else f'%i{f}_{lane-1}')+
                      f', float %e{f}_{lane}, i32 {lane}']
        lines += [f'  %r{f} = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i{f}_3) #0, !frisk.register_lifetimes !0',
                  f'  %h{f} = fptrunc <4 x float> %r{f} to <4 x half>',
                  f'  %p{f} = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %h{f}) #0, !frisk.register_lifetimes !0']
        if f%2:
            for lane in range(4):
                lines.append(f'  %sum{f}_{lane} = fadd float %e{f-1}_{lane}, %e{f}_{lane}')
    lines += ['  call void @llvm.amdgcn.s.barrier()']
    for f in (0,2,1,3):
        # Emit independent chains in PV consumption order; no SSA forward edge.
        acc = 'zeroinitializer' if f in (0,2) else f'%mma{f-1}'
        lines += [f'  %mma{f} = {MMA}(<4 x half> %p{f}, <4 x half> %v, <4 x float> {acc}) #0']
    lines += [f'  store float %sum{f}_{lane}, ptr %out' for f in (1,3) for lane in range(4)]
    lines += ['  store <4 x float> %mma1, ptr %out', '  store <4 x float> %mma3, ptr %out',
              '  ret void', '}', 'attributes #0 = { convergent nounwind willreturn memory(none) }', '!0 = !{}']
    return '\n'.join(lines)+'\n'


def insts(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and '!frisk.softmax_fragments' not in l]


def main():
    with tempfile.TemporaryDirectory(prefix='softmax-fragments-') as tmp:
        p = Path(tmp)/'input.ll'
        def run(text,mode='pv'):
            p.write_text(text)
            out = subprocess.run([TOOL,'--softmax-fragments='+mode,p],text=True,capture_output=True)
            assert out.returncode == 0, out.stderr
            return out.stdout
        base = fixture()
        off = run(base,'off')
        for mode in ('fragment','pv'):
            out = run(base,mode)
            assert Counter(insts(out)) == Counter(insts(off))
            assert out.count('call void @llvm.amdgcn.sched.barrier') == 4
            assert out.index('%p0 =') < out.index('%sub1 =')
            for f in (1,3):
                for lane in range(4):
                    assert out.index(f'%sum{f}_{lane} =') < out.index(f'%r{f} =')
            expected = [0,1,2,3] if mode == 'fragment' else [0,2,1,3]
            assert re.findall(r'%p(\d) =',out) == list(map(str,expected))
            assert re.findall(r'%mma(\d) =',out) == ['0','2','1','3']
            if mode == 'fragment':
                assert out.index('%p3 =') < out.index('%mma0 =')
            else:
                assert out.index('%mma0 =') < out.index('%p2 =')
            twice = run(out,mode)
            assert insts(twice) == insts(out)
            assert twice.count('call void @llvm.amdgcn.sched.barrier') == 4
        renamed = re.sub(r'%([A-Za-z]\w*)',r'%rename_\1',base)
        assert re.findall(r'%rename_p(\d) =',run(renamed)) == ['0','2','1','3']
        rejected = {
            'unknown_call': base.replace('  call void @llvm.amdgcn.s.barrier()', '  call void @unknown()'),
            'store_boundary': base.replace('  call void @llvm.amdgcn.s.barrier()', '  store volatile <4 x float> %score, ptr %out'),
            'strictfp': base.replace('ptr %out) {','ptr %out) strictfp {'),
            'optnone': base.replace('ptr %out) {','ptr %out) noinline optnone {'),
            'unowned_identity': base.replace(', !frisk.register_lifetimes !0',''),
            'nonidentity_asm': base.replace('asm sideeffect "",','asm sideeffect "s_nop 0",'),
            'nonexp': base.replace('llvm.amdgcn.exp2.f32','other_exp'),
            'sideeffect_mma': base.replace('asm "v_mmac','asm sideeffect "v_mmac'),
        }
        for name,text in rejected.items():
            out = run(text)
            assert insts(out) == insts(run(text,'off')), name
            assert '"frisk.softmax_fragments"' not in out, name
        # An unrelated live use of a shifted score must remain dominated.
        extra = base.replace('  %x0_0 =', '  %extra = fadd <4 x float> %sub1, %score\n  %x0_0 =')
        extra = extra.replace('  ret void', '  store <4 x float> %extra, ptr %out\n  ret void')
        assert Counter(insts(run(extra))) == Counter(insts(run(extra,'off')))
        for mode in ('', 'true', '1', 'Fragment', 'pv-more'):
            p.write_text(base)
            assert subprocess.run([TOOL,'--softmax-fragments='+mode,p],capture_output=True).returncode != 0
    print('PASS softmax fragment scheduling: both modes, exact DAG, sum-before-pack, PV order, idempotence, renaming, 8 rejection cases, live external use, invalid CLI')


if __name__ == '__main__': main()
