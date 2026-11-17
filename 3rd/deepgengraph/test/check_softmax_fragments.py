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
    # Keep fixture() for the independent LLVM half-packing tests.
    for mode in ('off', 'fragment', 'pv'):
        p = subprocess.run([TOOL, '--softmax-fragments='+mode], text=True, capture_output=True)
        assert p.returncode != 0 and 'moved to ThreadIR' in p.stderr, p.stderr
        assert not p.stdout
    print('PASS LLVM softmax scheduling entry removed; see check_thread_fragment_schedules.py')


if __name__ == '__main__': main()
