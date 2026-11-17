#!/usr/bin/env python3
"""Retired LLVM entry point; scheduling coverage is in check_thread_fragment_schedules.py."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
OPT = Path('/data2/xsl/install/bin/opt')
MMA = 'asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\\0A\\09", "=v,v,v,0"'


def fixture(count=8, spacing=128, reduction='maxnum', boundary=''):
    lines = ['declare float @llvm.maxnum.f32(float, float)',
             'declare float @llvm.minnum.f32(float, float)',
             'declare float @llvm.maximum.f32(float, float)',
             'declare float @llvm.minimum.f32(float, float)',
             'declare void @opaque()',
             'declare void @pure_unknown() memory(none)',
             'declare void @llvm.amdgcn.s.barrier()',
             'declare void @llvm.amdgcn.sched.barrier(i32)',
             'declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)',
             'define void @not_attention(ptr addrspace(3) %base, ptr %out) {', 'entry:']
    for i in range(count):
        offset = (i // 2) * 4096 + (i % 2) * spacing
        lines.append(f'  %p{i} = getelementptr i8, ptr addrspace(3) %base, i64 {offset}')
    for i in range(count):
        acc = 'zeroinitializer' if i == 0 else f'%m{i-1}'
        prev = '0.0' if i == 0 else f'%r{i-1}'
        lines += [f'  %k{i} = load <4 x half>, ptr addrspace(3) %p{i}, align 8',
                  f'  %m{i} = call <4 x float> {MMA}(<4 x half> zeroinitializer, <4 x half> %k{i}, <4 x float> {acc}) #0',
                  f'  %x{i} = extractelement <4 x float> %m{i}, i32 0']
        if reduction == 'sum':
            lines.append(f'  %r{i} = fadd float %x{i}, {prev}')
        else:
            lines.append(f'  %r{i} = call float @llvm.{reduction}.f32(float %x{i}, float {prev})')
        if boundary:
            lines.append('  ' + boundary.replace('$i', str(i)))
    lines += [f'  store float %r{count-1}, ptr %out', f'  store <4 x float> %m{count-1}, ptr %out',
              '  ret void', '}', 'attributes #0 = { convergent nounwind willreturn memory(none) }', '!0 = !{}']
    return '\n'.join(lines) + '\n'


def instructions(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and '!frisk.k_tail_prefetch' not in l]


class PairedLDSTail(unittest.TestCase):
    def test_llvm_entry_removed(self):
        for option in ['--k-tail-prefetch=0', '--k-tail-prefetch=4']:
            result = subprocess.run([TOOL, option], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('moved to ThreadIR', result.stderr)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
