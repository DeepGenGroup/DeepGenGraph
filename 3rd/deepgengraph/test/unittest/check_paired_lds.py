#!/usr/bin/env python3
"""Retired LLVM entry point; scheduling coverage is in check_thread_fragment_schedules.py."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
OPT=Path('/data2/xsl/install/bin/opt')
MMA='asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\\0A\\09", "=v,v,v,0"'


def fixture(boundary='', spacing=128, align=8):
    lines=['declare void @opaque()', 'declare void @llvm.amdgcn.s.barrier()',
           'define void @arbitrary(ptr addrspace(3) %base, ptr %out) {', 'entry:']
    for i,offset in enumerate((0,spacing,1024,1024+spacing)):
        lines += [f'  %p{i} = getelementptr i8, ptr addrspace(3) %base, i64 {offset}',
                  f'  %k{i} = load <4 x half>, ptr addrspace(3) %p{i}, align {align}',
                  f'  %m{i} = call <4 x float> {MMA}(<4 x half> zeroinitializer, <4 x half> %k{i}, <4 x float> '+
                  ('zeroinitializer' if i==0 else f'%m{i-1}')+') #0']
        if boundary: lines.append('  '+boundary)
    lines+=['  store <4 x float> %m3, ptr %out', '  ret void','}',
            'attributes #0 = { convergent nounwind willreturn memory(none) }']
    return '\n'.join(lines)+'\n'


class PairedLDS(unittest.TestCase):
    def test_llvm_entry_removed(self):
        for option in ['--k-lds-prefetch=0', '--k-lds-prefetch=8']:
            result = subprocess.run([TOOL, option], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('moved to ThreadIR', result.stderr)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
