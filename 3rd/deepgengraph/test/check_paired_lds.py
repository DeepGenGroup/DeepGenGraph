#!/usr/bin/env python3
"""Pairing, hoist boundaries, idempotence and O3 verification."""
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
    def export(self, text, distance='2', ok=True):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'in.ll'; source.write_text(text)
            result=subprocess.run([TOOL,'--k-lds-prefetch='+distance,source],text=True,capture_output=True)
            if ok:
                self.assertEqual(result.returncode,0,result.stderr)
                source.write_text(result.stdout)
                subprocess.run([OPT,'-passes=default<O3>,verify',source,'-disable-output'],check=True,capture_output=True)
            else:self.assertNotEqual(result.returncode,0)
            return result

    def test_pair_hoist_and_distance(self):
        for distance in ('1','2','4','8'):
            r=self.export(fixture(),distance)
            self.assertIn('pairs=2, moved=1, anchors=1',r.stderr)
            self.assertEqual(r.stdout.count('call void @llvm.amdgcn.sched.barrier'),2)
            self.assertEqual(r.stdout.count('load <4 x half>'),4)
            self.assertEqual(re.findall(r'%m\d = call',r.stdout),['%m0 = call','%m1 = call','%m2 = call','%m3 = call'])

    def test_side_effect_boundaries(self):
        for boundary in ('call void @opaque()', 'call void @llvm.amdgcn.s.barrier()',
                         'fence syncscope("workgroup") acquire',
                         'store i32 1, ptr addrspace(3) %base',
                         'call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()',
                         'call void asm sideeffect "", "~{memory}"()'):
            r=self.export(fixture(boundary))
            self.assertIn('moved=0, anchors=0',r.stderr)
            self.assertNotIn('call void @llvm.amdgcn.sched.barrier',r.stdout)

    def test_volatile_alignment_and_operand_role(self):
        for text in (fixture().replace('load <4 x half>','load volatile <4 x half>'),
                     fixture(align=4),
                     re.sub(r'<4 x half> zeroinitializer, <4 x half> (%k\d)',r'<4 x half> \1, <4 x half> zeroinitializer',fixture())):
            self.assertIn('moved=0, anchors=0',self.export(text).stderr)

    def test_off_idempotent_invalid(self):
        r=self.export(fixture(),'0')
        self.assertNotIn('frisk.k_lds_prefetch',r.stdout)
        r=self.export(fixture())
        again=self.export(r.stdout)
        normalize=lambda s:'\n'.join(re.sub(r', ![\w.]+ !\d+', '',l) for l in s.splitlines()
                                    if not l.startswith(('; ModuleID','!')))
        self.assertEqual(normalize(r.stdout),normalize(again.stdout))
        self.export(r.stdout,'4',ok=False)
        for distance in ('-1','9','on','abc'):self.export(fixture(),distance,ok=False)

    def test_unknown_pointer_relation(self):
        text=fixture().replace('ptr %out)', 'ptr %out, i64 %dynamic)')
        text=text.replace('i64 128','i64 %dynamic').replace('i64 1152','i64 %dynamic')
        # Duplicate dynamic loads may CSE, but cannot introduce guessed offsets.
        r=self.export(text)
        self.assertNotIn('inttoptr',r.stdout)


if __name__=='__main__':unittest.main()
