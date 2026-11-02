#!/usr/bin/env python3
"""Cross-iteration K0 pass: generic shapes, barriers, aliases and recurrence anchors."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
OPT = Path('/data2/xsl/install/bin/opt')
DMA = 'asm sideeffect "s_mov_b32 $0, $1\\0A\\09buffer_load_dwordx{words} $2, $3, 0 offen lds", "={{m0}},s,v,s,~{{memory}}"'
MMA = 'asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0"'
SYNC = '  fence syncscope("workgroup") release\n  call void @llvm.amdgcn.s.barrier()\n  fence syncscope("workgroup") acquire\n'
WAIT = '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()\n'


def fixture(threads=64, packets=1, width=8, tiles=5):
    low = threads * packets * width
    text = f'''@shared = addrspace(3) global [{low + threads*16} x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)
declare void @llvm.amdgcn.s.barrier()
declare void @unknown()
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32)
define amdgpu_kernel void @generic(ptr addrspace(1) %out, <4 x i32> %resource, <4 x half> %q) !reqd_work_group_size !0 {{
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lane = mul i32 %tid, {width}
  %kp = getelementptr i8, ptr addrspace(3) @shared, i32 %lane
  %ka = ptrtoint ptr addrspace(3) %kp to i32
  %kd = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ka)
  %vlane = mul i32 %tid, 16
  %voffset = add i32 %vlane, {low}
  %vp = getelementptr i8, ptr addrspace(3) @shared, i32 %voffset
  %va = ptrtoint ptr addrspace(3) %vp to i32
  %vd = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %va)
{SYNC}  br label %header
header:
  %iv = phi i32 [0, %entry], [%next, %body]
  %sum = phi float [0.0, %entry], [%sum.next, %body]
  %acc = phi <4 x float> [zeroinitializer, %entry], [%pv, %body]
  %more = icmp ult i32 %iv, {tiles}
  br i1 %more, label %body, label %exit
body:
  %global.base = mul i32 %iv, {low}
  %global = add i32 %global.base, %lane
'''
    for p in range(packets):
        text += f'  %dst{p} = add i32 %kd, {p*threads*width}\n'
        text += f'  %src{p} = add i32 %global, {p*threads*width}\n'
        text += f'  %copy{p} = call i32 {DMA.format(words=width//4)}(i32 %dst{p}, i32 %src{p}, <4 x i32> %resource)\n'
    text += WAIT + SYNC + f'''  %k = load <4 x half>, ptr addrspace(3) %kp, align 8
  %qk = call <4 x float> {MMA}(<4 x half> %q, <4 x half> %k, <4 x float> zeroinitializer) #0
  %score = extractelement <4 x float> %qk, i32 0
  %sum.next = fadd float %sum, %score
  %v.copy = call i32 {DMA.format(words=4)}(i32 %vd, i32 %global, <4 x i32> %resource)
{WAIT}{SYNC}  %v.read = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vp, i32 0)
  %v.bits = bitcast <4 x i32> %v.read to <8 x half>
  %v.half = shufflevector <8 x half> %v.bits, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %pv = call <4 x float> {MMA}(<4 x half> %q, <4 x half> %v.half, <4 x float> %acc) #0
{SYNC}  %next = add i32 %iv, 1
  br label %header
exit:
  store float %sum, ptr addrspace(1) %out
  store <4 x float> %acc, ptr addrspace(1) %out
  ret void
}}
attributes #0 = {{ convergent nounwind willreturn memory(none) }}
!0 = !{{i32 {threads}, i32 1, i32 1}}
'''
    return text


def instructions(s):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in s.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')]


class CrossBNK0(unittest.TestCase):
    def export(self, text, option='1', ok=True):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.ll'; p.write_text(text)
            r = subprocess.run([TOOL, '--cross-bn-k0='+option, p], text=True, capture_output=True)
            if not ok:
                self.assertNotEqual(r.returncode, 0)
                return r
            self.assertEqual(r.returncode, 0, r.stderr)
            p.write_text(r.stdout)
            verify = subprocess.run([OPT, '-passes=default<O3>,verify', '-disable-output', p], text=True, capture_output=True)
            self.assertEqual(verify.returncode, 0, verify.stderr)
            return r

    def assertSkipped(self, text):
        off = self.export(text, '0'); on = self.export(text)
        self.assertIn('Cross BN K0: skipped', on.stderr)
        self.assertEqual(instructions(off.stdout), instructions(on.stdout))
        self.assertNotIn('"frisk.cross_bn_k0"=', on.stdout)

    def test_generic_and_renamed(self):
        for threads, packets, width, tiles in [(64,1,8,1), (128,2,8,7), (256,1,16,19), (64,4,16,3)]:
            with self.subTest(threads=threads, packets=packets, width=width, tiles=tiles):
                text = fixture(threads, packets, width, tiles)
                result = self.export(text)
                self.assertIn(f'packets={packets}, slot-bytes={threads*packets*width}, overlap-mmac=1, sum-anchors=1', result.stderr)
                self.assertIn('"frisk.cross_bn_k0"="sum-anchor-v1"', result.stdout)
                self.assertEqual(result.stdout.count('call void @llvm.amdgcn.s.barrier'), text.count('call void @llvm.amdgcn.s.barrier'))
                self.assertEqual(result.stdout.count('s_waitcnt vmcnt(0)'), text.count('s_waitcnt vmcnt(0)'))
                self.assertIn(f'icmp ult i32 %next, {tiles}', result.stdout)
                self.assertEqual(result.stdout.count('call float asm sideeffect "", "=v,0"'), 1)
                renamed = re.sub(r'%([\w.]+)', r'%renamed_\1', text)
                renamed = re.sub(r'^(entry|header|body|exit):', r'renamed_\1:', renamed, flags=re.M)
                self.assertIn('sum-anchors=1', self.export(renamed.replace('@generic','@different')).stderr)

    def test_idempotence_and_options(self):
        first = self.export(fixture()).stdout
        second = self.export(first).stdout
        self.assertEqual(instructions(first), instructions(second))
        for value in ('', '-1', '2', 'true', '1x'):
            self.export(fixture(), value, ok=False)

    def test_loop_bounds_and_control(self):
        s = fixture()
        for bad in [s.replace('%iv, 5\n', '%iv, 0\n'),
                    s.replace('%iv, 5\n', '%iv, %tid\n'),
                    s.replace('[0, %entry]', '[1, %entry]'),
                    s.replace('add i32 %iv, 1', 'add i32 %iv, 2'),
                    s.replace('icmp ult', 'icmp slt'),
                    s.replace('!reqd_work_group_size !0', 'strictfp !reqd_work_group_size !0'),
                    s.replace('!reqd_work_group_size !0', 'noinline optnone !reqd_work_group_size !0'),
                    s.replace('i32 64, i32 1, i32 1', 'i32 32, i32 1, i32 1')]:
            self.assertSkipped(bad)

    def test_dynamic_causal_bounds(self):
        s = fixture().replace('declare i32 @llvm.amdgcn.workitem.id.x()',
            'declare i32 @llvm.amdgcn.workgroup.id.x()\ndeclare i32 @llvm.amdgcn.workitem.id.x()')
        s = s.replace('entry:\n', 'entry:\n  %bx = call i32 @llvm.amdgcn.workgroup.id.x()\n'
            '  %last = and i32 %bx, 31\n  %bound = or i32 %last, 1\n')
        for condition, branch, guard in [
            ('ult i32 %iv, %bound', 'br i1 %more, label %body, label %exit', 'ult i32 %next, %bound'),
            ('ule i32 %iv, %last', 'br i1 %more, label %body, label %exit', 'ule i32 %next, %last'),
            ('ugt i32 %iv, %last', 'br i1 %more, label %exit, label %body', 'ule i32 %next, %last'),
            ('uge i32 %last, %iv', 'br i1 %more, label %body, label %exit', 'ule i32 %next, %last'),
            ('ule i32 %iv, 0', 'br i1 %more, label %body, label %exit', 'ule i32 %next, 0')]:
            with self.subTest(condition=condition):
                text = s.replace('ult i32 %iv, 5',condition).replace(
                    'br i1 %more, label %body, label %exit',branch)
                result = self.export(text)
                self.assertIn('packets=1',result.stderr)
                self.assertIn('icmp '+guard,result.stdout)
                self.assertEqual(result.stdout.count('call void @llvm.amdgcn.s.barrier'),text.count('call void @llvm.amdgcn.s.barrier'))
                self.assertEqual(instructions(result.stdout),instructions(self.export(result.stdout).stdout))
        # A potentially zero-trip bound, unbounded sign bit, lane-varying
        # control, or a loop-carried bound cannot justify unconditional priming.
        for condition in ('ult i32 %iv, %last', 'ule i32 %iv, %bx',
                          'ule i32 %iv, %tid', 'ule i32 %iv, %iv'):
            with self.subTest(rejected=condition):
                self.assertSkipped(s.replace('ult i32 %iv, 5',condition))

    def test_missing_sync_and_aliases(self):
        s = fixture()
        last = s.rindex(SYNC)
        ready = s.index(SYNC, s.index('%v.copy ='))
        variants = [s.replace(SYNC, '', 1), s[:last]+s[last:].replace(SYNC,'',1),
                    s[:ready]+s[ready:].replace(SYNC,'',1),
                    s.replace(WAIT,'',1),
                    s.replace('ptr addrspace(3) %vp, i32 0)', 'ptr addrspace(3) %kp, i32 0)'),
                    s.replace('load <4 x half>', 'load volatile <4 x half>'),
                    s.replace('  %pv =', '  store i32 0, ptr addrspace(3) %kp\n  %pv ='),
                    s.replace('  %pv =', '  call void @unknown()\n  %pv ='),
                    s.replace('  %sum.next =', '  call void @unknown()\n  %sum.next ='),
                    s.replace('  %copy0 =', '  %unknown = load i32, ptr addrspace(1) %out\n  %copy0 =')]
        for bad in variants:
            with self.subTest(change=variants.index(bad)):
                self.assertSkipped(bad)

    def test_dma_and_missing_sum(self):
        s = fixture(packets=2)
        self.assertSkipped(s.replace('%kd, 512', '%kd, 0'))
        self.assertSkipped(s.replace('i32 %src0, <4 x i32>', 'i32 0, <4 x i32>').replace('i32 %src1, <4 x i32>', 'i32 0, <4 x i32>'))
        self.assertSkipped(s.replace('fadd float %sum, %score', 'fmul float %sum, %score'))
        self.assertSkipped(s.replace('buffer_load_dwordx2', 'buffer_load_dword'))

    def test_tagged_register_identity(self):
        s = fixture().replace('  %sum.next =',
            '  %tagged = call i32 asm sideeffect "", "=v,0,i"(i32 %tid, i32 1) #0\n  %sum.next =')
        self.assertIn('packets=1',self.export(s).stderr)
        self.assertSkipped(s.replace('(i32 %tid, i32 1) #0','(i32 %tid, i32 %iv) #0'))

    def test_measured_fixture(self):
        p = ROOT / 'generated_ir/noncausal_qk_tail_prefetch_pass/k_tail_prefetch4.ll'
        result = self.export(p.read_text())
        self.assertIn('packets=4, slot-bytes=8192, overlap-mmac=32, sum-anchors=64', result.stderr)


if __name__ == '__main__':
    unittest.main()
