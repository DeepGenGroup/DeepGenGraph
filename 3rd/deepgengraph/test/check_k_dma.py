#!/usr/bin/env python3
"""Compiler K-DMA configuration, generic source maps, rejection and idempotence."""
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')
DMA = 'asm sideeffect "s_mov_b32 $0, $1\\0A\\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"'


def fixture(threads=64, key_arg=1):
    args = ['ptr addrspace(1) %other', 'ptr addrspace(1) %key', 'ptr addrspace(1) %out']
    args[1], args[key_arg] = args[key_arg], args[1]
    lines = ['@scratch = addrspace(3) global [16384 x i8] undef, align 16',
             'declare i32 @llvm.amdgcn.workitem.id.x()',
             'declare i32 @llvm.amdgcn.readfirstlane.i32(i32)',
             'declare void @llvm.amdgcn.s.barrier()',
             'define amdgpu_kernel void @unrelated_name('+', '.join(args)+') !reqd_work_group_size !0 {',
             'entry:', '  %lane = call i32 @llvm.amdgcn.workitem.id.x()',
             '  %dst = mul i32 %lane, 16',
             '  %ptr = getelementptr i8, ptr addrspace(3) @scratch, i32 %dst',
             '  %bits = ptrtoint ptr addrspace(3) %ptr to i32',
             '  %leader = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %bits)',
             # readfirstlane(tid*16) is the wave's 1024-byte packet base.
             '  %stride = mul i32 %lane, 64',
             '  %src = add i32 %stride, 128']
    for tag, arg in [('k','key'),('o','other')]:
        lines += [f'  %{tag}.address = ptrtoint ptr addrspace(1) %{arg} to i64',
                  f'  %{tag}.low = trunc i64 %{tag}.address to i32',
                  f'  %{tag}.shift = lshr i64 %{tag}.address, 32',
                  f'  %{tag}.high = trunc i64 %{tag}.shift to i32',
                  f'  %{tag}.r0 = insertelement <4 x i32> poison, i32 %{tag}.low, i32 0',
                  f'  %{tag}.r1 = insertelement <4 x i32> %{tag}.r0, i32 %{tag}.high, i32 1',
                  f'  %{tag}.r2 = insertelement <4 x i32> %{tag}.r1, i32 65536, i32 2',
                  f'  %{tag}.resource = insertelement <4 x i32> %{tag}.r2, i32 131072, i32 3']
    lines += [f'  %copy = call i32 {DMA}(i32 %leader, i32 %src, <4 x i32> %k.resource)',
              '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()',
              '  call void @llvm.amdgcn.s.barrier()',
              '  %value = load <4 x half>, ptr addrspace(3) %ptr, align 8',
              '  store <4 x half> %value, ptr addrspace(1) %out',
              '  call void @llvm.amdgcn.s.barrier()',
              f'  %release = call i32 {DMA}(i32 %leader, i32 %src, <4 x i32> %o.resource)',
              '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()',
              '  ret void', '}', f'!0 = !{{i32 {threads}, i32 1, i32 1}}']
    return '\n'.join(lines)+'\n'


class KDMATest(unittest.TestCase):
    def export(self, text, *options, ok=True):
        with tempfile.TemporaryDirectory(prefix='kdma-test-') as directory:
            p=Path(directory)/'in.ll'; p.write_text(text)
            result=subprocess.run([TOOL,*options,p],text=True,capture_output=True)
            if ok:
                self.assertEqual(result.returncode,0,result.stderr)
                q=Path(directory)/'out.ll'; q.write_text(result.stdout)
                subprocess.run([LLVM/'opt','-passes=default<O3>,verify',q,'-disable-output'],check=True,capture_output=True)
            else: self.assertNotEqual(result.returncode,0)
            return result

    def test_generic_threads_and_modes(self):
        for threads in (64,128,256,512):
            for mode in ('64','64-swizzle'):
                with self.subTest(threads=threads,mode=mode):
                    out=self.export(fixture(threads),f'--k-dma={mode}')
                    self.assertEqual(out.stdout.count('buffer_load_dwordx2 '),2)
                    self.assertEqual(out.stdout.count('buffer_load_dwordx4 '),1)
                    self.assertIn('packets=1->2, reads=1',out.stderr)

    def test_abi_binding(self):
        self.export(fixture(128,2),'--k-dma=64-swizzle','--k-dma-arg=2')
        self.export(fixture(),'--k-dma=64','--k-dma-arg=3',ok=False)

    def test_uniform_source_without_existing_thread_intrinsic(self):
        source=fixture().replace('declare i32 @llvm.amdgcn.workitem.id.x()\n','')
        source=source.replace('%lane = call i32 @llvm.amdgcn.workitem.id.x()', '%lane = add i32 0, 0')
        self.export(source,'--k-dma=64-swizzle')

    def test_off_and_idempotence(self):
        off=self.export(fixture(),'--k-dma=128')
        self.assertNotIn('buffer_load_dwordx2 ',off.stdout)
        first=self.export(fixture(),'--k-dma=64-swizzle')
        twice=self.export(first.stdout,'--k-dma=64-swizzle')
        self.assertEqual(twice.stdout.count('buffer_load_dwordx2 '),2)
        self.assertNotIn('packets=',twice.stderr)
        self.export(first.stdout,'--k-dma=64',ok=False)

    def test_reject_unsafe_shapes_and_footprints(self):
        for source in [fixture(32), fixture().replace('i32 64, i32 1, i32 1','i32 64, i32 2, i32 1'),
                       fixture().replace('[16384 x i8]','[512 x i8]'),
                       fixture().replace('  %stride =', '  %bad = add i32 %leader, 16\n  %stride =').replace('(i32 %leader,','(i32 %bad,'),
                       fixture().replace('  %release =','  %unused =').replace('<4 x i32> %o.resource)','<4 x i32> %k.resource)'),
                       fixture().replace('load <4 x half>','load volatile <4 x half>'),
                       fixture().replace('<4 x half>','<8 x half>')]:
            self.export(source,'--k-dma=64-swizzle',ok=False)

    def test_invalid_flags(self):
        for arg in ('--k-dma=32','--k-dma=on','--k-dma','--k-dma-arg=-1','--k-dma-arg=no'):
            self.export(fixture(),arg,ok=False)


if __name__=='__main__': unittest.main()
