#!/usr/bin/env python3
"""Check real exported dword DMA addresses, packed LDS data and code generation.

The full-wave precondition is deliberate: partial-wave tails are rejected by
the packed-layout planner. No GPU timing is simulated here.
"""
import argparse
from pathlib import Path
import re
import resource
import subprocess
import tempfile

import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel

HERE = Path(__file__).resolve().parent
DMA = 'llvm.amdgcn.global.load.async.lds'


def fixture(participants):
    return f'''target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [4352 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @{DMA}(ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)
declare void @llvm.amdgcn.s.barrier()
define amdgpu_kernel void @copy_probe(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 {{
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %active = icmp ult i32 %tid, {participants}
  br i1 %active, label %copy, label %done
copy:
  %permuted = xor i32 %tid, 37
  %source.offset = mul i32 %permuted, 32
  %source.plus = add i32 %source.offset, 16
  %src = getelementptr i8, ptr addrspace(1) %0, i32 %source.plus
  %wave = lshr i32 %tid, 6
  %base = mul i32 %wave, 1024
  %padded = add i32 %base, 128
  %dst = getelementptr i8, ptr addrspace(3) @shm_0, i32 %padded
  call void @{DMA}(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 16, i32 0, i32 0)
  br label %done
done:
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.s.barrier()
  br i1 %active, label %readback, label %exit
readback:
  %out.offset = mul i32 %tid, 16
  %read.offset = add i32 %out.offset, 128
  %read = getelementptr i8, ptr addrspace(3) @shm_0, i32 %read.offset
  %v = load <8 x half>, ptr addrspace(3) %read, align 16
  %out = getelementptr i8, ptr addrspace(1) %3, i32 %out.offset
  store <8 x half> %v, ptr addrspace(1) %out, align 16
  br label %exit
exit:
  ret void
}}
attributes #0 = {{ "amdgpu-flat-work-group-size"="256,256" }}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-tool', type=Path, default=HERE.parent / 'build/test/LegacyLLVMExportTest')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    parser.add_argument('--arch', default='gfx900', help='Use gfx936 with the target DTK')
    parser.add_argument('--probe-output', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def export(text):
        return subprocess.run([args.export_tool, '--split-full-wave-dma', '-'], input=text,
                              text=True, capture_output=True, check=True).stdout

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for participants in (128, 256):
            original = fixture(participants)
            split = export(original)
            assert export(split) == split
            calls = re.findall(r'call void @' + re.escape(DMA) + r'\((.*)\)', split)
            assert len(calls) == 4 and all('i32 4, i32 0, i32 0)' in c + ')' for c in calls)
            assert split.count('asm sideeffect "", "=v,0"') == 4
            assert 'llvm.hcu.' not in split
            assert split.count('call void @llvm.amdgcn.asyncmark()') == 1
            rng = np.random.default_rng(31)
            q, k, v = [rng.normal(size=(1, 1024, 8)).astype(np.float16) for _ in range(3)]
            models = [LLVMWorkgroupModel(ir, q, k, v) for ir in (original, split)]
            output = [model.run() for model in models]
            np.testing.assert_array_equal(output[0], output[1])
            # Independent expected source-to-destination mapping, including
            # non-contiguous source packets, multiple waves and an LDS prefix.
            for tid in range(participants):
                start = (tid ^ 37) * 16 + 8
                np.testing.assert_array_equal(output[1][tid * 8:(tid + 1) * 8], q.ravel()[start:start + 8])
            assert np.isnan(output[1][participants * 8:]).all()
            assert np.isnan(models[1].mem[0][:64]).all()
            assert np.isnan(models[1].mem[0][64 + participants * 8:]).all()
            ir, assembly = root / 'probe.ll', root / 'probe.s'
            ir.write_text(split)
            subprocess.run([args.llvm_bin / 'opt', '-passes=verify', '-disable-output', ir], check=True)
            subprocess.run([args.llvm_bin / 'llc', '-mtriple=amdgcn-amd-amdhsa', f'-mcpu={args.arch}',
                            '-O3', '-filetype=asm', ir, '-o', assembly], check=True)
            text = assembly.read_text()
            dma = [line for line in text.splitlines() if re.search(r'global_load_dword\b.*\blds\b', line)]
            assert len(dma) == 4 and all('offset:' not in line for line in dma), dma
            assert DMA not in text and 'llvm.amdgcn.ds.bpermute' not in text
            print(f'PASS {participants}/256 lanes: exact data, untouched guards/inactive waves, four zero-offset dword LDS instructions ({args.arch})')
            if args.probe_output:
                args.probe_output.write_text(split)
        rejected = subprocess.run([args.export_tool, '--split-full-wave-dma', '-'],
                                  input=fixture(256).replace('i32 16, i32 0, i32 0)', 'i32 16, i32 32, i32 0)'),
                                  text=True, capture_output=True)
        assert rejected.returncode != 0 and 'requires zero immediate offset' in rejected.stderr
        print('PASS nonzero immediate offset rejected')


if __name__ == '__main__':
    main()
