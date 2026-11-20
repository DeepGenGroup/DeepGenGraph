#!/usr/bin/env python3
"""Validate buffer inline-asm DMA export: addresses, bounds, waits, M0 and encoding."""
import argparse
from pathlib import Path
import re
import resource
import subprocess
import tempfile

import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel
from check_split_dma_export import fixture

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-tool', type=Path, default=HERE.parent / 'build/test/LegacyLLVMExportTest')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    parser.add_argument('--arch', default='gfx950', help='Local encoding check; use gfx936 with target DTK')
    parser.add_argument('--probe-output', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def export(text, check=True):
        return subprocess.run([args.export_tool, '--buffer-asm-dma', '-'], input=text,
                              text=True, capture_output=True, check=check)

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for participants in (128, 256):
            original = fixture(participants)
            result = export(original).stdout
            assert export(result).stdout == result, 'export must be idempotent'
            assert result.count('buffer_load_dwordx4') == 1
            assert '"={m0},s,v,s,~{memory}"' in result and '~{m0}' not in result
            assert result.count('s_waitcnt vmcnt(0)') == 1
            assert not re.search(r'call void @llvm.amdgcn.(asyncmark|wait.asyncmark|global.load.async.lds)', result)
            assert result.index('s_waitcnt vmcnt(0)') < result.index('call void @llvm.amdgcn.s.barrier')
            rng = np.random.default_rng(74)
            q, k, v = [rng.normal(size=(1, 1024, 8)).astype(np.float16) for _ in range(3)]
            models = [LLVMWorkgroupModel(ir, q, k, v) for ir in (original, result)]
            output = [m.run() for m in models]
            np.testing.assert_array_equal(output[0], output[1])
            for tid in range(participants):
                start = (tid ^ 37) * 16 + 8
                np.testing.assert_array_equal(output[1][tid * 8:(tid + 1) * 8], q.ravel()[start:start + 8])
            assert np.isnan(output[1][participants * 8:]).all()
            assert np.isnan(models[1].mem[0][:64]).all()
            assert np.isnan(models[1].mem[0][64 + participants * 8:]).all()
            ir, optimized, obj = root / 'probe.ll', root / 'probe.opt.ll', root / 'probe.o'
            ir.write_text(result)
            subprocess.run([args.llvm_bin / 'opt', '-passes=verify,default<O3>,verify', '-S', ir, '-o', optimized], check=True)
            assert optimized.read_text().count('buffer_load_dwordx4') == 1
            assert optimized.read_text().count('s_waitcnt vmcnt(0)') == 1
            compilation = subprocess.run([args.llvm_bin / 'llc', '-mtriple=amdgcn-amd-amdhsa', f'-mcpu={args.arch}',
                                         '-O3', '-filetype=obj', '-amdhsa-code-object-version=4', optimized, '-o', obj],
                                        text=True, capture_output=True, check=True)
            assert 'reserved registers' not in compilation.stderr
            dis = subprocess.run([args.llvm_bin / 'llvm-objdump', '-d', f'--mcpu={args.arch}', obj],
                                 text=True, capture_output=True, check=True).stdout
            assert re.search(r'buffer_load_dwordx4[^\n]*\blds\b', dis), dis
            assert 's_waitcnt vmcnt(0)' in dis
            print(f'PASS {participants}/256 threads: noncontiguous GM, descriptor/bounds, LDS guards, explicit wait, object encoding ({args.arch}), no M0 clobber warning')
            if args.probe_output:
                args.probe_output.write_text(result)
        # Automatic packet selection can retain narrow Q/K while widening V.
        # The explicit VM wait must cover both, and narrow-only functions keep
        # their original async markers. Check actual copied LDS bytes as well.
        mixed = fixture(256).replace('[4352 x i8]', '[5504 x i8]')
        mixed = mixed.replace('  br label %done', '''
  %narrow.base = mul i32 %wave, 256
  %narrow.offset = add i32 %narrow.base, 4352
  %narrow.dst = getelementptr i8, ptr addrspace(3) @shm_0, i32 %narrow.offset
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src, ptr addrspace(3) %narrow.dst, i32 4, i32 0, i32 0)
  br label %done''')
        result = export(mixed).stdout
        assert result.count('buffer_load_dwordx4') == 1
        assert len(re.findall(r'call void @llvm.amdgcn.global.load.async.lds\(', result)) == 1
        assert not re.search(r'call void @llvm.amdgcn.(?:asyncmark|wait.asyncmark)\(', result)
        assert result.count('s_waitcnt vmcnt(0)') == 1
        models = [LLVMWorkgroupModel(ir, q, k, v) for ir in (mixed, result)]
        output = [m.run() for m in models]
        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(models[0].mem[0], models[1].mem[0])
        narrow = fixture(256).replace('i32 16, i32 0, i32 0)', 'i32 4, i32 0, i32 0)')
        narrow_export = export(narrow).stdout
        assert 'call void @llvm.amdgcn.wait.asyncmark(i16 0)' in narrow_export
        assert 'buffer_load_dwordx4' not in narrow_export
        ir.write_text(result)
        subprocess.run([args.llvm_bin / 'opt', '-passes=verify,default<O3>,verify', '-S', ir, '-o', optimized], check=True)
        subprocess.run([args.llvm_bin / 'llc', '-mtriple=amdgcn-amd-amdhsa', f'-mcpu={args.arch}',
                        '-O3', '-filetype=obj', optimized, '-o', obj], check=True)
        print('PASS mixed 4B/16B DMA bytes, common VM wait, narrow-only markers and object export')
        for bad in [fixture(256).replace('i32 16, i32 0, i32 0)', 'i32 16, i32 32, i32 0)'),
                    fixture(256).replace('wait.asyncmark(i16 0)', 'wait.asyncmark(i16 1)')]:
            rejected = export(bad, check=False)
            assert rejected.returncode != 0 and 'asm128 requires' in rejected.stderr
        print('PASS rejects unsupported DMA controls and nonzero async group waits')


if __name__ == '__main__':
    main()
