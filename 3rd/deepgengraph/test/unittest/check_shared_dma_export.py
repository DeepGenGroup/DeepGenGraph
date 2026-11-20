#!/usr/bin/env python3
"""Emit planned widths directly; no late ownership or LDS-layout rewrite."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def main():
    with tempfile.TemporaryDirectory(prefix='shared-dma-export-') as directory:
        work = Path(directory)
        source = '''target triple = "amdgcn-amd-amdhsa"
@scratch = addrspace(3) global [8192 x i8] undef, align 16
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
define amdgpu_kernel void @unrelated(ptr addrspace(1) %source) {
entry:
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %source, ptr addrspace(3) @scratch, i32 WIDTH, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
'''
        for width, instruction in ((8, 'buffer_load_dwordx2'), (16, 'buffer_load_dwordx4')):
            inp = work / 'input.ll'
            text = source.replace('WIDTH', str(width))
            if width == 8:
                text = text.replace('llvm.amdgcn.global.load.async.lds', '__frisk_async_load_lds')
            inp.write_text(text)
            subprocess.run([LLVM / 'opt', '-passes=verify', inp, '-disable-output'], check=True)
            result = subprocess.run([EXPORT, '--buffer-asm-dma', inp], text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            assert result.stdout.count(instruction) == 1
            assert 'kdma.source' not in result.stdout
            assert 's_waitcnt vmcnt(0)' in result.stdout
            out = work / 'output.ll'
            out.write_text(result.stdout)
            subprocess.run([LLVM / 'opt', '-passes=default<O3>,verify', out, '-disable-output'], check=True)
        removed = subprocess.run([EXPORT, '--k-dma=64-swizzle', inp], text=True, capture_output=True)
        assert removed.returncode != 0 and 'moved to Frisk Base IR' in removed.stderr
    print('PASS direct 8B/16B buffer DMA emission, waits and removal of late layout rewriting')


if __name__ == '__main__':
    main()
