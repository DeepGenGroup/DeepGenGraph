#!/usr/bin/env python3
"""Check final DTK DMA export ABI and preservation through host optimization.

Upstream LLVM verifies syntax, not HCU target instruction selection. The
optional --target-llvm-bin checks a minimal gfx936 probe with a DTK backend.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
GENERIC = 'llvm.amdgcn.global.load.async.lds'
HCU = 'llvm.hcu.global.load.dwordx4.lds'


def run(command, source=None):
    return subprocess.run(list(map(str, command)), input=source, text=True,
                          capture_output=True, check=True).stdout


def fixture(wide_only=False):
    narrow = '' if wide_only else f'  call void @{GENERIC}(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 4, i32 0, i32 0)\n'
    # Nonzero controls distinguish remapping from a name-only substitution.
    offset, aux = (0, 0) if wide_only else (32, 2)
    return f'''target triple = "amdgcn-amd-amdhsa"
@shared = addrspace(3) global [1024 x i8] undef, align 16
declare void @{GENERIC}(ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)
declare void @llvm.amdgcn.s.barrier()
define amdgpu_kernel void @hcu_dma_probe(ptr addrspace(1) %src) {{
  %dst = getelementptr i8, ptr addrspace(3) @shared, i32 0
  call void @{GENERIC}(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 16, i32 {offset}, i32 {aux})
{narrow}  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.s.barrier()
  ret void
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-tool', type=Path, default=HERE.parent / 'build/test/LegacyLLVMExportTest')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    parser.add_argument('--target-llvm-bin', type=Path)
    parser.add_argument('--target-mattr', default='', help='Explicit target feature override for the compile-only DTK probe')
    parser.add_argument('--probe-output', type=Path)
    args = parser.parse_args()

    def export(text):
        return run([args.export_tool, '--hcu-wide-dma', '-'], text)

    mixed = export(fixture())
    calls = [line for line in mixed.splitlines() if 'call void @llvm.' in line]
    assert len(calls) == 5
    assert HCU in calls[0] and 'i32 32, i32 2, i32 0)' in calls[0]
    assert GENERIC in calls[1] and 'i32 4, i32 0, i32 0)' in calls[1]
    assert 'asyncmark()' in calls[2] and 'wait.asyncmark(i16 0)' in calls[3] and 's.barrier()' in calls[4]
    assert mixed.count('asm sideeffect "", "=v,0"') == 2
    assert export(mixed) == mixed, 'export must be idempotent, including guards'
    probe = export(fixture(wide_only=True))
    assert GENERIC not in probe and probe.count('call void @' + HCU) == 1
    assert export(probe) == probe
    if args.probe_output:
        args.probe_output.write_text(probe)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, text in [('mixed', mixed), ('probe', probe)]:
            source, optimized = root / f'{name}.ll', root / f'{name}.o3.ll'
            source.write_text(text)
            run([args.llvm_bin / 'opt', '-passes=verify,default<O3>,verify', '-S', source, '-o', optimized])
            result = optimized.read_text()
            assert len(re.findall(r'\bcall void @' + re.escape(HCU) + r'\(', result)) == 1
            assert 'asm sideeffect "", "=v,0"' in result
        if args.target_llvm_bin:
            assembly = root / 'probe.s'
            run([args.target_llvm_bin / 'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx936',
                 *([f'-mattr={args.target_mattr}'] if args.target_mattr else []),
                 '-O3', root / 'probe.ll', '-o', assembly])
            text = assembly.read_text()
            assert HCU not in text, 'HCU must select an instruction, not an external call'
            assert re.search(r'global_load[^\n]*dwordx4[^\n]*lds|global_load[^\n]*lds[^\n]*dwordx4', text)
            print('PASS DTK gfx936 dwordx4 LDS instruction selection')
    print('PASS HCU operand ABI, narrow-path preservation, guards, waits/barrier, idempotence and host O3')
    if not args.target_llvm_bin:
        print('NOT RUN: DTK gfx936 instruction selection (supply --target-llvm-bin)')


if __name__ == '__main__':
    main()
