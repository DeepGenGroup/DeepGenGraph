#!/usr/bin/env python3
"""Emit/run an isolated gfx936 m32x16_b16 lane-layout probe.

  python check_matrix_read_hardware.py --emit /tmp/matrix_probe.ll
  # Compile that file with the same HCU compiler used for attention.
  python check_matrix_read_hardware.py --hsaco /tmp/matrix_probe.hsaco

One wave, no DMA or MMAC. Tests plain rows and the exact row-packet packing
used for attention B with K=64/128. Every read has an explicit completion wait.
"""
import argparse
from pathlib import Path
import sys


def probe_ir():
    lines = [
        '@probe_plain = internal addrspace(3) global [512 x half] undef, align 16',
        '@probe_k64 = internal addrspace(3) global [2048 x half] undef, align 16',
        '@probe_k128 = internal addrspace(3) global [4096 x half] undef, align 16',
        'declare i32 @llvm.amdgcn.workitem.id.x()',
        'declare void @llvm.amdgcn.s.barrier() convergent',
        'declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) convergent memory(read)',
        'define amdgpu_kernel void @MatrixReadProbe(ptr addrspace(1) %input, '
        'ptr addrspace(1) %unused1, ptr addrspace(1) %unused2, ptr addrspace(1) %out) '
        '#0 !reqd_work_group_size !0 {',
        'entry:',
        '  %tid = call i32 @llvm.amdgcn.workitem.id.x()',
        '  %logical = shl i32 %tid, 3',
        '  %row = lshr i32 %tid, 2',
        '  %colgroup = and i32 %tid, 3',
        '  %src = getelementptr half, ptr addrspace(1) %input, i32 %logical',
        '  %data = load <8 x half>, ptr addrspace(1) %src, align 16',
        '  %plain = getelementptr half, ptr addrspace(3) @probe_plain, i32 %logical',
        '  store <8 x half> %data, ptr addrspace(3) %plain, align 16',
    ]
    for k in (64, 128):
        lines += [
            f'  %k{k}.group = mul i32 %colgroup, {k}',
            f'  %k{k}.row = add i32 %k{k}.group, %row',
            f'  %k{k}.offset = shl i32 %k{k}.row, 3',
            f'  %k{k} = getelementptr half, ptr addrspace(3) @probe_k{k}, i32 %k{k}.offset',
            f'  store <8 x half> %data, ptr addrspace(3) %k{k}, align 16',
        ]
    lines += ['  call void @llvm.amdgcn.s.barrier()']
    for case, pointer in enumerate(('plain', 'k64', 'k128')):
        lines += [
            '  call void asm sideeffect "s_waitcnt vmcnt(0) lgkmcnt(0)", "~{memory}"()',
            f'  %read{case} = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %{pointer}, i32 0)',
            '  call void asm sideeffect "s_waitcnt lgkmcnt(0)", "~{memory}"()',
            f'  %output{case}.offset = add i32 %logical, {case * 512}',
            f'  %output{case}.ptr = getelementptr half, ptr addrspace(1) %out, i32 %output{case}.offset',
            f'  store <4 x i32> %read{case}, ptr addrspace(1) %output{case}.ptr, align 16',
        ]
    lines += ['  ret void', '}',
              'attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }',
              '!0 = !{i32 64, i32 1, i32 1}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--emit', type=Path)
    mode.add_argument('--hsaco', type=Path)
    args = parser.parse_args()
    if args.emit:
        args.emit.write_text(probe_ir())
        print(args.emit)
        return
    # Also support copying this standalone probe next to kernelLauncher.py.
    for directory in (*Path(__file__).resolve().parents, Path.cwd()):
        if (directory / 'kernelLauncher.py').is_file():
            sys.path.insert(0, str(directory))
            break
    else:
        parser.error('Cannot find kernelLauncher.py. Put it next to this script '
                     'or run from the directory containing kernelLauncher.py.')
    import torch
    from kernelLauncher import loaded_hsaco
    source = torch.arange(512, device='cuda', dtype=torch.float16)
    output = torch.full((3, 64, 8), float('nan'), device='cuda', dtype=torch.float16)
    with loaded_hsaco(str(args.hsaco), 'MatrixReadProbe',
                      [source, source, source, output], (1, 1, 1), (64, 1, 1)) as launch:
        launch()
    lanes = torch.arange(64, device='cuda')[:, None]
    regs = torch.arange(8, device='cuda')[None, :]
    expected = ((lanes // 16 * 4 + regs % 4) * 32 + lanes % 16 + regs // 4 * 16).half()
    passed = True
    for name, actual in zip(('plain', 'packed K=64', 'packed K=128'), output):
        mismatch = actual != expected
        count = mismatch.sum().item()
        print(f'{name}: mismatch={count}/512, exact={count == 0}')
        for lane, reg in mismatch.nonzero()[:16].tolist():
            print(f'  lane={lane} half-register={reg}: actual={actual[lane, reg].item()}, '
                  f'expected={expected[lane, reg].item()}')
        passed &= count == 0
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
