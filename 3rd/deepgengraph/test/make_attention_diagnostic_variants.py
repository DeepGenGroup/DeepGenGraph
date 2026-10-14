#!/usr/bin/env python3
"""Produce controlled LLVM diagnostics or export only synchronous DMA packets.

Each variant preserves computation/addresses/launch ABI. One replaces 4-byte
DMA with ordinary GM loads + LDS stores including DMA's implicit lane offset;
one replaces MMAC NOP padding with a deliberately conservative interval; one
combines both. Counts and provenance are recorded for hardware A/B testing.
--sync-output exports the same sync_copy transformation without MMAC changes.
--guard-dma-output retains DMA and makes each complete global address opaque
to backend constant-offset folding, which otherwise also shifts LDS writes.
Hardware validation is specific to the tested artifact, not every future input.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from check_gemm_pv_addresses import split_args


def synchronous_packets(text):
    count = 0
    if re.search(r'%diag\.copy\d+\.', text):
        raise ValueError('Input already uses the synchronous-copy SSA namespace')

    def replace(match):
        nonlocal count
        source, destination, size, offset, aux = split_args(match[1])
        if (size, offset, aux) != ('i32 4', 'i32 0', 'i32 0'):
            raise ValueError('Synchronous export supports only direct 4-byte, zero-offset DMA')
        if not source.startswith('ptr addrspace(1) ') or not destination.startswith('ptr addrspace(3) '):
            raise ValueError('Synchronous export requires global source and LDS destination pointers')
        name = f'%diag.copy{count}'
        count += 1
        return '\n'.join([
            f'  {name}.tid = call i32 @llvm.amdgcn.workitem.id.x()',
            f'  {name}.lane = and i32 {name}.tid, 63',
            f'  {name}.offset = shl i32 {name}.lane, 2',
            f'  {name}.ptr = getelementptr i8, {destination}, i32 {name}.offset',
            f'  {name}.value = load <2 x half>, {source}, align 4',
            f'  store <2 x half> {name}.value, ptr addrspace(3) {name}.ptr, align 4',
        ])

    result = re.sub(r'^  call void @llvm.amdgcn.global.load.async.lds\((.*)\)[^\n]*$',
                    replace, text, flags=re.M)
    if count == 0:
        raise ValueError('No supported DMA calls found')
    if re.search(r'\bcall\b[^\n]*@llvm\.amdgcn\.global\.load\.async\.lds\(', result):
        raise ValueError('Unsupported DMA call remains; refusing a partial synchronous export')
    return result, count


def conservative_mmac_spacing(text):
    count = 0

    def replace(match):
        nonlocal count
        assembly = match[1]
        instructions = [x.strip() for x in assembly.split(r'\0A\09') if x.strip()]
        mma = [x for x in instructions if x.startswith('v_mmac_f32_16x16x16_f16 ')]
        if not mma:
            return match[0]
        if len(mma) != 1 or any(x != mma[0] and not re.fullmatch(r's_nop \d+', x)
                               for x in instructions):
            raise ValueError('MMAC asm contains unexpected instructions')
        count += 1
        padded = r'\0A\09'.join(['s_nop 7'] * 8 + mma + ['s_nop 7'] * 8) + r'\0A\09'
        return 'asm sideeffect "' + padded + '"'

    result = re.sub(r'asm sideeffect "([^"]*)"', replace, text)
    if count == 0:
        raise ValueError('No supported MMAC calls found')
    return result, count


def guard_dma_addresses(text):
    """Pin the final 64-bit source, not just an intermediate address base.

    A global-to-LDS instruction's immediate affects both addresses. An empty
    tied VGPR asm preserves pointer bits but prevents load/store optimization
    from sharing a source base and introducing uncompensated immediates.
    Keep every original DMA, mark/wait, barrier and MMAC instruction.
    """
    count = 0
    attribute = max((int(n) for n in re.findall(r'#(\d+)', text)), default=-1) + 1
    if re.search(r'%diag\.dma\d+\.', text):
        raise ValueError('Input already uses the DMA address guard SSA namespace')

    def replace(match):
        nonlocal count
        source, destination, size, offset, aux = split_args(match[1])
        if (size, offset, aux) != ('i32 4', 'i32 0', 'i32 0'):
            raise ValueError('DMA address guard requires 4-byte, zero-offset, zero-aux packets')
        if not source.startswith('ptr addrspace(1) ') or not destination.startswith('ptr addrspace(3) '):
            raise ValueError('DMA address guard requires global and LDS pointers')
        name = f'%diag.dma{count}'
        count += 1
        return '\n'.join([
            f'  {name}.bits = ptrtoint {source} to i64',
            f'  {name}.opaque = call i64 asm sideeffect "", "=v,0"(i64 {name}.bits) #{attribute}',
            f'  {name}.ptr = inttoptr i64 {name}.opaque to ptr addrspace(1)',
            f'  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) {name}.ptr, {destination}, {size}, {offset}, {aux}){match[2]}',
        ])

    result = re.sub(r'^  call void @llvm.amdgcn.global.load.async.lds\((.*)\)([^\n]*)$',
                    replace, text, flags=re.M)
    calls = len(re.findall(r'\bcall\b[^\n]*@llvm\.amdgcn\.global\.load\.async\.lds\(', text))
    if not count or count != calls:
        raise ValueError('Unsupported DMA call format; refusing a partial address guard')
    result += f'\nattributes #{attribute} = {{ convergent nounwind memory(none) }}\n'
    return result, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output_dir', type=Path, nargs='?')
    parser.add_argument('--sync-output', type=Path,
                        help='Export only sync_copy to this LLVM file (input may equal output)')
    parser.add_argument('--guard-dma-output', type=Path,
                        help='Export a DMA-preserving address-guard candidate; requires hardware validation')
    args = parser.parse_args()
    if sum(x is not None for x in (args.output_dir, args.sync_output, args.guard_dma_output)) != 1:
        parser.error('Provide exactly one of output_dir, --sync-output, --guard-dma-output')
    text = args.input.read_text()
    if args.guard_dma_output is not None:
        guarded, count = guard_dma_addresses(text)
        args.guard_dma_output.write_text(guarded)
        print(f'DMA address guard: {args.guard_dma_output}; guarded {count} calls; '
              f'SHA256={hashlib.sha256(guarded.encode()).hexdigest()}')
        return
    sync, copies = synchronous_packets(text)
    if args.sync_output is not None:
        args.sync_output.write_text(sync)
        print(f'Synchronous export: {args.sync_output}; replaced {copies} DMA calls; '
              f'SHA256={hashlib.sha256(sync.encode()).hexdigest()}')
        return
    padded, mmas = conservative_mmac_spacing(text)
    combined, _ = conservative_mmac_spacing(sync)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'input': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'status': 'diagnostic_only_not_gpu_validated', 'variants': {}}
    for name, ir, changes in [
        ('sync_copy', sync, {'dma_calls_replaced': copies}),
        ('mmac_wait', padded, {'mmac_padding_sites': mmas}),
        ('sync_copy_mmac_wait', combined, {'dma_calls_replaced': copies, 'mmac_padding_sites': mmas}),
    ]:
        path = args.output_dir / (name + '.ll')
        path.write_text(ir)
        manifest['variants'][name] = {'path': str(path.resolve()),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     **changes}
        print(path)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
