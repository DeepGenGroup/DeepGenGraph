#!/usr/bin/env python3
"""Isolate HCU matrix-read semantics/waits without changing attention packing.

The scalar control exchanges the actual lane pointers with ds_bpermute and
loads each output half directly. It does not assume a common matrix tile base.
These are hardware diagnostics, not production fixes or performance candidates.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from check_gemm_pv_addresses import split_args
from make_attention_diagnostic_variants import synchronous_packets

READ = re.compile(r'^  (%[\w.-]+) = (?:tail )?call <4 x i32> '
                  r'@llvm.hcu.ds.read.m32x16.b16.v4i32.p3\((.*)\)[^\n]*$', re.M)


def scalar_reads(text):
    count = 0
    if '%diag.matrix' in text:
        raise ValueError('Input already contains matrix diagnostics')

    def replace(match):
        nonlocal count
        pointer, offset = split_args(match[2])
        assert pointer.startswith('ptr addrspace(3) ') and offset == 'i32 0'
        p = f'%diag.matrix{count}'
        count += 1
        lines = [
            f'{p}.tid = call i32 @llvm.amdgcn.workitem.id.x()',
            f'{p}.lane = and i32 {p}.tid, 63',
            f'{p}.group = lshr i32 {p}.lane, 4',
            f'{p}.row = shl i32 {p}.group, 2',
            f'{p}.col = and i32 {p}.lane, 15',
            f'{p}.pointer = ptrtoint {pointer} to i32',
        ]
        for reg in range(8):
            r = f'{p}.r{reg}'
            lines += [
                f'{r}.row = add i32 {p}.row, {reg % 4}',
                f'{r}.col = add i32 {p}.col, {reg // 4 * 16}',
                f'{r}.packet = lshr i32 {r}.col, 3',
                f'{r}.rowlane = shl i32 {r}.row, 2',
                f'{r}.owner = add i32 {r}.rowlane, {r}.packet',
                f'{r}.shuffle = shl i32 {r}.owner, 2',
                f'{r}.base = call i32 @llvm.amdgcn.ds.bpermute(i32 {r}.shuffle, i32 {p}.pointer)',
                f'{r}.element = and i32 {r}.col, 7',
                f'{r}.bytes = shl i32 {r}.element, 1',
                f'{r}.address = add i32 {r}.base, {r}.bytes',
                f'{r}.ptr = inttoptr i32 {r}.address to ptr addrspace(3)',
                f'{r}.value = load half, ptr addrspace(3) {r}.ptr, align 2',
                f'{r}.packed = insertelement <8 x half> '
                f'{p + ".r" + str(reg - 1) + ".packed" if reg else "poison"}, half {r}.value, i32 {reg}',
            ]
        lines += [f'{match[1]} = bitcast <8 x half> {p}.r7.packed to <4 x i32>']
        return '\n'.join('  ' + line for line in lines)

    result = READ.sub(replace, text)
    assert count and not READ.search(result)
    if not re.search(r'^declare i32 @llvm.amdgcn.ds.bpermute\(', result, re.M):
        result += '\ndeclare i32 @llvm.amdgcn.ds.bpermute(i32, i32) convergent nounwind memory(none)\n'
    return result, count


def wait_reads(text):
    # The memory clobber prevents motion across the read; no block barrier or
    # buffer lifetime is added. Explicitly drain both input and output hazards.
    wait = '  call void asm sideeffect "s_waitcnt vmcnt(0) lgkmcnt(0)", "~{memory}"()'
    result, count = READ.subn(lambda m: wait + '\n' + m[0] + '\n' + wait, text)
    assert count
    return result, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    text = args.input.read_text()
    scalar, count = scalar_reads(text)
    waited, _ = wait_reads(text)
    sync, copies = synchronous_packets(text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'input': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'status': 'diagnostic_only_not_gpu_validated', 'variants': {}}
    for name, ir, changes in [
        ('scalar_matrix', scalar, {'matrix_calls_replaced': count}),
        ('matrix_wait', waited, {'matrix_calls_waited': count}),
        ('sync_dma', sync, {'dma_calls_replaced': copies}),
    ]:
        path = args.output_dir / (name + '.ll')
        path.write_text(ir)
        manifest['variants'][name] = {'path': str(path.resolve()),
            'sha256': hashlib.sha256(ir.encode()).hexdigest(), **changes}
        print(path)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
