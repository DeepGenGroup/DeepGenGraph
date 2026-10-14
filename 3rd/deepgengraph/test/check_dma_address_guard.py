#!/usr/bin/env python3
"""Verify DMA preservation and zero-offset codegen on an available gfx900 backend.

The small reproducer has no vendor MMAC instructions. This is a backend
regression check, not execution of the gfx936 attention kernel.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from check_dma_assembly import dma_offsets
from make_attention_diagnostic_variants import guard_dma_addresses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--export-tool', type=Path,
                        help='Also check the integrated C++ guard and its idempotence')
    args = parser.parse_args()
    source = Path(__file__).with_name('test_dma_offset_folding.ll').read_text()
    guarded, count = guard_dma_addresses(source)
    assert count == 8
    assert guarded.count('call void @llvm.amdgcn.global.load.async.lds(') == 8
    assert guarded.count('call void @llvm.amdgcn.asyncmark(') == 1
    assert guarded.count('call void @llvm.amdgcn.wait.asyncmark(') == 1
    # Removing the guards and restoring each original pointer recovers all
    # original calls, arguments, barriers and computations byte for byte.
    restored = guarded
    for i in range(count):
        name = f'%diag.dma{i}'
        pointer = re.search(re.escape(name) + r'\.bits = ptrtoint ptr addrspace\(1\) (.+) to i64', restored)[1]
        restored = re.sub(r'^  ' + re.escape(name) + r'\.(?:bits|opaque|ptr) = .*\n', '', restored, flags=re.M)
        restored = restored.replace(name + '.ptr', pointer)
    restored = re.sub(r'\nattributes #\d+ = \{ convergent nounwind memory\(none\) \}\n$', '', restored)
    assert restored == source
    for invalid in (guarded, source.replace('i32 4, i32 0, i32 0)', 'i32 8, i32 0, i32 0)', 1),
                    source.replace('  call void @llvm.amdgcn.global.load.async.lds',
                                   '  tail call void @llvm.amdgcn.global.load.async.lds', 1)):
        try:
            guard_dma_addresses(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported/repeated guard must be rejected')
    variants = [('original', source), ('guarded', guarded)]
    if args.export_tool:
        def export(text):
            return subprocess.run([args.export_tool, '--guard-dma-addresses', '-'],
                                  input=text, text=True, capture_output=True, check=True).stdout
        integrated = export(source)
        assert export(integrated) == integrated, 'Repeated export must be idempotent'
        imported = export(guarded)
        for ir in (integrated, imported):
            assert ir.count('call void @llvm.amdgcn.global.load.async.lds(') == 8
            assert ir.count('asm sideeffect "", "=v,0"') == 8, 'No missing or duplicate guards'
        variants += [('integrated', integrated), ('imported', imported)]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, text in variants:
            ir, optimized, asm = (root / (name + ext) for ext in ('.ll', '.o3.ll', '.s'))
            ir.write_text(text)
            subprocess.run([args.llvm_bin / 'opt', '-passes=verify,default<O3>,verify', '-S', ir, '-o', optimized], check=True)
            subprocess.run([args.llvm_bin / 'llc', '-march=amdgcn', '-mcpu=gfx900', '-O3', optimized, '-o', asm], check=True)
            offsets = [offset for _, offset, _ in dma_offsets(asm.read_text(), 'dma_offset_probe')]
            assert len(offsets) == 8, 'Backend must recognize DMA rather than emit external calls'
            print(name, 'offsets:', offsets)
            if name != 'original':
                assert offsets == [0] * 8
        # Future backends may fix the original path; only the guarded invariant
        # is required. The supplied failing gfx936 log remains the bug witness.
    print('PASS pointer identity, instruction preservation, rejection cases and gfx900 codegen')


if __name__ == '__main__':
    main()
