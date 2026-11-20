#!/usr/bin/env python3
"""Full attention export/O3 regression for read packets and MMA memory effects.

Usage: check_attention_codegen.py [MyTest] [LLVM bin directory]
Checks LLVM IR and gfx90a instructions, including LLVM 15 when installed.
This is not a gfx936 machine-code or GPU performance test.
"""
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/MyTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def optimize(path):
    result = subprocess.run([str(LLVM/'opt'), '-S', '-mtriple=amdgcn-amd-amdhsa',
                             '-mcpu=gfx90a', '-passes=default<O3>',
                             str(path), '-o', '-'], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def reads(text):
    counts = collections.Counter(re.findall(
        r' = load (half|<\d+ x half>), ptr addrspace\(3\)', text))
    elements = sum(n * (1 if ty == 'half' else int(ty.split()[0][1:]))
                   for ty, n in counts.items())
    return sum(counts.values()), elements


def invariants(text):
    return (text.count('v_mmac_f32_16x16x16_f16'),
            collections.Counter(re.findall(r's_nop \d+', text)),
            len(re.findall(r'call void @llvm.amdgcn.s.barrier\(', text)))


def main():
    from check_fragment_schedule import check_mma_addresses
    from check_direct_shared_copy import check as check_direct_shared_copy
    from check_rowsum_forward import check as check_rowsum_forward
    from check_packed_attention_copy import check as check_packed_copy
    from check_packed_lds_banks import check as check_banks, check_row_packets
    import ctypes.util
    from check_legacy_llvm_export import load_parser
    library = ctypes.util.find_library('LLVM-15')
    parse_legacy = load_parser(library) if library else None
    results = {}
    with tempfile.TemporaryDirectory(prefix='attention-codegen-') as tmp:
        directory = Path(tmp)
        for enabled in (0, 1):
            path = directory/f'reorder-{enabled}.ll'
            result = subprocess.run([str(TOOL), str(HERE/'test_input.mlir'),
                                     '0', str(path), str(enabled)], cwd=directory,
                                    capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
            exported = path.read_text()
            assert 'readnone' in exported and 'convergent' in exported
            if parse_legacy:
                failed, error = parse_legacy(exported)
                assert not failed, error
            check_mma_addresses(exported)
            check_packed_copy(exported, k_layout="bhsd")
            check_banks(exported)
            check_row_packets(exported)
            check_direct_shared_copy(exported)
            check_rowsum_forward(exported)
            # A wide SLP result alone is insufficient: Q used to become a
            # <64 x half> load at O3, then 64 scalar global loads at codegen.
            # Require bounded explicit packets even before LLVM optimization,
            # without strengthening the external pointer's alignment contract.
            global_loads = re.findall(r' = load (half|<\d+ x half>), ptr addrspace\(1\)', exported)
            assert global_loads == ['<8 x half>'] * 16, collections.Counter(global_loads)
            signature = re.search(r'define amdgpu_kernel void @Attn_p2\((.*)\)', exported)[1]
            assert 'align ' not in signature, signature
            optimized = optimize(path)
            assert invariants(optimized) == invariants(exported)
            assert re.search(r'load <8 x half>, ptr addrspace\(3\)', exported), \
                'paired K fragments must form explicit 128-bit LDS reads'
            assert not re.search(r'load half, ptr addrspace\(1\)', optimized), \
                'packing must preserve vectorized global Q/K/V reads'
            # Isolate packet scheduling from load reuse: temporarily simulate
            # the old printer dropping readnone in a diagnostic copy only.
            opaque = directory/f'opaque-{enabled}.ll'
            opaque.write_text(exported.replace(' readnone', ''))
            original_effects = optimize(opaque)
            assert reads(optimized)[1] < reads(original_effects)[1]
            results[enabled] = reads(optimized), reads(original_effects)
            print(f'PASS reorder={enabled}: LDS half loads/elements '
                  f'{results[enabled][1]} -> {results[enabled][0]}; '
                  'MMA, padding and barriers preserved')
            assembly = directory/f'attention-{enabled}.gfx90a.s'
            optimized_path = directory/f'optimized-{enabled}.ll'
            optimized_path.write_text(optimized)
            result = subprocess.run([
                str(LLVM/'llc'), '-march=amdgcn', '-mcpu=gfx90a', '-O3',
                str(optimized_path), '-o', str(assembly)],
                capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stderr
            instructions = collections.Counter(re.findall(
                r'^\s*(ds_read\w+)', assembly.read_text(), re.M))
            assert instructions['ds_read_b128'] > 0, instructions
            assert instructions['ds_read_u16'] == 0, instructions
            loads = collections.Counter(re.findall(
                r'^\s*(global_load_\w+)', assembly.read_text(), re.M))
            assert loads == {'global_load_dwordx4': 16}, loads
            print(f'PASS gfx90a LDS codegen: {dict(instructions)}; 16 global_load_dwordx4')
            if library and enabled:
                assembly = directory/'attention.gfx90a.s'
                result = subprocess.run([
                    sys.executable, str(HERE/'check_legacy_llvm_export.py'),
                    '--codegen', str(path), str(assembly), library, '3'],
                    capture_output=True, text=True, timeout=120)
                assert result.returncode == 0, result.stderr
                loads = collections.Counter(re.findall(
                    r'^\s*(global_load_\w+)', assembly.read_text(), re.M))
                assert loads == {'global_load_dwordx4': 16}, loads
                assert re.search(r'^\s*ds_read_b128\b', assembly.read_text(), re.M)
                print('PASS LLVM 15 gfx90a: all Q/K/V global reads use 128-bit instructions')
        # Scheduling must preserve SLP opportunities. With precise effects the
        # reused element count stays equal; in the deliberately opaque control,
        # grouping wide reads can enable extra CSE before an asm boundary.
        for variant in (0, 1):
            disabled, enabled = results[0][variant], results[1][variant]
            assert enabled[0] <= disabled[0], (disabled, enabled)
            if variant == 0:
                assert enabled[1] == disabled[1], (disabled, enabled)
            else:
                assert enabled[1] <= disabled[1], (disabled, enabled)
        print('PASS grouped scheduling has no O3 load-count regression')


if __name__ == '__main__':
    main()
