#!/usr/bin/env python3
"""Full attention export/O3 regression for read packets and MMA memory effects.

Usage: check_attention_codegen.py [MyTest] [LLVM bin directory]
This checks LLVM IR, not gfx936 machine code or GPU performance.
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
    result = subprocess.run([str(LLVM/'opt'), '-S', '-passes=default<O3>',
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
            check_direct_shared_copy(exported)
            check_rowsum_forward(exported)
            optimized = optimize(path)
            assert invariants(optimized) == invariants(exported)
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
        # Scheduling must preserve SLP opportunities both with and without
        # precise MMA effects, as well as preserve the total accessed elements.
        for variant in (0, 1):
            disabled, enabled = results[0][variant], results[1][variant]
            assert enabled[0] <= disabled[0], (disabled, enabled)
            assert enabled[1] == disabled[1], (disabled, enabled)
        print('PASS grouped scheduling has no O3 load-count regression')


if __name__ == '__main__':
    main()
