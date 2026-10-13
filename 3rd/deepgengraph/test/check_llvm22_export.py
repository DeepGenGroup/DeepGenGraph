#!/usr/bin/env python3
"""Check LLVM 24 -> LLVM 22 export with both versions' actual parsers.

Usage: check_llvm22_export.py LegacyLLVMExportTest LLVM22_BIN LLVM24_BIN
"""
import subprocess
import sys
from pathlib import Path


def run(args, text):
    result = subprocess.run(list(map(str, args)), input=text, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    helper, llvm22, llvm24 = map(Path, sys.argv[1:])
    source = '''@half_value = constant half 1.275630e-01
@float_value = constant float f0x3E029666
@double_value = constant double f0x3FC052CCC0000000
@negative_zero = constant half -0.000000e+00
@infinity = constant float -inf
@nan = constant half +qnan
@payload = constant float +nan(0x123)
@signaling_nan = constant float +snan(0x123)
@bfloat_value = constant bfloat f0x3E03
@"half 1.275630e-01" = constant i32 0
@vec = constant <2 x half> splat (half 1.275630e-01)
declare noundef range(i32 0, 1024) i32 @thread_id() nocreateundeforpoison memory(none)
define float @scalar(float %double) {
  %v = fadd float %double, f0x3E029666 ; half 1.275630e-01
  ret float %v
}
define ptr @address(ptr captures(none) %p, i64 %i) {
  %flags = or disjoint i64 %i, 16
  %a = getelementptr inbounds nuw i8, ptr %p, i64 %flags
  ret ptr %a
}
'''
    # Work from the LLVM 24 printer's actual syntax, as production export does.
    canonical = run([llvm24 / 'opt', '-S', '-o', '-', '-'], source)
    converted = run([helper, '--llvm22-text', '-'], canonical)
    for token in ('splat (', 'range(i32 0, 1024)', 'nocreateundeforpoison',
                  'memory(none)', 'captures(none)', 'or disjoint',
                  'getelementptr inbounds nuw', '@"half 1.275630e-01"'):
        assert token in converted, (token, converted)
    assert 'half 0xH3015' in converted, converted
    assert 'half 0xH8000' in converted, converted
    assert run([helper, '--llvm22-text', '-'], converted) == converted
    verified = run([llvm22 / 'opt', '-passes=verify', '-S', '-o', '-', '-'], converted)
    roundtrip = run([llvm24 / 'opt', '-S', '-o', '-', '-'], converted)
    assert canonical == roundtrip, 'floating-point bits or attributes changed'
    via22 = run([llvm24 / 'opt', '-S', '-o', '-', '-'], verified)
    assert canonical == via22, 'LLVM 22 roundtrip changed floating-point bits or attributes'
    # Text handling must not modify names or comments containing FP spellings.
    literal_text = run([helper, '--llvm22-text', '-'], source)
    assert '; half 1.275630e-01' in literal_text
    assert '%double' in literal_text and '@infinity' in literal_text
    print('PASS LLVM 22 parser/verifier, FP bit preservation, attributes and splats')


if __name__ == '__main__':
    main()
