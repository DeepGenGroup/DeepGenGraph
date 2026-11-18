#!/usr/bin/env python3
"""Compare MyTest's default export with --llvm-opt=0 followed by external O3.

Usage: check_mytest_export_o3.py raw.ll optimized.ll /path/to/llvm/bin
Generate both files from the same input and tuning arguments. LLVM tools must
match the LLVM libraries linked into MyTest. No GPU is needed.
"""
import difflib
from pathlib import Path
import re
import subprocess
import sys


def canonical(path, llvm, pipeline):
    result = subprocess.run(
        [str(llvm / 'opt'), '-S', '-passes=' + pipeline, str(path), '-o', '-'],
        check=True, capture_output=True, text=True, timeout=120)
    return re.sub(r'^; ModuleID = .*\n', '', result.stdout)


def main():
    raw, optimized, llvm = map(Path, sys.argv[1:])
    expected = canonical(raw, llvm, 'default<O3>,verify')
    # Read both optimized outputs exactly once. LLVM's reader resets intrinsic
    # attributes (including inferred mustprogress) and predecessor use order;
    # comparing freshly optimized text to reread text is not symmetric.
    reread = subprocess.run(
        [str(llvm / 'opt'), '-S', '-passes=verify', '-', '-o', '-'],
        input=expected, check=True, capture_output=True, text=True, timeout=120)
    expected = re.sub(r'^; ModuleID = .*\n', '', reread.stdout)
    actual = canonical(optimized, llvm, 'verify')
    if actual != expected:
        diff = ''.join(difflib.unified_diff(
            expected.splitlines(True), actual.splitlines(True),
            fromfile='external O3', tofile='MyTest O3'))
        raise SystemExit('MyTest export differs from external O3:\n' + diff[:12000])
    if canonical(raw, llvm, 'verify') == actual:
        raise SystemExit('Fixture must exercise O3: raw and optimized IR are identical')
    print('PASS: MyTest O3 matches external default<O3>; both modules verified')


if __name__ == '__main__':
    main()
