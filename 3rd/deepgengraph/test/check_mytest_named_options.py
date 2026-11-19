#!/usr/bin/env python3
"""Check named/legacy CLI validation without running the lowering pipeline.

Usage: check_mytest_named_options.py /path/to/MyTest
A deliberately missing input distinguishes successful option parsing from errors.
Full IR equivalence should additionally be checked on an attention fixture.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = str(Path(sys.argv[1]).resolve())
    env = dict(os.environ, FRISK_GM_TO_SHM_MODE='async')
    with tempfile.TemporaryDirectory(prefix='mytest-cli-') as directory:
        missing = str(Path(directory) / 'missing.mlir')

        def check(args, expected, code=1):
            result = subprocess.run([binary, *args], cwd=directory, env=env,
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == code, (args, result.returncode, result.stderr)
            assert expected in result.stderr, (args, result.stderr)

        parsed = 'Failed to parse input MLIR file: ' + missing
        check(['--help'], '--block-warps=VALUE', code=0)
        check([], 'usage:')
        check(['--threads=256'], 'expected input.mlir or --input=PATH')
        check([missing], parsed)
        check([f'--input={missing}'], parsed)
        check([missing, 'out.ll', '--threads=256'], parsed)
        check([missing, '0', 'out.ll', '--threads=256', '--block-warps=4,1'], parsed)
        pairs = [
            ('input', missing), ('pipeline', '0'), ('output', 'out.ll'),
            ('reorder', '1'), ('pack-shared', '1'), ('wmma-layout', '3.3'),
            ('math', 'approx'), ('threads', '256'), ('block-warps', '4,1'),
            ('register-lifetimes', '1'), ('uniform-masks', '0'),
            ('prefetch-packets', '1'), ('fuse-mask-add', '0'),
            ('lds-reuse-words', '0'), ('dma-pack', 'useKPack'),
            ('dma-export', 'asm128'), ('qk-tile', '128'), ('vo-tile', '128')]
        check([value for _, value in pairs], parsed)
        named = [f'--{name}={value}' for name, value in pairs]
        check(named + ['--llvm-opt=0'], parsed)
        check(list(reversed(named)), parsed)
        check([f'--input={missing}', '--wmma-layout=b-permute'], parsed)
        # Sparse options preserve defaults, including the implicit 128 threads.
        check([f'--input={missing}', '--block-warps=2,1'], parsed)
        check([f'--input={missing}', '--block-warps=4,1'], 'block-warps must be')
        for args, expected in [
            (['--threads'], 'use --threads=VALUE'),
            (['--threads='], 'use --threads=VALUE'),
            (['--threads=256', '--threads=128'], 'duplicate option --threads'),
            (['--pipeline=2'], 'pipeline must be 0 or 1'),
            (['--threads=65'], 'threads must be a multiple of 64'),
            (['--qk-tile=-1'], 'QK/VO tile must be nonnegative'),
            (['--vo-tile=abc'], 'QK/VO tile must be nonnegative'),
            (['--prefetch-packets=2'], 'prefetch-packets must be 0 or 1'),
            (['--math=strict', '--local-reduce=approx'], 'requires math=approx'),
            (['--typo=1'], 'unknown option: --typo=1'),
            (['--input=another.mlir'], '--input also supplied positionally'),
            (['0', '--pipeline=0'], '--pipeline also supplied positionally'),
            (['out.ll', '--output=another.ll'], '--output also supplied positionally'),
        ]:
            check([missing, *args], expected)
    print('PASS: named options, legacy positions, shorthand, defaults and diagnostics')


if __name__ == '__main__':
    main()
