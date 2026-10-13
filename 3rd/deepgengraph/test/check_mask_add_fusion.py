#!/usr/bin/env python3
"""Check fragment fusion, operand order, fallbacks and repeated conversion."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

tool = Path(sys.argv[1]).resolve()
source = Path(__file__).with_name('test_friskBase.mlir').read_text()
source = re.sub(r'%25 = scf.if %24 -> \(f32\) \{.*?\} else \{.*?\n        \}',
                '%25 = arith.select %24, %cst_0, %cst : f32', source, flags=re.S)
consumer = '%16 = frisk.add %qk, %maskTIle'
multi = source.replace(consumer, '%extra = frisk.add %qk, %maskTIle : '
    'memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>\n      ' + consumer)
multi = multi.replace('%qkme = frisk.exp2 %16',
    '%sum = frisk.add %extra, %16 : memref<64x32xf32>, memref<64x32xf32> '
    '-> memref<64x32xf32>\n      %qkme = frisk.exp2 %sum')
cases = [
    ('rhs', source, True, True, True),
    ('lhs', source.replace(consumer, '%16 = frisk.add %maskTIle, %qk'), True, True, True),
    ('no-fast-path', source, True, False, True),
    ('64-threads', source.replace('thread_num = 128', 'thread_num = 64'), True, True, True),
    ('256-threads', source.replace('thread_num = 128 : i32',
        'thread_num = 256 : i32, frisk.block_warps = array<i64: 4, 1>'), True, True, True),
    ('disabled', source, False, True, False),
    ('multi-use', multi, True, True, False),
    ('barrier', source.replace(consumer, 'gpu.barrier\n      ' + consumer), True, True, False),
    ('signed-predicate', source.replace('cmpi ule', 'cmpi sle'), True, True, True),
    ('negative-offset', source.replace('%c1 = arith.constant 1 : index',
                                      '%c1 = arith.constant -1 : index'), True, True, True),
    ('non-triangular', source.replace('%25 = arith.select %24, %cst_0, %cst : f32',
                                    '%25 = arith.addf %cst_0, %cst : f32'), True, True, True),
]
with tempfile.TemporaryDirectory() as directory:
    path, out = Path(directory) / 'test.mlir', Path(directory) / 'out.mlir'
    for name, text, fused, uniform, expected in cases:
        path.write_text(text)
        options = f'fuse-mask-add={str(fused).lower()} uniform-masks={str(uniform).lower()}'
        command = [str(tool), str(path), '--convert-friskbase-to-thread=' + options,
                   '--verify-each', '-o', str(out)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (name, result.stderr[-6000:])
        transformed = out.read_text()
        assert ('frisk.fused_mask_add' in transformed) == expected, name
        if expected:
            # Each arithmetic instruction resides inside the mask fragment;
            # the branch produces summed values, never a standalone mask tile.
            for region in transformed.split('frisk.fragment'):
                if 'frisk.fused_mask_add' not in region:
                    continue
                assert 'arith.addf' in region and 'vector.extract' in region, name
            # The matched operation keeps both operands (including +0).
            assert not re.search(r'arith.addf[^\n]*fastmath', transformed), name
        path.write_text(transformed)
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (name, result.stderr[-4000:])
        assert out.read_text() == transformed, (name, 'not idempotent')
        print('PASS', name, flush=True)
