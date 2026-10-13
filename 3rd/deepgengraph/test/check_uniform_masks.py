#!/usr/bin/env python3
"""Mask specialization at a second tile shape, with conservative fallbacks."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

tool = Path(sys.argv[1]).resolve()
source = Path(__file__).with_name('test_friskBase.mlir').read_text()
source = re.sub(r'%25 = scf.if %24 -> \(f32\) \{.*?\} else \{.*?\n        \}',
                '%25 = arith.select %24, %cst_0, %cst : f32', source, flags=re.S)
assert 'arith.select %24' in source
with tempfile.TemporaryDirectory() as directory:
    path, out = Path(directory) / 'test.mlir', Path(directory) / 'out.mlir'
    for name, text, enabled, expected in [
        ('64x32', source, True, True), ('disabled', source, False, False),
        ('signed', source.replace('cmpi ule', 'cmpi sle'), True, False),
        ('negative-diagonal', source.replace('%c1 = arith.constant 1 : index',
                                             '%c1 = arith.constant -1 : index'), True, False)]:
        path.write_text(text)
        result = subprocess.run([str(tool), str(path),
            '--convert-friskbase-to-thread=fuse-mask-add=false uniform-masks=' + str(enabled).lower(),
            '--verify-each', '-o', str(out)], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-4000:]
        transformed = out.read_text()
        assert ('frisk.uniform_mask' in transformed) == expected, name
        if expected:
            assert transformed.count('arith.cmpi uge') >= 3  # wrapping spans must fall back
            assert 'vector.broadcast' in transformed
print('PASS mask lowering: 64x32 tile, disable switch, signed/negative-offset fallbacks, overflow guards and IR verification')
