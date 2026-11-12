#!/usr/bin/env python3
"""Check direct exp2 selection, vector scalarization and strict fallbacks."""
import subprocess
import sys
from pathlib import Path

tool = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parent.parent / 'build/test/ThreadTilingTest')


def lower(ty, native, attrs=''):
    dialect = 'llvm' if attrs else 'func'
    source = f'''module {{
      {dialect}.func @exp2(%x: {ty}) -> {ty} {attrs} {{
        %y = math.exp2 %x : {ty}
        {dialect}.return %y : {ty}
      }}
    }}'''
    result = subprocess.run(
        [str(tool), '--verify-each', f'--test-exp2-to-rocdl=native-exp2={str(native).lower()}'],
        input=source, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'math.exp2' not in result.stdout, result.stdout
    return result.stdout


for ty, count in [('f32', 1), ('vector<4xf32>', 4), ('vector<2x4xf32>', 8)]:
    native = lower(ty, True)
    assert native.count('rocdl.exp2') == count, native
    assert '__ocml_exp2' not in native, native
    strict = lower(ty, False)
    assert strict.count('llvm.call @__ocml_exp2_f32') == count, strict
    assert 'rocdl.exp2' not in strict, strict

for ty in ['f16', 'f64']:
    ir = lower(ty, True)
    assert f'__ocml_exp2_{ty}' in ir and 'rocdl.exp2' not in ir, ir

for attrs in ['attributes {passthrough = ["strictfp"]}',
              'attributes {passthrough = ["optnone", "noinline"]}',
              'attributes {optimize_none, no_inline}']:
    ir = lower('f32', True, attrs)
    assert '__ocml_exp2_f32' in ir and 'rocdl.exp2' not in ir, ir

print('PASS native/strict scalar and multidimensional vector exp2; f16/f64 and protected function fallbacks')
