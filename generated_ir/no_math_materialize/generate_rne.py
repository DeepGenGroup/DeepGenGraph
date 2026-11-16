#!/usr/bin/env python3
"""Restore ordinary softmax fptrunc in the two asm-boundary-free PKRTZ IRs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LLVM = Path('/data2/xsl/install/bin')
PAIRS = [
    ('best_noncausal_early_wave/kernel.no_math_materialize.o3.ll',
     'best_noncausal_early_wave/kernel.rne.no_math_materialize.o3.ll'),
    ('causal_mask_lowering/mask_finite_pk_rtz.no_math_materialize.o3.ll',
     'causal_mask_lowering/mask_finite_rne.no_math_materialize.o3.ll'),
]
SSA = r'%[-a-zA-Z$._0-9]+'
START = re.compile(r'  (%\.cvtpack\.f0\d*) = extractelement <4 x float> ('
                   + SSA + r'), i64 0(, !dbg !\d+)')
MARKER = ' "frisk.softmax_cvt_pack"="rtz"'


def digest(data):
    return hashlib.sha256(data).hexdigest()


records = []
for input_name, output_name in PAIRS:
    source, target = ROOT / input_name, ROOT / output_name
    original = source.read_bytes()
    text = original.decode()
    lines = text.splitlines(keepends=True)
    retained, changes = [], []
    i = 0
    while i < len(lines):
        start = START.fullmatch(lines[i].rstrip('\n'))
        if not start:
            retained.append(lines[i])
            i += 1
            continue
        _, vector, metadata = start.groups()
        block = [line.rstrip('\n') for line in lines[i:i + 9]]
        elements, pairs, packed = [], [], []
        for lane in range(4):
            match = re.fullmatch(r'  (' + SSA + r') = extractelement <4 x float> '
                                 + re.escape(vector) + f', i64 {lane}'
                                 + re.escape(metadata), block[lane])
            assert match, block[lane]
            elements.append(match[1])
        for pair in range(2):
            expression = (' = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", '
                          f'"=v,v,v"(float {elements[2 * pair]}, float {elements[2 * pair + 1]})')
            match = re.fullmatch(r'  (' + SSA + ')' + re.escape(expression)
                                 + r' #\d+' + re.escape(metadata), block[4 + pair])
            assert match, block[4 + pair]
            pairs.append(match[1])
        for lane in range(2):
            base = 'poison' if lane == 0 else packed[0]
            expression = f' = insertelement <2 x i32> {base}, i32 {pairs[lane]}, i64 {lane}{metadata}'
            match = re.fullmatch(r'  (' + SSA + ')' + re.escape(expression), block[6 + lane])
            assert match, block[6 + lane]
            packed.append(match[1])
        match = re.fullmatch(r'  (' + SSA + ')' + re.escape(
            f' = bitcast <2 x i32> {packed[1]} to <4 x half>{metadata}'), block[8])
        assert match, block[8]
        replacement = f'  {match[1]} = fptrunc <4 x float> {vector} to <4 x half>{metadata}\n'
        retained.append(replacement)
        changes.append((''.join(lines[i:i + 9]), replacement))
        i += 9
    assert len(changes) == 16
    output = ''.join(retained)
    # Reversing only these 16 replacements must recover the exact input.
    restored = output
    for block, replacement in changes:
        assert restored.count(replacement) == 1
        restored = restored.replace(replacement, block)
    assert restored == text
    assert output.count(MARKER) == 1
    output = output.replace(MARKER, '')
    assert 'cvtpack' not in output and 'v_cvt_pkrtz_f16_f32' not in output
    assert not re.search(r'%math\.(?:result|packed)\d*\b', output)
    assert output.count(' = fptrunc ') == text.count(' = fptrunc ') + 16 == 32
    target.write_text(output)
    commands = [
        [str(LLVM / 'llvm-as'), str(target), '-o', '/dev/null'],
        [str(LLVM / 'opt'), '-passes=verify', '-disable-output', str(target)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    config = target.with_suffix('.launch.json')
    config.write_bytes(source.with_suffix('.launch.json').read_bytes())
    assert source.read_bytes() == original
    records.append(dict(source=str(source), output=str(target),
                        source_sha256=digest(original), output_sha256=digest(output.encode()),
                        restored_fptrunc=16, total_fptrunc=32, remaining_pkrtz=0,
                        launch_config=str(config), checks=commands, rerun_o3=False,
                        gpu_precision_tested=False))
    print(f'{target}: 16 softmax fptrunc restored, 0 PKRTZ; llvm-as and verifier passed')
(HERE / 'rne_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
