#!/usr/bin/env python3
"""Remove only the math.result/math.packed identity asms from two saved IRs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LLVM = Path('/data2/xsl/install/bin')
SOURCES = [
    ROOT / 'best_noncausal_early_wave/kernel.o3.ll',
    ROOT / 'causal_mask_lowering/mask_finite_pk_rtz.o3.ll',
]
BOUNDARY = re.compile(
    r'^  (%math\.(result|packed)\d*) = (?:tail )?call '
    r'(<4 x (?:float|half)>) asm sideeffect "", "=v,0"'
    r'\(\3 (%[-a-zA-Z$._0-9]+)\).*$'
)
SSA = re.compile(r'%[-a-zA-Z$._0-9]+')


def digest(data):
    return hashlib.sha256(data).hexdigest()


records = []
for source in SOURCES:
    original = source.read_bytes()
    text = original.decode()
    replacements, counts, retained = {}, {'result': 0, 'packed': 0}, []
    for line in text.splitlines(keepends=True):
        match = BOUNDARY.fullmatch(line.rstrip('\n'))
        if match:
            name, kind, _, operand = match.groups()
            replacements[name] = operand
            counts[kind] += 1
        else:
            retained.append(line)
    assert counts == {'result': 16, 'packed': 16}, (source, counts)
    assert not (set(replacements) & set(replacements.values()))
    output = SSA.sub(lambda m: replacements.get(m[0], m[0]), ''.join(retained))
    assert not re.search(r'%math\.(?:result|packed)\d*\b', output)
    # Check that real conversion instructions and all other empty asms survive.
    for marker in (' = fptrunc ', 'v_cvt_pkrtz_f16_f32'):
        assert text.count(marker) == output.count(marker)
    other_asms = lambda s: [line for line in s.splitlines()
                           if 'asm sideeffect ""' in line
                           and not BOUNDARY.fullmatch(line)]
    assert other_asms(text) == other_asms(output)
    target = source.with_name(source.name.replace('.o3.ll', '.no_math_materialize.o3.ll'))
    target.write_text(output)
    commands = [
        [str(LLVM / 'llvm-as'), str(target), '-o', '/dev/null'],
        [str(LLVM / 'opt'), '-passes=verify', '-disable-output', str(target)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    config = source.with_suffix('.launch.json')
    target_config = target.with_suffix('.launch.json')
    target_config.write_bytes(config.read_bytes())
    assert source.read_bytes() == original
    records.append(dict(source=str(source), output=str(target),
                        source_sha256=digest(original),
                        output_sha256=digest(output.encode()), removed=counts,
                        launch_config=str(target_config), checks=commands,
                        rerun_o3=False, gpu_precision_tested=False))
    print(f'{target}: removed {counts}; llvm-as and verifier passed')
(HERE / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
