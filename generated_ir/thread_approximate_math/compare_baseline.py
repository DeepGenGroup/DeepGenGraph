#!/usr/bin/env python3
"""Compare dataflow after aligning only the two reciprocal instruction positions.

The delivered kernels are not modified. LLVM-diff is sensitive to the earlier
reciprocal placement; comparison copies place each reciprocal at its first use.
This check is specific to the two same-block reciprocals in this attention IR.
"""
import json
from pathlib import Path
import re
import subprocess

here = Path(__file__).resolve().parent
llvm = Path('/data2/xsl/install/bin')
records = {}
for suffix in ('ll', 'o3.ll'):
    outputs = []
    for name, source in [('before', here / 'baseline' / f'kernel.{suffix}'),
                         ('after', here / f'kernel.{suffix}')]:
        text = subprocess.check_output([llvm / 'opt', '-S', '-passes=instnamer', source], text=True)
        lines = text.splitlines()
        reciprocals = [line for line in lines if '= fdiv arcp ' in line]
        assert len(reciprocals) == 2
        for line in reciprocals:
            value = line.strip().split(' = ', 1)[0]
            original = lines.index(line)
            lines.remove(line)
            first_use = next(i for i, text in enumerate(lines)
                             if re.search(re.escape(value) + r'(?![\w.])', text))
            assert first_use >= original
            assert not any(re.match(r'^[-\w.$]+:', text) or text == '}'
                           for text in lines[original:first_use]), 'cross-block movement'
            lines.insert(first_use, line)
        output = here / 'baseline' / f'{name}.sunk.{suffix}'
        output.write_text('\n'.join(lines) + '\n')
        subprocess.run([llvm / 'opt', '-disable-output', '-passes=verify', output], check=True)
        outputs.append(output)
    diff = subprocess.run([llvm / 'llvm-diff', *outputs], capture_output=True, text=True)
    (here / 'baseline' / f'{suffix}.sunk-diff.log').write_text(diff.stdout + diff.stderr)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    records[suffix] = dict(equal_after_sinking_reciprocals_to_first_use=True)
    print('PASS', suffix, 'llvm-diff after aligning reciprocal positions')
(here / 'baseline/comparison.json').write_text(json.dumps(records, indent=2) + '\n')
