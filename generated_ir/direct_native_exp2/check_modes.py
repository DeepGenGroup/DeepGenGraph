#!/usr/bin/env python3
"""Generate comparable attention kernels for all five existing math modes."""
import json
import os
from pathlib import Path
import re
import subprocess

here = Path(__file__).resolve().parent
repo = here.parent.parent
tool = repo / '3rd/deepgengraph/build/test/MyTest'
opt = Path('/data2/xsl/install/bin/opt')
summary = {}
for mode in ('strict', 'mmac', 'mmac-recip', 'mmac-exp2', 'approx'):
    out = here / 'checks' / mode
    out.mkdir(parents=True, exist_ok=True)
    command = [str(tool), str(here / 'input.mlir'), '0', str(out / 'kernel.ll'),
               '1', '1', '3.3', '34816', mode, '256', '4,1',
               '--dump-llvm-passes=1']
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'lowering.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async',
                            'FRISK_GM_TO_SHM_ALIGNMENT': '16'}, check=True)
    log = (out / 'lowering.log').read_text()
    translated = log.split('---- after llvm translateToLLVM -----', 1)[1].split(
        '---- after llvm optimizeRegisterMath -----', 1)[0]
    lines = translated.splitlines()
    end = max(i for i, line in enumerate(lines) if line.startswith('!'))
    translated = '\n'.join(lines[:end + 1]) + '\n'
    (out / 'translated.ll').write_text(translated)
    ir = (out / 'kernel.ll').read_text()
    expect_native = mode in ('approx', 'mmac-exp2')
    for stage, text in [('translated', translated), ('exported', ir)]:
        native = len(re.findall(r'call float @llvm\.amdgcn\.exp2\.f32\(', text))
        ocml = len(re.findall(r'call float @__ocml_exp2_f32\(', text))
        assert native > 0 and ocml == 0 if expect_native else ocml > 0 and native == 0, (mode, stage, native, ocml)
        summary[f'{mode}/{stage}'] = {'native_exp2_calls': native, 'ocml_exp2_calls': ocml}
    subprocess.run([str(opt), '-disable-output', '-passes=verify', str(out / 'kernel.ll')], check=True)
    print('PASS', mode, summary[f'{mode}/translated'], flush=True)
(here / 'checks/modes.json').write_text(json.dumps(summary, indent=2) + '\n')
