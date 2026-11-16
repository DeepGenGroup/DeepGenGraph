#!/usr/bin/env python3
"""Regenerate current-compiler IR using the best_noncausal_early_wave settings."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REFERENCE = HERE.parent / 'best_noncausal_early_wave'
LLVM = Path('/data2/xsl/install/bin')
TOOL = REPO / '3rd/deepgengraph/build/test/MyTest'


def sha256(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    for name in ('input.mlir', 'command.sh', 'kernel.launch.json', 'kernel.o3.launch.json'):
        assert (HERE / name).read_bytes() == (REFERENCE / name).read_bytes(), name
    command = ['bash', str(HERE / 'command.sh')]
    with (HERE / 'lowering.log').open('w') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify = [str(LLVM / 'llvm-as'), str(HERE / 'kernel.ll'), '-o', '/dev/null']
    subprocess.run(verify, check=True)
    optimize = [str(LLVM / 'opt'), '-S', '-passes=default<O3>,verify',
                str(HERE / 'kernel.ll'), '-o', str(HERE / 'kernel.o3.ll')]
    with (HERE / 'o3.log').open('w') as log:
        subprocess.run(optimize, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(LLVM / 'llvm-as'), str(HERE / 'kernel.o3.ll'), '-o', '/dev/null'], check=True)
    text = (HERE / 'lowering.log').read_text()
    assert text.index('after frisk-thread-local-reduction') < text.index('after llvm translateToLLVM')
    assert 'after llvm optimizeLocalReductions' not in text
    reports = re.findall(r'^LocalReduction: .*$', text, re.M)
    assert any('4 groups, 128 inputs, 4 chains (approx)' in report for report in reports), reports
    stages = [{'line': i, 'stage': line.strip()} for i, line in enumerate(text.splitlines(), 1)
              if re.match(r'^-+ after |^-+ secondLowering done', line)]
    (HERE / 'stages.txt').write_text(''.join(f"{s['line']}: {s['stage']}\n" for s in stages))
    metadata = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'reference': str(REFERENCE / 'kernel.o3.ll'),
        'reference_sha256': sha256(REFERENCE / 'kernel.o3.ll'),
        'reference_parameters_and_input_identical': True,
        'input': str(HERE / 'input.mlir'), 'input_sha256': sha256(HERE / 'input.mlir'),
        'tool': str(TOOL), 'tool_sha256': sha256(TOOL),
        'command': command, 'command_script_sha256': sha256(HERE / 'command.sh'),
        'mytest_exit_code': 0,
        'kernel_sha256': sha256(HERE / 'kernel.ll'),
        'o3_sha256': sha256(HERE / 'kernel.o3.ll'),
        'verification': {'llvm_as_command': verify, 'llvm_as_exit_code': 0,
                         'o3_command': optimize, 'o3_exit_code': 0,
                         'thread_reduction_stage_order_checked': True},
        'local_reduction_reports': reports, 'gpu_tested': False, 'stages': stages,
        'note': 'Current compiler, including thread-level multichain reduction; '
                'all tuning arguments and input copied from the reference. '
                'This is not an isolated comparison against the historical compiler.',
    }
    (HERE / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('\n'.join(reports))
    print('Verified:', HERE / 'kernel.o3.ll')


if __name__ == '__main__':
    main()
