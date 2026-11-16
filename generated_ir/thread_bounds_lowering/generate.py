#!/usr/bin/env python3
"""Regenerate both tuned attention presets with the current MyTest."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LLVM = Path('/data2/xsl/install/bin')
TOOL = ROOT / '3rd/deepgengraph/build/test/MyTest'
PRESETS = {
    'noncausal': ('best_noncausal_early_wave', 'command.sh', 'kernel'),
    'causal': ('causal_mask_lowering', 'mask_finite_pk_rtz.command.sh',
               'mask_finite_pk_rtz'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate(mode):
    folder, script, stem = PRESETS[mode]
    reference = HERE.parent / folder
    source_script = reference / script
    text = source_script.read_text().replace('\\\n', ' ')
    command = shlex.split(text[text.index('env FRISK_GM_TO_SHM_MODE='):])
    command = [arg.replace('$repo_root', str(ROOT)).replace('$here', str(reference))
               for arg in command if arg != '$@']
    tool_index = command.index(str(TOOL))
    source_input = Path(command[tool_index + 1])
    original_command = command.copy()
    destination = HERE / mode
    destination.mkdir(exist_ok=True)
    work = destination / 'work'
    work.mkdir(exist_ok=True)
    source = destination / 'input.mlir'
    shutil.copyfile(source_input, source)
    raw = destination / (stem + '.ll')
    optimized = destination / (stem + '.o3.ll')
    command[tool_index + 1] = str(source)
    command[tool_index + 3] = str(raw)
    # Only relocate the input/output; retain every positional/named tuning flag.
    assert command[tool_index + 4:] == original_command[tool_index + 4:]
    assert command[:tool_index + 1] == original_command[:tool_index + 1]
    assert digest(source) == digest(source_input)
    launch = reference / (stem + '.o3.launch.json')
    for path in (raw, optimized):
        shutil.copyfile(launch, path.with_suffix('.launch.json'))
    checks = [
        [str(LLVM / 'llvm-as'), str(raw), '-o', '/dev/null'],
        [str(LLVM / 'opt'), '-S', '-passes=default<O3>,verify',
         str(raw), '-o', str(optimized)],
        [str(LLVM / 'llvm-as'), str(optimized), '-o', '/dev/null'],
    ]
    replay = ('#!/usr/bin/env bash\nset -euo pipefail\n'
              + 'mkdir -p -- ' + shlex.quote(str(work)) + '\n'
              + 'cd -- ' + shlex.quote(str(work)) + '\n'
              + shlex.join(command) + ' > ' + shlex.quote(str(destination / 'lowering.log'))
              + ' 2>&1\n' + '\n'.join(shlex.join(c) for c in checks) + '\n')
    (destination / 'command.sh').write_text(replay)
    with (destination / 'lowering.log').open('w') as log:
        subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (destination / 'verification.log').open('w') as log:
        for check in checks:
            subprocess.run(check, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (destination / 'lowering.log').read_text()
    reports = [line for line in log.splitlines() if line.startswith((
        'Compact thread indices:', 'Address shared bases:', 'OptimizeRegisterLifetimes:',
        'HoistWaveInvariants:', 'Causal mask lowering:', 'LocalReduction:',
        'ThreadDeferLaneSum:', 'V double buffer:', 'K DMA:', 'K LDS prefetch:',
        'K tail LDS prefetch:', 'Cross BN K0:', 'SoftmaxFragments:',
        'Lazy softmax base:', 'Softmax cvt pack:'))]
    o3 = optimized.read_text()
    identities = re.findall(r'^.*%address.thread[\w.]* = .*$', o3, re.M)
    assert identities, 'Expected local thread materialization'
    for identity in identities:
        slot = re.search(r'!range !(\d+)', identity)
        assert slot and f'!{slot[1]} = !{{i32 0, i32 256}}' in o3, identity
    packed_pairs = o3.count('asm "v_cvt_pkrtz_f16_f32')
    assert packed_pairs == 32, (mode, packed_pairs, reports)
    metadata = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'mode': mode,
        'reference': str(reference / (stem + '.o3.ll')),
        'reference_sha256': digest(reference / (stem + '.o3.ll')),
        'source_command': str(source_script),
        'source_command_sha256': digest(source_script),
        'source_input': str(source_input), 'input_sha256': digest(source),
        'tuning_parameters_unchanged': True,
        'command': command, 'working_directory': str(work),
        'compiler': str(TOOL), 'compiler_sha256': digest(TOOL),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'verification_commands': checks, 'LLVM_and_O3_verified': True,
        'raw_sha256': digest(raw), 'o3_sha256': digest(optimized),
        'thread_identity_range': [0, 256], 'thread_identities': len(identities),
        'packed_rtz_pairs': packed_pairs, 'pass_reports': reports,
        'gpu_tested': False,
    }
    (destination / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Verified:', optimized, flush=True)
    print('\n'.join(reports), flush=True)


if __name__ == '__main__':
    for mode in sys.argv[1:] or PRESETS:
        generate(mode)
