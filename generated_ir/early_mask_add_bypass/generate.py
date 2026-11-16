#!/usr/bin/env python3
"""Regenerate the exact pmc mask_finite_pk_rtz preset with early mask bypass."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'causal_mask_lowering'
LLVM = Path('/data2/xsl/install/bin')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path):
    return re.sub(r'^; ModuleID = .*\n', '', path.read_text())


def main():
    command = shlex.split((ORIGINAL / 'mask_finite_pk_rtz.command.sh').read_text().splitlines()[-1])
    name = 'mask_finite_pk_rtz'
    command[6] = str(HERE / (name + '.ll'))
    # Preserve every positional/named tuning parameter, including uniform=0
    # and fuse-mask-add=0. finite mode already fuses the eligible mask/add.
    work = HERE / (name + '.work')
    work.mkdir(exist_ok=True)
    (HERE / (name + '.command.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        + 'cd -- ' + shlex.quote(str(work)) + '\n' + shlex.join(command) + '\n')
    with (HERE / (name + '.log')).open('w') as log:
        subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw, opt = HERE / (name + '.ll'), HERE / (name + '.o3.ll')
    subprocess.run([LLVM/'llvm-as', raw, '-o', '/dev/null'], check=True)
    subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', raw, '-o', opt], check=True)
    for path in (raw, opt):
        path.with_suffix('.launch.json').write_bytes((ORIGINAL / (name + '.o3.launch.json')).read_bytes())
    log = (HERE / (name + '.log')).read_text()
    reports = [line for line in log.splitlines() if line.startswith((
        'Mask add bypass lowering:', 'MaskAddBypass:', 'Causal mask lowering:',
        'K DMA:', 'K LDS prefetch:', 'K tail LDS prefetch:', 'Cross BN K0:',
        'Softmax cvt pack:', 'V double buffer:'))]
    assert any('groups=0, elements=0, reason=no-uniform-branch' in line for line in reports), reports
    assert any('0 groups, 0 additions bypassed' in line for line in reports), reports
    assert 'frisk.mask_add_bypass' not in log, 'preset must not introduce a uniform bypass branch'
    text = opt.read_text()
    assert text.count('asm "v_cvt_pkrtz_f16_f32') == 32
    assert text.count('fadd <4 x float>') == 0
    comparisons = {}
    for suffix in ('.ll', '.o3.ll'):
        candidate = HERE / (name + suffix)
        baseline = HERE / ('baseline' + suffix)
        tested = ORIGINAL / (name + suffix)
        comparisons[suffix] = dict(
            equals_pre_change_except_module_id=canonical(candidate) == canonical(baseline),
            equals_pmc_artifact_except_module_id=canonical(candidate) == canonical(tested),
            pre_change_sha256=digest(baseline), pmc_artifact_sha256=digest(tested))
        assert comparisons[suffix]['equals_pre_change_except_module_id'], comparisons
    manifest = dict(command=command, source_command=str(ORIGINAL/'mask_finite_pk_rtz.command.sh'),
        input_sha256=digest(Path(command[4])), compiler_sha256=digest(Path(command[3])),
        raw_sha256=digest(raw), o3_sha256=digest(opt), comparisons=comparisons,
        LLVM_and_O3_verified=True, tuning_parameters_unchanged=True,
        score_plus_mask_vector_adds=0, packed_pairs=32, pass_reports=reports,
        gpu_tested=False, target_codegen_tested=False)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS exact pmc preset: LLVM/O3 verified, identical to pre-change IR except ModuleID')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
