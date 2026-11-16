#!/usr/bin/env python3
"""Regenerate both user-selected configurations through early shared planning."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/data2/xsl/install/bin')
TOOL = ROOT/'3rd/deepgengraph/build/test/MyTest'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(kind):
    target = HERE/kind
    target.mkdir(exist_ok=True)
    if kind == 'noncausal':
        original = HERE.parent/'best_noncausal_early_wave'
        script = (original/'command.sh').read_text()
        script = script[script.index('env FRISK_'):].replace('\\\n', '')
        script = script.replace('$repo_root', str(ROOT)).replace('$here', str(original)).replace('"$@"', '')
        command = shlex.split(script)
        reference = original/'kernel.o3.ll'
    else:
        original = HERE.parent/'causal_mask_lowering'
        data = json.loads((original/'manifest.json').read_text())
        command = data['variants']['mask_finite_pk_rtz']['command'].copy()
        reference = original/'mask_finite_pk_rtz.o3.ll'
    original_command = command.copy()
    (target/'input.mlir').write_bytes(Path(command[4]).read_bytes())
    command[3] = str(TOOL)
    command[4] = str(target/'input.mlir')
    command[6] = str(target/'kernel.ll')
    # Equivalent early binding. No LLVM pass may split or swizzle these copies.
    command = [arg for arg in command if not arg.startswith(('--k-dma=', '--k-dma-arg='))]
    command.append('--shared-copy-constraints=1:8:9:3')
    work = target/'work'
    work.mkdir(exist_ok=True)
    (target/'command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- '
        +shlex.quote(str(work))+'\n'+shlex.join(command)+'\n')
    tool_sha256 = digest(TOOL)
    with (target/'lowering.log').open('w') as log:
        subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = target/'kernel.ll'
    optimized = target/'kernel.o3.ll'
    subprocess.run([LLVM/'llvm-as', raw, '-o', '/dev/null'], check=True)
    with (target/'o3.log').open('w') as log:
        subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', raw, '-o', optimized],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([LLVM/'llvm-as', optimized, '-o', '/dev/null'], check=True)
    for path in (raw, optimized):
        path.with_suffix('.launch.json').write_bytes(reference.with_suffix('.launch.json').read_bytes())
    text = optimized.read_text()
    log = (target/'lowering.log').read_text()
    assert 'narrowKDMA' not in log and 'K DMA:' not in log
    assert 'frisk.dma_destination_map' in log and 'frisk.dma_source_map' in log
    assert 'buffer_load_dwordx2' in text and 'buffer_load_dwordx4' in text
    assert 'frisk.k_dma' not in text and '__frisk_async_load_lds' not in text
    prefixes = ('V double buffer:', 'K LDS prefetch:', 'K tail LDS prefetch:', 'Cross BN K0:',
                'SoftmaxFragments:', 'Softmax cvt pack:', 'Lazy softmax base:', 'Causal mask lowering:',
                'Async copy lowering:', 'Shared memory plan:')
    reports = [line for line in log.splitlines() if line.startswith(prefixes)]
    assert any(line.startswith('SoftmaxFragments:') and '16 fragments' in line for line in reports), reports
    assert any(line.startswith('V double buffer: 4 slices') for line in reports), reports
    assert text.count('asm \"v_cvt_pkrtz_f16_f32') == 32
    metadata = dict(generated_at=datetime.datetime.now().astimezone().isoformat(),
        kind=kind, reference=str(reference), reference_sha256=digest(reference),
        original_command=original_command, command=command,
        parameter_translation='--k-dma=64-swizzle --k-dma-arg=1 -> --shared-copy-constraints=1:8:9:3',
        input_sha256=digest(target/'input.mlir'), mytest_sha256=tool_sha256,
        llvm_sha256=digest(raw), o3_sha256=digest(optimized), LLVM_and_O3_verified=True,
        dma8_instructions=text.count('buffer_load_dwordx2'), dma16_instructions=text.count('buffer_load_dwordx4'),
        packed_rtz_pairs=text.count('asm "v_cvt_pkrtz_f16_f32'),
        pass_reports=reports, gpu_tested=False, target_codegen_tested=False)
    (target/'generation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('PASS', kind, optimized, flush=True)


if __name__ == '__main__':
    for kind in sys.argv[1:] or ['noncausal', 'causal']:
        if kind not in ('noncausal', 'causal'):
            raise ValueError(kind)
        generate(kind)
