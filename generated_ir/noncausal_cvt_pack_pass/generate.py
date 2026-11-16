#!/usr/bin/env python3
"""Export the compiled pass and compare O3 IR against the GPU-measured winner."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = HERE.parent/'noncausal_cvt_pack'
TOOL = ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def run(command):
    return subprocess.run(list(map(str, command)), text=True, capture_output=True, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(text):
    # These are the ONLY tolerated differences against the measured winner.
    return re.sub(r'^; ModuleID = .*\n', '', text).replace(' "frisk.softmax_cvt_pack"="rtz"', '')


def main():
    experiment = json.loads((EXPERIMENT/'manifest.json').read_text())
    for name in ('control', 'lazy_t8', 'lazy_t8_pk_rtz'):
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            assert digest(EXPERIMENT/(name+suffix+'.ll')) == experiment['variants'][name][key]
    records = {}
    variants = {
        'pass_off': ('lazy_t8', ['--softmax-cvt-pack=off']),
        'pass_pk_rtz': ('lazy_t8', ['--softmax-cvt-pack=rtz']),
        'pipeline_t8_pk_rtz': ('control', ['--lazy-softmax-base=8', '--softmax-cvt-pack=rtz']),
    }
    for name, (source_name, options) in variants.items():
        source = EXPERIMENT/(source_name+'.ll')
        command = [TOOL, *options, source]
        result = run(command)
        raw = HERE/(name+'.raw.ll')
        raw.write_text(result.stdout)
        (HERE/(name+'.pass.log')).write_text(result.stderr)
        out = HERE/(name+'.ll')
        out.write_text(run([TOOL, '--llvm22-text', raw]).stdout)
        config = (EXPERIMENT/(source_name+'.launch.json')).read_bytes()
        out.with_suffix('.launch.json').write_bytes(config)
        run([LLVM/'llvm-as', out, '-o', '/dev/null'])
        opt = HERE/(name+'.o3.ll')
        run([LLVM/'opt', '-S', '-passes=default<O3>,verify', out, '-o', opt])
        opt.with_suffix('.launch.json').write_bytes(config)
        pairs = opt.read_text().count('asm "v_cvt_pkrtz_f16_f32')
        assert pairs == (0 if name == 'pass_off' else 32), (name, result.stderr)
        reference = EXPERIMENT/('lazy_t8.o3.ll' if name == 'pass_off' else 'lazy_t8_pk_rtz.o3.ll')
        same = canonical(opt.read_text()) == canonical(reference.read_text())
        if name != 'pipeline_t8_pk_rtz':
            assert same, ('unexpected difference from measured IR', name)
        records[name] = dict(command=list(map(str, command)), source_sha256=digest(source),
            sha256=digest(out), o3_sha256=digest(opt), packed_pairs=pairs,
            diagnostics=result.stderr.strip(), LLVM_and_O3_verified=True,
            O3_equal_to_reference_except_module_id_and_pass_marker=same,
            reference=str(reference.relative_to(ROOT)))
        print('PASS export', name, 'packed pairs', pairs, 'O3 matches reference', same)
    # A second application must not add conversions or perturb the output.
    out = HERE/'pass_pk_rtz.ll'
    twice = run([TOOL, '--softmax-cvt-pack=rtz', out])
    once = run([TOOL, '--softmax-cvt-pack=off', out])
    assert canonical(twice.stdout) == canonical(once.stdout)
    assert not twice.stderr
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=records, pass_binary_sha256=digest(TOOL),
        measured_winner=str((EXPERIMENT/'lazy_t8_pk_rtz.o3.ll').relative_to(ROOT)),
        measured_median_ms=1.5057590007781982, measured_vgpr=228, measured_scratch_bytes=0,
        measurement_source='pmc_test.log', pass_artifacts_gpu_tested=False,
        idempotence_checked=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
