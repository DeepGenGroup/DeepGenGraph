#!/usr/bin/env python3
"""Generate real C++ pass artifacts and compare with the measured experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'noncausal_qk_tail_prefetch'
SOURCE = HERE.parent / 'noncausal_softmax_fragments/softmax_pv.ll'
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')
spec = importlib.util.spec_from_file_location('tail_structure', PREVIOUS / 'generate.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def run(cmd, **kwargs):
    return subprocess.run(list(map(str, cmd)), text=True, capture_output=True, check=True, **kwargs)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def instructions(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')]


def main():
    assert digest(SOURCE) == previous.EXPECTED_SHA
    variants = {}
    for name, d in [('control', 0)] + [(f'k_tail_prefetch{d}', d) for d in (2, 4, 8)]:
        p = HERE / (name + '.ll')
        cmd = [TOOL, f'--k-tail-prefetch={d}', SOURCE]
        result = run(cmd)
        p.with_suffix('.pass.log').write_text(result.stderr)
        raw = p.with_suffix('.raw.ll')
        raw.write_text(result.stdout)
        p.write_text(run([TOOL, '--llvm22-text', raw]).stdout)
        if d == 0:
            assert instructions(p.read_text()) == instructions(SOURCE.read_text())
            p.write_bytes(SOURCE.read_bytes())
        else:
            assert 'regions=1, pairs=8, moved=7' in result.stderr
            assert f'"frisk.k_tail_prefetch"="{d}"' in p.read_text()
            assert instructions(p.read_text()) == instructions((PREVIOUS / p.name).read_text())
        p.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        run([LLVM / 'llvm-as', p, '-o', '/dev/null'])
        run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', p, '-o', p.with_suffix('.o3.ll')])
        schedule = previous.inspect(p.read_text())
        assert previous.inspect(p.with_suffix('.o3.ll').read_text()) == schedule
        variants[name] = dict(distance=d, command=list(map(str, cmd)), sha256=digest(p),
            llvm_and_O3_verified=True, identical_instructions_to_measured_experiment=True,
            measured_experiment=str(PREVIOUS / p.name), measured_sha256=digest(PREVIOUS / p.name),
            diagnostics=result.stderr.strip(), schedule=schedule)
        print('PASS', name, result.stderr.strip(), flush=True)
    (HERE / 'manifest.json').write_text(json.dumps(dict(source=str(SOURCE), source_sha256=digest(SOURCE),
        exporter_sha256=digest(TOOL), variants=variants, gpu_tested=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
