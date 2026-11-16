#!/usr/bin/env python3
"""Regenerate real attention pressure logs with unchanged existing tuning options."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    'noncausal': ROOT / 'generated_ir/fragment_level_optimized_noncausal/command.json',
    'causal': ROOT / 'generated_ir/thread_mask_add_bypass/manifest.json',
}


def extract_module(log, marker):
    tail = log.split(marker, 1)[1].split('\n', 1)[1]
    start = re.search(r'^(?:#\w+\s*=|module\b)', tail, re.M).start()
    module = re.search(r'^module\b', tail[start:], re.M).start() + start
    end = re.search(r'^}\s*$', tail[module:], re.M).end() + module
    return tail[start:end].rstrip() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=SOURCES, action='append')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    records = []
    for case in args.case or SOURCES:
        source = SOURCES[case]
        original = json.loads(source.read_text())
        command = list(original.get('argv', original.get('command')))
        tool = next(i for i, part in enumerate(command) if Path(part).name == 'MyTest')
        output = HERE / f'{case}.ll'
        command[tool + 3] = str(output)
        command.append('--analyze-register-pressure=1')
        (HERE / f'{case}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (HERE / f'{case}.lowering.log').open('w') as log:
            p = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        assert p.returncode == 0, f'{case}: see lowering.log'
        text = (HERE / f'{case}.lowering.log').read_text()
        reports = re.findall(r'^RegisterPressure:.*?^EndRegisterPressure\n', text, re.M | re.S)
        assert reports, case
        (HERE / f'{case}.pressure.log').write_text(''.join(reports))
        record = {'case': case, 'source_command': str(source.relative_to(ROOT)),
                  'input': command[tool + 1],
                  'compiler_sha256': hashlib.sha256(Path(command[tool]).read_bytes()).hexdigest(),
                  'input_sha256': hashlib.sha256(Path(command[tool + 1]).read_bytes()).hexdigest(),
                  'reports': []}
        for report in reports:
            header = re.match(r'RegisterPressure: @(\S+) stage=(\S+)', report)
            fields = re.search(r'^  summary:.*$', report, re.M)[0]
            entry = dict(function=header[1], stage=header[2],
                         **{k: int(v) for k, v in re.findall(r'([a-z-]+)=(\d+)', fields)})
            record['reports'].append(entry)
            print(case, entry, flush=True)
        for stage, marker in [('fragment', 'after fragment-level optimization'),
                              ('thread', 'after frisk-hoist-wave-invariants')]:
            ir = extract_module(text, marker)
            path = HERE / f'{case}.{stage}.mlir'
            path.write_text(ir)
            # Re-analyze a serialized snapshot: its locations now identify exact
            # source lines, making hotspot/contributor diagnostics navigable.
            with (HERE / f'{case}.{stage}.pressure.log').open('w') as log:
                p = subprocess.run([str(ROOT / '3rd/deepgengraph/build/test/ThreadTilingTest'),
                    '--mlir-disable-threading', str(path),
                    f'--frisk-analyze-register-pressure=stage={stage} top=8', '-o', '/dev/null'],
                    stderr=log, stdout=subprocess.DEVNULL, timeout=180)
            assert p.returncode == 0, f'{case} {stage} replay failed'
            replay = (HERE / f'{case}.{stage}.pressure.log').read_text()
            expected = [re.search(r'^  summary:.*$', r, re.M)[0] for r in reports
                        if f'stage={stage}\n' in r]
            actual = re.findall(r'^  summary:.*$', replay, re.M)
            assert sorted(expected) == sorted(actual), (case, stage, expected, actual)
        baseline = HERE / f'{case}.baseline.ll'
        if args.baseline:
            command.pop()
            command[tool + 3] = str(baseline)
            with (HERE / f'{case}.baseline.log').open('w') as log:
                p = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            assert p.returncode == 0, f'{case} baseline failed'
        if baseline.exists():
            normalize = lambda path: re.sub(r'^; ModuleID =.*\n', '', path.read_text(), flags=re.M)
            assert normalize(output) == normalize(baseline), f'{case}: analysis changed exported LLVM'
            record['analysis_on_off_llvm_identical'] = True
        record['llvm_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        records.append(record)
    (HERE / 'validation.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
