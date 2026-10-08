#!/usr/bin/env python3
"""Isolate MMAC padding cost in exported attention IR on a DTK/gfx936 host.

The default only creates diagnostic LLVM files. --run compiles and checks each
variant with the existing attention launcher, retaining all logs and metadata.
It never changes compiler defaults or treats a failed correctness run as a win.
A passing run is evidence for this input, not a hardware latency specification.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MMA = 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'
SEP = r'\0A\09'
PAD = 's_nop 7' + SEP
ORIGINAL = PAD * 8 + MMA + SEP + PAD * 8
ASM = re.compile(r'(asm sideeffect ")([^"]*)(", "=v,v,v,0")')


def variant(source, padding):
    if not 0 <= padding <= 8:
        raise ValueError('padding must be between 0 and 8')
    count = 0

    def replace(match):
        nonlocal count
        if match[2] != ORIGINAL:
            return match[0]
        count += 1
        return match[1] + PAD * padding + MMA + SEP + PAD * padding + match[3]

    result = ASM.sub(replace, source)
    if not count or count != source.count('v_mmac_f32_16x16x16_f16'):
        raise ValueError('expected only the exact eight-nop register MMAC export')
    # No operands, constraints, attributes, memory or control flow may change.
    assert result.replace(PAD, '') == source.replace(PAD, '')
    return result, count


def command(args, cwd, log, timeout):
    with log.open('w') as stream:
        stream.write('COMMAND: ' + json.dumps(list(map(str, args))) + '\n')
        stream.flush()
        try:
            return subprocess.run(list(map(str, args)), cwd=cwd, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(str(error) + '\n')
            return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path,
                        help='new directory; existing results are never overwritten')
    parser.add_argument('--padding', type=int, nargs='+', default=[8, 4, 2, 1],
                        help='nop count per side; 8 is always run first as control')
    parser.add_argument('--run', action='store_true', help='compile/check on DTK GPU host')
    parser.add_argument('--bracket-control', action='store_true',
                        help='check/time pad-8 immediately before and after each shorter-padding trial')
    parser.add_argument('--grid', default='32,64,1')
    parser.add_argument('--block', default='128,1,1')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=50)
    args = parser.parse_args()
    if any(p < 0 or p > 8 for p in args.padding):
        parser.error('--padding must be between 0 and 8')
    if args.warmup < 0 or args.repeat <= 0:
        parser.error('--warmup must be nonnegative and --repeat must be positive')
    source = args.input.resolve().read_text()
    variant(source, 8)  # Validate before creating any artifacts.
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paddings = list(dict.fromkeys([8] + args.padding))
    report = {'input': str(args.input.resolve()),
              'sha256': hashlib.sha256(source.encode()).hexdigest(),
              'note': 'Diagnostic only; do not deploy from one passing input.',
              'results': []}
    summary = out/'summary.json'

    def save():
        summary.write_text(json.dumps(report, indent=2) + '\n')

    def benchmark(hsaco, work, filename):
        code = command([sys.executable, ROOT/'kernelLauncher.py', '--hsaco', hsaco,
                        '--kernel', 'Attn_p2', '--grid', args.grid, '--block', args.block,
                        '--device', args.device, '--warmup', args.warmup,
                        '--repeat', args.repeat], work, work/filename, 600)
        log = (work/filename).read_text()
        passed = not code and 'torch.allclose(atol=1e-2, rtol=1e-2): True' in log
        result = {'status': 'passed' if passed else 'check_failed'}
        if passed:
            for field in ('time_our_mid', 'time_base_mid'):
                match = re.search(field + r'=([0-9.eE+-]+)', log)
                if match:
                    result[field + '_ms'] = float(match[1])
        return result

    if args.run:
        for tool in ('opt', 'llc', 'llvm-link', 'llvm-readobj', 'llvm-objdump'):
            path = shutil.which(tool)
            report.setdefault('tools', {})[tool] = path
            if path:
                command([path, '--version'], out, out/(tool + '-version.txt'), 30)
    for padding in paddings:
        work = out/f'pad-{padding}'
        work.mkdir()
        text, count = variant(source, padding)
        ll = work/'kernel.ll'
        ll.write_text(text)
        entry = {'padding_per_side': padding, 'mma_count': count,
                 'nop_count': 2 * padding * count, 'status': 'generated'}
        report['results'].append(entry)
        save()
        if not args.run:
            print(f'Generated {ll} ({entry["nop_count"]} nops)', flush=True)
            continue
        hsaco = work/'kernel.hsaco'
        print(f'Compile padding={padding} at O3', flush=True)
        status = command(['bash', ROOT/'GenHsaco.sh', ll, '3', hsaco], work,
                         work/'compile.log', 600)
        if status:
            entry['status'] = 'compile_failed'
            save()
            if padding == 8:
                raise SystemExit(f'Control failed; see {work / "compile.log"}')
            continue
        # GenHsaco.sh also leaves this variant's merged.bc, opt.bc and kernel.o
        # in work. Separate working directories prevent cross-variant artifacts.
        for tool, flags, filename in (
            ('llvm-readobj', ['--notes'], 'metadata.txt'),
            ('llvm-objdump', ['-d', '--mcpu=gfx936'], 'isa.txt'),
        ):
            code = command([tool, *flags, hsaco], work, work/filename, 120)
            entry[filename + '_exit_code'] = code
        metadata = (work/'metadata.txt').read_text()
        for field in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count',
                      'sgpr_spill_count', 'private_segment_fixed_size',
                      'group_segment_fixed_size'):
            match = re.search(r'\.' + field + r':\s*(\d+)', metadata)
            if match:
                entry[field] = int(match[1])
        control_hsaco = out/'pad-8/kernel.hsaco'
        if args.bracket_control and padding != 8:
            print(f'Check pad-8 control before padding={padding}', flush=True)
            entry['control_before'] = benchmark(control_hsaco, work, 'control-before.log')
            if entry['control_before']['status'] != 'passed':
                entry['status'] = 'control_failed'
                save()
                raise SystemExit(f'Control failed; see {work / "control-before.log"}')
        print(f'Check and time padding={padding}', flush=True)
        entry.update(benchmark(hsaco, work, 'benchmark.log'))
        if args.bracket_control and padding != 8:
            print(f'Check pad-8 control after padding={padding}', flush=True)
            entry['control_after'] = benchmark(control_hsaco, work, 'control-after.log')
            if entry['control_after']['status'] != 'passed':
                entry['status'] = 'control_failed'
                save()
                raise SystemExit(f'Control failed; see {work / "control-after.log"}')
            before = entry['control_before'].get('time_our_mid_ms', 0)
            after = entry['control_after'].get('time_our_mid_ms', 0)
            trial = entry.get('time_our_mid_ms', 0)
            if before > 0 and after > 0:
                entry['control_drift_percent'] = 100 * (after / before - 1)
                if entry['status'] == 'passed' and trial > 0:
                    entry['speedup_vs_bracket_mean'] = (before + after) / (2 * trial)
        save()
        print(json.dumps(entry), flush=True)
        if padding == 8 and entry['status'] != 'passed':
            raise SystemExit(f'Control failed; see {work / "benchmark.log"}')
    print(f'Results: {summary}')


if __name__ == '__main__':
    main()
