#!/usr/bin/env python3
"""Generate causal reverse-group4 candidates through MyTest, with no IR patches."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path(os.environ.get('IR_VERIFY_LLVM_DIR', '/data2/xsl/install/bin'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE.parent/'causal_best_mytest/causal.input.mlir'
    shutil.copyfile(source, HERE/'causal.input.mlir')
    base = json.loads((source.parent/'commands.json').read_text())['group4']
    historical = HERE.parent/'retained32_group_prefetch/run-20261023-163255-p16epokv/grid-000001/config.json'
    safe = HERE.parent/'causal_v_double_buffer_reverse/manifest.json'
    (HERE/'historical_best.config.json').write_bytes(historical.read_bytes())
    (HERE/'historical_safe_best.config.json').write_text(json.dumps(
        json.loads(safe.read_text())['variants']['group4'], indent=2)+'\n')
    log = (ROOT/'pmc_test.log').read_text(encoding='utf-16').splitlines()
    evidence = []
    for start, end in [(1, 29), (187, 216), (484, 504), (1133, 1173)]:
        evidence.extend(f'{i+1}: {log[i]}' for i in range(start-1, end))
    (HERE/'pmc_evidence.txt').write_text('\n'.join(evidence)+'\n')
    records = {}
    for name, mode in [('group4_reverse_off', 'off'), ('group4_reverse_pk_rtz', 'rtz')]:
        cmd = base.copy()
        cmd[4] = str(HERE/'causal.input.mlir')
        cmd[6] = str(HERE/(name+'.ll'))
        cmd = ['--workgroup-reverse=1' if a == '--workgroup-reverse=0' else a for a in cmd]
        cmd += ['--simplify-bounded-affine=1', '--hoist-lds-bases=1',
                '--hoist-lane-shuffles=1', '--elide-backedge-barrier=1',
                '--lazy-softmax-base=off', '--softmax-cvt-pack='+mode]
        work = HERE/(name+'.work')
        work.mkdir(exist_ok=True)
        (HERE/(name+'.command.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            + 'cd -- '+shlex.quote(str(work))+'\n'+shlex.join(cmd)+'\n')
        with (HERE/(name+'.log')).open('w') as output:
            subprocess.run(cmd, cwd=work, stdout=output, stderr=subprocess.STDOUT, check=True)
        reports = (HERE/(name+'.log')).read_text()
        for required in ['V double buffer: 4 slices', 'K DMA: mode=64-swizzle',
                         'K LDS prefetch: distance=8', 'K tail LDS prefetch: distance=4',
                         'Cross BN K0: skipped, function=Attn_p2: requires canonical unsigned loop condition']:
            assert required in reports, (name, required)
        out = HERE/(name+'.ll')
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'llvm-as', out, '-o', '/dev/null'], check=True)
        subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', out, '-o', optimized], check=True)
        for path in (out, optimized):
            text = path.read_text()
            assert text.count('asm "v_cvt_pkrtz_f16_f32') == (32 if mode == 'rtz' else 0)
            assert '"frisk.lazy_softmax_base"' not in text
            assert '0xFFF0000000000000' in text or 'float -inf' in text
            config = dict(schema_version=1, kernel='Attn_p2', shape=[1,32,4096,128],
                          grid=[32,32,1], block=[256,1,1], shared_mem=0,
                          k_layout='bhsd', causal=True)
            path.with_suffix('.launch.json').write_text(json.dumps(config, indent=2)+'\n')
        prefixes = ('V double buffer:', 'K DMA:', 'K LDS prefetch:', 'SoftmaxFragments:',
                    'K tail LDS prefetch:', 'Cross BN K0:', 'DeferLaneSum:',
                    'Softmax cvt pack:', 'Elide', 'PV interleave:')
        records[name] = dict(command=cmd, sha256=digest(out), o3_sha256=digest(optimized),
            softmax_cvt_pack=mode, LLVM_and_O3_verified=True,
            cross_bn_k0_applied=False,
            cross_bn_k0_skip_reason='reverse produces a signed loop condition; current pass requires unsigned',
            pass_reports=[line for line in reports.splitlines() if line.startswith(prefixes)])
        print('PASS MyTest export', name, flush=True)
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=records,
        input_sha256=digest(HERE/'causal.input.mlir'), mytest_sha256=digest(Path(base[3])),
        shape=[1,32,4096,128], tiles=dict(BM=128,BN=128,BK=32), causal=True,
        group=4, reverse=True, query_axis='x', head_axis='y',
        historical_best_median_ms=1.168558955192566,
        historical_best_softmax='unsafe (explicitly labeled in pmc_test.log)',
        historical_safe_best_median_ms=1.1768704652786255,
        noncausal_reference='generated_ir/noncausal_cvt_pack_pass/pass_pk_rtz.o3.ll',
        lazy_softmax_base='off: current pass does not model causal -inf masks',
        gpu_tested=False, target_codegen_tested=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
