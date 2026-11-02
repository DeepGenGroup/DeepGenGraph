#!/usr/bin/env python3
"""Generate forward group4/group8 causal LLVM through the complete MyTest path."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/data2/xsl/install/bin')


def main():
    spec = importlib.util.spec_from_file_location('attention_input', ROOT/'generate_attention_ir.py')
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    source, shape = frontend.generate_ir(frontend.TEMPLATE.read_text(), 128, 128, 32, softmax='online')
    # Match the best input's explicit specialization of QK's four D slices.
    end = '            scf.yield %nextQK : tensor<128x128xf32>\n          }'
    assert source.count(end) == 1
    source = source.replace(end, end+' {frisk.simplify_loop}')
    (HERE/'causal.input.mlir').write_text(source)
    base = json.loads((HERE.parent/'noncausal_cross_bn_k0_pass/mytest_smoke/commands.json').read_text())['cross_bn_k0']
    commands = {}
    for group in (4, 8):
        name = f'group{group}'
        cmd = base.copy()
        cmd[4] = str(HERE/'causal.input.mlir')
        cmd[6] = str(HERE/(name+'.ll'))
        # Causal uniform-mask fast paths introduce additional compute blocks;
        # use elementwise selects so the DMA/scheduling passes can match.
        cmd[15] = '0'
        cmd = [f'--workgroup-group-size={group}' if a.startswith('--workgroup-group-size=') else a for a in cmd]
        assert '--workgroup-reverse=0' in cmd
        commands[name] = cmd
    (HERE/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    records = {}
    for name, cmd in commands.items():
        work = HERE/(name+'.work'); work.mkdir(exist_ok=True)
        with (HERE/(name+'.log')).open('w') as log:
            subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (HERE/(name+'.log')).read_text()
        reports = [line for line in log.splitlines() if line.startswith((
            'V double buffer:', 'K DMA:', 'K LDS prefetch:', 'SoftmaxFragments:',
            'K tail LDS prefetch:', 'Cross BN K0:', 'DeferLaneSum:'))]
        for required in ('V double buffer: 4 slices', 'K DMA: mode=64-swizzle',
                         'K LDS prefetch: distance=8', 'K tail LDS prefetch: distance=4',
                         'overlap-mmac=32, sum-anchors=64'):
            assert required in log, (name, required, reports)
        path = HERE/(name+'.ll')
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', path, '-o', HERE/(name+'.o3.ll')], check=True)
        config = dict(schema_version=1, kernel='Attn_p2', shape=list(shape), grid=[32,32,1],
                      block=[256,1,1], shared_mem=0, k_layout='bhsd', causal=True)
        (HERE/(name+'.launch.json')).write_text(json.dumps(config, indent=2)+'\n')
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            group=int(name[5:]), reverse=False, llvm_verified=True, O3_verified=True,
                            o3_sha256=hashlib.sha256((HERE/(name+'.o3.ll')).read_bytes()).hexdigest(),
                            pass_reports=reports)
        print('Generated', path, flush=True)
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=records, commands=commands), indent=2)+'\n')


if __name__ == '__main__':
    main()
