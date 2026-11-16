#!/usr/bin/env python3
"""Reproduce the two requested tuning configurations with current MyTest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/data2/xsl/install/bin')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(kind):
    if kind == 'noncausal':
        script = (HERE.parent/'best_noncausal_early_wave/command.sh').read_text()
        script = script[script.index('env FRISK_'):].replace('\\\n', ' ')
        script = script.replace('$repo_root', str(ROOT)).replace('$here', str(HERE.parent/'best_noncausal_early_wave'))
        argv = shlex.split(script)
        argv = [a for a in argv if a not in ('$@', '--dump-llvm-passes=1')]
    else:
        script = (HERE.parent/'causal_mask_lowering/mask_finite_pk_rtz.command.sh').read_text()
        argv = shlex.split(script[script.index('env FRISK_'):])
    return argv

def generate(kind, mode):
    name = kind + ('' if mode == 'thread' else '.' + mode)
    argv = command(kind)
    index = next(i for i,a in enumerate(argv) if a.endswith('/MyTest'))
    source = Path(argv[index+1])
    snapshot = HERE/(kind+'.input.mlir')
    snapshot.write_bytes(source.read_bytes())
    argv[index+1] = str(snapshot)
    argv[index+3] = str(HERE/(name+'.ll'))
    if mode == 'legacy': argv[index] = '/tmp/MyTest-before-thread-vdb'
    if mode == 'off': argv = [a.replace('--v-double-buffer=1', '--v-double-buffer=0') for a in argv]
    work = HERE/(name+'.work');work.mkdir(exist_ok=True)
    (HERE/(name+'.command.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- '+shlex.quote(str(work))+'\n'+shlex.join(argv)+'\n')
    with (HERE/(name+'.log')).open('w') as log:
        subprocess.run(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = HERE/(name+'.ll'); opt = HERE/(name+'.o3.ll')
    subprocess.run([LLVM/'llvm-as',raw,'-o','/dev/null'],check=True)
    subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',raw,'-o',opt],check=True)
    launch_source = HERE.parent/('causal_mask_lowering/mask_finite_pk_rtz.o3.launch.json' if kind == 'causal' else 'best_noncausal_early_wave/kernel.o3.launch.json')
    launch = raw.with_suffix('.launch.json')
    launch.write_bytes(launch_source.read_bytes())
    opt.with_suffix('.launch.json').write_bytes(launch_source.read_bytes())
    reports=[line for line in (HERE/(name+'.log')).read_text().splitlines() if line.startswith(('V double buffer:', 'PV interleave:', 'K DMA:', 'K LDS prefetch:', 'K tail LDS prefetch:', 'Cross BN K0:', 'SoftmaxFragments:', 'Lazy softmax base:', 'Softmax cvt pack:'))]
    if mode != 'off': assert any(re.match(r'V double buffer: 4 slices, slot-bytes=8192, displacement=8192, removed-barriers=3, LDS=16384,',line) for line in reports), reports
    reference = HERE.parent/('causal_mask_lowering/mask_finite_pk_rtz.o3.ll' if kind == 'causal' else 'best_noncausal_early_wave/kernel.o3.ll')
    record=dict(reference_ir=str(reference),reference_sha256=digest(reference),command=argv,source=str(source),input_sha256=digest(snapshot),mytest_sha256=digest(Path(argv[index])),llvm_sha256=digest(raw),o3_sha256=digest(opt),reports=reports,LLVM_and_O3_verified=True,gpu_tested=False)
    if mode == 'thread':
        log_text=(HERE/(name+'.log')).read_text()
        assert 'after llvm doubleBufferV' not in log_text
        for suffix,marker in [('before','after lower-frisk-fragments'),('after','after thread-double-buffer-v')]:
            tail=log_text.split(marker,1)[1].split('\n',1)[1]
            start=re.search(r'^(?:#\w+\s*=|module\b)',tail,re.M).start()
            module_start=re.search(r'^module\b',tail[start:],re.M).start()+start
            end=re.search(r'^}\s*$',tail[module_start:],re.M).end()+module_start
            (HERE/(name+'.thread.'+suffix+'.mlir')).write_text(tail[start:end].rstrip()+'\n')
        before=(HERE/(name+'.thread.before.mlir')).read_text()
        after=(HERE/(name+'.thread.after.mlir')).read_text()
        assert before.count('memref.alloc') == after.count('memref.alloc') == 1
        assert before.count('gpu.barrier') - after.count('gpu.barrier') == 3
        assert before.count('frisk.wait_buffer_ready') == after.count('frisk.wait_buffer_ready')
        assert 'memref<16384xi8, 3>' in after
        old=json.loads((HERE/(kind+'.legacy.generation.json')).read_text())
        assert sorted(reports)==sorted(old['reports']), (reports,old['reports'])
        record.update(thread_stage_capacity_and_sync_checked=True,other_pass_reports_equal_to_pre_migration=True)
    (HERE/(name+'.generation.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(name, '\n'+'\n'.join(reports),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['thread','off','legacy'],default='thread')
    parser.add_argument('--kind',choices=['causal','noncausal','both'],default='both')
    args=parser.parse_args()
    for kind in (['causal','noncausal'] if args.kind=='both' else [args.kind]): generate(kind,args.mode)
