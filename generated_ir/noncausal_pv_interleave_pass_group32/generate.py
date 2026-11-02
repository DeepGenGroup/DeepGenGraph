#!/usr/bin/env python3
"""Export the best-baseline PV experiments through the compiled LLVM Pass."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPORTER = ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')
SOURCE = HERE.parent/'noncausal_v_handoff_sweep_group32/dma_after4.ll'
SOURCE_SHA = '57a8f633c911ce4579b0e8a076d9ce08c05053d6071b025a7f0db39c3a1baecd'
PREVIOUS = HERE.parent/'noncausal_pv_interleave_group32'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def instructions(text):
    # Ignore only attribute-group numbering and metadata references. SSA names,
    # instruction order, operands, predicates, FP flags and constants must match.
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')]


def canonical(text):
    # LLVM22 export rewrites decimal half literals to their exact hexadecimal
    # bit patterns. Round-trip both sides before comparing instruction text.
    return subprocess.run([LLVM/'opt','-passes=verify','-S','-o','-','-'],
                          input=text,text=True,capture_output=True,check=True).stdout


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    helper = module('previous_pv', PREVIOUS/'generate.py')
    kdir = HERE.parent/'noncausal_grouped_lds_prefetch'
    kgen = module('kgen', kdir/'generate.py')
    km = json.loads((kdir/'manifest_protected.json').read_text())
    source = SOURCE.read_text()
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail']
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                    exporter=str(EXPORTER), exporter_sha256=hashlib.sha256(EXPORTER.read_bytes()).hexdigest(),
                    group_size=32, reverse=0, BM=128, BN=128, BK=32,
                    threads=256, lds_bytes=16384, reader_handoff_after=4,
                    gpu_tested=False, variants={})
    for name, width in [('baseline_after4',0),('pv_serial_guarded',1),('pv_interleave2',2),('pv_interleave4',4)]:
        path = HERE/(name+'.ll')
        if not width:
            path.write_bytes(SOURCE.read_bytes())
        else:
            command = [str(EXPORTER),f'--pv-interleave={width}',str(SOURCE)]
            raw = HERE/(name+'.raw.ll')
            log = HERE/(name+'.pass.log')
            with raw.open('w') as output, log.open('w') as errors:
                subprocess.run(command,stdout=output,stderr=errors,check=True)
            assert f'4 regions, width={width}' in log.read_text()
            commands.append(shlex.join(command)+' > '+shlex.quote(str(raw))+' 2> '+shlex.quote(str(log)))
            legalize = [str(EXPORTER),'--llvm22-text',str(raw)]
            with path.open('w') as output:
                subprocess.run(legalize,stdout=output,check=True)
            commands.append(shlex.join(legalize)+' > '+shlex.quote(str(path)))
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        text = path.read_text()
        assert instructions(canonical(text)) == instructions(canonical((PREVIOUS/(name+'.ll')).read_text())), name
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        optimized = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S',path,'-o',optimized],check=True)
        opt_text = optimized.read_text()
        schedule = helper.audit(text,source,width) if width else None
        if width: assert helper.audit(opt_text,source,width) == schedule
        kgen.assert_groups_unbroken(opt_text.splitlines(),km['groups'])
        d = helper.h.defs(opt_text.splitlines())
        for group in km['moved_groups']:
            assert max(d[n][0] for n in group['loads']) < d[group['anchor']][0]
            assert 'sched.barrier(i32 0)' in opt_text.splitlines()[d[group['anchor']][0]-1]
        # Do not compare ordered MMACs with the serial baseline; their SSA input
        # edges are unchanged and independently checked by the Python candidate.
        if not width:
            memory_sync = helper.h.operations(opt_text.splitlines())
        else:
            normalize = lambda ls: [re.sub(r', ![\w.]+ !\d+','',re.sub(r' #\d+','',l)) for l in ls]
            assert normalize(helper.h.operations(opt_text.splitlines())) == normalize(memory_sync)
        manifest['variants'][name] = dict(width=width,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            llvm_and_o3_verified=True,instructions_match_python_candidate=True,
            k_prefetch_preserved=True,memory_sync_order_preserved=True,schedule=schedule)
        print(f'PASS {name}: compiled Pass; matches Python candidate; LLVM/O3 and scheduling checks',flush=True)
    (HERE/'commands.sh').write_text('\n'.join(commands)+'\n')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
