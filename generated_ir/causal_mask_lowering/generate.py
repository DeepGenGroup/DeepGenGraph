#!/usr/bin/env python3
"""A/B the causal mask lowering through the complete current MyTest pipeline."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent/'causal_cvt_pack_mytest'
LLVM = Path('/data2/xsl/install/bin')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(text):
    return re.sub(r'^; ModuleID = .*\n','',text)


def main():
    old = json.loads((BASE/'manifest.json').read_text())['variants']['group4_reverse_pk_rtz']
    assert digest(BASE/'group4_reverse_pk_rtz.o3.ll') == old['o3_sha256']
    def generate(mode):
        name = 'mask_'+mode+'_pk_rtz'
        command = old['command'].copy()
        command[6] = str(HERE/(name+'.ll'))
        command += ['--causal-mask='+mode]
        work = HERE/(name+'.work'); work.mkdir(exist_ok=True)
        (HERE/(name+'.command.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            +'cd -- '+shlex.quote(str(work))+'\n'+shlex.join(command)+'\n')
        with (HERE/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        out = HERE/(name+'.ll'); opt = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'llvm-as',out,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',out,'-o',opt],check=True)
        text = opt.read_text()
        reports = [l for l in (HERE/(name+'.log')).read_text().splitlines()
                   if l.startswith(('Causal mask lowering:', 'Compact integer comparisons:',
                     'V double buffer:', 'K DMA:', 'Cross BN K0:', 'Softmax cvt pack:',
                     'K LDS prefetch:', 'K tail LDS prefetch:', 'SoftmaxFragments:'))]
        copies=len(re.findall(r'^  %compare.local.base.* = .*asm sideeffect',text,re.M))
        adds=text.count('fadd <4 x float>')
        assert text.count('asm "v_cvt_pkrtz_f16_f32')==32
        assert copies==(62 if mode=='off' else 0),(name,copies,reports)
        assert adds==(0 if mode=='finite' else 16),(name,adds,reports)
        tail_counts='pairs=7, moved=5, anchors=4' if mode=='off' else 'pairs=8, moved=7, anchors=5'
        assert any(l.startswith('K tail LDS prefetch:') and tail_counts in l
                   for l in reports),(name,'K-tail prefetch regressed',reports)
        same=canonical(text)==canonical((BASE/'group4_reverse_pk_rtz.o3.ll').read_text())
        if mode=='off': assert same,'disabled mode changed the user-tested artifact'
        for path in (out,opt):
            path.with_suffix('.launch.json').write_bytes((BASE/'group4_reverse_pk_rtz.o3.launch.json').read_bytes())
        info=dict(command=command,mode=mode,sha256=digest(out),o3_sha256=digest(opt),
            compare_base_copies=copies,score_plus_mask_vector_adds=adds,packed_pairs=32,
            LLVM_and_O3_verified=True,equals_user_tested_except_module_id=same,pass_reports=reports)
        print('PASS',name,'comparison copies',copies,'mask vector adds',adds,flush=True)
        return name,info
    with ThreadPoolExecutor(max_workers=3) as pool:
        variants=dict(pool.map(generate,('off','i32','finite')))
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=variants,
        input=old['command'][4],input_sha256=digest(Path(old['command'][4])),
        mytest_sha256=digest(Path(old['command'][3])),
        finite_contract='finite pre-mask scores; signed zero may change',
        group=4,reverse=True,causal=True,gpu_tested=False,target_codegen_tested=False),indent=2)+'\n')


if __name__=='__main__':
    main()
