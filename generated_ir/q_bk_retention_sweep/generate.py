#!/usr/bin/env python3
"""Compiler exports for 2/3/4 Q BK fragments retained across BN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT/'generated_ir/q_two_bk_retained'
LLVM = Path('/data2/xsl/install/bin')
SOURCE_HASH = 'eb04526e34435cc8e9766a76c4e72fcacc57e26969988a522e1af072a3d90925'
TOKENS = dict(DMA='buffer_load_dwordx4 ',waits='s_waitcnt vmcnt(0)',
              barriers='call void @llvm.amdgcn.s.barrier(',
              lds_reads='= load <4 x half>',q_scales='= fmul <4 x half>',
              mmac='v_mmac_f32_16x16x16_f16 ',hints='call void @llvm.amdgcn.sched.barrier(')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing',action='store_true')
    args = parser.parse_args()
    source = OLD/'retained_16words.ll'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    old = json.loads((OLD/'manifest.json').read_text())
    template = old['variants']['retained_16words']['reproduction_command']
    env = dict(os.environ,FRISK_GM_TO_SHM_MODE='async',FRISK_GM_TO_SHM_ALIGNMENT='16')
    records = {}
    for name,words in [('baseline_16words',16),('retained_24words',24),('retained_32words',32)]:
        path = HERE/(name+'.ll')
        cmd = [x.replace('--retain-fragment-words=16',f'--retain-fragment-words={words}') for x in template]
        cmd[3] = str(path)
        if words == 16:
            path.write_bytes(source.read_bytes())
            (HERE/(name+'.compile.log')).write_bytes((OLD/'retained_16words.compile.log').read_bytes())
        elif not args.existing:
            print('Exporting',name,flush=True)
            with (HERE/(name+'.compile.log')).open('w') as log:
                subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        text = path.read_text()
        log = (HERE/(name+'.compile.log')).read_text()
        retained = words//8
        assert f'RetainInvariantFragments: {retained} operand tiles retained; budget {words} words/thread' in log
        assert f'frisk.retained_words = {words} : i32' in log
        label = re.search(r'^([\w.]+):',text,re.M)[1]
        entry,loop = text.split('\n'+label+':',1)
        counts = {part:{k:s.count(t) for k,t in TOKENS.items()} for part,s in [('entry',entry),('loop',loop)]}
        assert counts['entry']['DMA'] == 2*retained
        assert counts['loop']['DMA'] == 24-2*retained
        assert counts['entry']['lds_reads'] == counts['entry']['q_scales'] == 4*retained
        assert counts['loop']['lds_reads'] == 144-4*retained
        assert counts['loop']['q_scales'] == 16-4*retained
        assert counts['loop']['mmac'] == 256 and counts['loop']['hints'] == 8
        assert counts['entry']['mmac'] == counts['entry']['hints'] == 0
        storage = re.findall(r'^@\S+ = addrspace\(3\) global \[(\d+) x i8\]',text,re.M)
        lds_bytes = sum(map(int,storage))
        assert lds_bytes and lds_bytes<=65536,storage
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify',path,'-S','-o',HERE/(name+'.opt.ll')],check=True)
        (HERE/(name+'.launch.json')).write_bytes(source.with_suffix('.launch.json').read_bytes())
        q_bytes = 32*8192*(32*retained+528*(4-retained))
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            retain_words=words,retained_BK_fragments=retained,retained_Q_columns=[0,32*retained],
                            counts=counts,static_lds_bytes=lds_bytes,llvm_assemble_and_O3_verify=True,
                            logical_Q_read_MiB=q_bytes//(1024**2),logical_QKV_read_MiB=1056+q_bytes//(1024**2),
                            reproduction_command=cmd,
                            provenance='Byte copy of measured 16-word candidate and its compile log' if words==16 else 'Fresh MyTest export')
        print(name,json.dumps({k:v for k,v in records[name].items() if 'command' not in k}),flush=True)
    manifest = dict(source=str(source.relative_to(ROOT)),source_sha256=SOURCE_HASH,
                    baseline_byte_identical=True,baseline_measured_median_ms=1.5019190311431885,
                    input_mlir=template[1],input_mlir_sha256=hashlib.sha256(Path(template[1]).read_bytes()).hexdigest(),
                    compiler_pass='frisk-retain-invariant-fragments',
                    sole_configuration_change='--retain-fragment-words=16 / 24 / 32',variants=records,
                    gpu_tested=False,target_gfx936_codegen_tested=False,
                    limitations='Retention budget is logical state, not allocated VGPRs. Target registers, scratch, LDS metadata and performance must be measured. Logical bytes are not HBM traffic.')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
