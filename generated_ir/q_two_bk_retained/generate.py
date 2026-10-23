#!/usr/bin/env python3
"""Single-variable compiler experiment: retain 8 versus 16 operand words."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LLVM = Path('/data2/xsl/install/bin')
OLD = ROOT/'generated_ir/invariant_fragment_pass'
SOURCE_HASH = '8923ce4d910b6c2e6ce91fa3345c5f2e702b57d2bef5ef37cd12c726b0efea19'
TOKENS = dict(DMA='buffer_load_dwordx4 ',waits='s_waitcnt vmcnt(0)',
              barriers='call void @llvm.amdgcn.s.barrier(',
              lds_reads='= load <4 x half>',q_scales='= fmul <4 x half>',
              mmac='v_mmac_f32_16x16x16_f16 ',hints='call void @llvm.amdgcn.sched.barrier(')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing',action='store_true',help='Validate already exported files and their recorded commands.')
    args = parser.parse_args()
    source = OLD/'retained_8words.ll'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    original = json.loads((OLD/'manifest.json').read_text())
    template = original['variants']['retained_8words']['command']
    env = dict(os.environ,FRISK_GM_TO_SHM_MODE='async',FRISK_GM_TO_SHM_ALIGNMENT='16')
    commands = json.loads((HERE/'export_commands.json').read_text()) if args.existing else {}
    records = {}
    for name,words in [('baseline_8words',8),('retained_16words',16)]:
        path = HERE/(name+'.ll')
        cmd = [x.replace('--retain-fragment-words=8',f'--retain-fragment-words={words}') for x in template]
        cmd[3] = str(path)
        if not args.existing:
            with (HERE/(name+'.compile.log')).open('w') as log:
                subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            commands[name] = cmd
        actual_cmd = commands[name]
        # Only the output path and retention budget differ across exports.
        assert [x for i,x in enumerate(actual_cmd) if i != 3] == [x for i,x in enumerate(cmd) if i != 3]
        text = path.read_text()
        log = (HERE/(name+'.compile.log')).read_text()
        retained = words//8
        assert f'RetainInvariantFragments: {retained} operand tiles retained; budget {words} words/thread' in log
        assert f'frisk.retained_words = {words} : i32' in log
        if words == 8: assert path.read_bytes() == source.read_bytes()
        label = re.search(r'^([\w.]+):',text,re.M)[1]
        entry,loop = text.split('\n'+label+':',1)
        counts = {part:{k:s.count(t) for k,t in TOKENS.items()} for part,s in [('entry',entry),('loop',loop)]}
        assert counts['entry']['DMA'] == 2*retained
        assert counts['loop']['DMA'] == 24-2*retained
        assert counts['entry']['lds_reads'] == counts['entry']['q_scales'] == 4*retained
        assert counts['loop']['lds_reads'] == 144-4*retained
        assert counts['loop']['q_scales'] == 16-4*retained
        assert counts['entry']['waits'] == retained and counts['loop']['waits'] == 12-retained
        assert counts['entry']['barriers'] == retained+1
        assert counts['loop']['barriers'] == (16 if retained==1 else 14)
        assert counts['loop']['mmac'] == 256 and counts['loop']['hints'] == 8
        assert counts['entry']['mmac'] == counts['entry']['hints'] == 0
        storage = re.findall(r'^@\S+ = addrspace\(3\) global \[(\d+) x i8\]',text,re.M)
        assert storage == ['16384'],storage
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify',path,'-S','-o',HERE/(name+'.opt.ll')],check=True)
        (HERE/(name+'.launch.json')).write_bytes(source.with_suffix('.launch.json').read_bytes())
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            retain_words=words,retained_BK_fragments=retained,
                            counts=counts,static_lds_bytes=16384,llvm_assemble_and_O3_verify=True,
                            actual_export_command=actual_cmd,reproduction_command=cmd)
        print(name,json.dumps({k:v for k,v in records[name].items() if 'command' not in k}),flush=True)
    (HERE/'export_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    manifest = dict(source=str(source.relative_to(ROOT)),source_sha256=SOURCE_HASH,
                    input_mlir=template[1],input_mlir_sha256=hashlib.sha256(Path(template[1]).read_bytes()).hexdigest(),
                    baseline_byte_identical=True,compiler_pass='frisk-retain-invariant-fragments',
                    sole_configuration_change='--retain-fragment-words=8 -> 16',variants=records,
                    logical_Q_read_MiB=dict(baseline_8words=404,retained_16words=280),
                    logical_QKV_read_MiB=dict(baseline_8words=1460,retained_16words=1336),
                    gpu_tested=False,target_gfx936_codegen_tested=False,
                    limitations='Logical request bytes are not measured HBM traffic. Extra 8 retained words do not predict allocated VGPR delta; target scratch/occupancy and timing require GPU testing.')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
