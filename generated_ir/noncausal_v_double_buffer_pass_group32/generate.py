#!/usr/bin/env python3
"""Reproduce the best-case V double-buffer artifact through the compiled pass."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent/'noncausal_pv_interleave_pass_group32/pv_interleave4.ll'
SOURCE_SHA = '2bef36b60ab7309928d68bea12a033ee0615d907f577333871fae41f74156c51'
EXPORTER = ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA
    helper = module('vdb_experiment',HERE.parent/'noncausal_v_double_buffer_group32/generate.py')
    source = SOURCE.read_text()
    manifest = dict(source=str(SOURCE),source_sha256=SOURCE_SHA,
                    exporter=str(EXPORTER),exporter_sha256=sha(EXPORTER),
                    pass_option='--v-double-buffer=1',pv_interleave=4,
                    BM=128,BN=128,BK=32,threads=256,waves=[4,1],
                    group_size=32,reverse=0,retain_fragment_words=32,
                    lds_bytes=16384,gpu_tested=False,variants={})
    for name,enabled in [('baseline_pv_interleave4',False),('v_double_buffer',True)]:
        out = HERE/(name+'.ll')
        if enabled:
            command = [str(EXPORTER),'--v-double-buffer=1',str(SOURCE)]
            result = subprocess.run(command,text=True,capture_output=True,check=True)
            raw = HERE/(name+'.raw.ll')
            raw.write_text(result.stdout)
            (HERE/(name+'.pass.log')).write_text(result.stderr)
            assert '4 slices, slot-bytes=8192, displacement=8192, removed-barriers=3, LDS=16384' in result.stderr,result.stderr
            text = subprocess.run([str(EXPORTER),'--llvm22-text',str(raw)],text=True,capture_output=True,check=True).stdout
            out.write_text(text)
        else:
            out.write_bytes(SOURCE.read_bytes())
        out.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as',out,'-o','/dev/null'],check=True)
        opt = HERE/(name+'.o3.ll')
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S',out,'-o',opt],check=True)
        schedule = helper.audit(out.read_text(),source,enabled)
        assert helper.audit(opt.read_text(),source,enabled)==schedule
        # Compare the entire Q/K/softmax prefix after canonical text printing.
        def prefix(text):
            canonical = subprocess.run([LLVM/'opt','-passes=verify','-S','-o','-','-'],
                                       input=text,text=True,capture_output=True,check=True).stdout
            lines = canonical.splitlines()
            first_read = next(i for i,l in enumerate(lines) if helper.DEF.match(l) and helper.VREAD in l)
            end = max(i for i in range(first_read) if 'fence syncscope("workgroup") acquire' in lines[i])
            return [re.sub(r', ![\w.]+ !\d+','',re.sub(r' #\d+','',l)) for l in lines[:end+1]
                    if l.startswith('  ')]
        assert prefix(out.read_text())==prefix(source)
        manifest['variants'][name]=dict(sha256=sha(out),llvm_and_o3_verified=True,schedule=schedule)
        print('PASS',name,'compiled pass, LLVM/O3, unchanged MMAC order and V schedule',flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
