#!/usr/bin/env python3
"""Generate paired-LDS pass variants and verify their attention behavior."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE.parent/'noncausal_k_dma_pass/mytest_k_swizzle64.ll'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('kdma_validation',SOURCE.parent/'validate.py')
kdma=importlib.util.module_from_spec(spec); spec.loader.exec_module(kdma)
LazyAddresses=kdma.LazyAddresses
spec=importlib.util.spec_from_file_location('comparison',HERE.parent/'noncausal_k_dma_compare/compare.py')
comparison=importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
LLVM=Path('/data2/xsl/install/bin')
TOOL=ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'


def check_pairs(text):
    pairs={}; hints=[]
    for i,l in enumerate(text.splitlines()):
        if 'call void @llvm.amdgcn.sched.barrier' in l:hints.append(i)
        if m:=re.search(r'!frisk.k_lds_pair !(\d+)',l):pairs.setdefault(m[1],[]).append(i)
    for positions in pairs.values():
        assert len(positions)==2,positions
        assert not any(positions[0]<i<positions[1] for i in hints),positions
    return len(pairs)


def generate():
    records={}
    for d in (0,1,2,4):
        p=HERE/f'k_lds_prefetch{d}.ll'
        cmd=[str(TOOL),f'--k-lds-prefetch={d}',str(SOURCE)]
        result=subprocess.run(cmd,text=True,capture_output=True,check=True)
        raw=p.with_suffix('.raw.ll');raw.write_text(result.stdout)
        p.with_suffix('.pass.log').write_text(result.stderr)
        legal=subprocess.run([TOOL,'--llvm22-text',raw],text=True,capture_output=True,check=True)
        p.write_text(legal.stdout)
        shutil.copyfile(SOURCE.with_suffix('.launch.json'),p.with_suffix('.launch.json'))
        optimized=p.with_suffix('.o3.ll')
        subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',p,'-o',optimized],check=True)
        if d:assert check_pairs(p.read_text())==check_pairs(optimized.read_text())>0
        records[d]=dict(command=cmd,sha256=kdma.digest(p),diagnostics=result.stderr.strip(),
                        optimized=comparison.inspect(optimized.read_text()))
    baseline=records[0]['optimized']
    for d,r in records.items():
        o=r['optimized']
        assert o['memory_schedule']==baseline['memory_schedule']
        assert o['opcodes']['load']==80 and o['opcodes']['store']==64
        assert len(o['k_reads'])==64 and set(o['k_reads'])==set(baseline['k_reads'])
        a=comparison.Addresses((HERE/'k_lds_prefetch0.o3.ll').read_text());a.reset()
        b=comparison.Addresses((HERE/f'k_lds_prefetch{d}.o3.ll').read_text());b.reset()
        for k in o['k_reads']:
            assert np.array_equal(a.value(baseline['k_reads'][k]['pointer']),b.value(o['k_reads'][k]['pointer']))
        if d:assert len(o['qk_sched_barriers'])>0
    (HERE/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=kdma.digest(SOURCE),
        exporter_sha256=kdma.digest(TOOL),variants=records,gpu_tested=False),indent=2)+'\n')
    print('PASS 0/1/2/4: LLVM O3, identical K addresses and DMA/V-read/wait/barrier positions',flush=True)


def numerical():
    records=[]
    for kind in ('random','future_key_dominates'):
        rng=np.random.default_rng(97)
        arrays=[rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        head,query=1,31 if kind=='random' else 0
        if kind=='future_key_dominates':
            arrays[0].fill(1);arrays[1].fill(-1);arrays[1][:,-128:]=1
            arrays[2].fill(0);arrays[2][:,-128:]=1
        expected=kdma.vdb.reference(*arrays,head,query,causal=False)
        baseline=None
        names=['k_lds_prefetch0.ll','k_lds_prefetch2.ll']
        if kind=='random':names+=['k_lds_prefetch2.o3.ll','k_lds_prefetch1.ll','k_lds_prefetch4.ll']
        for name in names:
            p=HERE/name
            m=kdma.Model(re.sub(r'\bexact ','',p.read_text()),*arrays,threads=256,physical=(query,head))
            full=m.run().reshape(arrays[0].shape)
            actual=full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            if baseline is None:baseline=full.copy()
            assert np.array_equal(full,baseline,equal_nan=True)
            assert not m.pending and m.vm_waits==260 and m.barriers==293 and m.mmac_count==8192
            assert m.v_reads_with_pending_dma==768 and m.dma64_count==512
            assert len(m.q_elements)==16384 and set(m.q_elements.values())=={1}
            records.append(dict(kind=kind,path=str(p),sha256=kdma.digest(p),bitwise_equal_to_off=True,
                vm_waits=m.vm_waits,barriers=m.barriers,mmac=m.mmac_count,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS numerical / delayed DMA / LDS epochs:',kind,name,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,gpu_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__':generate();numerical()
