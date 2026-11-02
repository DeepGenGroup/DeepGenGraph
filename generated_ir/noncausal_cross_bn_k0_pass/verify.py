#!/usr/bin/env python3
"""Numerics/epochs on actual pass IR, plus the complete MyTest export path."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('cross_bn_model',HERE.parent/'noncausal_cross_bn_k0/verify.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    for name, info in manifest['variants'].items():
        for suffix,key in [('', 'sha256'), ('.o3','o3_sha256')]:
            assert hashlib.sha256((HERE/(name+suffix+'.ll')).read_bytes()).hexdigest() == info[key]
    records = []
    cases = [('random',0,0),('random',1,31),('max_changes',1,2),('future_key_dominates',1,0)]
    for kind,head,query in cases:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        q,k,v = arrays
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:,:128]=.25; k[:,128:256]=.5
        if kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:,-128:]=1; v.fill(0); v[:,-128:]=1
        expected = previous.helper.kdma.vdb.reference(*arrays,head,query,causal=False)
        names = ['control','cross_bn_k0'] + (['cross_bn_k0.o3'] if (kind,head,query)==cases[0] else [])
        baseline = model0 = None
        for name in names:
            path = HERE/(name+'.ll')
            model = previous.Model(previous.helper.model_text(path.read_text()),*arrays,threads=256,physical=(query,head))
            full = model.run().reshape(q.shape)
            actual = full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            if baseline is None: baseline,model0=full.copy(),model
            assert np.array_equal(full,baseline,equal_nan=True)
            assert model.read_records == model0.read_records
            assert model.global_trace.digest() == model0.global_trace.digest()
            assert (model.v_reads,model.v_dma) == (model0.v_reads,model0.v_dma)
            early=name!='control'
            assert model.k0_issues == [(t,t*256-(32 if early and t else 0)) for t in range(32) for _ in range(4)]
            assert model.k0_overlap == (992 if early else 0)
            assert (model.vm_waits,model.barriers,model.mmac_count,model.dma64_count)==(260,293,8192,512)
            assert not model.pending and not model.k0_pending
            assert len(model.q_elements)==16384 and set(model.q_elements.values())=={1}
            assert np.isnan(full).sum()==full.size-16384
            records.append(dict(kind=kind,head=head,query=query,variant=name,
                bitwise_equal=True,delayed_DMA_and_ownership_passed=True,max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS',kind,head,query,name,flush=True)
    # Exercise the option through the MLIR export pipeline independently of
    # standalone transformation of the archived artifact.
    out = HERE/'mytest_smoke'; out.mkdir(exist_ok=True)
    original=json.loads((HERE.parent/'noncausal_softmax_fragments/fresh-command.json').read_text())
    commands={}
    for name,enabled in [('control',0),('cross_bn_k0',1)]:
        path=out/(name+'.ll')
        cmd=[str(path) if a=='/tmp/softmax-fragments-fresh.ll' else a for a in original]
        cmd += ['--k-tail-prefetch=4', f'--cross-bn-k0={enabled}']; commands[name]=cmd
        result=subprocess.run(cmd,cwd=out,text=True,capture_output=True,check=True)
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        if enabled: assert 'overlap-mmac=32, sum-anchors=64' in result.stderr,result.stderr[-2000:]
        subprocess.run(['/data2/xsl/install/bin/opt','-passes=default<O3>,verify',path,'-disable-output'],check=True)
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    arrays=[np.random.default_rng(s).normal(size=(2,4096,128)).astype(np.float16) for s in (101,102,103)]
    baseline=model0=None
    for name in commands:
        model=previous.Model(previous.helper.model_text((out/(name+'.ll')).read_text()),*arrays,threads=256,physical=(31,1))
        full=model.run()
        if baseline is None: baseline,model0=full.copy(),model
        assert np.array_equal(full,baseline,equal_nan=True)
        assert model.read_records==model0.read_records and model.global_trace.digest()==model0.global_trace.digest()
        assert model.k0_overlap==(992 if name=='cross_bn_k0' else 0)
        assert not model.pending and (model.vm_waits,model.barriers,model.mmac_count)==(260,293,8192)
        records.append(dict(kind='MyTest',variant=name,bitwise_equal=True,delayed_DMA_and_ownership_passed=True))
        print('PASS MyTest',name,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,variants=manifest['variants'],
        gpu_tested=False,target_codegen_tested=False,independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__': main()
