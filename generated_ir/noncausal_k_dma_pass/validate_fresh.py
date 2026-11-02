#!/usr/bin/env python3
"""Verify real TuneAttention -> MyTest output for all three parameter values."""
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from validate import HERE, LLVM, Model, vdb, digest


def main():
    runs=sorted((HERE/'mytest').glob('run-*'))
    work=runs[-1]
    paths={}
    for directory in sorted(work.glob('grid-*')):
        config=json.loads((directory/'config.json').read_text())
        paths[config['parameters']['k-dma']]=directory/'kernel.ll'
    assert set(paths)=={'128','64','64-swizzle'}
    optimized=paths['64-swizzle'].with_suffix('.o3.ll')
    subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',paths['64-swizzle'],'-o',optimized],check=True)
    paths['64-swizzle.o3']=optimized
    rng=np.random.default_rng(73)
    arrays=[rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    head,query=1,31
    expected=vdb.reference(*arrays,head,query,causal=False)
    baseline=values=None
    records=[]
    for name,path in paths.items():
        text=re.sub(r'\bexact ','',path.read_text())
        model=Model(text,*arrays,threads=256,physical=(query,head))
        full=model.run().reshape(arrays[0].shape)
        actual=full[head,query*128:(query+1)*128]
        np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
        assert not model.pending and model.vm_waits==260 and model.barriers==293 and model.mmac_count==8192
        assert model.v_reads_with_pending_dma==768
        assert len(model.q_elements)==16384 and set(model.q_elements.values())=={1}
        assert getattr(model,'dma64_count',0)==(0 if name=='128' else 512)
        if baseline is None:baseline=full.copy();values=model.lds_values.hexdigest()
        assert np.array_equal(full,baseline,equal_nan=True)
        if not name.endswith('.o3'):assert model.lds_values.hexdigest()==values
        assert np.isnan(full).sum()==full.size-16384
        records.append(dict(mode=name,path=str(path),sha256=digest(path),bitwise_equal_to_fresh_baseline=True,
            barrier_epoch_ownership=True,vm_waits=model.vm_waits,barriers=model.barriers,
            dma64_count=getattr(model,'dma64_count',0),max_abs_error=float(np.max(np.abs(actual-expected)))))
        print('PASS fresh MyTest',name,'bitwise / reference / delayed DMA / epochs',flush=True)
    (HERE/'fresh_validation.json').write_text(json.dumps(dict(cases=records,gpu_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__':main()
