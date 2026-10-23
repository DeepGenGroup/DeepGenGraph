#!/usr/bin/env python3
"""Execute both IRs, proving output equivalence and removed Q DMA traffic."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0,str(ROOT/'3rd/deepgengraph/test'))
from check_pipeline_cleanup_attention import CountedDMA
from check_attention_llvm_numerics import reference
from generate import transform


class MeasuredBlock(CountedDMA):
    def __init__(self,*args,physical,**kwargs):
        self.physical=physical
        self.read_calls=Counter()
        self.read_bytes=Counter()
        self.q0_addresses=Counter()
        self.other_reads=hashlib.sha256()
        self.lds_loads=self.q_scales=0
        super().__init__(*args,**kwargs)

    def call(self,body):
        for axis,value in zip('xyz',(*self.physical,0)):
            if f'@llvm.amdgcn.workgroup.id.{axis}(' in body:
                return np.full(self.t,value,dtype=np.int64)
        return super().call(body)

    def operation(self,expr):
        if expr.startswith(';'):
            return
        self.lds_loads += expr.startswith('load <4 x half>, ptr addrspace(3)')
        self.q_scales += expr.startswith('fmul <4 x half>')
        return super().operation(expr)

    def memory(self,ptr,ty,write=None):
        live=ptr[self.active]
        if write is None and np.all(live >= (1<<32)):
            tensor=int(live[0]>>32)
            assert np.all(live>>32 == tensor)
            size=self.size(ty)
            assert ty=='<8 x half>' and size==16
            key={1:'Q',2:'K',3:'V'}[tensor]
            if tensor==1:
                column=((live-(1<<32))//2)%128
                cached=column < 32
                assert np.all(cached == cached[0])
                key='Q0' if cached[0] else 'Qrest'
                if cached[0]:
                    assert np.all(column+size//2 <= 32)
                    for offset in range(0,size,2):
                        self.q0_addresses.update((live+offset).tolist())
            self.read_calls[key]+=1
            self.read_bytes[key]+=len(live)*size
            if key!='Q0':
                self.other_reads.update(key.encode())
                self.other_reads.update(live.tobytes())
        return super().memory(ptr,ty,write)


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    texts=[(HERE/name).read_text() for name in ('baseline.ll','q_bk32_retained.ll')]
    assert texts[1]==transform(texts[0])
    assert hashlib.sha256(texts[0].encode()).hexdigest()==manifest['baseline_sha256']
    assert hashlib.sha256(texts[1].encode()).hexdigest()==manifest['candidate_sha256']
    cases=[]
    for seed,head,query in ((42,0,0),(42,1,1),(7,1,7),(7,1,8),(42,1,31)):
        rng=np.random.default_rng(seed)
        arrays=[rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        f=31-query
        p=(f//8)*256+head*8+f%8
        models=[MeasuredBlock(t,*arrays,physical=(p%32,p//32)) for t in texts]
        outputs=[m.run().reshape(arrays[0].shape) for m in models]
        assert np.array_equal(*outputs,equal_nan=True),'output differs bitwise'
        n=query+1
        a,b=models
        assert a.other_reads.digest()==b.other_reads.digest(),'other global reads changed'
        assert a.read_calls==dict(Q0=2*n,Qrest=6*n,K=8*n,V=8*n),a.read_calls
        assert b.read_calls==dict(Q0=2,Qrest=6*n,K=8*n,V=8*n),b.read_calls
        # Each of the selected 128x32 FP16 elements is loaded exactly once.
        want={(1<<32)+2*((head*4096+row)*128+col):1
              for row in range(query*128,(query+1)*128) for col in range(32)}
        assert b.q0_addresses==want
        assert a.q0_addresses=={address:n for address in want}
        assert a.lds_loads==144*n and b.lds_loads==140*n+4
        assert a.q_scales==16*n and b.q_scales==12*n+4
        assert a.barriers==16*n and b.barriers==16*n+1
        assert a.vm_waits==12*n and b.vm_waits==12*n+1
        assert a.mmac_count==b.mmac_count==256*n
        assert not a.pending and not b.pending
        rows=slice(query*128,(query+1)*128)
        expected=reference(*arrays,head,query)
        actual=outputs[1][head,rows]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        written=np.zeros(outputs[1].shape,dtype=bool)
        written[head,rows]=True
        assert np.isnan(outputs[1][~written]).all()
        case=dict(seed=seed,head=head,query=query,bitwise_equal=True,
            kv_and_remaining_q_read_trace_identical=True,
            retained_q_elements=4096,each_retained_q_element_loaded_once=True,
            baseline_global_read_bytes=dict(a.read_bytes),candidate_global_read_bytes=dict(b.read_bytes),
            baseline_lds_loads=a.lds_loads,candidate_lds_loads=b.lds_loads,
            baseline_barriers=a.barriers,candidate_barriers=b.barriers,
            no_pending_dma=True,
            max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected))))
        cases.append(case)
        print('PASS',json.dumps(case),flush=True)
    result=dict(baseline_sha256=manifest['baseline_sha256'],candidate_sha256=manifest['candidate_sha256'],
        cpu_delayed_dma=cases,gpu_tested=False,target_gfx936_codegen_tested=False,
        limitations='Collective CPU model does not simulate independent wave timing or register allocation.')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
