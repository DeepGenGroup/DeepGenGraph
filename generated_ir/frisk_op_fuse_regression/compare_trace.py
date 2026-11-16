from pathlib import Path
import importlib.util, hashlib, json, re
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cvt',HERE.parent/'noncausal_cvt_pack/verify.py')
cvt=importlib.util.module_from_spec(spec); spec.loader.exec_module(cvt)
class Trace(cvt.Model):
    def __init__(self,*a,**k):
        self.mma_trace=[]; self.global_events=[]; self.output_addresses=[]
        super().__init__(*a,**k)
    def memory(self, ptr, ty, write=None):
        live=ptr[self.active]
        if live.size and np.all(live >> 32 != 0):
            self.global_events.append(("read" if write is None else "write", self.mmac_count, str(ty), hashlib.sha256(live.tobytes()).hexdigest()))
        if write is not None and live.size and np.all(live >> 32 != 0):
            self.output_addresses.append(live.copy())
        return super().memory(ptr,ty,write)
    def call(self,body):
        if 'asm "v_mmac_' in body:
            args=re.search(r'asm "[^"]+", "[^"]+"\((.*)\)',body)[1]
            values=[self.typed(x) for x in cvt.lazy.split_args(args)]
            self.mma_trace.append(tuple(hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in values))
        return super().call(body)
arrays=cvt.lazy.dataset('random',42)
models=[]; results=[]
for name in ('gemm_accumulator','frisk_op_fuse'):
    path=HERE.parent/('best_noncausal_'+name)/'kernel.o3.ll'
    m=Trace(cvt.lazy.previous.helper.model_text(path.read_text()),*arrays,threads=256,physical=(0,0))
    results.append(m.run());models.append(m)
    print(name,'done',m.mmac_count,flush=True)
a,b=models
r={'outputs_bit_equal':bool(np.array_equal(*results,equal_nan=True)),
'mma_inputs_equal_in_order':a.mma_trace==b.mma_trace,
'first_mma_mismatch':next((i for i,(x,y) in enumerate(zip(a.mma_trace,b.mma_trace)) if x!=y),None),
'lds_read_records_equal':a.read_records==b.read_records,'lds_values_equal':a.lds_values.digest()==b.lds_values.digest(),
'global_trace_equal':a.global_trace.digest()==b.global_trace.digest(),'k0_issues_equal':a.k0_issues==b.k0_issues,
'global_reads_equal':[x for x in a.global_events if x[0]=='read']==[x for x in b.global_events if x[0]=='read'],
'global_write_lane_vectors_equal_as_set':sorted(a.global_events)==sorted(b.global_events),
'collective_output_address_set_equal':bool(np.array_equal(np.sort(np.concatenate(a.output_addresses)),np.sort(np.concatenate(b.output_addresses)))),
'first_eight_store_columns_lanes_0_16_32_48':[[((x[[0,16,32,48]]%(1<<32))//2%128).tolist() for x in m.output_addresses[:8]] for m in models],
'first_global_event_mismatch':next(({'index':i,'before':x,'after':y} for i,(x,y) in enumerate(zip(a.global_events,b.global_events)) if x!=y),None),
'counts':[(m.vm_waits,m.barriers,m.mmac_count,m.dma64_count) for m in models]}
(HERE/'trace_comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
