#!/usr/bin/env python3
"""Execute the actual before/after LLVM using the existing DMA/LDS audit model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('causal_validation',HERE.parent/'causal_cvt_pack_mytest/verify.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    texts={}
    for name,info in manifest['variants'].items():
        for suffix,key in [('', 'sha256'),('.o3','o3_sha256')]:
            path=HERE/(name+suffix+'.ll')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==info[key]
            texts[name+suffix]=previous.causal.previous.helper.model_text(path.read_text())
    records=[]
    for kind,query in [('random',0),('random',8),('random',31),('max_changes',2),
                       ('diagonal_future_dominates',0),('positive_overflow',1),('negative_underflow',1)]:
        arrays=[np.random.default_rng(42+query+s).normal(size=(2,4096,128)).astype(np.float16) for s in range(3)]
        q,k,v=arrays;head=1
        if kind=='max_changes':
            q.fill(1);k.fill(-16);k[:,:128]=.25;k[:,128:256]=.5
        elif kind=='diagonal_future_dominates':
            q.fill(1);k[:]=(np.arange(4096,dtype=np.float32)/8)[None,:,None]
            v[:]=(np.arange(4096,dtype=np.float32)/128)[None,:,None]
        elif kind in ('positive_overflow','negative_underflow'):
            q.fill(16);k.fill(16 if kind=='positive_overflow' else -16)
        expected=previous.causal.previous.helper.kdma.vdb.reference(*arrays,head,query,causal=True)
        names=list(manifest['variants'])
        if kind=='random' and query in (0,31):names += [n+'.o3' for n in names]
        base=base_model=None;raw_outputs={}
        for name in names:
            model=previous.Model(texts[name],*arrays,threads=256,
                physical=previous.causal.physical(4,head,31-query),logical_head=head,logical_query=query)
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                full=model.run().reshape(q.shape)
            actual=full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            assert np.isfinite(actual).all() and np.isnan(full).sum()==full.size-16384
            if base is None:base,base_model=full.copy(),model
            bitwise=np.array_equal(full.view(np.uint8),base.view(np.uint8))
            if 'i32' in name:assert bitwise,(name,kind,'exact mode differs')
            assert model.global_trace.digest()==base_model.global_trace.digest()
            # Fusion changes scheduling opportunities, including K-tail LDS
            # prefetch. Match every read's address, value, barrier epoch and
            # multiplicity, while allowing independent reads to be reordered.
            assert model.read_records==base_model.read_records,(name,kind,'LDS provenance/epoch')
            ordered_lds_equal=model.lds_values.digest()==base_model.lds_values.digest()
            assert (model.k0_issues,model.v_reads,model.v_dma)==(base_model.k0_issues,base_model.v_reads,base_model.v_dma)
            assert (model.mmac_count,model.dma64_count,model.vm_waits,model.barriers)==(
                base_model.mmac_count,base_model.dma64_count,base_model.vm_waits,base_model.barriers)
            assert not model.pending and not model.k0_pending
            assert model.pack_calls==32*(query+1)
            if name.endswith('.o3'):assert np.array_equal(full.view(np.uint8),raw_outputs[name[:-3]].view(np.uint8))
            else:raw_outputs[name]=full.copy()
            records.append(dict(variant=name,case=kind,query=query,bitwise_equal_to_off=bool(bitwise),
                max_abs_error=float(np.max(np.abs(actual-expected))),
                causal_bounds_and_dma_lds_audit_passed=True,global_trace_and_v_schedule_equal_to_off=True,
                LDS_read_addresses_values_epochs_equal=True,ordered_LDS_values_equal=bool(ordered_lds_equal)))
            print('PASS',name,kind,query,'bitwise',bitwise,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,gpu_tested=False,
        artifacts={n:{k:v[k] for k in ('sha256','o3_sha256')} for n,v in manifest['variants'].items()},
        reference_atol=.01,reference_rtol=.01,independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__':main()
