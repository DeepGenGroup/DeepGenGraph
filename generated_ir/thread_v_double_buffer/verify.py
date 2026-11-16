#!/usr/bin/env python3
"""Execute actual LLVM with delayed DMA and LDS reader/barrier ownership audits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

pack=module('vdb_pack',HERE.parent/'noncausal_cvt_pack/verify.py')
causal=module('vdb_causal',HERE.parent/'causal_cvt_pack_mytest/verify.py')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    records=[]
    for kind in ('causal','noncausal'):
        variants=['legacy'] if args.baseline_only else ['legacy','thread','thread-o3']
        for query in ([0] if args.baseline_only else [0,1,31]):
            head=1
            arrays=[np.random.default_rng(42+query+i).normal(size=(2,4096,128)).astype(np.float16) for i in range(3)]
            expected=pack.lazy.previous.helper.kdma.vdb.reference(*arrays,head,query,causal=kind=='causal')
            baseline=base_model=None
            for variant in variants:
                suffix={'legacy':'.legacy.o3.ll','thread':'.ll','thread-o3':'.o3.ll'}[variant]
                path=HERE/(kind+suffix)
                text=pack.lazy.previous.helper.model_text(path.read_text())
                if kind=='causal':
                    model=causal.Model(text,*arrays,threads=256,physical=causal.causal.physical(4,head,31-query),logical_head=head,logical_query=query)
                else:
                    model=pack.Model(text,*arrays,threads=256,physical=(query,head))
                full=model.run().reshape(arrays[0].shape)
                actual=full[head,query*128:(query+1)*128]
                np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
                assert np.isfinite(actual).all()
                assert not model.pending and not model.k0_pending
                if baseline is None:baseline,base_model=full.copy(),model
                assert np.array_equal(full,baseline,equal_nan=True),(kind,query,variant,'output')
                # The scheduling rewrite may move independent reads, but never
                # changes global DMA data, total MMACs or completion counts.
                assert model.global_trace.digest()==base_model.global_trace.digest(),(kind,query,variant,'global DMA trace')
                assert model.mmac_count==base_model.mmac_count
                assert model.vm_waits==base_model.vm_waits
                assert model.barriers==base_model.barriers
                records.append(dict(kind=kind,query=query,variant=variant,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_abs_error=float(np.max(np.abs(actual-expected))),bitwise_equal_to_pre_migration=True,delayed_dma_and_lds_ownership_passed=True,barriers=model.barriers,vm_waits=model.vm_waits,mmac=model.mmac_count))
                print('PASS',records[-1],flush=True)
    (HERE/('baseline-validation.json' if args.baseline_only else 'validation.json')).write_text(json.dumps(dict(cases=records,gpu_tested=False,independent_wave_timing_simulated=False),indent=2)+'\n')

if __name__=='__main__':main()
