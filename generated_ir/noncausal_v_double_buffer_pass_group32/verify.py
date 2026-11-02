#!/usr/bin/env python3
"""Compare pass output to the validated experiment, including LDS ownership."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from generate import module

HERE = Path(__file__).resolve().parent
previous = module('vdb_verification',HERE.parent/'noncausal_v_double_buffer_group32/verify.py')


def main():
    records = []
    paths = dict(experiment=HERE.parent/'noncausal_v_double_buffer_group32/v_double_buffer.ll',
                 compiled_pass=HERE/'v_double_buffer.ll',
                 compiled_pass_o3=HERE/'v_double_buffer.o3.ll')
    for kind,head,query in [('random',0,0),('random',1,31),('max_changes',1,2)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        if kind == 'max_changes':
            arrays[0].fill(1); arrays[1].fill(-16)
            arrays[1][:,:128]=.25; arrays[1][:,128:256]=.5
        expected = previous.reference(*arrays,head,query,causal=False)
        baseline = trace = values = None
        for name,path in paths.items():
            if name.endswith('_o3') and query != 0: continue
            text = re.sub(r'\bexact ','',path.read_text())
            model = previous.AuditedBlock(text,*arrays,threads=256,physical=(query,head))
            full = model.run().reshape(arrays[0].shape)
            actual = full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            assert model.barriers==293 and model.vm_waits==260 and not model.pending
            assert model.v_reads_with_pending_dma==768
            assert model.mmac_count==8192
            if baseline is None:
                baseline=full.copy();trace=model.global_trace.hexdigest();values=model.lds_values.hexdigest()
            assert np.array_equal(full,baseline,equal_nan=True),name
            assert model.global_trace.hexdigest()==trace and model.lds_values.hexdigest()==values,name
            records.append(dict(kind=kind,head=head,query=query,variant=name,
                bitwise_equal_to_validated_experiment=True,barrier_epoch_ownership_passed=True,
                global_trace_sha256=trace,lds_values_sha256=values,
                barriers=model.barriers,vm_waits=model.vm_waits,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS',kind,head,query,name,'bitwise, memory values, slots, barrier ownership',flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,
        sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()},
        gpu_tested=False,independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
