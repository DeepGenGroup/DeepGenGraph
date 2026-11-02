#!/usr/bin/env python3
"""Validate full MyTest output against the same fresh pipeline with V DB off."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from generate import module, LLVM

HERE = Path(__file__).resolve().parent
helper = module('vdb_fresh_verify', HERE.parent/'noncausal_v_double_buffer_group32/verify.py')


def main():
    work = HERE/'mytest_smoke'
    subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S',work/'kernel.ll',
                    '-o',work/'kernel.o3.ll'],check=True)
    rng = np.random.default_rng(73)
    arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    records = []
    reference = helper.reference(*arrays,1,31,causal=False)
    baseline = global_trace = lds_values = None
    for name in ['baseline','kernel','kernel.o3']:
        path = work/(name+'.ll')
        model = helper.AuditedBlock(re.sub(r'\bexact ','',path.read_text()),*arrays,
                                   threads=256,physical=(31,1))
        full = model.run().reshape(arrays[0].shape)
        actual = full[1,31*128:]
        np.testing.assert_allclose(actual,reference,atol=.01,rtol=.01)
        assert not model.pending and model.vm_waits==260
        assert model.barriers==(389 if name=='baseline' else 293)
        if baseline is None:
            baseline=full.copy(); global_trace=model.global_trace.hexdigest(); lds_values=model.lds_values.hexdigest()
        assert np.array_equal(full,baseline,equal_nan=True),name
        if name=='kernel':
            assert global_trace==model.global_trace.hexdigest() and lds_values==model.lds_values.hexdigest()
        records.append(dict(variant=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bitwise_equal_to_fresh_baseline=True,barrier_epoch_ownership_passed=True,
            vm_waits=model.vm_waits,barriers=model.barriers,
            max_abs_error=float(np.max(np.abs(actual-reference)))))
        print('PASS fresh MyTest',name,'numerics, delayed DMA and barrier epochs',flush=True)
    (work/'validation.json').write_text(json.dumps(dict(cases=records,gpu_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
