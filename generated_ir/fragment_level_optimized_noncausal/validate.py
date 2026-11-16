#!/usr/bin/env python3
"""Execute delivered LLVM using the repository's wave/DMA numerical model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/data2/xsl/install/bin')
spec = importlib.util.spec_from_file_location('cross_bn_model', HERE.parent/'noncausal_cross_bn_k0/verify.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def main():
    records = []
    for kind, head, query in [('random', 1, 31), ('future_key_dominates', 1, 0)]:
        rng = np.random.default_rng(42 + query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        q,k,v = arrays
        if kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:,-128:] = 1; v.fill(0); v[:,-128:] = 1
        expected = model.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        baseline = None
        kernel_result = None
        for name in (['baseline', 'kernel', 'kernel.o3'] if kind == 'random' else ['baseline', 'kernel']):
            path = HERE / (name + '.ll')
            m = model.Model(model.helper.model_text(path.read_text()), *arrays,
                            threads=256, physical=(query,head))
            full = m.run().reshape(q.shape)
            actual = full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert np.isfinite(actual).all()
            if baseline is None: baseline = full.copy()
            baseline_equal = np.array_equal(full, baseline, equal_nan=True)
            if name == 'kernel': kernel_result = full.copy()
            if name == 'kernel.o3': assert np.array_equal(full, kernel_result, equal_nan=True)
            assert not m.pending and not m.k0_pending
            assert (m.vm_waits,m.barriers,m.mmac_count,m.dma64_count) == (260,293,8192,512)
            assert m.k0_overlap == 992
            assert m.k0_issues == [(t,t*256-(32 if t else 0)) for t in range(32) for _ in range(4)]
            assert len(m.q_elements) == 16384 and set(m.q_elements.values()) == {1}
            records.append(dict(case=kind,head=head,query=query,variant=name,
                max_abs_error=float(np.max(np.abs(actual-expected))),
                baseline_bitwise_equal=bool(baseline_equal),
                O3_bitwise_equal=True if name == "kernel.o3" else None,
                delayed_DMA_and_ownership_passed=True,
                mmac_count=m.mmac_count,k0_overlap_mmac=m.k0_overlap))
            print('PASS', records[-1], flush=True)
    report = dict(cases=records, gpu_tested=False, target_codegen_tested=False,
        independent_wave_timing_simulated=False, reference_atol=.01,reference_rtol=.01,
        sha256=hashlib.sha256((HERE/'kernel.ll').read_bytes()).hexdigest())
    (HERE/'numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
