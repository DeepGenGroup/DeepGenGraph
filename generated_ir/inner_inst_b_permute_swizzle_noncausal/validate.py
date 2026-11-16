#!/usr/bin/env python3
"""Check the swizzle against the same-layout baseline and FP64 attention.

Audits delayed DMA and LDS ownership using the existing interpreter. Independent
wave timing, target code generation and GPU performance are not simulated.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wave_validation', HERE.parent/'wave_invariant_noncausal/validate.py')
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)


def main():
    paths = {name: HERE/(name+'.ll') for name in ('baseline', 'kernel', 'kernel.o3')}
    texts = {name: wave.model.helper.model_text(p.read_text()) for name, p in paths.items()}
    records = []
    for kind, head, query in wave.CASES:
        arrays = wave.inputs(kind, query)
        expected = wave.model.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        base, base_full = wave.execute(texts['baseline'], arrays, head, query)
        for name in ('kernel', 'kernel.o3'):
            m, full = wave.execute(texts[name], arrays, head, query)
            actual = full[head, query*128:(query+1)*128]
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert np.isfinite(actual).all()
            assert np.array_equal(full, base_full, equal_nan=True), 'swizzle changed output'
            assert not m.pending and not m.k0_pending
            assert len(m.q_elements) == 16384 and set(m.q_elements.values()) == {1}
            assert (m.v_reads, m.v_dma) == (base.v_reads, base.v_dma)
            assert m.k0_issues == base.k0_issues and m.k0_overlap == base.k0_overlap == 992
            counts = (m.vm_waits, m.barriers, m.mmac_count, m.dma64_count)
            assert counts == (base.vm_waits, base.barriers, base.mmac_count, base.dma64_count)
            record = dict(case=kind, head=head, query=query, variant=name,
                          max_abs_error=float(np.max(np.abs(actual-expected))),
                          bitwise_equal_to_baseline=True, delayed_dma_and_ownership_passed=True,
                          v_schedule_equal=True, k0_schedule_equal=True, counts=counts)
            records.append(record)
            print('PASS', json.dumps(record), flush=True)
    result = dict(cases=records, gpu_tested=False, target_codegen_tested=False,
                  independent_wave_timing_simulated=False, reference_atol=.01, reference_rtol=.01,
                  sha256={name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()})
    (HERE/'numerical_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS all', len(records), 'cases', flush=True)


if __name__ == '__main__':
    main()
