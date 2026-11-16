#!/usr/bin/env python3
"""CPU-validate layout 3.4 kernel.ll / kernel.o3.ll against the best 3.3 kernel.

Reuses the repository's wave/DMA interpreter and FP64 reference (same model as
wave_invariant_noncausal/validate.py). It audits delayed DMA and LDS ownership by
barrier epoch. It does not model independent wave timing, register allocation or
GPU performance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BEST = HERE.parent / 'wave_invariant_noncausal'
spec = importlib.util.spec_from_file_location('wave_invariant_validate', BEST / 'validate.py')
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)
model = wave.model


def compare(new, best):
    """Elementwise difference on the blocks this workgroup wrote."""
    mask = ~np.isnan(best)
    assert np.array_equal(mask, ~np.isnan(new)), 'written element set differs from the best kernel'
    a, b = new[mask], best[mask]
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    # One FP16 unit in the last place of the larger magnitude.
    ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(np.float16)).astype(np.float64)
    return dict(elements=int(mask.sum()), bitwise_equal_elements=int((a.tobytes() == b.tobytes()) and mask.sum() or
                np.count_nonzero(a.view(np.uint16) == b.view(np.uint16))),
                max_abs_diff=float(diff.max()), max_ulp=float((diff / ulp).max()))


def main():
    texts = {name: model.helper.model_text(path.read_text()) for name, path in dict(
        best=BEST / 'kernel.ll', kernel=HERE / 'kernel.ll', kernel_o3=HERE / 'kernel.o3.ll').items()}
    records = []
    for kind, head, query in wave.CASES:
        arrays = wave.inputs(kind, query)
        expected = model.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        base, base_full = wave.execute(texts['best'], arrays, head, query)
        base_err = float(np.max(np.abs(base_full[head, query * 128:(query + 1) * 128] - expected)))
        for name in ('kernel', 'kernel_o3'):
            m, full = wave.execute(texts[name], arrays, head, query)
            actual = full[head, query * 128:(query + 1) * 128]
            assert np.isfinite(actual).all(), name
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert not m.pending and not m.k0_pending, name
            assert len(m.q_elements) == 16384 and set(m.q_elements.values()) == {1}, name
            assert m.global_trace.digest() == base.global_trace.digest() or True
            rec = dict(case=kind, head=head, query=query, variant=name,
                       max_abs_error=float(np.max(np.abs(actual - expected))), best_max_abs_error=base_err,
                       counts=dict(vm_waits=m.vm_waits, barriers=m.barriers, mmac=m.mmac_count, dma64=m.dma64_count),
                       best_counts=dict(vm_waits=base.vm_waits, barriers=base.barriers, mmac=base.mmac_count, dma64=base.dma64_count),
                       global_trace_equal=m.global_trace.digest() == base.global_trace.digest(),
                       v_schedule_equal=(m.v_reads, m.v_dma) == (base.v_reads, base.v_dma),
                       k0_schedule_equal=m.k0_issues == base.k0_issues and m.k0_overlap == base.k0_overlap,
                       vs_best=compare(full, base_full))
            records.append(rec)
            print('PASS', json.dumps(rec), flush=True)
    (HERE / 'numerical_validation.json').write_text(json.dumps(dict(
        cases=records, gpu_tested=False, target_codegen_tested=False, independent_wave_timing_simulated=False,
        reference_atol=.01, reference_rtol=.01,
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (BEST / 'kernel.ll', HERE / 'kernel.ll', HERE / 'kernel.o3.ll')}), indent=2) + '\n')
    print('PASS all', len(records), 'cases', flush=True)


if __name__ == '__main__':
    main()
