#!/usr/bin/env python3
"""Actual-IR numeric and delayed-DMA checks for lifetime repair candidates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cross_bn_checks', HERE.parent / 'noncausal_cross_bn_k0/verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    texts = {}
    for name, info in manifest['variants'].items():
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            raw = (HERE / (name + suffix + '.ll')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == info[key]
            texts[name + suffix] = previous.helper.model_text(raw.decode())
    records = []
    cases = [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2), ('future_key_dominates', 1, 0)]
    for kind, head, query in cases:
        rng = np.random.default_rng(42 + query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        q, k, v = arrays
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
        elif kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:, -128:] = 1; v.fill(0); v[:, -128:] = 1
        expected = previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        names = list(manifest['variants'])
        if (kind, head, query) == cases[0]:
            names += [n + '.o3' for n in names if n != 'control']
        baseline = base_model = None
        for name in names:
            overlap = manifest['variants'][name.removesuffix('.o3')]['overlap']
            model = previous.Model(texts[name], *arrays, threads=256, physical=(query, head))
            full = model.run().reshape(q.shape)
            actual = full[head, query * 128:(query + 1) * 128]
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert np.isfinite(actual).all()
            assert model.k0_issues == [(t, t * 256 - (overlap if t else 0)) for t in range(32) for _ in range(4)]
            assert model.k0_overlap == overlap * 31
            assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
            assert not model.pending and not model.k0_pending
            assert model.v_reads_with_pending_dma == (1016 if overlap == 32 else 768)
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert np.isnan(full).sum() == full.size - 16384
            if baseline is None:
                baseline, base_model = full.copy(), model
            assert np.array_equal(full, baseline, equal_nan=True)
            assert model.read_records == base_model.read_records
            assert model.lds_values.digest() == base_model.lds_values.digest()
            assert model.global_trace.digest() == base_model.global_trace.digest()
            assert (model.v_reads, model.v_dma) == (base_model.v_reads, base_model.v_dma)
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                bitwise_equal=True, global_trace_LDS_values_and_epochs_equal=True,
                K0_overlap_MMAC=model.k0_overlap, delayed_DMA_ownership_passed=True,
                max_abs_error=float(np.max(np.abs(actual - expected)))))
            print('PASS', kind, head, query, name, 'overlap', model.k0_overlap, flush=True)
    # The original cross-BN verifier already covers handoff/wait negative
    # controls; verify the final guard again after relocating the issue block.
    bad = texts['k0_sum_anchor_remat_after16'].replace(
        'br i1 %k0.has.next, label %k0.issue, label %k0.join', 'br label %k0.issue')
    try:
        previous.Model(bad, *arrays, threads=256, physical=(31, 1)).run()
    except AssertionError as error:
        assert 'extra K0 DMA past final tile' in str(error)
        negative = str(error)
    else:
        raise AssertionError('missing last-tile guard not detected')
    (HERE / 'validation.json').write_text(json.dumps(dict(cases=records,
        negative_control=negative, variants=manifest['variants'],
        reference_atol=.01, reference_rtol=.01, gpu_tested=False,
        target_codegen_tested=False, independent_wave_timing_simulated=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
