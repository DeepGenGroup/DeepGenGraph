#!/usr/bin/env python3
"""Numerically compare the compiled pass and composed passes with the winner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cvt_pack_model', HERE.parent/'noncausal_cvt_pack/verify.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    paths = {'measured': HERE.parent/'noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll',
             'pass_pk_rtz': HERE/'pass_pk_rtz.o3.ll',
             'pipeline_t8_pk_rtz': HERE/'pipeline_t8_pk_rtz.o3.ll'}
    for name, path in paths.items():
        if name != 'measured':
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['variants'][name]['o3_sha256']
    texts = {name: pack.lazy.previous.helper.model_text(path.read_text()) for name, path in paths.items()}
    records = []
    for kind, head, query in [('random', 0, 0), ('one_wave_triggers', 1, 1), ('large_jump', 1, 1)]:
        arrays = pack.lazy.dataset(kind, 42+query)
        expected = pack.lazy.previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        original = original_full = None
        for name, text in texts.items():
            model = pack.Model(text, *arrays, threads=256, physical=(query, head))
            full = model.run().reshape(arrays[0].shape)
            actual = full[head, query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            if original is None:
                original, original_full = model, full.copy()
            assert np.array_equal(full, original_full, equal_nan=True), (kind, name)
            assert model.read_records == original.read_records
            assert model.lds_values.digest() == original.lds_values.digest()
            assert model.global_trace.digest() == original.global_trace.digest()
            assert (model.v_reads, model.v_dma) == (original.v_reads, original.v_dma)
            assert model.k0_issues == original.k0_issues
            assert np.array_equal(model.slow_counts, original.slow_counts)
            assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
            assert not model.pending and not model.k0_pending
            assert model.pack_calls == 1024
            records.append(dict(case=kind, variant=name, bitwise_equal_to_measured=True,
                memory_and_schedule_audit_passed=True,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS', kind, name, flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,
        sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        gpu_tested=False, target_codegen_tested=False, reference_atol=.01, reference_rtol=.01), indent=2)+'\n')


if __name__ == '__main__':
    main()
