#!/usr/bin/env python3
"""Numerical and delayed-DMA checks for the actual delivered LLVM IR."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'tail_softmax_model', HERE.parent / 'noncausal_softmax_fragments/verify.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Model(helper.kdma.Model):
    def __init__(self, *args, **kwargs):
        self.read_records = Counter()
        super().__init__(*args, **kwargs)

    def memory(self, ptr, ty, write=None):
        result = super().memory(ptr, ty, write)
        live = ptr[self.active]
        if write is None and live.size and np.all(live >> 32 == 0):
            # LDS order intentionally changes. Compare address AND value AND
            # synchronization epoch, with multiplicity, independent of order.
            h = hashlib.sha256()
            h.update(str(self.epoch).encode())
            h.update(ty.encode())
            h.update(live.tobytes())
            h.update(result[self.active].tobytes())
            self.read_records[h.hexdigest()] += 1
        return result


def main():
    records = []
    cases = [('random', 0, 0), ('random', 1, 31), ('positive_overflow', 1, 1),
             ('negative_underflow', 1, 1), ('max_changes', 1, 2), ('future_key_dominates', 1, 0)]
    for kind, head, query in cases:
        rng = np.random.default_rng(42 + query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        q, k, v = arrays
        if kind == 'positive_overflow':
            q.fill(16); k.fill(16)
        if kind == 'negative_underflow':
            q.fill(16); k.fill(-16)
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
        if kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:, -128:] = 1; v.fill(0); v[:, -128:] = 1
        expected = helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        baseline = reads = global_trace = v_schedule = None
        names = ['control'] + [f'k_tail_prefetch{d}' for d in (2, 4, 8)]
        if kind == 'random' and query == 0:
            names += [f'k_tail_prefetch{d}.o3' for d in (2, 4, 8)]
        for name in names:
            path = HERE / (name + '.ll')
            model = Model(helper.model_text(path.read_text()), *arrays,
                          threads=256, physical=(query, head))
            full = model.run().reshape(q.shape)
            actual = full[head, query * 128:(query + 1) * 128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            if baseline is None:
                baseline = full.copy()
                reads = model.read_records
                global_trace = model.global_trace.hexdigest()
                v_schedule = (model.v_reads, model.v_dma)
            assert np.array_equal(full, baseline, equal_nan=True), (kind, name)
            assert reads == model.read_records, (kind, name, 'LDS provenance/epoch')
            assert global_trace == model.global_trace.hexdigest()
            assert v_schedule == (model.v_reads, model.v_dma)
            assert not model.pending
            assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
            assert model.v_reads_with_pending_dma == 768
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert np.isnan(full).sum() == full.size - 16384
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                sha256=helper.kdma.digest(path), bitwise_equal_to_control=True,
                reference_atol=.01, reference_rtol=.01,
                max_abs_error=float(np.max(np.abs(actual - expected))),
                identical_LDS_addresses_values_and_epochs=True,
                identical_global_trace_and_V_schedule=True,
                delayed_DMA_and_ownership_passed=True))
            print('PASS', kind, head, query, name, flush=True)
    (HERE / 'validation.json').write_text(json.dumps(dict(cases=records,
        gpu_tested=False, independent_wave_timing_simulated=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
