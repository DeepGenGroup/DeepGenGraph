#!/usr/bin/env python3
"""Actual LLVM CPU execution with delayed DMA; not a gfx936 simulator."""
import json
import hashlib
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'3rd/deepgengraph/test'))
from check_online_softmax import OnlineBlock, reference


def main():
    names = ['control', 'group_prefetch2', 'group_prefetch2_protected']
    texts = {name: re.sub(r'\bexact ', '', (HERE/(name+'.ll')).read_text()) for name in names}
    records = []
    for kind, head, query in [('random', 0, 0), ('random', 1, 31), ('large_scores', 1, 1)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        if kind == 'large_scores':
            arrays[0] *= np.float16(8)
            arrays[1] *= np.float16(8)
        linear = query//4*128 + head*4 + query%4  # group4, reverse=0
        outputs, counts = {}, {}
        for name, text in texts.items():
            model = OnlineBlock(text, *arrays, threads=256, physical=(linear%32, linear//32))
            full = model.run().reshape(arrays[0].shape)
            assert not model.pending
            assert model.vm_waits == 260 and model.barriers == 389
            mask = np.zeros(full.shape, dtype=bool)
            mask[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~mask]).all()
            outputs[name] = full
            counts[name] = dict(vm_waits=model.vm_waits, barriers=model.barriers,
                                q_elements=len(model.q_elements))
            print(f'Executed {kind} h={head} q={query} {name}', flush=True)
        for name in names[1:]:
            assert np.array_equal(outputs['control'], outputs[name], equal_nan=True), name
        actual = outputs['group_prefetch2_protected'][head, query*128:(query+1)*128]
        expected = reference(*arrays, head, query, causal=False)
        assert np.isfinite(actual).all()
        # Same reference tolerance as the existing checker. Large random scores
        # expose the BASELINE's approximate arithmetic error too: report it
        # explicitly rather than treating bitwise-equivalent scheduling as an
        # accuracy fix, or silently relaxing the tolerance.
        close = np.isclose(actual, expected, atol=.01, rtol=.01)
        reference_ok = bool(close.all())
        if kind != 'large_scores':
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        records.append(dict(kind=kind, head=head, query=query, bitwise_equal=True,
                            reference_atol=.01, reference_rtol=.01,
                            reference_tolerance_passed=reference_ok,
                            reference_mismatch_elements=int((~close).sum()),
                            baseline_has_identical_reference_error=True,
                            max_abs_error=float(np.max(np.abs(actual-expected))), counts=counts))
        print('PASS', records[-1], flush=True)
    (HERE/'validation_protected.json').write_text(json.dumps(dict(cases=records,
        variants={name: hashlib.sha256((HERE/(name+'.ll')).read_bytes()).hexdigest() for name in names},
        all_cases_bitwise_equal_to_control=True,
        reference_limitations=[r['kind'] for r in records if not r['reference_tolerance_passed']],
        gpu_tested=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
