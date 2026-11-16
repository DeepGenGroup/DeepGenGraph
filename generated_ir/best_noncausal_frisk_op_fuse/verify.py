#!/usr/bin/env python3
"""Execute delivered raw/O3 LLVM with the existing delayed-DMA CPU model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cvt_model', HERE.parent / 'noncausal_cvt_pack/verify.py')
cvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvt)
records = []
for name in ('kernel.ll', 'kernel.o3.ll'):
    path = HERE / name
    text = cvt.lazy.previous.helper.model_text(path.read_text())
    for kind, head, query in [('random', 0, 0), ('future_key_dominates', 1, 0)]:
        arrays = cvt.lazy.dataset(kind, 42 + query)
        expected = cvt.lazy.previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        model = cvt.Model(text, *arrays, threads=256, physical=(query, head))
        full = model.run().reshape(arrays[0].shape)
        actual = full[head, query * 128:(query + 1) * 128]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        assert not model.pending and not model.k0_pending
        record = dict(file=name, case=kind, head=head, query=query,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      max_abs_error=float(np.max(np.abs(actual - expected))),
                      mmac_count=model.mmac_count, dma64_count=model.dma64_count)
        records.append(record)
        print('PASS', json.dumps(record), flush=True)
(HERE / 'validation.json').write_text(json.dumps(dict(cases=records,
    reference_atol=.01, reference_rtol=.01, gpu_tested=False,
    note='CPU delayed-DMA/LLVM model; not target GPU execution or timing'), indent=2) + '\n')
