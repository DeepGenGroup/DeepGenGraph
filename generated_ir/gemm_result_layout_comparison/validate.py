#!/usr/bin/env python3
"""Execute both actual LLVM exports with the repository's delayed-DMA model."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / '3rd/deepgengraph/test'))
from check_online_softmax import OnlineBlock, reference

# LDS f32 and leader-store support now live in the shared repository model.

def main():
    records = {}
    names = sys.argv[1:] or ['layout_3_1', 'layout_3_3']
    for name in names:
        directory = HERE / name
        ir = directory / 'kernel.ll'
        subprocess.run(['/data2/xsl/install/bin/opt', '-passes=default<O3>,verify',
                        '-S', str(ir), '-o', str(directory / 'kernel.o3.ll')], check=True)
        text = re.sub(r'\bexact ', '', ir.read_text())
        cases = []
        for kind, query in [('random', 0), ('random', 1), ('random', 31),
                            ('positive_overflow', 1), ('negative_underflow', 1),
                            ('max_changes', 2)]:
            rng = np.random.default_rng(42)
            q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
            if kind == 'positive_overflow': q.fill(16); k.fill(16)
            elif kind == 'negative_underflow': q.fill(16); k.fill(-16)
            elif kind == 'max_changes':
                q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
            head, group = 1, 4
            forward = 31 - query
            linear = forward // group * (32 * group) + head * group + forward % group
            model = OnlineBlock(text, q, k, v, physical=(linear % 32, linear // 32))
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = model.run().reshape(q.shape)
            actual = full[head, query*128:(query+1)*128]
            expected = reference(q, k, v, head, query)
            finite = bool(np.isfinite(actual).all())
            passed = bool(np.allclose(actual, expected, atol=.01, rtol=.01))
            written = np.zeros(full.shape, dtype=bool)
            written[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            assert not model.pending
            record = dict(case=kind, query=query,
                          max_abs_error=float(np.max(np.abs(actual-expected))),
                          q_load_count_min=min(model.q_elements.values()),
                          q_load_count_max=max(model.q_elements.values()),
                          delayed_DMA_passed=True, finite=finite, passed=passed,
                          mismatches=int(np.count_nonzero(~np.isclose(actual, expected, atol=.01, rtol=.01))))
            cases.append(record)
            print('PASS' if passed else 'FAIL', name, json.dumps(record), flush=True)
        records[name] = dict(sha256=hashlib.sha256(ir.read_bytes()).hexdigest(),
                             llvm_o3_verified=True, cases=cases, gpu_tested=False,
                             numerical_passed=all(c['passed'] for c in cases))
        (directory / 'validation.json').write_text(json.dumps(records[name], indent=2)+'\n')
    return 0 if all(r['numerical_passed'] for r in records.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
