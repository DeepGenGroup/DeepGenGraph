#!/usr/bin/env python3
"""在不同物理block上执行同一逻辑head/query，验证四组IR输出逐bit一致。"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'3rd/deepgengraph/test'))
from check_online_softmax import OnlineBlock, reference


def main():
    rng = np.random.default_rng(73)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    head, query = 1, 31  # 非零head、最后query，四种分组对应四个不同物理block。
    expected = reference(*arrays, head, query, causal=False)
    baseline = None
    records = []
    for group in (4, 8, 16, 32):
        path = HERE/f'group{group}.ll'
        text = re.sub(r'\bexact ', '', path.read_text())
        linear = query//group*(32*group) + head*group + query%group
        physical = (linear%32, linear//32)
        model = OnlineBlock(text, *arrays, threads=256, physical=physical)
        full = model.run().reshape(arrays[0].shape)
        assert not model.pending
        assert model.vm_waits == 260 and model.barriers == 389
        assert len(model.q_elements) == 16384
        mask = np.zeros(full.shape, dtype=bool)
        mask[head, query*128:(query+1)*128] = True
        assert np.isnan(full[~mask]).all()
        actual = full[head, query*128:(query+1)*128]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        if baseline is None:
            baseline = full.copy()
        else:
            assert np.array_equal(full, baseline, equal_nan=True)
        records.append(dict(group_size=group, physical=physical, head=head, query=query,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bitwise_equal_to_group4=True, max_abs_error=float(np.max(np.abs(actual-expected))),
                            vm_waits=model.vm_waits, barriers=model.barriers))
        print(f'PASS group{group}: physical={physical}, logical=(head={head}, query={query}), bitwise equal', flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,
        reference_atol=.01, reference_rtol=.01, gpu_tested=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
