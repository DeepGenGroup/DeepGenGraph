#!/usr/bin/env python3
"""Execute exported BM=128 online-attention LLVM with delayed DMA checks.

Usage: OPENBLAS_NUM_THREADS=1 python3 check_online_softmax.py kernel.ll --bn 128
For layout 3.1 without Q retention, add --q-loads per-tile. The random
multi-tile cases regress Exp2OpTiling's alpha row-layout mismatch: zero initial
state or row-uniform alpha alone cannot expose that bug.
The fixture uses BHSD=(1,32,4096,128), grouped reverse, 256 threads. This is a
CPU execution model of actual LLVM, not a replacement for gfx936 validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from check_pipeline_cleanup_attention import CountedDMA


class OnlineBlock(CountedDMA):
    def __init__(self, *args, physical, **kwargs):
        self.physical = physical
        self.q_elements = {}
        super().__init__(*args, **kwargs)

    def call(self, body):
        for axis, value in zip('xyz', (*self.physical, 0)):
            if f'@llvm.amdgcn.workgroup.id.{axis}(' in body:
                return np.full(self.t, value, dtype=np.int64)
        return super().call(body)

    def memory(self, ptr, ty, write=None):
        live = ptr[self.active]
        if write is None and live.size and np.all(live >> 32 == 1):
            for offset in range(0, self.size(ty), 2):
                for address in live + offset:
                    key = int(address)
                    self.q_elements[key] = self.q_elements.get(key, 0) + 1
        return super().memory(ptr, ty, write)


def reference(q, k, v, head, query):
    start, end = query*128, (query+1)*128
    scores = (q[head, start:end].astype(np.float64) @
              k[head, :end].astype(np.float64).T) / np.sqrt(128.)
    scores[np.arange(end)[None, :] > np.arange(start, end)[:, None]] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ v[head, :end].astype(np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ir', type=Path)
    p.add_argument('--bn', type=int, choices=(64, 128), default=128)
    p.add_argument('--group', type=int, choices=(4, 8, 16), default=8,
                   help='Query grouping used by the exported kernel (reverse=1)')
    p.add_argument('--output', type=Path)
    p.add_argument('--q-loads', choices=('once', 'per-tile'), default='once',
                   help='Expected Q loading: retained fragments or reload on each key tile')
    p.add_argument('--legacy', type=Path, help='Optional old IR, expected to fail the overflow case')
    a = p.parse_args()
    text = re.sub(r'\bexact ', '', a.ir.read_text())
    assert 'llvm.maxnum.' in text and 'fsub ' in text
    records = []
    kinds = [('random', 0), ('random', 1), ('random', 31),
             ('positive_overflow', 1), ('negative_underflow', 1), ('max_changes', 2)]
    for kind, query in kinds:
        rng = np.random.default_rng(42)
        q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        if kind == 'positive_overflow':
            q.fill(16)
            k.fill(16)
        elif kind == 'negative_underflow':
            q.fill(16)
            k.fill(-16)
        elif kind == 'max_changes':
            q.fill(1)
            k.fill(-16)
            k[:, :a.bn] = .25
            k[:, a.bn:2*a.bn] = .5
        head = 1
        forward = 31-query
        linear = forward//a.group*(32*a.group) + head*a.group + forward%a.group
        model = OnlineBlock(text, q, k, v, physical=(linear % 32, linear // 32))
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            full = model.run().reshape(q.shape)
        actual = full[head, query*128:(query+1)*128]
        expected = reference(q, k, v, head, query)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        written = np.zeros(full.shape, dtype=bool)
        written[head, query*128:(query+1)*128] = True
        assert np.isnan(full[~written]).all()
        assert not model.pending
        q_loads = 1 if a.q_loads == 'once' else (query+1)*128//a.bn
        want = {(1 << 32) + 2*((head*4096+row)*128+col): q_loads
                for row in range(query*128, (query+1)*128) for col in range(128)}
        assert model.q_elements == want, 'unexpected Q load coverage/count'
        record = dict(case=kind, head=head, query=query, bn=a.bn, group=a.group,
                      max_abs_error=float(np.max(np.abs(actual-expected))),
                      finite=True, full_Q_loaded_once=q_loads == 1,
                      q_loads_per_element=q_loads, delayed_DMA_passed=True)
        records.append(record)
        print('PASS', json.dumps(record), flush=True)
    legacy_failed = None
    if a.legacy:
        q.fill(16)
        k.fill(16)
        model = OnlineBlock(re.sub(r'\bexact ', '', a.legacy.read_text()), q, k, v,
                            physical=(15, 24))  # group8 reverse, head=1, query=0
        with np.errstate(all='ignore'):
            old = model.run().reshape(q.shape)[1, :128]
        assert not np.isfinite(old).all(), 'negative control unexpectedly stayed finite'
        legacy_failed = True
        print('PASS legacy overflow negative control', flush=True)
    output = a.output or a.ir.with_suffix('.online-validation.json')
    output.write_text(json.dumps(dict(ir=str(a.ir.resolve()),
        sha256=hashlib.sha256(a.ir.read_bytes()).hexdigest(), cases=records,
        legacy_overflow_reproduced=legacy_failed, gpu_tested=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
