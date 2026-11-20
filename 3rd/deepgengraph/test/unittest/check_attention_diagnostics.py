#!/usr/bin/env python3
"""Check diagnostic references and probes against independent prefix sums."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from attentionDiagnostics import dense_prefix_reference, diagnose_attention, report_prefix

torch.set_num_threads(1)
torch.manual_seed(37)
for depth in (32, 128):
    q, k, v = [torch.randn(1, 2, 128, depth).half() for _ in range(3)]
    baseline = dense_prefix_reference(q, k, v).half()
    calls = []

    def launch(probe_q, probe_k, probe_v):
        # Q is the same in every row, so causal weighted prefix sums provide
        # a reference without reusing the dense GEMM/mask implementation.
        active = torch.nonzero(probe_q[0, 0, 0], as_tuple=False).flatten()
        assert active.numel() <= 1
        assert torch.equal(probe_q, probe_q[..., :1, :].expand_as(probe_q))
        calls.append(active.tolist())
        scores = torch.zeros_like(probe_k[..., 0], dtype=torch.float32)
        if active.numel():
            dim = active.item()
            scores = probe_k[..., dim].float() * 4 / depth ** .5
        weights = scores.exp()
        output = (weights[..., None] * probe_v.float()).cumsum(dim=-2)
        return (output / weights.cumsum(dim=-1)[..., None]).half()

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / 'dummy.hsaco'
        artifact.write_bytes(b'CPU-only diagnostic test')
        with contextlib.redirect_stdout(io.StringIO()):
            results = diagnose_attention(q, k, v, baseline, baseline, launch,
                                         artifact, (1, 1, 1), (256, 1, 1))
        assert len(results) == 5 and all(results.values()), results
        assert calls == [[], [0], [depth // 4], [depth // 2], [3 * depth // 4]], calls
        snapshot = np.load(str(artifact) + '.prefix.npz', allow_pickle=False)
        assert snapshot['q'].shape == (1, 1, 128, depth)
        np.testing.assert_array_equal(snapshot['actual'], baseline[:, :1].numpy())
        with np.load(str(artifact) + '.probes.npz', allow_pickle=False) as probes:
            assert probes['k'].shape == (1, 2, 128, depth)
            np.testing.assert_array_equal(probes['k'], k.numpy())
            for name in ['zero_q'] + [f'q_dim_{d}' for d in (0, depth // 4, depth // 2, 3 * depth // 4)]:
                np.testing.assert_allclose(probes[name + '_actual'], probes[name + '_expected'],
                                           atol=.01, rtol=.01)
    print(f'PASS zero/one-dimension probes, independent prefix references and snapshot, D={depth}')

# A NaN must not hide a separate finite error or turn an invalid output into PASS.
expected = torch.zeros(1, 2, 3, 4)
actual = expected.clone()
actual[0, 0, 1, 2] = float('nan')
actual[0, 1, 0, 0] = float('inf')
actual[0, 1, 2, 3] = 2
log = io.StringIO()
with contextlib.redirect_stdout(log):
    assert not report_prefix('mixed invalid output', actual, expected)
text = log.getvalue()
assert 'actual_nan=1, actual_inf=1, expected_nonfinite=0' in text, text
assert 'finite_mismatch=1/22, finite_max_abs=2' in text, text
assert '[0, 0, 1, 2]' in text and '[0, 1, 0, 0]' in text, text
assert 'mismatch head=1: affected_rows=2' in text, text
with contextlib.redirect_stdout(io.StringIO()):
    assert not report_prefix('all invalid', torch.full_like(expected, float('nan')), expected)
print('PASS nonfinite locations, finite-only errors and all-invalid rejection')
