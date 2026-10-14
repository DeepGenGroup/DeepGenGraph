"""Numerical probes outside attention timing; no HIP dependency at import time."""
import hashlib
from pathlib import Path

import numpy as np
import torch


def dense_prefix_reference(q, k, v, rows=128, heads=2):
    """Independent FP32 reference for the first causal rows of selected heads."""
    q, k, v = [x[:1, :heads, :rows].float() for x in (q, k, v)]
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    mask = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(mask, -float("inf")), dim=-1) @ v


def report_prefix(label, actual, expected):
    actual = actual[:1, :expected.shape[1], :expected.shape[2]].float()
    error = (actual - expected).abs()
    passed = torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
    mismatch = ~torch.isclose(actual, expected, atol=1e-2, rtol=1e-2)
    mismatches = mismatch.sum().item()
    print(f"  {label}: allclose={passed}, max_abs={error.max().item():.8g}, "
          f"mean_abs={error.mean().item():.8g}, mismatch={mismatches}/{actual.numel()}")
    if not passed:
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        finite_error = error[finite]
        finite_max = finite_error.max().item() if finite_error.numel() else float("nan")
        finite_mean = finite_error.mean().item() if finite_error.numel() else float("nan")
        print(f"    actual_nan={torch.isnan(actual).sum().item()}, "
              f"actual_inf={torch.isinf(actual).sum().item()}, "
              f"expected_nonfinite={(~torch.isfinite(expected)).sum().item()}, "
              f"finite_mismatch={(mismatch & finite).sum().item()}/{finite.sum().item()}, "
              f"finite_max_abs={finite_max:.8g}, finite_mean_abs={finite_mean:.8g}")
        # Keep diagnostics bounded, while exposing row/column patterns hidden by NaN reductions.
        for name, mask in (("nonfinite", ~torch.isfinite(actual)), ("mismatch", mismatch)):
            locations = mask.nonzero(as_tuple=False)
            if locations.numel():
                print(f"    first {name} BHSD positions (up to 8): {locations[:8].tolist()}")
                for head in range(mask.shape[1]):
                    counts = mask[0, head].sum(dim=-1)
                    rows = counts.nonzero(as_tuple=False).flatten()
                    pairs = [(row, counts[row].item()) for row in rows[:12].tolist()]
                    print(f"    {name} head={head}: affected_rows={rows.numel()}, "
                          f"first row:column_count pairs={pairs}")
    return passed


def row_one_weights(q, k, v, actual):
    """Infer the two-key mixture from all output columns, not a single value."""
    if q.shape[-2] < 2:
        return
    a, b = v[0, 0, 0].float(), v[0, 0, 1].float()
    delta = a - b
    norm = delta.square().sum()
    if norm.item() == 0:
        return
    observed = actual[0, 0, 1].float()
    weight = ((observed - b) * delta).sum() / norm
    fit = b + weight * delta
    scores = q[0, 0, 1].float() @ k[0, 0, :2].float().T / q.shape[-1] ** 0.5
    expected = torch.softmax(scores, dim=-1)[0]
    uniform_error = (observed - (a + b) * .5).abs().max()
    print(f"  row 1 inferred p(V0)={weight.item():.8g}, expected={expected.item():.8g}, "
          f"mixture_fit_max_abs={(observed - fit).abs().max().item():.8g}, "
          f"uniform_mean_max_abs={uniform_error.item():.8g}")
    if torch.isfinite(weight).item() and 0 < weight.item() < 1:
        inferred_delta = torch.log(weight / (1 - weight)).item()
        print(f"  row 1 inferred score0-score1={inferred_delta:.8g}, "
              f"expected={(scores[0] - scores[1]).item():.8g} "
              "(inference assumes a normalized two-value mixture)")


def diagnose_attention(q, k_bhsd, v, actual, sdpa, launch, hsaco, grid, block):
    """Run controlled inputs with unchanged launch geometry and logical ABI.

    `launch(q, k_bhsd, v)` returns a fresh output. The caller handles physical K
    storage. These prefix probes never change the original benchmark verdict.
    """
    print("\nAttention diagnostics (outside timing; first 128 rows, up to 2 heads):")
    expected = dense_prefix_reference(q, k_bhsd, v)
    report_prefix("original HSACO vs independent dense reference", actual, expected)
    report_prefix("SDPA vs independent dense reference", sdpa, expected)
    row_one_weights(q, k_bhsd, v, actual)

    prefix = {name: tensor[:1, :1, :128].detach().cpu().numpy()
              for name, tensor in dict(q=q, k=k_bhsd, v=v, actual=actual, sdpa=sdpa).items()}
    snapshot = Path(str(hsaco) + ".prefix.npz")
    digest = hashlib.sha256(Path(hsaco).read_bytes()).hexdigest()
    np.savez_compressed(snapshot, **prefix, hsaco_sha256=digest,
                        full_shape_bhsd=np.array(q.shape), grid=np.array(grid), block=np.array(block))
    print(f"  Saved original inputs/output prefix: {snapshot} (HSACO sha256={digest})")

    # Zero Q makes every causal score equal: output must be the prefix mean
    # of V, independent of K. One active Q dimension then isolates reduction
    # dimensions without assuming the compiler's actual BK tile size.
    probes = [("zero Q / causal mean(V)", None)]
    probes += [(f"Q only dimension {dim}", dim)
               for dim in sorted({0, q.shape[-1] // 4, q.shape[-1] // 2,
                                  3 * q.shape[-1] // 4})]
    results = {}
    probe_snapshot = Path(str(hsaco) + ".probes.npz")
    probe_data = {name: tensor[:1, :2, :128].detach().cpu().numpy().copy()
                  for name, tensor in dict(k=k_bhsd, v=v).items()}
    probe_data.update(hsaco_sha256=digest, full_shape_bhsd=np.array(q.shape),
                      grid=np.array(grid), block=np.array(block), probe_q_value=4)
    for label, dim in probes:
        probe_q = torch.zeros_like(q)
        if dim is not None:
            probe_q[..., dim] = 4
        probe_out = launch(probe_q, k_bhsd, v)
        reference = dense_prefix_reference(probe_q, k_bhsd, v)
        results[label] = report_prefix(label, probe_out, reference)
        key = "zero_q" if dim is None else f"q_dim_{dim}"
        probe_data[key + "_actual"] = probe_out[:1, :2, :128].detach().cpu().numpy().copy()
        probe_data[key + "_expected"] = reference.detach().cpu().numpy().copy()
        # Save after each launch so completed probes survive a later diagnostic failure.
        np.savez_compressed(probe_snapshot, **probe_data)
    print(f"  Saved probe outputs/references and K/V prefix: {probe_snapshot}")
    print("  Probe results are diagnostic only; the original correctness result is unchanged.")
    return results
