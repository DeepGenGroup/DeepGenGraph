#!/usr/bin/env python3
"""Execute delivered IR using the existing delayed-DMA attention interpreter.

Models conversion semantics, not gfx936 execution/rounding internals or timing.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lazy_cvt_model', HERE.parent/'noncausal_lazy_base/verify.py')
lazy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lazy)


def convert(values, rtz):
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = values.astype(np.float16)
        if rtz:
            overshot = np.isfinite(values) & (np.abs(rounded.astype(np.float32)) > np.abs(values))
            rounded = np.where(overshot, np.nextafter(rounded, np.float16(0)), rounded).astype(np.float16)
    return rounded


class Model(lazy.Model):
    def __init__(self, *args, **kwargs):
        self.pack_calls = 0
        super().__init__(*args, **kwargs)

    def call(self, body):
        if re.search(r'asm "v_cvt_', body):
            match = re.search(r'asm "([^"]+)", "([^"]+)"\((.*)\)', body)
            assert match, body
            asm, constraints, operands = match.groups()
            if asm.startswith('v_cvt_f16_f32 '):
                assert constraints == '=&v,v,v' and 'dst_unused:UNUSED_PRESERVE' in asm
            else:
                assert constraints == '=v,v,v'
            a, b = map(self.typed, lazy.split_args(operands))
            values = np.stack([a, b], axis=1).astype(np.float32)
            assert np.isfinite(values).all() and (values >= 0).all()
            self.max_w = max(self.max_w, float(values.max()))
            halves = convert(values, 'pkrtz' in asm).view(np.uint16).astype(np.uint32)
            self.pack_calls += 1
            return (halves[:, 0] | (halves[:, 1] << 16)).astype(np.int64)
        return super().call(body)


def conversion_edges():
    # Every positive finite half, plus its midpoint and adjacent FP32 values.
    h = np.arange(0x7bff, dtype=np.uint16).view(np.float16).astype(np.float32)
    up = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
    mid = (h + up) * np.float32(.5)
    values = np.concatenate([h, mid, np.nextafter(mid, np.float32(0)),
                             np.nextafter(mid, np.float32(np.inf)),
                             np.array([65504, 65520, 65536, 1e10], np.float32)])
    values = np.concatenate([values, -values])
    # An independent RTZ oracle: binary-search representable positive halves.
    table = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
    indices = np.searchsorted(table, np.abs(values), side='right') - 1
    expected = table[indices].astype(np.float16)
    expected = np.copysign(expected, values).astype(np.float16)
    np.testing.assert_array_equal(convert(values, True).view(np.uint16), expected.view(np.uint16))
    rne = convert(values, False).view(np.uint16)
    # Explicit tie-even check for all adjacent finite positive half values.
    want_mid = np.arange(len(h), dtype=np.uint16)
    want_mid += want_mid & 1
    np.testing.assert_array_equal(convert(mid, False).view(np.uint16), want_mid)
    assert np.any(rne != convert(values, True).view(np.uint16)), 'RTZ must not masquerade as RNE'
    return dict(finite_values_tested=len(values), all_positive_half_midpoints_checked=True,
                independent_rtz_oracle_passed=True)


def main():
    edges = conversion_edges()
    manifest = json.loads((HERE/'manifest.json').read_text())
    texts = {}
    for name, record in manifest['variants'].items():
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            path = HERE/(name+suffix+'.ll')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record[key]
            texts[name+suffix] = lazy.previous.helper.model_text(path.read_text())
    records = []
    # Exercise all existing lazy-base edge cases; the non-lazy family is also
    # run on random data. O3 variants are executed on the first random case.
    for case_index, (kind, head, query) in enumerate(lazy.CASES):
        arrays = lazy.dataset(kind, 42+query)
        expected = lazy.previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        for base in (('control', 'lazy_t8') if case_index == 0 else ('lazy_t8',)):
            names = [base] + [base+'_'+m for m in ('pk_rne', 'sdwa_rne', 'pk_rtz')]
            if case_index == 0:
                names += [n+'.o3' for n in names[1:]]
            original = original_full = None
            for name in names:
                model = Model(texts[name], *arrays, threads=256, physical=(query, head))
                full = model.run().reshape(arrays[0].shape)
                actual = full[head, query*128:(query+1)*128]
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
                if original is None:
                    original, original_full = model, full.copy()
                rtz = '_pk_rtz' in name
                if not rtz:
                    assert np.array_equal(full, original_full, equal_nan=True), (kind, name)
                assert model.read_records == original.read_records
                assert model.lds_values.digest() == original.lds_values.digest()
                assert model.global_trace.digest() == original.global_trace.digest()
                assert (model.v_reads, model.v_dma) == (original.v_reads, original.v_dma)
                assert model.k0_issues == original.k0_issues
                assert np.array_equal(model.slow_counts, original.slow_counts)
                assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
                assert not model.pending and not model.k0_pending
                assert model.pack_calls == (0 if name == base else 32*32)
                assert model.max_w <= (256 if base == 'lazy_t8' else 1)*(1+1e-6)
                diff = np.max(np.abs(actual.astype(np.float32)-original_full[head, query*128:(query+1)*128].astype(np.float32)))
                record = dict(case=kind, head=head, query=query, variant=name,
                    max_abs_error=float(np.max(np.abs(actual-expected))),
                    max_abs_diff_from_baseline=float(diff),
                    bitwise_equal_to_baseline=bool(np.array_equal(full, original_full, equal_nan=True)),
                    memory_and_schedule_audit_passed=True, pack_calls=model.pack_calls)
                records.append(record)
                print('PASS', kind, name, 'error', record['max_abs_error'], 'baseline_diff', float(diff), flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records, conversion_edges=edges,
        sha256={name: record['sha256'] for name, record in manifest['variants'].items()},
        reference_atol=.01, reference_rtol=.01, gpu_tested=False, target_codegen_tested=False,
        independent_wave_timing_simulated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
