#!/usr/bin/env python3
"""Execute the exported causal IR, including delayed DMA/LDS ownership audits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


causal = module('causal_model', HERE.parent/'causal_best_mytest/verify.py')
pack = module('pack_model', HERE.parent/'noncausal_cvt_pack/verify.py')


class Model(causal.Model):
    def __init__(self, *args, **kwargs):
        self.pack_calls = 0
        super().__init__(*args, **kwargs)

    def call(self, body):
        if 'asm "v_cvt_pkrtz_f16_f32 ' in body:
            match = re.search(r'asm "[^"]+", "=v,v,v"\((.*)\)', body)
            assert match, body
            a, b = map(self.typed, causal.split_args(match[1]))
            values = np.stack([a, b], axis=1).astype(np.float32)
            assert np.isfinite(values).all() and (values >= 0).all()
            halves = pack.convert(values, True).view(np.uint16).astype(np.uint32)
            self.pack_calls += 1
            return (halves[:, 0] | (halves[:, 1] << 16)).astype(np.int64)
        return super().call(body)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    texts = {}
    for name, record in manifest['variants'].items():
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            path = HERE/(name+suffix+'.ll')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record[key]
            texts[name+suffix] = causal.previous.helper.model_text(path.read_text())
    edges = pack.conversion_edges()
    records = []
    cases = [('random',0), ('random',8), ('random',31), ('max_changes',2),
             ('diagonal_future_dominates',0), ('positive_overflow',1), ('negative_underflow',1)]
    for kind, query in cases:
        arrays = [np.random.default_rng(42+query+s).normal(size=(2,4096,128)).astype(np.float16) for s in range(3)]
        q, k, v = arrays
        head = 1
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:,:128]=.25; k[:,128:256]=.5
        elif kind == 'diagonal_future_dominates':
            q.fill(1); k[:] = (np.arange(4096,dtype=np.float32)/8)[None,:,None]
            v[:] = (np.arange(4096,dtype=np.float32)/128)[None,:,None]
        elif kind in ('positive_overflow','negative_underflow'):
            q.fill(16); k.fill(16 if kind=='positive_overflow' else -16)
        expected = causal.previous.helper.kdma.vdb.reference(*arrays, head, query, causal=True)
        # Reverse logical query after grouping: physical forward index = 31-query.
        physical = causal.physical(4, head, 31-query)
        baseline = baseline_model = None
        names = list(manifest['variants'])
        if kind == 'random' and query in (0,31):
            names += [name+'.o3' for name in names]
        raw_outputs = {}
        for name in names:
            m = Model(texts[name], *arrays, threads=256, physical=physical,
                      logical_head=head, logical_query=query)
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = m.run().reshape(q.shape)
            actual = full[head,query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert np.isnan(full).sum() == full.size-16384
            if baseline is None:
                baseline, baseline_model = full.copy(), m
            assert m.global_trace.digest() == baseline_model.global_trace.digest()
            assert m.lds_values.digest() == baseline_model.lds_values.digest()
            assert (m.v_reads, m.v_dma, m.k0_issues) == (baseline_model.v_reads, baseline_model.v_dma, baseline_model.k0_issues)
            tiles = query+1
            assert m.k0_issues == [(t,t*256) for t in range(tiles) for _ in range(4)]
            assert m.k0_overlap == 0
            assert not m.pending and not m.k0_pending
            assert (m.mmac_count, m.dma64_count, m.vm_waits) == (256*tiles,16*tiles,4+8*tiles)
            assert m.pack_calls == (32*tiles if 'pk_rtz' in name else 0)
            if name.endswith('.o3'):
                assert np.array_equal(full, raw_outputs[name[:-3]], equal_nan=True)
            else:
                raw_outputs[name] = full.copy()
            records.append(dict(variant=name,case=kind,head=head,query=query,physical=physical,
                max_abs_error=float(np.max(np.abs(actual-expected))),
                max_abs_difference_from_off=float(np.max(np.abs(actual.astype(np.float32)-baseline[head,query*128:(query+1)*128].astype(np.float32)))),
                causal_kv_bounds_checked=True,delayed_dma_and_lds_ownership_checked=True,
                memory_and_schedule_equal_to_off=True,pack_calls=m.pack_calls,barriers=m.barriers))
            print('PASS',name,kind,query,'error',records[-1]['max_abs_error'],flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,conversion_edges=edges,
        reference_atol=.01,reference_rtol=.01,gpu_tested=False,target_codegen_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
