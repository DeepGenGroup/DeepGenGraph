#!/usr/bin/env python3
"""Check exact output/traffic and cross-wave LDS hazards for K0 rescheduling."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('prior_prefetch_audits',
    ROOT/'generated_ir/retained_cross_bn_k0/verify.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sys.path.insert(0, str(ROOT/'3rd/deepgengraph/test'))
from check_online_softmax import reference


def main():
    texts = {n: re.sub(r'\bexact ', '', (HERE/(n+'.ll')).read_text())
             for n in ('baseline', 'k0_late_control', 'k0_prefetch')}
    rng = np.random.default_rng(73)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    head, query = 1, 31
    forward = 31-query
    linear = forward//4*128+head*4+forward%4
    physical = (linear % 32, linear // 32)
    models, outputs, records = {}, {}, {}
    for name, text in texts.items():
        model = prior.PrefetchBlock(text, *arrays, physical=physical)
        full = model.run().reshape(arrays[0].shape)
        models[name], outputs[name] = model, full
        assert not model.pending and not model.k0_pending
        assert model.read_calls == dict(Q0=2, Qrest=6, K=256, V=256)
        assert model.vm_waits == 260 and model.barriers == 389
        assert model.mmac_count == 8192
        early = name == 'k0_prefetch'
        want = [dict(tile=t, mmac_before=256*t-(32 if early and t else 0))
                for t in range(32) for packet in range(2)]
        assert model.k0_issues == want
        assert model.k0_overlap == (992 if early else 0)
        assert np.array_equal(full, outputs['baseline'], equal_nan=True)
        assert model.global_trace.digest() == models['baseline'].global_trace.digest()
        assert model.lds_trace.digest() == models['baseline'].lds_trace.digest()
        expected = reference(*arrays, head, query, causal=False)
        actual = full[head, query*128:(query+1)*128]
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        records[name] = dict(bitwise_equal=True, ordered_global_and_lds_traces_equal=True,
            cross_wave_barrier_epoch_audit_passed=True, no_pending_dma=True,
            vm_waits=model.vm_waits, barriers=model.barriers,
            overlap_mmac=model.k0_overlap, max_abs_error=float(np.max(np.abs(actual-expected))))
        print('PASS', name, json.dumps(records[name]), flush=True)

    text = texts['k0_prefetch']
    marker = text.index('; V3 readers complete')
    handoff = re.search(r'^  call void @llvm.amdgcn.s.barrier\(\)[^\n]*\n', text[marker:], re.M)
    start, end = marker+handoff.start(), marker+handoff.end()
    missing_handoff = text[:start]+text[end:]
    body = text.index('\nnc.n134:')
    wait = re.search(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n', text[body:], re.M)
    start, end = body+wait.start(), body+wait.end()
    missing_wait = text[:start]+text[end:]
    unguarded = text.replace('br i1 %k0.has.next, label %k0.issue, label %k0.join',
                            'br label %k0.issue')
    negatives = []
    for tag, bad in [('missing_reader_handoff', missing_handoff),
                     ('missing_k0_wait', missing_wait), ('missing_final_guard', unguarded)]:
        try:
            broken = prior.PrefetchBlock(bad, *arrays, physical=physical)
            broken.run()
            assert not broken.pending, 'extra DMA at kernel exit'
        except (AssertionError, IndexError) as exc:
            reason = str(exc)
            assert any(s in reason for s in ('DMA', 'cross-wave', 'out of bounds')), reason
            negatives.append(dict(case=tag, rejected=reason))
            print('PASS negative', tag, reason, flush=True)
        else:
            raise AssertionError('negative control passed: '+tag)
    (HERE/'prefetch-audit.json').write_text(json.dumps(dict(cases=records,
        negative_controls=negatives, gpu_tested=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
