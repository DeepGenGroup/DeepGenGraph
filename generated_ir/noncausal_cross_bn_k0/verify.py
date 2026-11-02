#!/usr/bin/env python3
"""Execute delivered IR with delayed DMA, byte ownership, and trace audits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tail_checks', HERE.parent / 'noncausal_qk_tail_prefetch/verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
helper = previous.helper


class Model(previous.Model):
    def __init__(self, *args, **kwargs):
        self.expected_head = kwargs['physical'][1]
        self.k0_pending = False
        self.k0_issues = []
        self.k0_overlap = 0
        super().__init__(*args, **kwargs)

    def call(self, body):
        if 's_waitcnt vmcnt(0)' in body:
            self.k0_pending = False
        if 'v_mmac_' in body and self.k0_pending:
            assert self.pending
            assert self.mmac_count % 256 >= 224, 'K0 overlaps outside final PV stage'
            self.k0_overlap += 1
        return super().call(body)

    def memory(self, ptr, ty, write=None):
        live = ptr[self.active]
        if write is None and live.size and np.all(live >> 32 == 2):
            elements = (live - (2 << 32)) // 2
            if np.all(elements % 128 < 32):
                # The address is formed with OR: an illegal tile 32 can alias
                # tile 0 of an odd head instead of exceeding the allocation.
                assert len(self.k0_issues) < 32 * 4, 'extra K0 DMA past final tile'
                assert np.all(elements // (4096 * 128) == self.expected_head), 'K0 DMA out of bounds for head'
                tile = elements // 128 % 4096 // 128
                assert np.all(tile == tile[0])
                assert int(tile[0]) == len(self.k0_issues) // 4, 'K0 tile sequence changed'
                self.k0_issues.append((int(tile[0]), self.mmac_count))
                self.k0_pending = True
        # The inherited audit marks LDS writes at DMA issue (not only at wait),
        # so DMA cannot be issued before all waves release the previous reader.
        return super().memory(ptr, ty, write)


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    texts = {}
    for name, info in manifest['variants'].items():
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            path = HERE / (name + suffix + '.ll')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info[key]
            texts[name + suffix] = helper.model_text(path.read_text())
    records = []
    cases = [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2),
             ('future_key_dominates', 1, 0), ('positive_overflow', 1, 1), ('negative_underflow', 1, 1)]
    for kind, head, query in cases:
        rng = np.random.default_rng(42 + query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        q, k, v = arrays
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
        elif kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:, -128:] = 1; v.fill(0); v[:, -128:] = 1
        elif kind in ('positive_overflow', 'negative_underflow'):
            q.fill(16); k.fill(16 if kind == 'positive_overflow' else -16)
        expected = helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        names = list(manifest['variants'])
        if (kind, head, query) == cases[0]:
            names += ['k0_prefetch_pv_tail.o3', 'k0_late_control.o3']
        baseline = baseline_model = None
        for name in names:
            model = Model(texts[name], *arrays, threads=256, physical=(query, head))
            full = model.run().reshape(q.shape)
            actual = full[head, query * 128:(query + 1) * 128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            early = name.startswith('k0_prefetch_pv_tail')
            want = [(tile, tile * 256 - (32 if early and tile else 0))
                    for tile in range(32) for _ in range(4)]
            assert model.k0_issues == want, (name, model.k0_issues)
            assert model.k0_overlap == (31 * 32 if early else 0)
            assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
            assert not model.pending and not model.k0_pending
            assert model.v_reads_with_pending_dma == (1016 if early else 768)
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert np.isnan(full).sum() == full.size - 16384
            if baseline is None:
                baseline, baseline_model = full.copy(), model
            assert np.array_equal(full, baseline, equal_nan=True), name
            assert model.read_records == baseline_model.read_records, 'LDS addresses/values/epochs'
            assert model.global_trace.digest() == baseline_model.global_trace.digest(), 'global trace'
            assert model.lds_values.digest() == baseline_model.lds_values.digest(), 'ordered LDS values'
            assert (model.v_reads, model.v_dma) == (baseline_model.v_reads, baseline_model.v_dma), 'V schedule'
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                bitwise_equal_to_control=True, global_trace_and_LDS_values_epochs_equal=True,
                V_schedule_equal=True, K0_overlapped_MMAC=model.k0_overlap,
                max_abs_error=float(np.max(np.abs(actual - expected))),
                delayed_DMA_and_ownership_passed=True))
            print('PASS', kind, head, query, name, 'K0 overlap', model.k0_overlap, flush=True)

    good = texts['k0_prefetch_pv_tail']
    anchor = good.index('; V2 readers released low LDS')
    barrier = good.rfind('  call void @llvm.amdgcn.s.barrier()', 0, anchor)
    assert barrier >= 0
    missing_barrier = good[:barrier] + good[good.index('\n', barrier) + 1:]
    body = good.index('\nbb95:')
    wait = re.search(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n', good[body:], re.M)
    assert wait
    missing_wait = good[:body + wait.start()] + good[body + wait.end():]
    unguarded = good.replace('br i1 %k0.has.next, label %k0.issue, label %k0.join', 'br label %k0.issue')
    negatives = []
    for label, text, reason in [('missing_V3_handoff', missing_barrier, 'reader barrier'),
                                ('missing_K0_wait', missing_wait, 'DMA'),
                                ('missing_final_guard', unguarded, 'extra K0 DMA past final tile')]:
        try:
            Model(text, *arrays, threads=256, physical=(31, 1)).run()
        except AssertionError as error:
            assert reason in str(error), (label, str(error))
            negatives.append(dict(case=label, detected=str(error)))
            print('PASS negative', label, str(error), flush=True)
        else:
            raise AssertionError('negative control unexpectedly passed: ' + label)
    (HERE / 'validation.json').write_text(json.dumps(dict(cases=records, negative_controls=negatives,
        source_sha256=manifest['source_sha256'], variants=manifest['variants'],
        reference_atol=.01, reference_rtol=.01, gpu_tested=False,
        target_codegen_tested=False, independent_wave_timing_simulated=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
