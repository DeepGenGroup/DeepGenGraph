#!/usr/bin/env python3
"""CPU-validate kernel.ll and kernel.o3.ll with the repository's wave/DMA model.

Uses the same interpreter and FP64 reference as async_task_noncausal/validate.py,
plus a stronger baseline comparison and negative controls. The model audits LDS
ownership by barrier epoch. It does not model independent wave timing, register
allocation, or GPU performance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cross_bn_model', HERE.parent / 'noncausal_cross_bn_k0/verify.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

BASELINE_COUNTS = (260, 293, 8192, 512)  # vm waits, barriers, MMAC, 64-bit DMA
ELIDED_BARRIERS = 32                     # one per iteration
CASES = [('random', 0, 0), ('random', 1, 31), ('random', 1, 17), ('max_changes', 1, 2),
         ('future_key_dominates', 1, 0), ('positive_overflow', 1, 1), ('negative_underflow', 1, 1)]


def inputs(kind, query):
    rng = np.random.default_rng(42 + query)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    q, k, v = arrays
    if kind == 'max_changes':
        q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
    elif kind == 'future_key_dominates':
        q.fill(1); k.fill(-1); k[:, -128:] = 1; v.fill(0); v[:, -128:] = 1
    elif kind in ('positive_overflow', 'negative_underflow'):
        q.fill(16); k.fill(16 if kind == 'positive_overflow' else -16)
    return arrays


def execute(text, arrays, head, query):
    m = model.Model(text, *arrays, threads=256, physical=(query, head))
    return m, m.run().reshape(arrays[0].shape)


def check_variant(name, m, full, base, base_full, expected, head, query):
    actual = full[head, query * 128:(query + 1) * 128]
    assert np.isfinite(actual).all(), name
    np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
    assert np.array_equal(full, base_full, equal_nan=True), name + ': output differs from baseline'
    assert not m.pending and not m.k0_pending, name
    # Same memory behaviour. Only barrier epochs may renumber, so read_records,
    # which hash the epoch, are deliberately not compared.
    assert m.global_trace.digest() == base.global_trace.digest(), name + ': global trace'
    assert m.lds_values.digest() == base.lds_values.digest(), name + ': ordered LDS values'
    assert (m.v_reads, m.v_dma) == (base.v_reads, base.v_dma), name + ': V schedule'
    assert m.k0_issues == base.k0_issues and m.k0_overlap == base.k0_overlap == 992, name + ': K0 schedule'
    assert m.v_reads_with_pending_dma == base.v_reads_with_pending_dma, name
    assert len(m.q_elements) == 16384 and set(m.q_elements.values()) == {1}, name
    assert (m.vm_waits, m.barriers, m.mmac_count, m.dma64_count) == (
        BASELINE_COUNTS[0], BASELINE_COUNTS[1] - ELIDED_BARRIERS, BASELINE_COUNTS[2], BASELINE_COUNTS[3]), name
    return float(np.max(np.abs(actual - expected)))


def remove_line(text, needle, position):
    """Delete one occurrence of a full line. position indexes the matches."""
    lines = text.split('\n')
    matches = [i for i, line in enumerate(lines) if needle in line]
    del lines[matches[position]]
    return '\n'.join(lines)


def loop_line(text, needle, label, position=0):
    """Find the nth occurrence of needle after a block label and delete it."""
    start = text.index('\n' + label + ':')
    offset = text[:start].count(needle)
    return remove_line(text, needle, offset + position)


def expect_rejected(label, text, arrays, reason, run=None):
    try:
        run(text) if run else execute(text, arrays, 1, 31)
    except AssertionError as error:
        assert reason in str(error), (label, str(error))
        print('PASS negative', label, '->', error, flush=True)
        return dict(case=label, detected=str(error))
    raise AssertionError('negative control unexpectedly passed: ' + label)


def main():
    texts = {name: model.helper.model_text((HERE / (name + '.ll')).read_text())
             for name in ('baseline', 'kernel', 'kernel.o3')}
    records = []
    for kind, head, query in CASES:
        arrays = inputs(kind, query)
        expected = model.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        base, base_full = execute(texts['baseline'], arrays, head, query)
        assert (base.vm_waits, base.barriers, base.mmac_count, base.dma64_count) == BASELINE_COUNTS
        for name in ('kernel', 'kernel.o3'):
            m, full = execute(texts[name], arrays, head, query)
            error = check_variant(name, m, full, base, base_full, expected, head, query)
            records.append(dict(case=kind, head=head, query=query, variant=name, max_abs_error=error,
                                bitwise_equal_to_baseline=True, global_trace_equal=True, lds_values_equal=True,
                                v_schedule_equal=True, k0_schedule_equal=True, barriers=m.barriers,
                                baseline_barriers=base.barriers, vm_waits=m.vm_waits, mmac_count=m.mmac_count,
                                k0_overlap_mmac=m.k0_overlap))
            print('PASS', records[-1], flush=True)

    # The audit must catch the bugs this change could have introduced. Each case
    # breaks only the synchronization that the elision relies on.
    arrays = inputs('random', 31)
    good = texts['kernel']
    barrier, wait = 'call void @llvm.amdgcn.s.barrier()', 's_waitcnt vmcnt(0)'
    body = _body_label(good)
    negatives = [
        expect_rejected('head_barrier_removed', loop_line(good, barrier, body), arrays, 'barrier'),
        expect_rejected('head_wait_removed', loop_line(good, wait, body), arrays, 'DMA'),
    ]
    # What the MLIR stage sees: no V double buffering and no K0 prefetch, so K0
    # and V3 share an LDS slot. Deleting the tail barrier there must be detected,
    # which is why the elision cannot run on MLIR.
    no_split = model.helper.model_text((HERE / 'no_lds_split.ll').read_text())
    # The K0-tracking subclass assumes K0 prefetch, so use its parent model here.
    def audit(text):
        return model.helper.kdma.Model(text, *arrays, threads=256, physical=(31, 1)).run()
    audit(no_split)  # Intact, it must pass, or the control proves nothing.
    tail = len([line for line in no_split.split('\n') if barrier in line]) - 1
    negatives.append(expect_rejected('tail_barrier_removed_without_lds_slot_split',
                                     remove_line(no_split, barrier, tail), arrays, 'reader barrier', audit))
    (HERE / 'numerical_validation.json').write_text(json.dumps(dict(
        cases=records, negative_controls=negatives, gpu_tested=False, target_codegen_tested=False,
        independent_wave_timing_simulated=False, reference_atol=.01, reference_rtol=.01,
        sha256={name: hashlib.sha256((HERE / (name + '.ll')).read_bytes()).hexdigest()
                for name in ('baseline', 'kernel', 'kernel.o3')}), indent=2) + '\n')
    print('PASS all', len(records), 'cases and', len(negatives), 'negative controls', flush=True)


def _body_label(text):
    """The block holding the K0 guard's branch is the loop body."""
    lines = text.split('\n')
    guard = next(i for i, line in enumerate(lines) if line.startswith('  br i1 %k0.has.next'))
    for line in reversed(lines[:guard]):
        if line and not line[0].isspace() and ':' in line:
            return line.split(':')[0]
    raise AssertionError('loop body label not found')


if __name__ == '__main__':
    main()
