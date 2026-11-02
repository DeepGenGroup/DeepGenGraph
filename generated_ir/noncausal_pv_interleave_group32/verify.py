#!/usr/bin/env python3
"""Execute actual LLVM with delayed DMA and compare outputs and memory traces.

This checks functional execution, not gfx936 timing or independent-wave
progress. The generator separately proves unchanged memory/sync order.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'3rd/deepgengraph/test'))
from check_online_softmax import OnlineBlock, reference


class AuditedBlock(OnlineBlock):
    def __init__(self, *args, **kwargs):
        self.memory_trace = hashlib.sha256()
        self.v_reads = []
        self.v_dma = []
        super().__init__(*args, **kwargs)

    def memory(self, ptr, ty, write=None):
        self.memory_trace.update(('R' if write is None else 'W').encode())
        self.memory_trace.update(ty.encode())
        self.memory_trace.update(np.asarray(ptr[self.active], dtype=np.int64).tobytes())
        return super().memory(ptr, ty, write)

    def call(self, body):
        if '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3(' in body:
            self.v_reads.append(self.mmac_count)
        if 'buffer_load_dwordx4' in body and '%dma.buffer.rsrc1349' in body:
            self.v_dma.append(self.mmac_count)
        return super().call(body)


def expected_schedule(name):
    reads = [tile*256+128+stage*32 for tile in range(32) for stage in range(4) for _ in range(8)]
    issues = [tile*256 + (96 if stage == 0 else 128+(stage-1)*32+4)
              for tile in range(32) for stage in range(4) for _ in range(2)]
    return reads, issues


def main():
    names = ['baseline_after4', 'pv_serial_guarded', 'pv_interleave2', 'pv_interleave4']
    texts = {n: re.sub(r'\bexact ', '', (HERE/(n+'.ll')).read_text()) for n in names}
    records = []
    for kind, head, query in [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        if kind == 'max_changes':
            arrays[0].fill(1)
            arrays[1].fill(-16)
            arrays[1][:, :128] = .25
            arrays[1][:, 128:256] = .5
        expected = reference(*arrays, head, query, causal=False)
        baseline_output = baseline_trace = None
        for name in names:
            model = AuditedBlock(texts[name], *arrays, threads=256, physical=(query, head))
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = model.run().reshape(arrays[0].shape)
            assert not model.pending
            assert model.vm_waits == 260 and model.barriers == 389
            assert model.overlapped_mma == 6784
            assert model.mmac_count == 8192 and len(model.q_elements) == 16384
            reads, issues = expected_schedule(name)
            assert model.v_reads == reads, (name, 'V read schedule', model.v_reads[:32], reads[:32])
            assert model.v_dma == issues, (name, 'V DMA schedule', model.v_dma[:8], issues[:8])
            mask = np.zeros(full.shape, dtype=bool)
            mask[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~mask]).all()
            actual = full[head, query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            trace = model.memory_trace.hexdigest()
            if name == 'baseline_after4':
                baseline_output, baseline_trace = full.copy(), trace
            assert np.array_equal(full, baseline_output, equal_nan=True)
            assert trace == baseline_trace, (name, 'ordered memory accesses changed')
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                bitwise_equal_to_baseline=True, ordered_memory_trace_sha256=trace,
                v_read_and_dma_schedule_verified=True, vm_waits=model.vm_waits,
                hardware_barriers=model.barriers, mmac_with_pending_dma=model.overlapped_mma,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS', kind, head, query, name, 'bitwise and memory trace equal; V schedule checked', flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,
        sha256={n: hashlib.sha256((HERE/(n+'.ll')).read_bytes()).hexdigest() for n in names},
        reference_atol=.01, reference_rtol=.01, all_cases_bitwise_equal=True,
        gpu_tested=False, independent_wave_timing_simulated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
