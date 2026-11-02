#!/usr/bin/env python3
"""执行实际LLVM，核对输出、访存轨迹、V读取及DMA发起位置。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v_overlap_verify',
    HERE.parent/'noncausal_v_lds_overlap_group32/verify.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def main():
    texts = {n: re.sub(r'\bexact ', '', (HERE/f'dma_after{n}.ll').read_text()) for n in (8, 2, 4)}
    records = []
    for kind, head, query in [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        if kind == 'max_changes':
            arrays[0].fill(1)
            arrays[1].fill(-16)
            arrays[1][:, :128] = .25
            arrays[1][:, 128:256] = .5
        expected = prior.reference(*arrays, head, query, causal=False)
        base_output = base_trace = None
        for after, text in texts.items():
            model = prior.AuditedBlock(text, *arrays, threads=256, physical=(query, head))
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = model.run().reshape(arrays[0].shape)
            assert not model.pending
            assert model.vm_waits == 260 and model.barriers == 389
            assert model.mmac_count == 8192 and len(model.q_elements) == 16384
            want_reads = [tile*256+128+stage*32 for tile in range(32) for stage in range(4) for _ in range(8)]
            want_dma = [tile*256 + (96 if stage == 0 else 128+(stage-1)*32+after)
                        for tile in range(32) for stage in range(4) for _ in range(2)]
            assert model.v_reads == want_reads
            assert model.v_dma == want_dma
            assert model.overlapped_mma == 7168-96*after
            mask = np.zeros(full.shape, dtype=bool)
            mask[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~mask]).all()
            actual = full[head, query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            trace = model.memory_trace.hexdigest()
            if after == 8:
                base_output, base_trace = full.copy(), trace
            assert np.array_equal(full, base_output, equal_nan=True)
            assert trace == base_trace
            records.append(dict(kind=kind, head=head, query=query, after_mmac_count=after,
                bitwise_equal_to_after8=True, ordered_memory_trace_sha256=trace,
                v_read_and_dma_schedule_verified=True, vm_waits=model.vm_waits,
                hardware_barriers=model.barriers, mmac_with_pending_dma=model.overlapped_mma,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print(f'PASS {kind} h={head} q={query} after{after}: bitwise/memory trace equal; DMA schedule verified', flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,
        sha256={f'dma_after{n}': hashlib.sha256((HERE/f'dma_after{n}.ll').read_bytes()).hexdigest() for n in texts},
        reference_atol=.01, reference_rtol=.01, all_cases_bitwise_equal=True,
        gpu_tested=False, independent_wave_timing_simulated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
