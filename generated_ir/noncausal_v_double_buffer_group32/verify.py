#!/usr/bin/env python3
"""Execute actual IR; audit delayed DMA, buffer ownership and barrier epochs.

Epoch checks require a workgroup barrier between conflicting LDS accesses,
covering all 256 lanes. They do not simulate machine scheduling or GPU timing.
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
    def operation(self, expr):
        if expr.startswith(';'):
            return None
        return super().operation(expr)

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.last_read = np.full(16384, -1, dtype=np.int64)
        self.last_write = np.full(16384, -1, dtype=np.int64)
        self.global_trace = hashlib.sha256()
        self.lds_values = hashlib.sha256()
        self.v_reads, self.v_dma = [], []
        self.v_dma_context = self.v_read_context = False
        self.v_reads_with_pending_dma = 0
        super().__init__(*args, **kwargs)

    def memory(self, ptr, ty, write=None):
        live = ptr[self.active]
        shared = bool(live.size and np.all(live >> 32 == 0))
        if shared:
            touched = self.addresses(ptr, ty, self.active)
            assert touched.min() >= 0 and touched.max() < 16384, 'LDS out of bounds'
            if write is not None:
                assert (self.last_read[touched] < self.epoch).all(), 'overwrite without reader barrier'
                assert (self.last_write[touched] < self.epoch).all(), 'writes overlap in barrier epoch'
                self.last_write[touched] = self.epoch
                if self.v_dma_context:
                    slot = int(touched.min() // 8192)
                    assert np.all(touched // 8192 == slot)
                    self.v_dma.append((self.mmac_count, slot))
            else:
                assert (self.last_write[touched] < self.epoch).all(), 'read before publishing barrier'
                self.last_read[touched] = self.epoch
                if self.v_read_context:
                    slot = int(touched.min() // 8192)
                    assert np.all(touched // 8192 == slot)
                    self.v_reads.append((self.mmac_count, slot))
                    self.v_reads_with_pending_dma += bool(self.pending)
        else:
            self.global_trace.update(('R' if write is None else 'W').encode())
            self.global_trace.update(ty.encode())
            self.global_trace.update(np.asarray(live, dtype=np.int64).tobytes())
        result = super().memory(ptr, ty, write)
        if shared and write is None:
            self.lds_values.update(ty.encode())
            self.lds_values.update(result[self.active].tobytes())
        return result

    def call(self, body):
        if '@llvm.amdgcn.s.barrier(' in body:
            self.epoch += 1
        self.v_read_context = '@llvm.hcu.ds.read.m32x16.b16.v4i32.p3(' in body
        self.v_dma_context = 'buffer_load_dwordx4' in body and '%dma.buffer.rsrc1349' in body
        try:
            return super().call(body)
        finally:
            self.v_read_context = self.v_dma_context = False


def main():
    names = ['baseline_pv_interleave4', 'v_double_buffer']
    texts = {n: re.sub(r'\bexact ', '', (HERE/(n+'.ll')).read_text()) for n in names}
    records = []
    cases = [('random',0,0), ('random',1,31), ('max_changes',1,2), ('future_key_dominates',1,0)]
    for kind, head, query in cases:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        if kind == 'max_changes':
            arrays[0].fill(1)
            arrays[1].fill(-16)
            arrays[1][:,:128] = .25
            arrays[1][:,128:256] = .5
        elif kind == 'future_key_dominates':
            arrays[0].fill(1)
            arrays[1].fill(-1)
            arrays[1][:,-128:] = 1
            arrays[2].fill(0)
            arrays[2][:,-128:] = 1
        expected = reference(*arrays, head, query, causal=False)
        baseline_output = baseline_global = baseline_values = None
        variants = names + (['v_double_buffer.o3'] if (kind,head,query) == cases[0] else [])
        for name in variants:
            double = name != names[0]
            text = texts.get(name) or re.sub(r'\bexact ', '', (HERE/(name+'.ll')).read_text())
            model = AuditedBlock(text, *arrays, threads=256, physical=(query,head))
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = model.run().reshape(arrays[0].shape)
            assert not model.pending
            assert model.vm_waits == 260 and model.barriers == (293 if double else 389)
            assert model.mmac_count == 8192
            assert model.q_elements and set(model.q_elements.values()) == {1}
            assert len(model.q_elements) == 16384
            want_reads = [(tile*256+128+s*32, s%2 if double else 0)
                          for tile in range(32) for s in range(4) for _ in range(8)]
            want_dma = [(tile*256 + (96 if s == 0 else 128+(s-1)*32+(0 if double else 4)),
                         s%2 if double else 0) for tile in range(32) for s in range(4) for _ in range(2)]
            assert model.v_reads == want_reads, (name, 'V reads')
            assert model.v_dma == want_dma, (name, 'V DMA')
            assert model.v_reads_with_pending_dma == (768 if double else 0)
            actual = full[head,query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            mask = np.zeros(full.shape, dtype=bool)
            mask[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~mask]).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            global_trace, values = model.global_trace.hexdigest(), model.lds_values.hexdigest()
            if not double:
                baseline_output, baseline_global, baseline_values = full.copy(), global_trace, values
            assert np.array_equal(full, baseline_output, equal_nan=True), name
            assert global_trace == baseline_global, (name, 'global memory trace')
            assert values == baseline_values, (name, 'LDS read values')
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                bitwise_equal_to_best=True, global_trace_sha256=global_trace,
                lds_read_values_sha256=values, barrier_epoch_ownership_passed=True,
                vm_waits=model.vm_waits, hardware_barriers=model.barriers,
                v_reads_with_pending_dma=model.v_reads_with_pending_dma,
                mmac_with_pending_dma=model.overlapped_mma,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS', kind, head, query, name, 'bitwise, global trace, LDS values, slots and barrier epochs', flush=True)
    # Negative controls ensure the hazard checks can detect the two key bugs:
    # V1 writes low while V0 is being read, or V1 reads lack publication.
    good = texts['v_double_buffer']
    broken_slot = good.replace('i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1342,',
                               'i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1342,', 1)
    assert broken_slot != good
    before = good.index('  %vdb.read.s1.p0 =')
    barrier = good.rfind('  tail call void @llvm.amdgcn.s.barrier()', 0, before)
    end = good.index('\n', barrier)
    broken_barrier = good[:barrier] + good[end+1:]
    negative = []
    for label, broken, message in [('wrong_slot', broken_slot, 'read before publishing barrier'),
                                    ('missing_ready_barrier', broken_barrier, 'reader barrier')]:
        try:
            AuditedBlock(broken, *arrays, threads=256, physical=(query,head)).run()
        except AssertionError as error:
            assert message in str(error), (label,str(error))
            negative.append(dict(case=label, detected=str(error)))
            print('PASS negative control', label, str(error), flush=True)
        else:
            raise AssertionError('Negative control unexpectedly passed: '+label)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records, negative_controls=negative,
        sha256={n:hashlib.sha256((HERE/(n+'.ll')).read_bytes()).hexdigest()
                for n in names+['v_double_buffer.o3']},
        reference_atol=.01, reference_rtol=.01, gpu_tested=False,
        independent_wave_timing_simulated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
