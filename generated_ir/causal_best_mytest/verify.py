#!/usr/bin/env python3
"""Execute actual causal LLVM with DMA completion and LDS ownership auditing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cross_bn_model', HERE.parent/'noncausal_cross_bn_k0/verify.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
from check_gemm_pv_addresses import split_args


def physical(group, head, query):
    linear = query//group*(32*group)+head*group+query%group
    return linear%32, linear//32


class Model(previous.Model):
    def __init__(self, *args, logical_head, logical_query, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_head = logical_head
        self.logical_query = logical_query

    def operation(self, expr):
        match = re.fullmatch(r'icmp (slt|sle|sgt|sge) (.*)', expr)
        if match:
            ty, operands = self.parse_type(match[2])
            bits = int(re.search(r'i(\d+)', ty)[1])
            a, b = [self.value(x, ty) for x in split_args(operands)]
            if bits == 32:
                a, b = [np.where(x % (1<<32) >= (1<<31), x % (1<<32)-(1<<32), x % (1<<32)) for x in (a,b)]
            elif bits != 64:
                raise AssertionError(ty)
            return {'slt':np.less, 'sle':np.less_equal, 'sgt':np.greater, 'sge':np.greater_equal}[match[1]](a,b)
        return super().operation(expr)

    def memory(self, ptr, ty, write=None):
        live = ptr[self.active]
        if live.size and np.all(np.isin(live>>32, [2,3])):
            offsets = live & ((1<<32)-1)
            begin = self.expected_head*4096*128*2
            end = begin+(self.logical_query+1)*128*128*2
            assert offsets.min() >= begin and (offsets+self.size(ty)).max() <= end, 'K/V read exceeds causal tile bound'
        return super().memory(ptr, ty, write)


def main():
    records = []
    cases = [('random',0), ('random',3), ('random',8), ('random',31),
             ('max_changes',2), ('diagonal_future_dominates',0),
             ('positive_overflow',1), ('negative_underflow',1)]
    for kind, query in cases:
        arrays = [np.random.default_rng(42+query+s).normal(size=(2,4096,128)).astype(np.float16) for s in range(3)]
        q,k,v = arrays; head = 1
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:,:128]=.25; k[:,128:256]=.5
        elif kind == 'diagonal_future_dominates':
            q.fill(1); k[:] = (np.arange(4096,dtype=np.float32)/8)[None,:,None]
            v[:] = (np.arange(4096,dtype=np.float32)/128)[None,:,None]
        elif kind in ('positive_overflow','negative_underflow'):
            q.fill(16); k.fill(16 if kind=='positive_overflow' else -16)
        expected = previous.helper.kdma.vdb.reference(*arrays, head, query, causal=True)
        variants = [(g, False) for g in (4,8)]
        if kind == 'random' and query in (0,31):
            variants += [(g, True) for g in (4,8)]
        baseline = trace0 = None
        for group, optimized in variants:
            name = f'group{group}'+('.o3' if optimized else '')
            path = HERE/(name+'.ll'); raw = path.read_text()
            model = Model(previous.helper.model_text(raw), *arrays, threads=256,
                          physical=physical(group,head,query), logical_head=head, logical_query=query)
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                full = model.run().reshape(q.shape)
            actual = full[head,query*128:(query+1)*128]; tiles=query+1
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert np.isfinite(actual).all()
            assert np.isnan(full).sum() == full.size-16384
            if baseline is None: baseline,trace0 = full.copy(),model.global_trace.digest()
            assert np.array_equal(full, baseline, equal_nan=True), name
            assert model.global_trace.digest() == trace0, name
            early = '"frisk.cross_bn_k0"=' in raw
            assert model.k0_issues == [(t,t*256-(32 if early and t else 0)) for t in range(tiles) for _ in range(4)]
            assert model.k0_overlap == (32*(tiles-1) if early else 0)
            assert (model.vm_waits,model.barriers,model.mmac_count,model.dma64_count) == (4+8*tiles,5+9*tiles,256*tiles,16*tiles)
            assert not model.pending and not model.k0_pending
            assert len(model.q_elements)==16384 and set(model.q_elements.values())=={1}
            assert model.v_reads_with_pending_dma == 24*tiles+(8*(tiles-1) if early else 0)
            # Compare to the separately saved MyTest export before enabling
            # dynamic-bound K0 prefetch: all FP bits and memory values agree.
            control_checked = early and group == 4 and not optimized and kind == 'random' and query in (0,31)
            if control_checked:
                control_path = HERE/'before_cross_bn/group4.ll'
                control = Model(previous.helper.model_text(control_path.read_text()), *arrays, threads=256,
                    physical=physical(group,head,query), logical_head=head, logical_query=query)
                assert np.array_equal(control.run().reshape(q.shape),full,equal_nan=True)
                assert control.global_trace.digest() == model.global_trace.digest()
                assert control.read_records == model.read_records
            records.append(dict(variant=name,kind=kind,query=query,head=head,physical=physical(group,head,query),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),key_tiles=tiles,cross_bn_k0=early,
                max_abs_error=float(np.max(np.abs(actual-expected))),bitwise_equal=True,
                before_cross_bn_bitwise_and_trace_comparison=control_checked,
                causal_kv_bounds_checked=True,delayed_dma_and_lds_ownership_checked=True))
            print('PASS',name,kind,query,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,reference_atol=.01,reference_rtol=.01,
        gpu_tested=False,target_codegen_tested=False,independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
