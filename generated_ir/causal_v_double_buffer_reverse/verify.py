#!/usr/bin/env python3
"""Execute actual LLVM with delayed DMA and LDS ownership checks; no GPU timing."""
import json
import re

import numpy as np
from generate import HERE, SOURCE, GROUPS, score_layout, digest, module

audit = module('vdb_audit', HERE.parent/'noncausal_v_double_buffer_group32/verify.py')
AuditedBlock, reference = audit.AuditedBlock, audit.reference
from check_gemm_pv_addresses import split_args


def check_score_coordinates():
    source = SOURCE.read_text()
    mmas = [re.match(r'  (%[\w.]+) = ',l)[1] for l in source.splitlines()
            if 'call <4 x float> asm "v_mmac_' in l][:128]
    coordinates = {}

    class Probe(AuditedBlock):
        def call(self, body):
            if 'v_mmac_' in body:
                args = re.search(r'"=v,v,v,0"\((.*)\)',body)[1]
                a,b,_ = [self.typed(x) for x in split_args(args)]
                scale = np.array([0x3015],dtype=np.uint16).view(np.float16)[0]
                a = np.rint(a.astype(np.float32)/scale).astype(int)
                b = b.astype(int)
                coords = []
                for w in range(4):
                    aa = a[w*64:(w+1)*64].reshape(4,16,4).transpose(1,0,2).reshape(16,16)
                    bb = b[w*64:(w+1)*64].reshape(4,16,4).transpose(1,0,2).reshape(16,16)
                    assert np.all(aa==aa[:,:1]) and np.all(bb==bb[:,:1])
                    rr = np.broadcast_to(aa[:,:1],(16,16))
                    cc = np.broadcast_to(bb[:,:1].T,(16,16))
                    coords.append(np.stack([rr,cc],axis=-1).reshape(16,4,4,2)
                                  .transpose(2,0,1,3).reshape(64,4,2))
                coordinates[mmas[self.mmac_count]] = np.concatenate(coords)
            return super().call(body)

    tags = np.broadcast_to((np.arange(4096)%128)[None,:,None],(1,4096,128)).astype(np.float16).copy()
    probe = Probe(re.sub(r'\bexact ','',source), tags, tags.copy(), np.zeros_like(tags),physical=(0,0))
    block = probe.blocks['grp.n128']
    end = next(i for i,l in enumerate(block) if l.startswith(mmas[-1]+' ='))+1
    probe.blocks['grp.n128'] = block[:end]+['ret void']
    probe.run()
    tid = np.arange(256)
    covered = set()
    for record in score_layout(source):
        actual = coordinates[record['vector']][:,record['element']]
        expected = np.stack([tid%16+tid//64*16+record['row'],
                             (tid%64//16)*16+record['column']],axis=-1)
        np.testing.assert_array_equal(actual,expected)
        covered.update(map(tuple,actual))
    assert covered == {(r,c) for r in range(128) for c in range(128)}
    print('PASS all 16384 score coordinates traced from actual Q/K DMA, LDS and MMAC operands',flush=True)


def physical(group,head,query):
    forward = 31-query
    linear = forward//group*(32*group)+head*group+forward%group
    return linear%32,linear//32


class BoundedBlock(AuditedBlock):
    def __init__(self,*args,logical_head,logical_query,**kwargs):
        self.logical_head = logical_head
        self.logical_query = logical_query
        super().__init__(*args,**kwargs)

    def operation(self,expr):
        # The shared interpreter stores i32 arithmetic modulo 2**32; signed
        # comparisons must reinterpret negative row-minus-column thresholds.
        match = re.fullmatch(r'icmp (slt|sle|sgt|sge) (.*)',expr)
        if match:
            ty,operands = self.parse_type(match[2])
            assert ty == 'i32' or re.fullmatch(r'<\d+ x i32>',ty),ty
            a,b = [self.value(x,ty) for x in split_args(operands)]
            a,b = [np.where(x % (1<<32) >= (1<<31),x % (1<<32)-(1<<32),x % (1<<32)) for x in (a,b)]
            return {'slt':np.less,'sle':np.less_equal,'sgt':np.greater,'sge':np.greater_equal}[match[1]](a,b)
        return super().operation(expr)

    def memory(self,ptr,ty,write=None):
        live = ptr[self.active]
        if live.size and np.all(np.isin(live>>32,[2,3])):
            offsets = live & ((1<<32)-1)
            begin = self.logical_head*4096*128*2
            end = begin+(self.logical_query+1)*128*128*2
            assert offsets.min() >= begin and (offsets+self.size(ty)).max() <= end, 'K/V read exceeds causal tile bound'
        return super().memory(ptr,ty,write)


def main():
    check_score_coordinates()
    records = []
    cases = [('random',0),('random',1),('random',15),('random',16),('random',31),
             ('max_changes',2),('diagonal_future_dominates',0),
             ('positive_overflow',1),('negative_underflow',1)]
    for kind,query in cases:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        q,k,v = arrays
        head = 1
        if kind == 'max_changes':
            q.fill(1);k.fill(-16);k[:,:128]=.25;k[:,128:256]=.5
        elif kind == 'diagonal_future_dominates':
            q.fill(1)
            k[:] = (np.arange(4096,dtype=np.float32)/8)[None,:,None]
            v[:] = (np.arange(4096,dtype=np.float32)/128)[None,:,None]
        elif kind in ('positive_overflow','negative_underflow'):
            q.fill(16);k.fill(16 if kind=='positive_overflow' else -16)
        expected = reference(*arrays,head,query,causal=True)
        groups = (32,) if kind.endswith(('overflow','underflow')) else GROUPS
        variants = [(g,False) for g in groups]
        if (kind,query) in [('random',0),('random',31),('positive_overflow',1),('negative_underflow',1)]:
            variants += [(g,True) for g in groups]
        baseline = baseline_global = None
        for group,optimized in variants:
            name = f'group{group}'+('.o3' if optimized else '')
            path = HERE/(name+'.ll')
            text = re.sub(r'\bexact ','',path.read_text())
            model = BoundedBlock(text,*arrays,threads=256,physical=physical(group,head,query),
                                 logical_head=head,logical_query=query)
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                full = model.run().reshape(q.shape)
            tiles = query+1
            assert not model.pending
            assert model.mmac_count == 256*tiles
            assert model.vm_waits == 4+8*tiles
            assert model.barriers == 5+9*tiles
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert model.v_reads == [(t*256+128+s*32,s%2) for t in range(tiles) for s in range(4) for _ in range(8)]
            assert model.v_dma == [(t*256+n,s%2) for t in range(tiles)
                                   for s,n in enumerate([96,128,160,192]) for _ in range(2)]
            assert model.v_reads_with_pending_dma == 24*tiles
            actual = full[head,query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            mask = np.zeros(full.shape,dtype=bool)
            mask[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~mask]).all()
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            trace = model.global_trace.hexdigest()
            if baseline is None:
                baseline,baseline_global = full.copy(),trace
            assert np.array_equal(full,baseline,equal_nan=True),name
            assert trace == baseline_global,name
            records.append(dict(kind=kind,head=head,query=query,variant=name,
                sha256=digest(path),physical=list(physical(group,head,query)),key_tiles=tiles,
                bitwise_equal_across_groups_and_o3=True,global_trace_sha256=trace,
                max_abs_error=float(np.max(np.abs(actual-expected))),
                vm_waits=model.vm_waits,barriers=model.barriers,mmac=model.mmac_count,
                causal_kv_bounds_checked=True,lds_ownership_checked=True))
            print(f'PASS {name} {kind} query={query}: causal numerics, bounds, DMA and LDS ownership',flush=True)
    # These controls prove the semantic checks reject missing diagonal masking
    # and a noncausal loop bound, independently of the generated config flag.
    text = (HERE/'group32.ll').read_text()
    text = re.sub(r'\bexact ','',text)
    rng = np.random.default_rng(77)
    arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    expected = reference(*arrays,1,0,causal=True)
    broken = re.sub(r'icmp sge i32 %causal.delta, -?\d+',
                    'icmp eq i32 0, 0',text)
    model = BoundedBlock(broken,*arrays,threads=256,physical=physical(32,1,0),logical_head=1,logical_query=0)
    actual = model.run().reshape(arrays[0].shape)[1,:128]
    assert not np.allclose(actual,expected,atol=.01,rtol=.01)
    broken = text.replace('icmp samesign ult i64 %grp.n12010, %grp.n14,',
                          'icmp samesign ult i64 %grp.n12010, 31,')
    try:
        BoundedBlock(broken,*arrays,threads=256,physical=physical(32,1,0),logical_head=1,logical_query=0).run()
    except AssertionError as error:
        assert 'K/V read exceeds causal tile bound' in str(error)
    else:
        raise AssertionError('Missing loop bound not detected')
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,score_coordinates_checked=16384,
        negative_controls=['missing_diagonal_mask','missing_causal_loop_bound'],
        reference_atol=.01,reference_rtol=.01,gpu_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')
    print(f'PASS {len(records)} numerical executions and 2 negative controls',flush=True)


if __name__ == '__main__':
    main()
