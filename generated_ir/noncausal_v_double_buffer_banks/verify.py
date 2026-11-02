#!/usr/bin/env python3
"""Verify producer provenance, bank addresses, numerical results and LDS epochs."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
from generate import HERE, ROOT, RESOURCES, digest

sys.path.insert(0, str(ROOT/'3rd/deepgengraph/test'))
from check_attention_llvm_numerics import LLVMWorkgroupModel
from check_gemm_pv_addresses import split_args
from check_packed_lds_banks import multiplicity

spec = importlib.util.spec_from_file_location('vdb_audit', HERE.parent/'noncausal_v_double_buffer_group32/verify.py')
vdb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vdb)


class AuditedBlock(vdb.AuditedBlock):
    def call(self, body):
        if 'buffer_load_dword ' not in body:
            return super().call(body)
        dst, off, resource = [self.typed(x) for x in split_args(re.search(r'"\((.*)\)', body)[1])]
        assert np.all(resource[:, 3] == 0x20000)
        assert np.all(off >= 0) and np.all(off+4 <= resource[:, 2])
        for wave in range(0, self.t, 64):
            assert np.all(dst[wave:wave+64] == dst[wave])
            assert np.all(resource[wave:wave+64] == resource[wave])
        src = resource[:, 0] + (resource[:, 1] << 32) + off
        self.issuing = True
        try:
            self.memory(dst+np.arange(self.t)%64*4, '<2 x half>', self.memory(src, '<2 x half>'))
        finally:
            self.issuing = False
        return dst


class LazyAddresses(LLVMWorkgroupModel):
    def __init__(self, text):
        text = re.sub(r'\bexact ', '', text)
        dummy = np.zeros(1, dtype=np.float16)
        super().__init__(text, dummy, dummy, dummy, threads=256)
        self.defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+.*)?$', text, re.M))
        self.initial = dict(self.values)

    def reset(self, head=0, query=0, tile=0):
        self.values = dict(self.initial)
        for name, expr in self.defs.items():
            value = None
            if '@llvm.amdgcn.workgroup.id.x()' in expr: value = query
            if '@llvm.amdgcn.workgroup.id.y()' in expr: value = head
            if expr.startswith('phi i64 '): value = tile
            if value is not None: self.values[name] = np.full(self.t, value, dtype=np.int64)

    def value(self, name, ty=None):
        if name not in self.values and name in self.defs:
            self.values[name] = self.operation(self.defs[name])
        return super().value(name, ty)


def dma_args(text, kind):
    return [split_args(re.search(r'"\((.*)\)', line)[1]) for line in text.splitlines()
            if 'buffer_load_dword' in line and RESOURCES[kind] in line]


def address_audit(texts):
    models = {name: LazyAddresses(text) for name, text in texts.items()}
    baseline = models['baseline']
    records = {}
    tid = np.arange(256)
    for name, model in models.items():
        for m in models.values(): m.reset()
        banks = {}
        for kind, anchor in [('Q', '%grp.n51'), ('K', '%grp.n159')]:
            dbg = re.search(r'!dbg !(\d+)', next(l for l in texts[name].splitlines() if l.startswith('  '+anchor+' =')))[1]
            ptrs = re.findall(r'load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 8, !dbg !'+dbg+r'\b', texts[name])
            addresses = [model.value(p) for p in ptrs]
            count = collections.Counter(max(multiplicity(a[i:i+16].tolist(), 8) for i in range(0,256,16)) for a in addresses)
            want = 1 if kind == 'K' and name in ('k_swizzle32','qk_swizzle32') or kind == 'Q' and name == 'qk_swizzle32' else 2
            assert dict(count) == {want: 64 if kind == 'K' else 16}, (name, kind, count)
            banks[kind] = dict(multiplicity=dict(count), first_16_bytes=addresses[0][:16].tolist())
        if name == 'baseline':
            # Exhaustively search single-XOR settings that preserve 16B packets,
            # restricted to permutations internal to a 1024B DMA wave packet.
            a = [baseline.value(p) for p in re.findall(r'load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 8, !dbg !41\b', texts[name])]
            assert len(a) == 64
            sweep = {}
            for src in range(4, 10):
                for dst in range(4, 10):
                    if src == dst: continue
                    # Undo current 8:6, then apply a replacement setting.
                    transformed = [(x ^ ((x>>2)&64)) for x in a]
                    transformed = [x ^ (((x>>src)&1)<<dst) for x in transformed]
                    sweep[f'{src}:{dst}'] = max(multiplicity(x[i:i+16].tolist(),8) for x in transformed for i in range(0,256,16))
            assert min(sweep.values()) == 2
            banks['K_packet_preserving_single_xor_sweep'] = sweep
        packets = 0
        for kind in ('Q', 'K'):
            old_args, new_args = dma_args(texts['baseline'], kind), dma_args(texts[name], kind)
            if len(old_args) == len(new_args):
                assert old_args == new_args
                continue
            assert len(old_args) == 8 and len(new_args) == 32
            # All heads and relevant query/key tiles. The other axis does not
            # participate in this resource's offsets.
            for head in range(32):
                for tile in range(32):
                    baseline.reset(head, tile if kind == 'Q' else 0, tile if kind == 'K' else 0)
                    model.reset(head, tile if kind == 'Q' else 0, tile if kind == 'K' else 0)
                    for i, args in enumerate(old_args):
                        m0, off, rsrc = map(baseline.typed, args)
                        covered = []
                        for j in range(4):
                            m1, off1, rsrc1 = map(model.typed, new_args[4*i+j])
                            assert np.array_equal(rsrc, rsrc1)
                            dst = m1+(tid%64)*4
                            swizzle = kind == 'K' and name in ('k_swizzle32','qk_swizzle32') or kind == 'Q' and name == 'qk_swizzle32'
                            bit = 9 if kind == 'K' else 7
                            original = dst ^ (((dst>>bit)&1)<<3) if swizzle else dst
                            relative = original-m0
                            owner = tid//64*64+relative//16
                            assert np.all((owner//64) == tid//64)
                            assert np.array_equal(off1, off[owner]+relative%16)
                            assert np.all(off1%4 == 0) and np.all(off1 >= 0) and np.all(off1+4 <= rsrc1[:,2])
                            assert np.all((dst >= 0) & (dst+4 <= 16384))
                            covered.extend(dst.tolist())
                            packets += 256
                        expected = (m0[:,None]+(tid%64)[:,None]*16+np.arange(0,16,4)[None,:]).ravel()
                        assert sorted(covered) == sorted(expected.tolist())
        records[name] = dict(banks=banks, checked_dword_packets=packets, source_provenance_and_coverage=True)
        print('PASS addresses', name, packets, flush=True)
    return records


def main():
    names = ['baseline','k_split32_control','k_swizzle32','qk_swizzle32']
    texts = {name: (HERE/(name+'.ll')).read_text() for name in names}
    result = dict(gpu_tested=False, gfx936_codegen_tested=False,
                  model='32 banks x 4 bytes; b64 evaluated per contiguous 16-lane phase; V excluded by assumption',
                  addresses=address_audit(texts), cases=[])
    for kind, head, query in [('random',0,0), ('future_key_dominates',1,31), ('max_changes',1,2)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        if kind == 'future_key_dominates':
            arrays[0].fill(1); arrays[1].fill(-1); arrays[1][:,-128:] = 1
            arrays[2].fill(0); arrays[2][:,-128:] = 1
        if kind == 'max_changes':
            arrays[0].fill(1); arrays[1].fill(-16)
            arrays[1][:,:128] = .25; arrays[1][:,128:256] = .5
        expected = vdb.reference(*arrays, head, query, causal=False)
        output = values = v_reads = v_dma = None
        variants = names + (['k_swizzle32.o3','qk_swizzle32.o3'] if kind == 'random' else [])
        for name in variants:
            text = re.sub(r'\bexact ', '', (HERE/(name+'.ll')).read_text())
            model = AuditedBlock(text, *arrays, threads=256, physical=(query,head))
            full = model.run().reshape(arrays[0].shape)
            actual = full[head,query*128:(query+1)*128]
            assert model.vm_waits == 260 and model.barriers == 293 and model.mmac_count == 8192 and not model.pending
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert model.v_reads_with_pending_dma == 768
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            if output is None:
                output, values = full.copy(), model.lds_values.hexdigest()
                v_reads, v_dma = model.v_reads, model.v_dma
            assert np.array_equal(full, output, equal_nan=True)
            assert model.lds_values.hexdigest() == values
            assert model.v_reads == v_reads and model.v_dma == v_dma
            assert np.isnan(full).sum() == full.size-16384
            result['cases'].append(dict(kind=kind, head=head, query=query, variant=name,
                bitwise_equal=True, identical_lds_read_values=True, barrier_epoch_ownership=True,
                v_schedule_unchanged=True, max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS numerical / LDS epochs / V schedule', kind, name, flush=True)
    result['sha256'] = {n:digest(HERE/(n+'.ll')) for n in names}
    (HERE/'validation.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
