#!/usr/bin/env python3
"""Numerical/DMA-trace equivalence plus cross-wave LDS barrier-epoch auditing."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'generated_ir/q_bk32_retained'))
from validate import MeasuredBlock, reference


class AuditedBlock(MeasuredBlock):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.readers = np.zeros(16384,dtype=np.uint8)
        self.writers = np.zeros(16384,dtype=np.uint8)
        self.global_trace = hashlib.sha256()
        self.lds_trace = hashlib.sha256()
        self.epoch_accesses = 0
        self.hints = 0

    def call(self,body):
        self.hints += '@llvm.amdgcn.sched.barrier(' in body
        result = super().call(body)
        if '@llvm.amdgcn.s.barrier(' in body:
            assert np.all(self.active), 'non-collective barrier'
            self.readers.fill(0)
            self.writers.fill(0)
        return result

    def memory(self,ptr,ty,write=None):
        live = ptr[self.active]
        if live.size and np.all(live >= 1<<32):
            if write is None:
                self.global_trace.update(ty.encode())
                self.global_trace.update(live.tobytes())
        elif live.size:
            size = self.size(ty)
            assert live.min() >= 0 and live.max()+size <= 16384
            self.lds_trace.update(str((ty,write is not None)).encode())
            self.lds_trace.update(live.tobytes())
            # Within a barrier epoch, a cross-wave read/write or write/write
            # overlap is unsafe without extra synchronization. Check DMA writes
            # at issue (earliest possible overwrite), not only at completion.
            for wave in range(self.t//64):
                mask = self.active.copy()
                mask[:wave*64] = False
                mask[(wave+1)*64:] = False
                addresses = np.unique((ptr[mask,None]+np.arange(size)[None,:]).ravel())
                other = np.uint8(15 ^ (1<<wave))
                if write is None:
                    assert not np.any(self.writers[addresses] & other), 'cross-wave RAW without publication barrier'
                else:
                    conflict = addresses[(self.readers[addresses] & other) != 0]
                    assert not len(conflict), f'cross-wave WAR without reader-completion barrier: epoch={self.barriers} mma={self.mmac_count} lds_reads={self.lds_loads} wave={wave} bytes={conflict[:8]}'
                    assert not np.any(self.writers[addresses] & other), 'cross-wave WAW without barrier'
            for wave in range(self.t//64):
                mask = self.active.copy()
                mask[:wave*64] = False
                mask[(wave+1)*64:] = False
                addresses = np.unique((ptr[mask,None]+np.arange(size)[None,:]).ravel())
                target = self.readers if write is None else self.writers
                target[addresses] |= np.uint8(1<<wave)
            self.epoch_accesses += 1
        # Parent also rejects accessing a pending DMA destination and executes
        # the real IR instructions (including matrix-read lane permutations).
        return super().memory(ptr,ty,write)


def check_o3_schedule(manifest):
    source = (ROOT/manifest['source']).read_text().splitlines()
    def ident(line):
        name = re.match(r'  (%[\w.]+) =',line)[1]
        return '%sched.n'+name[1:] if name[1:].isdigit() else name
    results = {}
    for name in ('wait_tail2','fragment_pair','combined'):
        lines = (HERE/(name+'.opt.ll')).read_text().splitlines()
        definitions = {re.match(r'  (%[\w.]+) =',l)[1]:i for i,l in enumerate(lines)
                       if re.match(r'  (%[\w.]+) =',l)}
        waits = [i for i,l in enumerate(lines) if 's_waitcnt vmcnt(0)' in l]
        record = {}
        if name in ('wait_tail2','combined'):
            for change in manifest['wait_changes']:
                dma = definitions[ident(source[change['q_dma_lines'][-1]-1])]
                a,b = [definitions[ident(source[i-1])] for i in change['original_mmac_lines']]
                wait = next(i for i in waits if i>dma)
                assert dma<a<b<wait
            record['two_mmac_tails_between_dma_and_wait'] = 3
        if name in ('fragment_pair','combined'):
            for change in manifest['fragment_changes']:
                anchor = definitions[ident(source[change['mmac_original_line']-1])]
                assert all(definitions[v]<anchor for v in change['loads'])
            record['whole_read_pairs_before_prior_mmac'] = len(manifest['fragment_changes'])
        record['vm_waits_static'] = len(waits)
        record['hardware_barriers_static'] = sum('call void @llvm.amdgcn.s.barrier(' in l for l in lines)
        results[name] = record
    (HERE/'o3_schedule.json').write_text(json.dumps(results,indent=2)+'\n')


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    check_o3_schedule(manifest)
    names = list(manifest['variants'])
    texts = {}
    for name in names:
        raw = (HERE/(name+'.ll')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['variants'][name]['sha256']
        texts[name] = re.sub(r'\bexact ', '', raw.decode())
    cases = []
    for seed,head,query in [(42,0,0),(42,1,1),(7,1,7),(7,1,8),(42,1,31)]:
        rng = np.random.default_rng(seed)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        forward = 31-query
        linear = forward//8*256+head*8+forward%8
        models, outputs, counts = {}, {}, {}
        for name in names:
            model = AuditedBlock(texts[name],*arrays,physical=(linear%32,linear//32))
            full = model.run().reshape(arrays[0].shape)
            models[name],outputs[name] = model,full
            n = query+1
            assert not model.pending
            assert model.mmac_count == 256*n
            assert model.barriers == 16*n+2
            assert model.vm_waits == 11*n+1
            assert model.lds_loads == 140*n+4 and model.q_scales == 12*n+4
            assert model.hints == (62 if name in ('fragment_pair','combined') else 8)*n
            assert model.read_calls == dict(Q0=2,Qrest=6*n,K=8*n,V=8*n)
            assert np.array_equal(full,outputs['baseline'],equal_nan=True)
            assert model.global_trace.digest() == models['baseline'].global_trace.digest()
            assert model.lds_trace.digest() == models['baseline'].lds_trace.digest()
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            counts[name] = dict(waits=model.vm_waits,barriers=model.barriers,
                                hints=model.hints,lds_epoch_accesses=model.epoch_accesses,
                                bitwise_equal=True,global_and_lds_access_traces_identical=True)
            print(f'PASS {name} head={head} query={query} '+json.dumps(counts[name]),flush=True)
        expected = reference(*arrays,head,query)
        actual = outputs['baseline'][head,query*128:(query+1)*128]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        cases.append(dict(seed=seed,head=head,query=query,variants=counts,
                          max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected)))))
    # Ensure the audits really reject missing publication/old-reader protection.
    text = texts['baseline']
    waits = list(re.finditer(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n',text,re.M))
    barriers = list(re.finditer(r'^  call void @llvm.amdgcn.s.barrier\(\)[^\n]*\n',text,re.M))
    negatives = []
    for tag,match in [('missing_dma_wait',waits[0]),('missing_handoff_barrier',barriers[1])]:
        bad = text[:match.start()]+text[match.end():]
        try:
            AuditedBlock(bad,*arrays,physical=(7,24)).run()
        except AssertionError as e:
            assert ('DMA' in str(e) or 'cross-wave' in str(e)), str(e)
            negatives.append(dict(case=tag,rejected=str(e)))
            print('PASS negative',tag,str(e),flush=True)
        else:
            raise AssertionError('hazard audit failed to reject '+tag)
    result = dict(sha256={n:manifest['variants'][n]['sha256'] for n in names},
                  cases=cases,hazard_negative_controls=negatives,gpu_tested=False,
                  limitations='LDS byte/reader-wave checks establish barrier-epoch ordering for modeled paths; this is not a cycle-accurate GPU simulator or target register-allocation check.')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__': main()
