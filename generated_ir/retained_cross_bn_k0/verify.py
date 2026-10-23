#!/usr/bin/env python3
"""Execute actual IR with delayed DMA, LDS ownership and ordered trace audits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('retention_sweep_checks',ROOT/'generated_ir/q_bk_retention_sweep/verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class PrefetchBlock(previous.SweepBlock):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.k0_pending = False
        self.k0_issues = []
        self.k0_overlap = 0

    def call(self,body):
        if 's_waitcnt vmcnt(0)' in body:
            self.k0_pending = False
        if 'v_mmac_' in body and self.k0_pending:
            assert self.pending
            assert self.mmac_count % 256 >= 224, 'K0 overlapped outside final PV stage'
            self.k0_overlap += 1
        return super().call(body)

    def memory(self,ptr,ty,write=None):
        live = ptr[self.active]
        if write is None and live.size and np.all(live>>32 == 2):
            elements = (live-(2<<32))//2
            if np.all(elements % 128 < 32):
                tile = (elements//128 % 4096)//128
                assert np.all(tile == tile[0])
                self.k0_issues.append(dict(tile=int(tile[0]),mmac_before=self.mmac_count))
                self.k0_pending = True
        return super().memory(ptr,ty,write)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    texts = {}
    llvm = Path('/data2/xsl/install/bin')
    schedule = {}
    for name,info in manifest['variants'].items():
        path = HERE/(name+'.ll')
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        subprocess.run([llvm/'llvm-as',path,'-o','/dev/null'],check=True)
        subprocess.run([llvm/'opt','-passes=default<O3>,verify','-S',path,'-o',HERE/(name+'.opt.ll')],check=True)
        opt = (HERE/(name+'.opt.ll')).read_text()
        if name != 'baseline':
            dma = opt.index('%k0.next.bnk.n253 =')
            first, last = (opt.index('%bnk.n'+str(n)+' =') for n in (1513,1563))
            if name == 'k0_prefetch_pv_tail':
                assert dma < first < last
                assert opt[dma:last].count('v_mmac_') == 31
            else:
                assert first < last < dma
            schedule[name] = dict(O3_preserves_issue_side_of_final_PV=True)
        texts[name] = re.sub(r'\bexact ', '',raw.decode())
    cases = []
    for seed,head,query in [(42,0,0),(42,1,1),(7,1,7),(7,1,8),(42,1,31)]:
        rng = np.random.default_rng(seed)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        forward = 31-query
        linear = forward//8*256+head*8+forward%8
        models, outputs, counts = {}, {}, {}
        n = query+1
        for name,text in texts.items():
            model = PrefetchBlock(text,*arrays,physical=(linear%32,linear//32))
            full = model.run().reshape(arrays[0].shape)
            models[name],outputs[name] = model,full
            assert not model.pending and not model.k0_pending
            assert model.read_calls == dict(Q0=2,Qrest=6,K=8*n,V=8*n)
            assert model.q_fragment_reads == {f:2 for f in range(4)}
            assert model.q_scales == 16 and model.lds_loads == 16+128*n
            assert model.vm_waits == 4+8*n and model.barriers == 5+12*n
            assert model.mmac_count == 256*n and model.hints == 8*n
            early = name == 'k0_prefetch_pv_tail'
            want = [dict(tile=tile,mmac_before=256*tile-(32 if early and tile else 0))
                    for tile in range(n) for packet in range(2)]
            assert model.k0_issues == want,(name,model.k0_issues,want)
            assert model.k0_overlap == (32*(n-1) if early else 0)
            assert np.array_equal(full,outputs['baseline'],equal_nan=True)
            assert model.global_trace.digest() == models['baseline'].global_trace.digest()
            assert model.lds_trace.digest() == models['baseline'].lds_trace.digest()
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            counts[name] = dict(waits=model.vm_waits,barriers=model.barriers,
                k0_overlap_mmac=model.k0_overlap,k0_issues=model.k0_issues,
                bitwise_equal=True,ordered_global_and_LDS_traces_identical=True,
                no_outstanding_DMA=True,cross_wave_barrier_epoch_audit_passed=True)
            print(f'PASS {name} head={head} query={query} overlap={model.k0_overlap}',flush=True)
        expected = previous.previous.audits.reference(*arrays,head,query)
        actual = outputs['baseline'][head,query*128:(query+1)*128]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        cases.append(dict(seed=seed,head=head,query=query,variants=counts,
                          max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected)))))
    text = texts['k0_prefetch_pv_tail']
    marker = text.index('; V3 readers complete')
    barrier = re.search(r'^  call void @llvm.amdgcn.s.barrier\(\)[^\n]*\n',text[marker:],re.M)
    start,end = marker+barrier.start(),marker+barrier.end()
    missing_handoff = text[:start]+text[end:]
    header = text.index('\nbnk.n212:')
    wait = re.search(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n',text[header:],re.M)
    start,end = header+wait.start(),header+wait.end()
    missing_wait = text[:start]+text[end:]
    unguarded = text.replace('br i1 %k0.has.next, label %k0.issue, label %k0.join','br label %k0.issue')
    negatives = []
    for tag,bad in [('missing_reader_handoff',missing_handoff),('missing_K0_wait',missing_wait),('missing_last_iteration_guard',unguarded)]:
        try:
            broken = PrefetchBlock(bad,*arrays,physical=(8,0))  # head=1, query=31: last legal K tile
            broken.run()
            assert not broken.pending, 'extra K0 DMA at kernel exit'
        except (AssertionError,IndexError) as error:
            reason = str(error)
            assert any(s in reason for s in ('DMA','cross-wave','out of bounds')),reason
            negatives.append(dict(case=tag,rejected=reason))
            print('PASS negative',tag,reason,flush=True)
        else:
            raise AssertionError('failed to reject '+tag)
    (HERE/'validation.json').write_text(json.dumps(dict(
        variants=manifest['variants'],cases=cases,O3_schedule=schedule,negative_controls=negatives,
        gpu_tested=False,target_gfx936_codegen_tested=False,
        limitations='CPU execution of actual input IR with delayed DMA and per-byte cross-wave LDS audit; not cycle-accurate. O3 checked structurally. Hardware scheduling/register allocation/performance require DTK GPU testing.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
