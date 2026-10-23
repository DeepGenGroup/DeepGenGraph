#!/usr/bin/env python3
"""Check real LLVM execution, exact per-fragment Q traffic and LDS hazards."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('retention_hazard_audits',ROOT/'generated_ir/retained_local_schedule/verify.py')
audits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audits)


class FragmentBlock(audits.AuditedBlock):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.q_fragment_reads = Counter()
        self.q_elements = [Counter() for _ in range(4)]
        self.unchanged_reads = hashlib.sha256()

    def memory(self,ptr,ty,write=None):
        live = ptr[self.active]
        if write is None and live.size and np.all(live >= 1<<32):
            tensor = int(live[0]>>32)
            assert np.all(live>>32 == tensor)
            unchanged = tensor != 1
            if tensor == 1:
                col = ((live-(1<<32))//2)%128
                fragment = int(col[0]//32)
                assert np.all(col//32 == fragment) and ty == '<8 x half>'
                assert np.all(col%32+8 <= 32)
                self.q_fragment_reads[fragment] += 1
                for offset in range(0,16,2):
                    self.q_elements[fragment].update((live+offset).tolist())
                unchanged = fragment >= 2
            if unchanged:
                self.unchanged_reads.update(str((tensor,ty)).encode())
                self.unchanged_reads.update(live.tobytes())
        return super().memory(ptr,ty,write)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    names = list(manifest['variants'])
    texts = {}
    for name in names:
        raw = (HERE/(name+'.ll')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['variants'][name]['sha256']
        texts[name] = re.sub(r'\bexact ', '',raw.decode())
    cases = []
    for seed,head,query in [(42,0,0),(42,1,1),(7,1,7),(7,1,8),(42,1,31)]:
        rng = np.random.default_rng(seed)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        forward = 31-query
        linear = forward//8*256+head*8+forward%8
        outputs,models,counts = {},{},{}
        n = query+1
        for name in names:
            retained = manifest['variants'][name]['retained_BK_fragments']
            model = FragmentBlock(texts[name],*arrays,physical=(linear%32,linear//32))
            full = model.run().reshape(arrays[0].shape)
            models[name],outputs[name] = model,full
            assert not model.pending
            assert model.q_fragment_reads == {f:2*(1 if f<retained else n) for f in range(4)}
            for fragment in range(4):
                repetitions = 1 if fragment<retained else n
                want = {(1<<32)+2*((head*4096+row)*128+col):repetitions
                        for row in range(query*128,(query+1)*128)
                        for col in range(fragment*32,(fragment+1)*32)}
                assert model.q_elements[fragment] == want,(name,fragment)
            assert model.read_calls['K'] == model.read_calls['V'] == 8*n
            assert model.lds_loads == 4*retained+(144-4*retained)*n
            assert model.q_scales == 4*retained+(16-4*retained)*n
            assert model.vm_waits == retained+(12-retained)*n
            assert model.barriers == retained+1+(16 if retained==1 else 14)*n
            assert model.mmac_count == 256*n and model.hints == 8*n
            assert np.array_equal(full,outputs[names[0]],equal_nan=True)
            assert model.unchanged_reads.digest() == models[names[0]].unchanged_reads.digest()
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            counts[name] = dict(q_fragment_DMA_calls=dict(model.q_fragment_reads),
                                lds_reads=model.lds_loads,q_scales=model.q_scales,
                                waits=model.vm_waits,barriers=model.barriers,
                                retained_elements_each_loaded_once=4096*retained,
                                bitwise_equal=True,KV_and_Q64_128_trace_identical=True,
                                cross_wave_barrier_epoch_audit_passed=True)
            print(f'PASS {name} head={head} query={query} '+json.dumps(counts[name]),flush=True)
        expected = audits.reference(*arrays,head,query)
        actual = outputs[names[-1]][head,query*128:(query+1)*128]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        cases.append(dict(seed=seed,head=head,query=query,variants=counts,
                          max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected)))))
    # Ensure new prologue synchronization is actually covered by the audit.
    text = texts['retained_16words']
    label = re.search(r'^([\w.]+):',text,re.M)[1]
    prologue = text.split('\n'+label+':',1)[0]
    waits = list(re.finditer(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n',prologue,re.M))
    barriers = list(re.finditer(r'^  call void @llvm.amdgcn.s.barrier\(\)[^\n]*\n',prologue,re.M))
    assert len(waits)==2 and len(barriers)==3
    negatives = []
    for name,match in [('missing_second_Q_wait',waits[1]),('missing_second_Q_publication',barriers[1])]:
        bad = text[:match.start()]+text[match.end():]
        try: FragmentBlock(bad,*arrays,physical=(7,24)).run()
        except AssertionError as e:
            assert 'DMA' in str(e) or 'cross-wave' in str(e),str(e)
            negatives.append(dict(case=name,rejected=str(e)))
            print('PASS negative',name,str(e),flush=True)
        else: raise AssertionError('audit failed to reject '+name)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=cases,hazard_negative_controls=negatives,
        gpu_tested=False,target_gfx936_codegen_tested=False,
        limitations='Actual LLVM CPU execution with delayed DMA and per-byte cross-wave barrier-epoch checks, not cycle-accurate GPU execution or VGPR allocation.'),indent=2)+'\n')


if __name__ == '__main__': main()
