#!/usr/bin/env python3
"""Execute 2/3/4-fragment LLVM exports and verify exact Q request multiplicity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('two_fragment_execution_checks',ROOT/'generated_ir/q_two_bk_retained/verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class SweepBlock(previous.FragmentBlock):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.kv_reads = hashlib.sha256()

    def memory(self,ptr,ty,write=None):
        live = ptr[self.active]
        if write is None and live.size and np.all(live >= 1<<32):
            tensor = int(live[0]>>32)
            if tensor in (2,3):
                self.kv_reads.update(str((tensor,ty)).encode())
                self.kv_reads.update(live.tobytes())
        return super().memory(ptr,ty,write)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    names = list(manifest['variants'])
    texts = {}
    for name in names:
        info = manifest['variants'][name]
        raw = (HERE/(name+'.ll')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        # The inherited per-byte hazard auditor covers this entire allocation.
        assert info['static_lds_bytes'] == 16384
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
            info = manifest['variants'][name]
            retained = info['retained_BK_fragments']
            model = SweepBlock(texts[name],*arrays,physical=(linear%32,linear//32))
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
            for key,actual in [('waits',model.vm_waits),('barriers',model.barriers)]:
                assert actual == info['counts']['entry'][key]+n*info['counts']['loop'][key]
            assert model.mmac_count == 256*n and model.hints == 8*n
            assert np.array_equal(full,outputs[names[0]],equal_nan=True)
            assert model.kv_reads.digest() == models[names[0]].kv_reads.digest()
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            counts[name] = dict(q_fragment_DMA_calls=dict(model.q_fragment_reads),
                                lds_reads=model.lds_loads,q_scales=model.q_scales,
                                waits=model.vm_waits,barriers=model.barriers,
                                retained_elements_each_loaded_once=4096*retained,
                                all_Q_element_multiplicities_checked=True,
                                bitwise_equal=True,KV_trace_identical=True,
                                cross_wave_barrier_epoch_audit_passed=True)
            print(f'PASS {name} head={head} query={query} '+json.dumps(counts[name]),flush=True)
        expected = previous.audits.reference(*arrays,head,query)
        actual = outputs[names[-1]][head,query*128:(query+1)*128]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        cases.append(dict(seed=seed,head=head,query=query,variants=counts,
                          max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected)))))
    negatives = []
    for variant in ('retained_24words','retained_32words'):
        text = texts[variant]
        label = re.search(r'^([\w.]+):',text,re.M)[1]
        prologue = text.split('\n'+label+':',1)[0]
        waits = list(re.finditer(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n',prologue,re.M))
        barriers = list(re.finditer(r'^  call void @llvm.amdgcn.s.barrier\(\)[^\n]*\n',prologue,re.M))
        publication = next(b for b in barriers if b.start()>waits[-1].end())
        for name,match in [('missing_last_Q_wait',waits[-1]),('missing_last_Q_publication',publication)]:
            bad = text[:match.start()]+text[match.end():]
            try: SweepBlock(bad,*arrays,physical=(7,24)).run()
            except AssertionError as e:
                assert 'DMA' in str(e) or 'cross-wave' in str(e),str(e)
                negatives.append(dict(variant=variant,case=name,rejected=str(e)))
                print('PASS negative',variant,name,str(e),flush=True)
            else: raise AssertionError('audit failed to reject '+variant+' '+name)
    (HERE/'validation.json').write_text(json.dumps(dict(
        sha256={n:manifest['variants'][n]['sha256'] for n in names},
        cases=cases,hazard_negative_controls=negatives,gpu_tested=False,
        target_gfx936_codegen_tested=False,
        limitations='Actual LLVM CPU execution with delayed DMA and barrier-epoch hazard checks; not cycle-accurate GPU execution or target register allocation.'),indent=2)+'\n')


if __name__ == '__main__': main()
