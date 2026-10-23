#!/usr/bin/env python3
"""Execute all four LLVM inputs and audit DMA/LDS access equivalence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('previous_schedule_audits',ROOT/'generated_ir/retained_local_schedule/verify.py')
audits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audits)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
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
            model = audits.AuditedBlock(texts[name],*arrays,physical=(linear%32,linear//32))
            full = model.run().reshape(arrays[0].shape)
            models[name],outputs[name] = model,full
            n = query+1
            assert not model.pending
            assert model.mmac_count == 256*n
            assert model.barriers == 16*n+2 and model.vm_waits == 11*n+1
            assert model.lds_loads == 140*n+4 and model.q_scales == 12*n+4
            assert model.hints == manifest['variants'][name]['hints_static']*n
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
        expected = audits.reference(*arrays,head,query)
        actual = outputs['baseline'][head,query*128:(query+1)*128]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        cases.append(dict(seed=seed,head=head,query=query,variants=counts,
                          max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected)))))
    # The imported hazard checker must still detect a missing DMA wait.
    text = texts['baseline']
    match = re.search(r'^  call void asm sideeffect "s_waitcnt vmcnt\(0\)"[^\n]*\n',text,re.M)
    bad = text[:match.start()]+text[match.end():]
    try:
        audits.AuditedBlock(bad,*arrays,physical=(7,24)).run()
    except AssertionError as e:
        assert 'DMA' in str(e), str(e)
        negative = str(e)
        print('PASS negative missing_dma_wait',negative,flush=True)
    else:
        raise AssertionError('hazard checker failed to reject missing DMA wait')
    result = dict(sha256={n:manifest['variants'][n]['sha256'] for n in names},
                  cases=cases,negative_control=negative,gpu_tested=False,
                  limitations='CPU LLVM execution and barrier-epoch hazard checks; not cycle-accurate GPU execution or proof of target instruction combining.')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
