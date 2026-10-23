#!/usr/bin/env python3
"""Execute the generated LLVM pair and audit the eliminated global requests."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'generated_ir/q_bk32_retained'))
from validate import MeasuredBlock, reference


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    names = ['baseline', 'retained_8words', 'historical_best']
    paths = [HERE / (n+'.ll') for n in names[:2]] + [ROOT / manifest['historical_best']]
    texts = [re.sub(r'\bexact ', '', p.read_text()) for p in paths]
    for name, path in zip(names[:2], paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['variants'][name]['sha256']
    candidate = texts[1]
    label = re.search(r'^([\w.]+):', candidate, re.M)[1]
    prologue, body = candidate.split('\n'+label+':', 1)
    assert prologue.count('buffer_load_dwordx4 ') == 2 and body.count('buffer_load_dwordx4 ') == 22
    loads = list(re.finditer(r'= load <4 x half>, ptr addrspace\(3\)', prologue))
    barriers = list(re.finditer(r'call void @llvm.amdgcn.s.barrier\(', prologue))
    assert len(loads) == 4 and len(barriers) == 2
    assert barriers[0].start() < loads[0].start() < loads[-1].start() < barriers[1].start()
    cases = []
    for seed, head, query in [(42,0,0), (42,1,1), (7,1,7), (7,1,8), (42,1,31)]:
        rng = np.random.default_rng(seed)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        forward = 31-query
        physical = forward//8*256+head*8+forward%8
        models = [MeasuredBlock(t, *arrays, physical=(physical%32, physical//32)) for t in texts]
        outputs = []
        for name, model in zip(names, models):
            outputs.append(model.run().reshape(arrays[0].shape))
            print(f'Executed {name} head={head} query={query}', flush=True)
        assert np.array_equal(outputs[0], outputs[1], equal_nan=True), 'pass OFF/ON differs'
        assert np.array_equal(outputs[0], outputs[2], equal_nan=True), 'regenerated baseline differs from historical winner'
        n = query+1
        a, b, historical = models
        assert a.read_calls == historical.read_calls == dict(Q0=2*n,Qrest=6*n,K=8*n,V=8*n)
        assert b.read_calls == dict(Q0=2,Qrest=6*n,K=8*n,V=8*n), b.read_calls
        assert a.other_reads.digest() == b.other_reads.digest() == historical.other_reads.digest()
        want = {(1<<32)+2*((head*4096+row)*128+col):1
                for row in range(query*128,(query+1)*128) for col in range(32)}
        assert b.q0_addresses == want
        assert a.lds_loads == 144*n and b.lds_loads == 4+140*n
        assert a.q_scales == 16*n and b.q_scales == 4+12*n
        assert a.vm_waits == 12*n and b.vm_waits == 1+11*n
        assert a.barriers == 16*n and b.barriers == 2+16*n
        for model, output in zip(models, outputs):
            assert not model.pending
            assert model.mmac_count == 256*n
            written = np.zeros(output.shape, dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(output[~written]).all()
        expected = reference(*arrays,head,query)
        actual = outputs[1][head,query*128:(query+1)*128]
        assert np.allclose(actual, expected, atol=.01, rtol=.01)
        case = dict(seed=seed, head=head, query=query,
                    bitwise_equal_pass_off_on_and_historical=True,
                    kv_and_other_q_read_trace_identical=True,
                    retained_q_elements=4096, each_loaded_once=True,
                    baseline=dict(read_calls=dict(a.read_calls), lds_loads=a.lds_loads,
                                  barriers=a.barriers, vm_waits=a.vm_waits),
                    candidate=dict(read_calls=dict(b.read_calls), lds_loads=b.lds_loads,
                                   barriers=b.barriers, vm_waits=b.vm_waits),
                    max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected))))
        cases.append(case)
        print('PASS', json.dumps(case), flush=True)
    (HERE / 'validation.json').write_text(json.dumps(dict(cases=cases,
        prologue_publication_and_storage_handoff_barriers_checked=True,
        gpu_tested=False, target_gfx936_codegen_tested=False,
        limitations='Collective CPU execution checks DMA readiness and arithmetic, not independent-wave timing, VGPR allocation or device performance.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
