#!/usr/bin/env python3
"""Actual LLVM CPU execution with delayed DMA and LDS ownership checks."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
helper = HERE.parent/'noncausal_k_dma_pass/validate.py'
spec = importlib.util.spec_from_file_location('softmax_kdma_validation', helper)
kdma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kdma)


def model_text(text):
    # The existing interpreter initializes argument slots %0..%3 and entry %4.
    # instnamer only changes names; reserve these slots for this interpreter,
    # without touching delivered IR, arithmetic, control flow or scheduling.
    signature = re.search(r'define amdgpu_kernel void @Attn_p2\(([^\n]*)\)[^\n]*\{\n',text)
    args = re.findall(r'%[\w.]+',signature[1])
    assert len(args) == 4
    # O3 can introduce numeric SSA temporaries even when all inputs are named.
    # Reserve the interpreter's slots without colliding with those temporaries.
    assert '%model.slot' not in text
    names = {n:'%model.slot'+n[1:] for n in re.findall(r'%\d+\b',text)}
    names.update(zip(args,['%0','%1','%2','%3']))
    rest = text[signature.end():]
    entry = re.match(r'([\w.]+):',rest)
    if entry:
        names['%'+entry[1]] = '%4'
    else:
        names['%4'] = '%4'
    text = re.sub(r'^([\w.]+):',lambda m:names.get('%'+m[1],'%'+m[1])[1:]+':',text,flags=re.M)
    text = re.sub(r'%[\w.]+',lambda m:names.get(m[0],m[0]),text)
    return re.sub(r'\bexact ', '',text)


def main():
    records = []
    cases = [('random',0,0), ('random',1,31), ('positive_overflow',1,1),
             ('negative_underflow',1,1), ('max_changes',1,2), ('future_key_dominates',1,0)]
    for kind,head,query in cases:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        q,k,v = arrays
        if kind == 'positive_overflow': q.fill(16); k.fill(16)
        if kind == 'negative_underflow': q.fill(16); k.fill(-16)
        if kind == 'max_changes':
            q.fill(1); k.fill(-16); k[:,:128]=.25; k[:,128:256]=.5
        if kind == 'future_key_dominates':
            q.fill(1); k.fill(-1); k[:,-128:]=1; v.fill(0); v[:,-128:]=1
        expected = kdma.vdb.reference(*arrays,head,query,causal=False)
        baseline = trace = values = None
        names = ['control', 'softmax_fragment', 'softmax_pv']
        if kind == 'random' and query == 0:
            names += ['control.o3', 'softmax_fragment.o3', 'softmax_pv.o3']
        for name in names:
            path = HERE/(name+'.ll')
            model = kdma.Model(model_text(path.read_text()), *arrays,
                               threads=256, physical=(query,head))
            full = model.run().reshape(q.shape)
            actual = full[head,query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            if baseline is None:
                baseline = full.copy()
                trace = (model.v_reads,model.v_dma)
                values = model.lds_values.hexdigest()
            assert np.array_equal(full,baseline,equal_nan=True), (kind,name)
            assert trace == (model.v_reads,model.v_dma)
            assert values == model.lds_values.hexdigest()
            assert not model.pending
            assert (model.vm_waits,model.barriers,model.mmac_count,model.dma64_count) == (260,293,8192,512)
            assert model.v_reads_with_pending_dma == 768
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert np.isnan(full).sum() == full.size-16384
            records.append(dict(kind=kind,head=head,query=query,variant=name,sha256=kdma.digest(path),
                bitwise_equal_to_control=True, reference_atol=.01,reference_rtol=.01,
                max_abs_error=float(np.max(np.abs(actual-expected))),
                identical_LDS_values_and_V_schedule=True, delayed_DMA_and_ownership_passed=True))
            print('PASS',kind,head,query,name,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,gpu_tested=False,
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__ == '__main__': main()
