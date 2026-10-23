#!/usr/bin/env python3
"""Validate the best-kernel A/B using actual LLVM execution and memory traces."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'3rd/deepgengraph/test'))
from check_pipeline_cleanup_attention import CountedDMA
from check_attention_llvm_numerics import reference
from check_gemm_pv_addresses import split_args


class AuditedBlock(CountedDMA):
    def __init__(self, *args, physical, **kwargs):
        self.physical = physical
        self.trace = hashlib.sha256()
        self.aligned_loads_checked = 0
        super().__init__(*args, **kwargs)

    def call(self, body):
        for axis, value in zip('xyz', (*self.physical, 0)):
            if f'@llvm.amdgcn.workgroup.id.{axis}(' in body:
                return np.full(self.t, value, dtype=np.int64)
        return super().call(body)

    def memory(self, ptr, ty, write=None):
        self.trace.update(str((ty, write is not None)).encode())
        self.trace.update(np.asarray(ptr, dtype=np.int64).tobytes())
        self.trace.update(self.active.tobytes())
        return super().memory(ptr, ty, write)

    def operation(self, expr):
        if expr.startswith('load ') and 'ptr addrspace(3)' in expr:
            _, pointer, *_ = split_args(expr[5:])
            alignment = int(re.search(r', align (\d+)', expr)[1])
            assert np.all(self.typed(pointer)[self.active] % alignment == 0), expr
            self.aligned_loads_checked += int(np.sum(self.active))
        return super().operation(expr)


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    texts = []
    for name in ('baseline','optimized'):
        data = (HERE/f'{name}.ll').read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[f'{name}_sha256']
        texts.append(data.decode())
    assert (HERE/'baseline.ll').read_bytes() == Path(manifest['source']).read_bytes()
    cases = []
    for seed, head, query in ((42,0,0), (42,1,7), (7,1,8), (7,1,31)):
        rng = np.random.default_rng(seed)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        forward = 31-query
        linear = (forward//8)*256 + head*8 + forward%8
        models = [AuditedBlock(text, *arrays, physical=(linear%32,linear//32)) for text in texts]
        results = [m.run().reshape(arrays[0].shape) for m in models]
        assert np.array_equal(*results, equal_nan=True), 'A/B differs bitwise'
        assert models[0].trace.digest() == models[1].trace.digest(), 'memory access order/addresses differ'
        for m in models:
            assert not m.pending
            assert m.mmac_count == 256*(query+1)
            assert m.barriers == 16*(query+1)
            assert m.vm_waits == 12*(query+1)
        rows = slice(query*128,(query+1)*128)
        expected = reference(*arrays,head,query)
        actual = results[1][head,rows]
        assert np.allclose(actual,expected,atol=.01,rtol=.01)
        written = np.zeros(results[1].shape,dtype=bool)
        written[head,rows] = True
        assert np.isnan(results[1][~written]).all()
        case = dict(seed=seed, head=head, query=query, bitwise_equal=True,
            identical_memory_trace=True, no_pending_dma=True,
            aligned_lane_loads_checked=models[1].aligned_loads_checked,
            barriers=models[1].barriers, vm_waits=models[1].vm_waits,
            max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-expected))))
        cases.append(case)
        print('PASS',json.dumps(case),flush=True)
    result = dict(baseline_sha256=manifest['baseline_sha256'],
        optimized_sha256=manifest['optimized_sha256'], cpu_delayed_dma=cases,
        gpu_tested=False, target_gfx936_codegen_tested=False,
        limitations='CPU model does not simulate independently progressing waves or target backend timing.')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
