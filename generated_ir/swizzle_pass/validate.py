#!/usr/bin/env python3
"""Validate compiler-generated IR against the measured hand-patched winner."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'generated_ir/group8_reverse_k_swizzle'))
from validate import PhysicalBlock, reference
LLVM = Path('/data2/xsl/install/bin')
BASE = ROOT/'generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll'

def main():
    candidate = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'group8_reverse_k128.ll'
    before, after = BASE.read_text(), candidate.read_text()
    assert after.count('buffer_load_dwordx4 ') == 24
    assert len(re.findall(r'= load <4 x half>, ptr addrspace\(3\)', after)) == 144
    assert '[16384 x i8]' in after
    subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', candidate, '-disable-output'], check=True)
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    cases = []
    for head, query in ((0,0), (1,7), (1,31)):
        forward = 31-query
        linear = forward//8*256+head*8+forward%8
        outputs, counters = [], []
        for name, text in (('measured-winner',before), ('compiler-pass',after)):
            model = PhysicalBlock(re.sub(r"\bexact ", "", text), *arrays, physical=(linear%32,linear//32), head=head, block=query)
            full = model.run().reshape(arrays[0].shape)
            outputs.append(full[head, query*128:(query+1)*128].copy())
            written = np.zeros(full.shape, dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all() and not model.pending
            counters.append((model.vm_waits,model.barriers,model.mmac_count))
            print(f'Executed {name} head={head} query={query}', flush=True)
        assert np.array_equal(*outputs)
        assert counters[0] == counters[1]
        expected = reference(*arrays,head,query)
        assert np.allclose(outputs[1],expected,atol=.01,rtol=.01)
        cases.append({'head':head,'query':query,'bitwise_equal':True,'waits_barriers_mmac':counters[1],
                      'max_abs_error':float(np.max(np.abs(outputs[1].astype(np.float32)-expected)))})
        print('PASS', cases[-1],flush=True)
    result = {'candidate':str(candidate),'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'DMA128_sites':24,'b64_LDS_load_sites':144,
              'LDS_bytes':16384,'LLVM_O3_verify':True,'CPU_delayed_DMA':cases,
              'gpu_tested':False,'gfx936_codegen_tested':False}
    candidate.with_suffix('.validation.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__': main()
