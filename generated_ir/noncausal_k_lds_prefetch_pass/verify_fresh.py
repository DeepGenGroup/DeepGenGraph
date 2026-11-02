#!/usr/bin/env python3
"""Verify the final full-MyTest export and equivalence to standalone pass output."""
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from validate import HERE, LLVM, kdma, check_pairs


def main():
    source=HERE/'mytest_k_lds_prefetch2.ll'
    optimized=[]
    for p in (HERE/'k_lds_prefetch2.ll',source):
        q=p.with_suffix('.stripped.o3.ll')
        subprocess.run([LLVM/'opt','-S','-strip-debug','-passes=default<O3>,verify',p,'-o',q],check=True)
        assert check_pairs(q.read_text())==23
        optimized.append(q)
    diff=subprocess.run([LLVM/'llvm-diff',*optimized],text=True,capture_output=True)
    (HERE/'fresh_llvm_diff.log').write_text(diff.stdout+diff.stderr)
    assert diff.returncode==0,diff.stdout+diff.stderr
    rng=np.random.default_rng(97)
    arrays=[rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    m=kdma.Model(re.sub(r'\bexact ','',source.read_text()),*arrays,threads=256,physical=(31,1))
    actual=m.run().reshape(arrays[0].shape)[1,31*128:32*128]
    expected=kdma.vdb.reference(*arrays,1,31,causal=False)
    np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
    assert not m.pending and (m.vm_waits,m.barriers,m.mmac_count,m.dma64_count)==(260,293,8192,512)
    data=json.loads((HERE/'fresh_artifact.json').read_text())
    data.update(llvm_diff_after_O3_equal=True,numerical_reference_passed=True,
                max_abs_error=float(np.max(np.abs(actual-expected))),pair_protection_after_O3=True)
    (HERE/'fresh_artifact.json').write_text(json.dumps(data,indent=2)+'\n')
    print('PASS fresh MyTest: llvm-diff O3 equality, 23 protected pairs, numerical reference and delayed DMA')


if __name__=='__main__':main()
