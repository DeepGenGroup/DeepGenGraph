#!/usr/bin/env python3
"""Check QK accumulator chains in the two D=128/BK=32 attention exports."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def structure(path):
    text = path.read_text()
    mmacs = [line for line in text.splitlines() if ' = call ' in line and 'v_mmac_f32_' in line]
    assert len(mmacs) == 256, (path, len(mmacs))
    qk = {}
    for line in mmacs[:128]:
        result = line.split(' = ', 1)[0].strip()
        seed = re.findall(r'<4 x float> (%[-\w.]+|zeroinitializer)', line)[-1]
        qk[result] = seed
    terminals = qk.keys() - set(qk.values())
    lengths = []
    for result in terminals:
        length = 0
        while result in qk:
            length += 1
            result = qk[result]
        lengths.append(length)
    return dict(qk_chain_lengths=dict(sorted(Counter(lengths).items())),
                qk_terminal_fp32_elements=len(terminals)*4,
                mmac_calls=len(mmacs),
                lds_bytes=int(re.search(r'@shm_0 = .*?\[(\d+) x i8\]', text)[1]),
                barrier_calls=text.count('call void @llvm.amdgcn.s.barrier()'),
                dma128_calls=text.count('buffer_load_dwordx4'),
                bpermute_calls=len(re.findall(r'= call i32 @llvm.amdgcn.ds.bpermute', text)),
                float_lds_stores=len(re.findall(r'store float .*?ptr addrspace\(3\)', text)),
                float_lds_loads=len(re.findall(r'= load float, ptr addrspace\(3\)', text)))

if __name__ == '__main__':
    results = {}
    for name in ('layout_3_1', 'layout_3_3'):
        result = structure(ROOT / name / 'kernel.ll')
        assert result['qk_chain_lengths'] == {8: 16}, result
        results[name] = result
    diagnostic = structure(ROOT.parent / 'spill_analysis/kernel.ll')
    assert results['layout_3_1'] == diagnostic, (results, diagnostic)
    results['diagnostic_3_1'] = diagnostic
    (ROOT / 'structure.json').write_text(json.dumps(results, indent=2) + '\n')
    print('PASS QK chains: 16 x 8; 3.1 instruction counts match measured diagnostic')
