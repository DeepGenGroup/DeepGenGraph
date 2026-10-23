#!/usr/bin/env python3
"""Check native-accumulator IR structure; LDS totals are estimates, not GPU measurements."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

def llvm_shuffles(path):
    text = path.read_text().split('define amdgpu_kernel void @Attn_p2', 1)[1].split('\n}', 1)[0]
    blocks, current = {'entry': []}, 'entry'
    for line in text.splitlines()[1:]:
        match = re.match(r'([\w.]+):', line)
        if match:
            current = match[1]
            blocks[current] = []
        else:
            blocks[current].append(line)
    edges = {key: re.findall(r'label %([\w.]+)', '\n'.join(lines))
             for key, lines in blocks.items()}

    def cyclic(key):
        pending, seen = list(edges[key]), set()
        while pending:
            other = pending.pop()
            if other == key:
                return True
            if other not in seen:
                seen.add(other)
                pending.extend(edges.get(other, []))
        return False

    counts = Counter()
    for key, lines in blocks.items():
        counts['inside_loop' if cyclic(key) else 'outside_loop'] += sum(
            'call ' in line and '@llvm.amdgcn.ds.bpermute' in line for line in lines)
    return dict(counts)

path = Path(sys.argv[1]).resolve()
bn = int(sys.argv[2])
text = path.read_text()
body = text.split('define amdgpu_kernel void @Attn_p2', 1)[1].split('\n}', 1)[0]
shuffles = llvm_shuffles(path)
assert shuffles == {'outside_loop': 64, 'inside_loop': 8}, shuffles
mmac = [line for line in body.splitlines() if 'call ' in line and 'v_mmac_' in line]
zero = sum('<4 x float> zeroinitializer' in line for line in mmac)
fadds = len(re.findall(r'\bfadd <4 x float>', body))
assert zero == bn // 8, zero  # only QK chains start with zero
assert fadds == bn // 8, fadds  # only score masking; no PV + O adds
report = dict(ir=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              bn=bn, shuffle_sites=shuffles, zero_seed_mmac=zero,
              vector4_fadds=fadds, mmac_instructions=len(mmac), gpu_tested=False)
if bn == 128:
    waves = 32 * 32 * 4
    iterations = 32 * sum(range(1,33)) * 4
    report['estimated_shared_ops'] = (8+64)*waves + (64+8)*iterations
    report['previous_measured_shared_ops'] = 9224192
    report['shared_ops_note'] = 'Fixed causal 1x32x4096x128, BM=BN=128. Assumes unchanged QKV LDS codegen; requires GPU PMC validation.'
print(json.dumps(report, indent=2))
