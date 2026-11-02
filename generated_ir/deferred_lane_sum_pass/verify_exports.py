#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re
import subprocess

here = Path(__file__).resolve().parent
root = sorted((here/'compiler_final').glob('run-*'))[-1]
records = []
for case in sorted(root.glob('input-*')):
    results = [json.loads(p.read_text()) for p in sorted(case.glob('grid-*/result.json'))]
    assert len(results) == 3 and all(r['status'] == 'verified' for r in results)
    assert results[0]['normalized_sha256'] == results[1]['normalized_sha256']
    assert results[1]['defer_lane_sum_reports'][0]['states'] == 0
    report = results[2]['defer_lane_sum_reports'][0]
    assert report['states'] == 2 and report['loop_shuffles'] == 4
    ir = Path(results[2]['llvm_ir'])
    optimized = ir.with_name('kernel.o3.ll')
    subprocess.run(['/data2/xsl/install/bin/opt', '-passes=default<O3>,verify', '-S', str(ir), '-o', str(optimized)], check=True)
    text = optimized.read_text()
    body = text.split('define amdgpu_kernel void ', 1)[1].split('\n}', 1)[0]
    blocks, current = {}, '4'
    for line in body.splitlines()[1:]:
        match = re.match(r'^([\w.-]+):', line)
        if match: current = match[1]
        blocks.setdefault(current, []).append(line)
    edges = {b: re.findall(r'label %([\w.-]+)', '\n'.join(lines)) for b, lines in blocks.items()}
    def cyclic(start):
        pending, seen = list(edges[start]), set()
        while pending:
            node = pending.pop()
            if node == start: return True
            if node in seen: continue
            seen.add(node); pending.extend(edges[node])
        return False
    counts = [0, 0]
    for block, lines in blocks.items():
        counts[0 if cyclic(block) else 1] += '\n'.join(lines).count('call i32 @llvm.amdgcn.ds.bpermute(')
    assert counts == [4, 68], counts
    records.append(dict(input=case.name, off_legal_identical=True, report=report,
                        O3_loop_shuffles=counts[0], O3_outside_shuffles=counts[1], llvm_ir=str(ir)))
validation = json.loads((here/'validation.json').read_text())
assert validation['experiment_sha256'] == hashlib.sha256((here/'kernel.ll').read_bytes()).hexdigest()
result = dict(exports=records, gpu_tested=False, kernel_sha256=validation['experiment_sha256'],
              generic=json.loads((here/'generic-validation.json').read_text()),
              attention_cases=len(validation['cases']),
              full_pipeline_bn128=json.loads((here/'full-bn128-validation.json').read_text()),
              full_pipeline_bn64=json.loads((here/'full-bn64-validation.json').read_text()))
(here/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(records,indent=2))
