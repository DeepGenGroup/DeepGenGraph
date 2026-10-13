#!/usr/bin/env python3
"""Prefetch preserves synchronization/alias boundaries and the register budget."""
import re
import subprocess
import sys
from pathlib import Path

tool = Path(sys.argv[1]).resolve()
source = '''target triple = "amdgcn-amd-amdhsa"
declare float @__ocml_exp2_f32(float)
declare void @llvm.amdgcn.s.barrier()
declare void @opaque()
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
'''
for name, boundary in [('valid', ''), ('shuffle', '%s = call i32 @llvm.amdgcn.ds.bpermute(i32 0, i32 0)'),
                       ('barrier', 'call void @llvm.amdgcn.s.barrier()'),
                       ('unknown', 'call void @opaque()'),
                       ('alias_store', 'store half 0xH0000, ptr addrspace(1) %p'),
                       ('fence', 'fence seq_cst')]:
    source += f'''define amdgpu_kernel void @{name}(ptr addrspace(1) %p, ptr addrspace(1) %out, float %x) {{
  %e = call float @__ocml_exp2_f32(float %x)
  {boundary}
'''
    for i in range(5):
        source += f'''  %p{i} = getelementptr half, ptr addrspace(1) %p, i32 {i*4}
  %v{i} = load <4 x half>, ptr addrspace(1) %p{i}
'''
    for i in range(5):
        source += f'''  %o{i} = getelementptr half, ptr addrspace(1) %out, i32 {i*4}
  store <4 x half> %v{i}, ptr addrspace(1) %o{i}
'''
    source += '  store float %e, ptr addrspace(1) %out\n  ret void\n}\n'
source += '''define amdgpu_kernel void @volatile(ptr addrspace(1) %p, float %x) {
  %e = call float @__ocml_exp2_f32(float %x)
  %v = load volatile float, ptr addrspace(1) %p
  %r = fadd float %e, %v
  store float %r, ptr addrspace(1) %p
  ret void
}
'''

def run(text):
    p = subprocess.run([str(tool), '--prefetch-global-packets', '-'], input=text,
                       text=True, capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout

output = run(source)
assert run(output) == output
for name in ('valid', 'shuffle', 'barrier', 'unknown', 'alias_store', 'fence', 'volatile'):
    body = re.search(r'define [^@]*@' + name + r'\(.*?\n\}', output, re.S)[0]
    early = body.split('call float @__ocml_exp2_f32')[0]
    assert early.count(' = load ') == (4 if name in ('valid', 'shuffle') else 0), name
    assert body.count(' = load ') == (1 if name == 'volatile' else 5)
print('PASS packet prefetch: 8-word budget, aliasing writes/unknown calls/barriers/fences/volatile boundaries, idempotence')
