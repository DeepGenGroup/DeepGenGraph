#!/usr/bin/env python3
"""Bounded LDS reuse: synchronization, aliasing, budgets and CPU execution."""
import ctypes as C
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

tool = Path(sys.argv[1]).resolve()
llvm = Path('/data2/xsl/install/bin')


def run(text):
    result = subprocess.run([tool, '--reuse-lds-fragments', '-'], input=text,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


source = '''target triple = "amdgcn-amd-amdhsa"
declare void @llvm.amdgcn.s.barrier()
declare void @opaque(ptr addrspace(3))
declare void @readonly_opaque(ptr addrspace(3)) memory(read)
'''
expected = {}
for width in (2, 4, 8, 16):
    for name, boundary in [
        ('plain', ''), ('barrier', 'call void @llvm.amdgcn.s.barrier()'),
        ('write', 'store float 7.0, ptr addrspace(3) %alias'),
        ('opaque', 'call void @opaque(ptr addrspace(3) %p)'),
        ('readonly', 'call void @readonly_opaque(ptr addrspace(3) %p)'),
        ('fence', 'fence seq_cst'),
        ('volatile', '%v = load volatile float, ptr addrspace(3) %p'),
        ('atomic', '%a = atomicrmw fadd ptr addrspace(3) %p, float 1.0 seq_cst'),
    ]:
        name = f'{name}_{width}'
        vector = f'<{width} x float>'
        source += f'''define amdgpu_kernel void @{name}(ptr addrspace(3) %p, ptr addrspace(3) %alias, ptr %out) {{
  %x = load {vector}, ptr addrspace(3) %p, align 4
  {boundary}
  %y = load {vector}, ptr addrspace(3) %p, align 4
  %z = fadd {vector} %x, %y
  store {vector} %z, ptr %out, align 4
  ret void
}}
'''
        expected[name] = 1 if name.startswith('plain_') and width <= 8 else 2

# Three overlapping four-word packets exceed an eight-word retention budget.
source += '''define amdgpu_kernel void @budget(ptr addrspace(3) %p, ptr %out) {
  %p1 = getelementptr float, ptr addrspace(3) %p, i32 4
  %p2 = getelementptr float, ptr addrspace(3) %p, i32 8
  %a = load <4 x float>, ptr addrspace(3) %p
  %b = load <4 x float>, ptr addrspace(3) %p1
  %c = load <4 x float>, ptr addrspace(3) %p2
  %a1 = load <4 x float>, ptr addrspace(3) %p
  %b1 = load <4 x float>, ptr addrspace(3) %p1
  %c1 = load <4 x float>, ptr addrspace(3) %p2
  %ab = fadd <4 x float> %a, %b
  %abc = fadd <4 x float> %ab, %c
  %ab1 = fadd <4 x float> %a1, %b1
  %abc1 = fadd <4 x float> %ab1, %c1
  %r = fadd <4 x float> %abc, %abc1
  store <4 x float> %r, ptr %out
  ret void
}
define amdgpu_kernel void @branch(ptr addrspace(3) %p, ptr %out, i1 %cond) {
entry:
  %a = load <4 x float>, ptr addrspace(3) %p
  br i1 %cond, label %left, label %right
left:
  store float 9.0, ptr addrspace(3) %p
  br label %right
right:
  %b = load <4 x float>, ptr addrspace(3) %p
  %r = fadd <4 x float> %a, %b
  store <4 x float> %r, ptr %out
  ret void
}
'''
expected.update(budget=4, branch=2)
output = run(source)
assert run(output) == output, 'must not accumulate retention budgets on repeated runs'
for name, count in expected.items():
    body = re.search(r'define [^@]*@' + name + r'\(.*?\n\}', output, re.S)[0]
    assert len(re.findall(r' = load <', body)) == count, (name, body)
subprocess.run([llvm / 'opt', '-passes=verify', '-disable-output', '-'], input=output,
               text=True, check=True, capture_output=True)

# Execute the actual before/after LLVM as host functions, using real aliases
# and mutations. Only target plumbing is replaced, never the load/add/store.
def host_ir(ir):
    ir = re.sub(r'^target (triple|datalayout).*$', '', ir, flags=re.M)
    ir = ir.replace('amdgpu_kernel ', '').replace(' addrspace(3)', '')
    ir = ir.replace('@llvm.amdgcn.s.barrier()', '@cpu_barrier()')
    ir = re.sub(r'^declare void @(opaque|readonly_opaque|cpu_barrier)\([^\n]*$', '', ir, flags=re.M)
    return ir + '''
define void @cpu_barrier() { ret void }
define void @opaque(ptr %p) { store float 11.0, ptr %p ret void }
define void @readonly_opaque(ptr %p) { ret void }
'''


with tempfile.TemporaryDirectory() as tmp:
    libs = []
    for index, ir in enumerate((source, output)):
        base = Path(tmp) / f'kernel{index}'
        base.with_suffix('.ll').write_text(host_ir(ir))
        subprocess.run([llvm / 'llc', '-mtriple=x86_64-unknown-linux-gnu',
                        '-relocation-model=pic', '-filetype=obj', base.with_suffix('.ll'),
                        '-o', base.with_suffix('.o')], check=True, capture_output=True)
        subprocess.run(['gcc', '-shared', base.with_suffix('.o'), '-o', base.with_suffix('.so')],
                       check=True, capture_output=True)
        libs.append(C.CDLL(str(base.with_suffix('.so'))))
    random.seed(37)
    for name in expected:
        for _ in range(10):
            values = [random.uniform(-8, 8) for _ in range(32)]
            results = []
            for lib in libs:
                data, out = (C.c_float * 32)(*values), (C.c_float * 32)()
                fn = getattr(lib, name)
                if name == 'budget':
                    fn.argtypes = [C.c_void_p, C.c_void_p]
                    fn(data, out)
                elif name == 'branch':
                    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_bool]
                    fn(data, out, values[0] > 0)
                else:
                    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
                    fn(data, data, out)
                results.append((bytes(data), bytes(out)))
            assert results[0] == results[1], name
print('PASS LDS reuse: vector widths, overlapping budgets, alias writes, volatile/atomic/fence/barrier/call/CFG boundaries, idempotence and randomized CPU equivalence')
