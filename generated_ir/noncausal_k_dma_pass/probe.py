#!/usr/bin/env python3
"""Compile the exported K pointer DAG alone; this is not gfx936 codegen."""
import collections
import json
from pathlib import Path
import re
import subprocess
import sys
from validate import HERE, LLVM, LazyAddresses


class Addresses(LazyAddresses):
    def value(self,name,ty=None):
        m=re.fullmatch(r'ptrtoint \((ptr addrspace\(3\) @[\w.]+) to i(?:32|64)\)',name)
        if m:return self.typed(m[1])
        return super().value(name,ty)


def main():
    source=HERE/'k_swizzle64.ll'
    text=source.read_text();model=Addresses(text);model.reset()
    pointers=[];seen_addresses=set()
    # Archived fixture loads Q once in entry, then 64 K b64 sites in the loop.
    for p in re.findall(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+)',text)[16:]:
        key=tuple(model.value(p))
        if key not in seen_addresses:pointers.append(p);seen_addresses.add(key)
    assert len(pointers)==32
    seen=set();ordered=[]
    def emit(n):
        if n in seen:return
        seen.add(n)
        expr=re.sub(r', ![\w.]+ !\d+', '',model.defs[n])
        expr=re.sub(r' #\d+', ' #2',expr)
        assert not expr.startswith(('load ','phi '))
        for dep in re.findall(r'%[\w.]+',expr):emit(dep)
        ordered.append(f'  {n} = {expr}')
    for p in pointers:emit(p)
    tid=next(n for n in seen if '@llvm.amdgcn.workitem.id.x(' in model.defs[n])
    body=['target triple = "amdgcn-amd-amdhsa"',
          '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16',
          'declare i32 @llvm.amdgcn.workitem.id.x() #0',
          'define amdgpu_kernel void @probe(ptr addrspace(1) %probe.out) #1 {',*ordered]
    for i,p in enumerate(pointers):body.append(f'  %probe.value{i} = load <4 x half>, ptr addrspace(3) {p}, align 8')
    body += [f'  %probe.off = shl i32 {tid}, 8',
             '  %probe.base = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off']
    for i in range(32):body += [f'  %probe.dst{i} = getelementptr i8, ptr addrspace(1) %probe.base, i32 {8*i}',
                               f'  store <4 x half> %probe.value{i}, ptr addrspace(1) %probe.dst{i}, align 8']
    body += ['  ret void','}', 'attributes #0 = { nounwind memory(none) }',
             'attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }',
             'attributes #2 = { convergent nounwind memory(none) }']
    p=HERE/'read_probe.ll';p.write_text(re.sub(r'%(\d+)\b',r'%ssa\1','\n'.join(body))+'\n')
    subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',p,'-o',HERE/'read_probe.o3.ll'],check=True)
    subprocess.run([LLVM/'llc','-march=amdgcn','-mcpu=gfx900','-O3',HERE/'read_probe.o3.ll','-o',HERE/'read_probe.gfx900.s'],check=True)
    counts=dict(collections.Counter(re.findall(r'^\s+(ds_read\w*)\s',(HERE/'read_probe.gfx900.s').read_text(),re.M)))
    assert counts=={'ds_read2_b64':16},counts
    (HERE/'read_probe.json').write_text(json.dumps(dict(unique_K_pointers=32,gfx900_isolated_reads=counts,
        gfx936_full_kernel_tested=False),indent=2)+'\n')
    print('PASS gfx900 isolated K read probe:',counts)


if __name__=='__main__':main()
