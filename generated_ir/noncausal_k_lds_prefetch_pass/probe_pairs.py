#!/usr/bin/env python3
"""Isolated gfx900 codegen of every selected pair, not a full gfx936 kernel."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
from validate import HERE, LLVM, comparison, check_pairs


def main():
    text=(HERE/'k_lds_prefetch2.o3.ll').read_text()
    assert check_pairs(text)==23
    model=comparison.Addresses(text)
    pairs=defaultdict(list)
    for line in text.splitlines():
        m=re.search(r'load <4 x half>, ptr addrspace\(3\) (%[\w.]+).*!frisk.k_lds_pair !(\d+)',line)
        if m:pairs[m[2]].append(m[1])
    body=['target triple = "amdgcn-amd-amdhsa"',
          '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16']
    declarations=set()
    for index,pointers in enumerate(pairs.values()):
        seen=set(); ordered=[]
        def emit(n):
            if n in seen:return
            seen.add(n)
            expr=re.sub(r', ![\w.]+ !\d+', '',model.defs[n])
            expr=re.sub(r' #\d+', ' #2',expr)
            assert not expr.startswith('load ')
            if expr.startswith('phi i64 '):expr='add i64 0, 0'
            else:
                for dep in re.findall(r'%[\w.]+',expr):emit(dep)
            for call in re.findall(r'@(llvm\.amdgcn\.[\w.]+)\(',expr):
                assert call in ('llvm.amdgcn.workitem.id.x','llvm.amdgcn.workgroup.id.x','llvm.amdgcn.workgroup.id.y')
                declarations.add('declare i32 @'+call+'() #0')
            ordered.append(f'  {n} = {expr}')
        for p in pointers:emit(p)
        tid=next(n for n in seen if '@llvm.amdgcn.workitem.id.x(' in model.defs[n])
        body += [f'define amdgpu_kernel void @pair{index}(ptr addrspace(1) %probe.out) #1 {{',*ordered,
                 f'  %probe.a = load <4 x half>, ptr addrspace(3) {pointers[0]}, align 8',
                 f'  %probe.b = load <4 x half>, ptr addrspace(3) {pointers[1]}, align 8',
                 f'  %probe.off = shl i32 {tid}, 4',
                 '  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off',
                 '  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8',
                 '  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8',
                 '  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8','  ret void','}']
    body += sorted(declarations)+['attributes #0 = { nounwind memory(none) }',
             'attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }',
             'attributes #2 = { convergent nounwind memory(none) }']
    source=HERE/'pair_probe.ll'
    source.write_text(re.sub(r'%(\d+)\b',r'%ssa\1','\n'.join(body))+'\n')
    optimized=source.with_suffix('.o3.ll');asm=source.with_suffix('.gfx900.s')
    subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',source,'-o',optimized],check=True)
    subprocess.run([LLVM/'llc','-march=amdgcn','-mcpu=gfx900','-O3',optimized,'-o',asm],check=True)
    counts=Counter(re.findall(r'^\s+(ds_read\w*)\s',asm.read_text(),re.M))
    assert sum(counts.values())==23 and all(k.startswith('ds_read2') for k in counts),counts
    (HERE/'pair_probe.json').write_text(json.dumps(dict(isolated_pair_count=23,counts=dict(counts),
        target='gfx900',gfx936_full_kernel_tested=False),indent=2)+'\n')
    print('PASS isolated pair codegen:',dict(counts))


if __name__=='__main__':main()
