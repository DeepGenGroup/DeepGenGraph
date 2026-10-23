#!/usr/bin/env python3
"""Extract actual K pointer DAGs; inspect local gfx900 read2 selection only."""
import collections
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'3rd/deepgengraph/test'))
from check_attention_llvm_numerics import LLVMWorkgroupModel
from check_packed_lds_banks import multiplicity
class Addresses(LLVMWorkgroupModel):
    def __init__(self,text):
        dummy=np.zeros(1,dtype=np.float16)
        super().__init__(text,dummy,dummy,dummy)
        self.definitions=dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, ![\w.]+ !\d+.*)?$',text,re.M))
    def value(self,s,ty=None):
        if s not in self.values and s in self.definitions:
            self.values[s]=self.operation(re.sub(r'\bexact ', '',self.definitions[s]))
        return super().value(s,ty)
def main():
    candidate=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'group8_reverse_k128.ll'
    text=candidate.read_text()
    model=Addresses(text)
    selected=[]; seen_addresses=set(); banks=collections.Counter()
    for pointer in re.findall(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+)',text):
        addresses=model.value(pointer)
        if not np.all(addresses>=8192): continue
        assert np.all(addresses+8<=16384) and np.all(addresses%8==0)
        key=tuple(addresses)
        if key in seen_addresses: continue
        seen_addresses.add(key); selected.append(pointer)
        banks[max(multiplicity(addresses[i:i+16].tolist(),8) for i in range(0,256,16))]+=1
    assert len(selected)==16 and dict(banks)=={2:16}, (len(selected),banks)
    seen=set(); ordered=[]
    def emit(name):
        if name in seen: return
        seen.add(name)
        expr=model.definitions[name]
        assert not expr.startswith(('load ','phi ')) and 'workgroup.id' not in expr
        expr=re.sub(r', ![\w.]+ !\d+', '',expr)
        expr=re.sub(r' #\d+', ' #2', expr)
        for dependency in re.findall(r'%[\w.]+',expr): emit(dependency)
        ordered.append(f'  {name} = {expr}')
    for pointer in selected: emit(pointer)
    tid=next(name for name in seen if '@llvm.amdgcn.workitem.id.x(' in model.definitions[name])
    body=['target triple = "amdgcn-amd-amdhsa"',
          '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16',
          'declare i32 @llvm.amdgcn.workitem.id.x() #0',
          'define amdgpu_kernel void @probe(ptr addrspace(1) %out) #1 {',*ordered]
    for i,pointer in enumerate(selected):
        body.append(f'  %probe.value{i} = load <4 x half>, ptr addrspace(3) {pointer}, align 8')
    body.extend([f'  %probe.off = shl i32 {tid}, 7',
                 '  %probe.base = getelementptr i8, ptr addrspace(1) %out, i32 %probe.off'])
    for i in range(16):
        body.extend([f'  %probe.out{i} = getelementptr i8, ptr addrspace(1) %probe.base, i32 {8*i}',
                     f'  store <4 x half> %probe.value{i}, ptr addrspace(1) %probe.out{i}, align 8'])
    body.extend(['  ret void','}', 'attributes #0 = { nounwind memory(none) }',
                 'attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }',
                 'attributes #2 = { convergent nounwind memory(none) }'])
    probe=HERE/'k.read_probe.ll'; optimized=HERE/'k.read_probe.opt.ll'; assembly=HERE/'k.read_probe.gfx900.s'
    probe.write_text(re.sub(r'%(\d+)\b',r'%ssa\1','\n'.join(body))+'\n')
    llvm=Path('/data2/xsl/install/bin')
    subprocess.run([llvm/'opt','-S','-passes=default<O3>,verify',probe,'-o',optimized],check=True)
    subprocess.run([llvm/'llc','-march=amdgcn','-mcpu=gfx900','-O3',optimized,'-o',assembly],check=True)
    counts=collections.Counter(re.findall(r'^\s+(ds_read\w*)\s',assembly.read_text(),re.M))
    result={'K_unique_pointers':16,'threads':256,'bank_model':dict(banks),'gfx900_isolated_reads':dict(counts),
            'gfx936_full_kernel_tested':False}
    (HERE/'read_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__': main()
