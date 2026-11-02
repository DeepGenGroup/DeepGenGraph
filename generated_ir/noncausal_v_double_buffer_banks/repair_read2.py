#!/usr/bin/env python3
"""Restore shared base + constant GEPs after the measured K subpacket swizzle.

This is a read2 code-generation repair, not a claim that DWORD DMA is faster.
Measured candidates are never overwritten. No producer or schedule changes.
"""
import collections
import json
import re
import subprocess

import numpy as np
from generate import HERE, LLVM, digest
from verify import LazyAddresses, AuditedBlock, vdb


def repair(source):
    model = LazyAddresses(source)
    model.reset()
    pointers = list(dict.fromkeys(re.findall(
        r'= load <4 x half>, ptr addrspace\(3\) (%bank\.read\d+\.ptr)', source)))
    groups, replacements = {}, {}
    for pointer in pointers:
        addresses = model.value(pointer)
        # Explicitly prove the pointer DAG depends only on the local thread.
        visited = set()
        def audit(name):
            if name in visited: return
            visited.add(name)
            expr = model.defs[name]
            assert 'workgroup.id' not in expr and not expr.startswith(('phi ', 'load '))
            for dep in re.findall(r'%[\w.]+', expr): audit(dep)
        audit(pointer)
        key = tuple(addresses-addresses[0])
        if key not in groups:
            groups[key] = pointer, addresses.copy()
            continue
        base, base_addresses = groups[key]
        delta = int(addresses[0]-base_addresses[0])
        assert np.all(addresses == base_addresses+delta)
        replacements[pointer] = f'getelementptr i8, ptr addrspace(3) {base}, i64 {delta}'
    assert len(groups) == 2
    lines = source.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r'  (%[\w.]+) = ', line)
        if m and m[1] in replacements:
            lines[i] = f'  {m[1]} = {replacements[m[1]]}'
    # Delete only dead temporaries introduced by the earlier swizzle transform.
    while True:
        uses = collections.Counter(re.findall(r'%[\w.]+', '\n'.join(lines)))
        dead = [i for i, line in enumerate(lines)
                if (m := re.match(r'  (%bank\.read[\w.]+) = ', line)) and uses[m[1]] == 1]
        if not dead: break
        lines = [line for i, line in enumerate(lines) if i not in dead]
    candidate = '\n'.join(lines)+'\n'
    new = LazyAddresses(candidate)
    new.reset()
    for pointer in pointers:
        assert np.array_equal(model.value(pointer), new.value(pointer))
    # Loads, MMAC, DMA, synchronization and all V pointer definitions unchanged.
    unchanged = lambda text: [line for line in text.splitlines() if not re.match(r'  %bank\.read', line)]
    assert unchanged(source) == unchanged(candidate)
    return candidate, dict(unique_K_pointers=len(pointers), common_bases=len(groups),
                           rewritten_pointers=len(replacements), all_256_lanes_identical=True,
                           independent_of_head_query_and_key_iteration=True)


def probe(text, name):
    model = LazyAddresses(text)
    pointers = list(dict.fromkeys(re.findall(
        r'= load <4 x half>, ptr addrspace\(3\) (%bank\.read\d+\.ptr)', text)))
    visited, ordered = set(), []
    def emit(name):
        if name in visited: return
        visited.add(name)
        expr = re.sub(r', ![\w.]+ !\d+', '', model.defs[name])
        expr = re.sub(r' #\d+', ' #2', expr)
        for dependency in re.findall(r'%[\w.]+', expr): emit(dependency)
        ordered.append(f'  {name} = {expr}')
    for pointer in pointers: emit(pointer)
    tid = next(n for n in visited if '@llvm.amdgcn.workitem.id.x(' in model.defs[n])
    body = ['target triple = "amdgcn-amd-amdhsa"',
            '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16',
            'declare i32 @llvm.amdgcn.workitem.id.x() #0',
            'define amdgpu_kernel void @probe(ptr addrspace(1) %probe.out) #1 {', *ordered]
    for i, p in enumerate(pointers):
        body.append(f'  %probe.value{i} = load <4 x half>, ptr addrspace(3) {p}, align 8')
    body += [f'  %probe.off = mul i32 {tid}, {8*len(pointers)}',
             '  %probe.base = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off']
    for i in range(len(pointers)):
        body += [f'  %probe.dst{i} = getelementptr i8, ptr addrspace(1) %probe.base, i32 {8*i}',
                 f'  store <4 x half> %probe.value{i}, ptr addrspace(1) %probe.dst{i}, align 8']
    body += ['  ret void', '}', 'attributes #0 = { nounwind memory(none) }',
             'attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }',
             'attributes #2 = { convergent nounwind memory(none) }']
    path = HERE/(name+'.read_probe.ll')
    path.write_text(re.sub(r'%(\d+)\b', r'%probe.ssa\1', '\n'.join(body))+'\n')
    opt, asm = path.with_suffix('.opt.ll'), path.with_suffix('.gfx900.s')
    subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', path, '-o', opt], check=True)
    subprocess.run([LLVM/'llc', '-march=amdgcn', '-mcpu=gfx900', '-O3', opt, '-o', asm], check=True)
    return dict(collections.Counter(re.findall(r'^\s+(ds_read\w*)\s', asm.read_text(), re.M)))


def main():
    source_path = HERE/'k_swizzle32.ll'
    source = source_path.read_text()
    assert digest(source_path) == '9605d5eeb763f9a754863e07bc07a4e8560515974a2608f4c4f27b79f2f510ef'
    candidate, audit = repair(source)
    out = HERE/'k_swizzle32_gep.ll'
    out.write_text(candidate)
    out.with_suffix('.launch.json').write_bytes(source_path.with_suffix('.launch.json').read_bytes())
    opt = out.with_suffix('.o3.ll')
    subprocess.run([LLVM/'llvm-as', out, '-o', '/dev/null'], check=True)
    subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', out, '-o', opt], check=True)
    probes = {name:probe(text,name) for name,text in [('k_swizzle32',source),('k_swizzle32_gep',candidate)]}
    print('gfx900 isolated reads:', probes, flush=True)
    assert sum(probes['k_swizzle32_gep'].values()) < sum(probes['k_swizzle32'].values())
    results = []
    for head, query in [(0,0), (1,31)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        expected = vdb.reference(*arrays,head,query,causal=False)
        baseline_output = baseline_values = baseline_v = None
        variants = [('measured',source),('gep',candidate)] + ([('gep_o3',opt.read_text())] if head==0 else [])
        for name,text in variants:
            m = AuditedBlock(re.sub(r'\bexact ', '', text), *arrays, threads=256, physical=(query,head))
            full = m.run().reshape(arrays[0].shape)
            actual = full[head,query*128:(query+1)*128]
            assert m.vm_waits==260 and m.barriers==293 and m.mmac_count==8192 and not m.pending
            assert m.v_reads_with_pending_dma==768
            assert len(m.q_elements)==16384 and set(m.q_elements.values())=={1}
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            if baseline_output is None:
                baseline_output,baseline_values,baseline_v = full.copy(),m.lds_values.hexdigest(),(m.v_reads,m.v_dma)
            assert np.array_equal(full,baseline_output,equal_nan=True)
            assert m.lds_values.hexdigest()==baseline_values and (m.v_reads,m.v_dma)==baseline_v
            results.append(dict(head=head,query=query,variant=name,bitwise_equal=True,
                                identical_LDS_values_and_V_schedule=True,barrier_epoch_ownership=True))
            print('PASS numerics / ownership',head,query,name,flush=True)
    (HERE/'read2_repair.validation.json').write_text(json.dumps(dict(
        source_sha256=digest(source_path), candidate_sha256=digest(out),
        address_audit=audit, local_gfx900_isolated_read_probe=probes, cases=results,
        llvm_and_O3_verified=True,gpu_tested=False,gfx936_full_kernel_codegen_tested=False,
        limitation='DWORD DMA cost remains; restored read2 and VGPR count require gfx936 full-kernel measurement.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
