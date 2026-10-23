#!/usr/bin/env python3
"""Canonicalize the measured K128 swizzle's LDS reads and validate the result.

Run: OPENBLAS_NUM_THREADS=1 python3 generated_ir/group8_reverse_k_gep/generate.py
No target DTK or GPU execution is performed. A gfx900 isolated read probe is
only evidence about address matching in the locally available LLVM backend.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREVIOUS = ROOT / "generated_ir/group8_reverse_k_swizzle"
sys.path.insert(0, str(PREVIOUS))
from validate import LazyAddresses, PhysicalBlock, reference, multiplicity

SOURCE = PREVIOUS / "group8_reverse_k_swizzle128.ll"
SOURCE_SHA256 = "73234d33f58e848a06f363a42f1090a2952e71791ec6e2e409e762f944a880c1"
CONTROL = HERE / "group8_reverse_k128_control.ll"
CANDIDATE = HERE / "group8_reverse_k128_gep.ll"
LLVM = Path('/data2/xsl/install/bin')
# Index of the existing read pointer -> (base, constant byte displacement).
MAPPING = [(0, 0), (0, 4096), (1, 0), (1, 4096),
           (0, 128), (0, 4224), (1, 128), (1, 4224),
           (0, 1024), (0, 5120), (1, 1024), (1, 5120),
           (0, 1152), (0, 5248), (1, 1152), (1, 5248)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source):
    lines = []
    for line in source.splitlines():
        m = re.match(r'  %ksw.read(\d+)\.(address|xor|pointer) = ', line)
        if m and int(m[1]) != 0:
            if m[2] == 'pointer':
                base, offset = MAPPING[int(m[1])]
                pointer = '%ksw.read0.pointer' if base == 0 else '%kgep.base1'
                lines.append(f'  %ksw.read{m[1]}.pointer = getelementptr i8, ptr addrspace(3) {pointer}, i32 {offset}, !dbg !17')
            continue
        if '= load <4 x half>, ptr addrspace(3) %ksw.read' in line:
            line = line.replace(', align 2,', ', align 8,')
        lines.append(line)
        if line.startswith('  %ksw.read0.pointer = '):
            lines += [
                '  %kgep.base1.bits = xor i32 %ksw.read0.xor, 64, !dbg !17',
                '  %kgep.base1 = inttoptr i32 %kgep.base1.bits to ptr addrspace(3), !dbg !17',
            ]
    candidate = '\n'.join(lines) + '\n'

    def unchanged_body(text):
        return [l.replace(', align 8,', ', align 2,') if '%ksw.read' in l else l
                for l in text.splitlines()
                if not re.match(r'  %(?:ksw.read\d+\.|kgep\.)', l)]
    assert unchanged_body(candidate) == unchanged_body(source)
    assert candidate.count('buffer_load_dwordx4 ') == source.count('buffer_load_dwordx4 ') == 24
    assert candidate.count('= load <4 x half>, ptr addrspace(3) %ksw.read') == 128
    assert candidate.count('= getelementptr i8, ptr addrspace(3) %kgep.base1,') == 8
    return candidate


def audit_addresses(source, candidate):
    models = [LazyAddresses(text) for text in (source, candidate)]
    # Both address graphs depend only on thread id, with no block or loop input.
    for model in models:
        for i in range(16):
            seen = set()

            def visit(name):
                if name in seen:
                    return
                seen.add(name)
                expr = model.definitions.get(name, '')
                assert not expr.startswith('phi ') and 'workgroup.id.' not in expr
                assert name not in ('%0', '%1', '%2', '%3')
                for dependency in re.findall(r'%[\w.]+', expr):
                    visit(dependency)
            visit(f'%ksw.read{i}.pointer')
        model.reset(0, 0)
    stats = collections.Counter()
    for i, (base, offset) in enumerate(MAPPING):
        a, b = [model.value(f'%ksw.read{i}.pointer') for model in models]
        assert np.array_equal(a, b), i
        assert np.all(b % 8 == 0) and np.all((b >= 8192) & (b + 8 <= 16384))
        root = models[1].value('%ksw.read0.pointer' if base == 0 else '%kgep.base1')
        assert np.array_equal(b, root + offset)
        ways = max(multiplicity(b[j:j+16].tolist(), 8) for j in range(0,256,16))
        stats[ways] += 1
    assert dict(stats) == {2: 16}
    print('PASS addresses: all 16 pointers x 256 threads identical; align 8 valid; K remains 2-way', flush=True)
    return {'all_16_pointers_256_threads_identical': True, 'independent_of_head_query_loop': True,
            'all_128_load_sites_covered': True, 'align8_and_LDS_bounds_verified': True,
            'bank_model_32banks_b64_16lane_phases': dict(stats)}


def make_probe(text):
    """Extract the actual K pointer graph, not an invented address stand-in."""
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*)$', text, re.M))
    seen, ordered = set(), []

    def emit(name):
        if name in seen:
            return
        seen.add(name)
        expr = definitions[name]
        expr = re.sub(r', ![\w.]+ !\d+', '', expr).replace(' #9', ' #2')
        for dependency in re.findall(r'%[\w.]+', expr):
            emit(dependency)
        ordered.append(f'  {name} = {expr}')
    for i in range(16):
        emit(f'%ksw.read{i}.pointer')
    # Named SSA values avoid depending on gaps in the original numeric IDs.
    ordered = [re.sub(r'%(\d+)\b', r'%probe.ssa\1', line) for line in ordered]
    alignment = 8 if '%kgep.base1' in text else 2
    body = ['target triple = "amdgcn-amd-amdhsa"',
            '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16',
            'declare i32 @llvm.amdgcn.workitem.id.x() #0',
            'define amdgpu_kernel void @probe(ptr addrspace(1) %out) #1 {', *ordered]
    for i in range(16):
        body.append(f'  %probe.value{i} = load <4 x half>, ptr addrspace(3) %ksw.read{i}.pointer, align {alignment}')
    body += ['  %probe.output.offset = shl i32 %probe.ssa24, 7',
             '  %probe.output.base = getelementptr i8, ptr addrspace(1) %out, i32 %probe.output.offset']
    for i in range(16):
        body += [f'  %probe.out{i} = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 {i*8}',
                 f'  store <4 x half> %probe.value{i}, ptr addrspace(1) %probe.out{i}, align 8']
    body += ['  ret void', '}',
             'attributes #0 = { nounwind memory(none) }',
             'attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }',
             'attributes #2 = { convergent nounwind memory(none) }']
    return '\n'.join(body) + '\n'


def verify_and_probe(source, candidate):
    probe_results = {}
    with tempfile.TemporaryDirectory(prefix='k-gep-verify-') as directory:
        for path in (CONTROL, CANDIDATE):
            bc = Path(directory)/(path.stem + '.bc')
            subprocess.run([LLVM/'llvm-as', path, '-o', bc], check=True)
            subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', bc, '-disable-output'], check=True)
        for name, text in (('control',source), ('gep',candidate)):
            path = HERE/f'{name}.read_probe.ll'
            optimized = HERE/f'{name}.read_probe.opt.ll'
            assembly = HERE/f'{name}.read_probe.gfx900.s'
            path.write_text(make_probe(text))
            subprocess.run([LLVM/'opt', '-S', '-passes=default<O3>,verify', path, '-o', optimized], check=True)
            subprocess.run([LLVM/'llc', '-march=amdgcn', '-mcpu=gfx900', '-O3', optimized, '-o', assembly], check=True)
            counts = collections.Counter(re.findall(r'^\s+(ds_read\w*)\s', assembly.read_text(), re.M))
            probe_results[name] = dict(counts)
            print(f'Local gfx900 isolated read probe {name}: {dict(counts)}', flush=True)
    return probe_results


def numerics(source, candidate):
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    cases = []
    for head, query in ((0,0), (1,7), (1,31)):
        q = 31-query
        linear = q//8*256+head*8+q%8
        outputs, counters = [], []
        for name, text in (('control',source), ('gep',candidate)):
            model = PhysicalBlock(text,*arrays,physical=(linear%32,linear//32),head=head,block=query)
            full = model.run().reshape(arrays[0].shape)
            outputs.append(full[head,query*128:(query+1)*128].copy())
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all() and not model.pending
            assert model.mmac_count == 256*(query+1)
            counters.append((model.vm_waits,model.barriers,model.mmac_count))
            print(f'Executed {name} head={head} query={query}',flush=True)
        assert np.array_equal(*outputs) and counters[0] == counters[1]
        expected = reference(*arrays,head,query)
        assert np.allclose(outputs[1],expected,atol=.01,rtol=.01)
        error = float(np.max(np.abs(outputs[1].astype(np.float32)-expected)))
        cases.append({'head':head,'query':query,'bitwise_equal_to_control':True,
                      'matches_attention_reference':True,'max_abs_error':error,
                      'writes_only_selected_tile':True,'waits_barriers_mmac':counters[1]})
        print(f'PASS numerics head={head} query={query}, max error {error}',flush=True)
    return cases


def main():
    assert sha(SOURCE) == SOURCE_SHA256
    source = SOURCE.read_text()
    candidate = transform(source)
    CONTROL.write_text(source)
    CANDIDATE.write_text(candidate)
    result = {'source': str(SOURCE.relative_to(ROOT)), 'control_sha256': sha(CONTROL),
              'candidate': CANDIDATE.name, 'candidate_sha256': sha(CANDIDATE),
              'control_program_median_ms': 1.8204780220985413,
              'gpu_tested': False, 'target_gfx936_codegen_tested': False,
              'static_lds_bytes': 16384, 'all_24_DMA_sites_128bit_and_unchanged': True,
              'K_swizzle_layout_unchanged': True, 'Q_V_math_entry_waits_barriers_unchanged': True,
              'changes': ['K addresses use two bases plus constant GEP', 'K loads declare verified 8-byte alignment'],
              'pointer_base_byte_offset_mapping': MAPPING,
              'limitations': 'CPU delayed DMA does not simulate independent waves. Isolated gfx900 probe is not gfx936 full-kernel code generation or a performance measurement.',
              'launch': {'kernel':'Attn_p2','shape':[1,32,4096,128],'grid':[32,32,1],
                         'block':[256,1,1],'shared_mem':0,'k_layout':'bhsd'}}
    result['addresses'] = audit_addresses(source,candidate)
    result['local_gfx900_read_probe'] = verify_and_probe(source,candidate)
    result['llvm_assemble_O3_verify'] = True
    result['cpu_delayed_DMA_numerics'] = numerics(source,candidate)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Validated candidate: {CANDIDATE}',flush=True)


if __name__ == '__main__':
    main()
