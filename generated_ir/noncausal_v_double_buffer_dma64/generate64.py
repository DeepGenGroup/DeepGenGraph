#!/usr/bin/env python3
"""Generate and validate two 8-byte K DMA packets per original 16-byte packet."""
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'noncausal_v_double_buffer_banks'
sys.path.insert(0, str(PREVIOUS))
from generate import LLVM, digest
from verify import LazyAddresses, AuditedBlock as DWORDModel, dma_args, vdb
from check_gemm_pv_addresses import split_args


def transform(source, swizzle):
    """Physical lane l writes m0 + 512*h + 8*l.

    Original 16B packet owner: wave_base + 32*h + floor(l/2).
    Original intra-packet byte offset: 8*((l&1) XOR h) with swizzle,
    or 8*(l&1) without it. h is the original LDS byte-address bit9.
    Keep the existing producer 8:6 mapping inside H(owner).
    """
    lines, output, sites = source.splitlines(), [], 0
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'  (%bank\.K\.dma\d+)\.q0\.off = add i32 (%[\w.]+),', line)
        if match:
            prefix, original_offset = match.groups()
            block = lines[i:i+12]
            assert len(block) == 12 and sum('buffer_load_dword ' in x for x in block) == 4
            original_dst = re.fullmatch(re.escape('  '+prefix)+r'\.q0\.dst = add i32 (%[\w.]+), 0',block[1])[1]
            for half in range(2):
                p = prefix+f'.h{half}'
                output += [f'  {p}.off = add i32 {original_offset}, %bank.K.q{half}.source.delta',
                           f'  {p}.dst = add i32 {original_dst}, {512*half}']
                old_half = 0 if half == 0 else 3
                call = block[old_half*3+2].replace('buffer_load_dword ', 'buffer_load_dwordx2 ')
                call = call.replace(prefix+f'.q{old_half}.', p+'.')
                output.append(call)
            sites += 1
            i += 12
            continue
        if line.startswith(('  %bank.K.q2.', '  %bank.K.q3.')):
            i += 1
            continue
        line = line.replace('%bank.owner.low = lshr i32 %bank.lane, 2', '%bank.owner.low = lshr i32 %bank.lane, 1')
        line = line.replace('%bank.intra.word = and i32 %bank.lane, 3', '%bank.intra.word = and i32 %bank.lane, 1')
        line = line.replace('%bank.intra = shl i32 %bank.intra.word, 2', '%bank.intra = shl i32 %bank.intra.word, 3')
        line = line.replace('%bank.K.q1.owner = add i32 %bank.owner.base, 16', '%bank.K.q1.owner = add i32 %bank.owner.base, 32')
        if swizzle:
            line = line.replace('%bank.K.q1.intra = xor i32 %bank.intra, 0', '%bank.K.q1.intra = xor i32 %bank.intra, 8')
        output.append(line)
        i += 1
    assert sites == 8
    result = '\n'.join(output)+'\n'
    assert result.count('buffer_load_dwordx2 ') == 16
    assert result.count('buffer_load_dwordx4 ') == 16
    assert 'buffer_load_dword ' not in result
    # Only bank producer arithmetic / K DMA instructions may differ.
    def invariant(text):
        return [l for l in text.splitlines() if not l.startswith('  %bank.')
                and not ('buffer_load_dword' in l and '%dma.buffer.rsrc1373' in l)]
    assert invariant(source) == invariant(result)
    assert [l for l in source.splitlines() if l.startswith('  %bank.read')] == [l for l in result.splitlines() if l.startswith('  %bank.read')]
    return result


class DMA64Model(DWORDModel):
    def call(self, body):
        if 'buffer_load_dwordx2 ' not in body:
            return super().call(body)
        dst, off, resource = [self.typed(x) for x in split_args(re.search(r'"\((.*)\)',body)[1])]
        assert np.all(resource[:,3] == 0x20000)
        assert np.all(off % 8 == 0) and np.all(off >= 0) and np.all(off+8 <= resource[:,2])
        for wave in range(0,self.t,64):
            assert np.all(dst[wave:wave+64] == dst[wave])
            assert np.all(resource[wave:wave+64] == resource[wave])
        src = resource[:,0]+(resource[:,1]<<32)+off
        self.issuing = True
        try:
            self.memory(dst+np.arange(self.t)%64*8, '<4 x half>', self.memory(src,'<4 x half>'))
        finally:
            self.issuing = False
        self.dma64_count = getattr(self,'dma64_count',0)+1
        return dst


def addresses(text, swizzle):
    old_text = (PREVIOUS/'baseline.ll').read_text()
    old, new = LazyAddresses(old_text), LazyAddresses(text)
    old_args, new_args = dma_args(old_text,'K'), dma_args(text,'K')
    assert len(old_args)==8 and len(new_args)==16
    tid = np.arange(256)
    packets = 0
    for head in range(32):
        for tile in range(32):
            old.reset(head,0,tile); new.reset(head,0,tile)
            for i, args in enumerate(old_args):
                m0, offset, resource = map(old.typed,args)
                covered = []
                for h in range(2):
                    m1, off1, resource1 = map(new.typed,new_args[2*i+h])
                    dst = m1+(tid%64)*8
                    original = dst ^ ((dst>>6)&8) if swizzle else dst
                    relative = original-m0
                    owner = tid//64*64+relative//16
                    assert np.all(owner//64==tid//64)
                    assert np.array_equal(resource,resource1)
                    assert np.array_equal(off1,offset[owner]+relative%16)
                    assert np.all(off1%8==0) and np.all(off1>=0) and np.all(off1+8<=resource1[:,2])
                    assert np.all(dst>=0) and np.all(dst+8<=16384)
                    covered.extend(dst.tolist())
                    packets += 256
                expected = (m0[:,None]+(tid%64)[:,None]*16+np.array([0,8])[None,:]).ravel()
                assert sorted(covered)==sorted(expected.tolist())
    print('PASS source provenance / coverage',swizzle,packets,'8-byte packets',flush=True)
    return dict(all_32_heads_32_key_tiles_256_lanes=True, checked_8byte_packets=packets,
                source_provenance=True, exact_coverage=True, in_bounds_and_aligned=True)


def main():
    source = PREVIOUS/'k_swizzle32_gep.ll'
    expected = json.loads((PREVIOUS/'read2_repair.validation.json').read_text())['candidate_sha256']
    assert digest(source)==expected
    variants = {'k_swizzle64_gep':(source,True),
                'k_split64_control':(PREVIOUS/'k_split32_control.ll',False)}
    texts, records = {}, {}
    for name,(path,swizzle) in variants.items():
        text = transform(path.read_text(),swizzle)
        out = HERE/(name+'.ll'); out.write_text(text)
        out.with_suffix('.launch.json').write_bytes(source.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as',out,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',out,'-o',out.with_suffix('.o3.ll')],check=True)
        records[name] = dict(source=str(path),source_sha256=digest(path),sha256=digest(out),
                             llvm_and_O3_verified=True,address_validation=addresses(text,swizzle))
        texts[name] = text
    cases = []
    for kind,head,query in [('random',0,0),('future_key_dominates',1,0),('max_changes',1,31)]:
        rng = np.random.default_rng(42+query)
        arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        if kind=='future_key_dominates':
            arrays[0].fill(1); arrays[1].fill(-1); arrays[1][:,-128:]=1
            arrays[2].fill(0); arrays[2][:,-128:]=1
        if kind=='max_changes':
            arrays[0].fill(1); arrays[1].fill(-16)
            arrays[1][:,:128]=.25; arrays[1][:,128:256]=.5
        expected = vdb.reference(*arrays,head,query,causal=False)
        base_output = base_values = base_v = None
        runs = {'measured_swizzle32_gep':source.read_text(), **texts}
        if kind=='random': runs['k_swizzle64_gep.o3']=(HERE/'k_swizzle64_gep.o3.ll').read_text()
        for name,text in runs.items():
            model = DMA64Model(re.sub(r'\bexact ','',text),*arrays,threads=256,physical=(query,head))
            full = model.run().reshape(arrays[0].shape)
            actual = full[head,query*128:(query+1)*128]
            assert model.vm_waits==260 and model.barriers==293 and model.mmac_count==8192 and not model.pending
            assert model.v_reads_with_pending_dma==768
            assert len(model.q_elements)==16384 and set(model.q_elements.values())=={1}
            assert getattr(model,'dma64_count',0)==(0 if name=='measured_swizzle32_gep' else 512)
            assert np.isnan(full).sum()==full.size-16384
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            if base_output is None:
                base_output,base_values,base_v=full.copy(),model.lds_values.hexdigest(),(model.v_reads,model.v_dma)
            assert np.array_equal(full,base_output,equal_nan=True)
            assert model.lds_values.hexdigest()==base_values and (model.v_reads,model.v_dma)==base_v
            cases.append(dict(kind=kind,head=head,query=query,variant=name,bitwise_equal=True,
                identical_LDS_read_values=True,V_schedule_unchanged=True,barrier_epoch_ownership=True,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS numeric / ownership',kind,name,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(variants=records,cases=cases,
        K_DMA_sites_per_iteration=16,Q_V_DMA_width_bits=128,LDS_bytes=16384,
        gpu_tested=False,gfx936_codegen_tested=False,
        DMA64_semantics='buffer_load_dwordx2 offen lds: destination=m0+8*lane; supplied hardware support by user; CPU model assumes this lane stride',
        independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__':
    main()
