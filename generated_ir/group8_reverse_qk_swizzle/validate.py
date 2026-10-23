#!/usr/bin/env python3
"""Validate Q split-DWORD producer provenance and full delayed-DMA numerics."""
import collections
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
from generate import HERE, ROOT, SOURCE, SOURCE_SHA256, BASELINE, CANDIDATE, sha, transform

sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_attention_llvm_numerics import LLVMWorkgroupModel, reference
from check_pipeline_cleanup_attention import CountedDMA
from check_gemm_pv_addresses import split_args
from check_packed_lds_banks import multiplicity


class PhysicalBlock(CountedDMA):
    def __init__(self, *args, physical, **kwargs):
        self.physical = physical
        self.dma_dword = 0
        super().__init__(*args, **kwargs)

    def call(self, body):
        if "@llvm.amdgcn.workgroup.id.x(" in body:
            return np.full(self.t, self.physical[0], dtype=np.int64)
        if "@llvm.amdgcn.workgroup.id.y(" in body:
            return np.full(self.t, self.physical[1], dtype=np.int64)
        if "buffer_load_dword " in body:
            assert '"={m0},s,v,s,~{memory}"' in body
            lds, offset, resource = [self.typed(x) for x in split_args(re.search(r'"\((.*)\)', body)[1])]
            assert np.all(resource[:, 3] == 0x20000)
            assert np.all(offset[self.active] >= 0) and np.all(offset[self.active] + 4 <= resource[self.active, 2])
            for w in range(self.t // 64):
                assert np.all(lds[w*64:(w+1)*64] == lds[w*64])
                assert np.all(resource[w*64:(w+1)*64] == resource[w*64])
            source = resource[:, 0] + (resource[:, 1] << 32) + offset
            self.issuing = True
            try:
                self.memory(lds + np.arange(self.t) % 64 * 4, '<2 x half>', self.memory(source, '<2 x half>'))
            finally:
                self.issuing = False
            self.dma_dword += 1
            return lds
        return super().call(body)


class LazyAddresses(LLVMWorkgroupModel):
    def __init__(self, text):
        dummy = np.zeros(1, dtype=np.float16)
        super().__init__(text, dummy, dummy, dummy)
        self.definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+.*)?$', text, re.M))
        self.initial = dict(self.values)

    def reset(self, head, query):
        self.values = dict(self.initial)
        self.values.update({"%9": np.full(self.t, head, dtype=np.int64),
                            "%7": np.full(self.t, query, dtype=np.int64),
                            "%41": np.zeros(self.t, dtype=np.int64)})

    def value(self, s, ty=None):
        if s not in self.values and s in self.definitions:
            self.values[s] = self.operation(self.definitions[s])
        return super().value(s, ty)


def address_audit(source, candidate):
    old, new = LazyAddresses(source), LazyAddresses(candidate)
    dma = [[split_args(re.search(r'"\((.*)\)', line)[1]) for line in text.splitlines()
            if "buffer_load_dword" in line and line.endswith("!dbg !11")]
           for text in (source, candidate)]
    assert len(dma[0]) == 8 and len(dma[1]) == 32
    tid = np.arange(256)
    lane = tid % 64
    packets = 0
    for head in range(32):
        for query in range(32):
            old.reset(head, query)
            new.reset(head, query)
            for i, original in enumerate(dma[0]):
                m0, off, resource = [old.typed(x) for x in original]
                covered = []
                for j in range(4):
                    m1, off1, resource1 = [new.typed(x) for x in dma[1][i*4+j]]
                    assert np.array_equal(resource, resource1)
                    dst = m1 + lane * 4
                    original_dst = dst ^ ((dst >> 4) & 8)
                    relative = original_dst - m0
                    peer = tid // 64 * 64 + relative // 16
                    intra = relative % 16
                    assert np.array_equal(off1, off[peer] + intra)
                    assert np.all(off1 % 4 == 0) and np.all(off1 >= 0)
                    assert np.all(off1 + 4 <= resource1[:, 2])
                    assert np.all((dst >= 0) & (dst + 4 <= 8192))
                    covered.extend(dst.tolist())
                    packets += 256
                expected = (m0[:, None] + lane[:, None] * 16 + np.arange(0,16,4)[None, :]).ravel()
                assert sorted(covered) == sorted(expected.tolist())
    bank = {}
    for tag, debug_id in (("Q", 12), ("K", 17)):
        reads = [re.findall(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 2, !dbg !'+str(debug_id)+r'$', text, re.M)
                 for text in (source, candidate)]
        counts = [collections.Counter(), collections.Counter()]
        for pa, pb in zip(*reads):
            a, b = old.value(pa), new.value(pb)
            assert np.array_equal(b, a ^ ((a >> 4) & 8) if tag == "Q" else a)
            for addresses, c in zip((a, b), counts):
                c[max(multiplicity(addresses[k:k+16].tolist(), 8) for k in range(0,256,16))] += 1
        assert [dict(c) for c in counts] == ([{2:16}, {1:16}] if tag == "Q" else [{2:128}, {2:128}])
        bank[tag] = {"before": dict(counts[0]), "after": dict(counts[1])}
    print(f"PASS addresses: {packets} Q DWORD packets; Q 2-way -> 1-way, K remains 2-way", flush=True)
    return {"all_32_heads_32_queries_256_threads": True, "Q_dword_packets": packets,
            "exact_coverage_and_source_provenance": True, "bank_model": bank}


def main():
    assert sha(SOURCE) == sha(BASELINE) == SOURCE_SHA256
    source, candidate = BASELINE.read_text(), CANDIDATE.read_text()
    assert candidate == transform(source)
    result = {"baseline_sha256": sha(BASELINE), "candidate_sha256": sha(CANDIDATE),
              "gpu_tested": False, "target_gfx936_codegen_tested": False,
              "limitations": "CPU delayed DMA does not simulate independent wave timing. 32-bank address model is not a hardware counter prediction.",
              "static_lds_bytes": 16384, "only_Q_address_and_DMA_changes": True}
    llvm = Path('/data2/xsl/install/bin')
    with tempfile.TemporaryDirectory(prefix='q-swizzle-verify-') as directory:
        for path in (BASELINE, CANDIDATE):
            bc = Path(directory) / (path.stem + '.bc')
            subprocess.run([llvm/'llvm-as', path, '-o', bc], check=True)
            subprocess.run([llvm/'opt', '-passes=default<O3>,verify', bc, '-disable-output'], check=True)
        # Check the newly introduced opcode in isolation on the available
        # upstream backend. This is not gfx936 full-kernel code generation.
        probe = (ROOT/'generated_ir/dma_inlineasm_probe/buffer_dwordx2_lds.ll').read_text().replace('buffer_load_dwordx2 ', 'buffer_load_dword ')
        probe_path = Path(directory)/'probe.ll'
        probe_path.write_text(probe)
        subprocess.run([llvm/'llc', '-march=amdgcn', '-mcpu=gfx900', '-filetype=obj', probe_path, '-o', Path(directory)/'probe.o'], check=True)
    result['llvm_assemble_O3_verify'] = True
    result['isolated_buffer_dword_lds_gfx900_assembly'] = True
    result['addresses'] = address_audit(source, candidate)
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
    cases = []
    for head, query in ((0,0), (1,7), (1,31)):
        q = 31-query
        linear = (q//8)*256+head*8+q%8
        outputs, counters = [], []
        for name, text in (('control',source), ('QK',candidate)):
            model = PhysicalBlock(text, *arrays, physical=(linear%32,linear//32), head=head, block=query)
            full = model.run().reshape(arrays[0].shape)
            outputs.append(full[head,query*128:(query+1)*128].copy())
            written = np.zeros(full.shape,dtype=bool)
            written[head,query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all() and not model.pending
            assert model.mmac_count == 256*(query+1)
            assert model.dma_dword == (32*(query+1) if name == 'QK' else 0)
            counters.append((model.vm_waits,model.barriers,model.mmac_count))
            print(f'Executed {name} head={head} query={query}',flush=True)
        assert np.array_equal(*outputs) and counters[0] == counters[1]
        expected = reference(*arrays,head,query)
        assert np.allclose(outputs[1],expected,atol=.01,rtol=.01)
        error = float(np.max(np.abs(outputs[1].astype(np.float32)-expected)))
        cases.append({'head':head,'query':query,'bitwise_equal_to_K128_control':True,
                      'matches_attention_reference':True,'max_abs_error':error,
                      'writes_only_selected_tile':True,'waits_barriers_mmac':counters[1]})
        print(f'PASS numerics head={head} query={query}, max error {error}',flush=True)
    result['cpu_delayed_dma_numerics'] = cases
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
