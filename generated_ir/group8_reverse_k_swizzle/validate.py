#!/usr/bin/env python3
"""Check generated IR, exhaustive K DMA addresses, and delayed-DMA CPU numerics.

Does not model independent wave timing, gfx936 instruction phases or performance.
"""
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
        super().__init__(*args, **kwargs)

    def call(self, body):
        if "@llvm.amdgcn.workgroup.id.x(" in body:
            return np.full(self.t, self.physical[0], dtype=np.int64)
        if "@llvm.amdgcn.workgroup.id.y(" in body:
            return np.full(self.t, self.physical[1], dtype=np.int64)
        return super().call(body)


class LazyAddresses(LLVMWorkgroupModel):
    def __init__(self, text):
        dummy = np.zeros(1, dtype=np.float16)
        super().__init__(text, dummy, dummy, dummy)
        self.definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+.*)?$', text, re.M))
        self.initial = dict(self.values)

    def reset(self, head, step):
        self.values = dict(self.initial)
        # Address audit need not execute the block entry: the logical head,
        # query tile and K-loop step are explicit inputs to the address graph.
        self.values.update({"%9": np.full(self.t, head, dtype=np.int64),
                            "%7": np.full(self.t, 31, dtype=np.int64),
                            "%41": np.full(self.t, step, dtype=np.int64)})

    def value(self, s, ty=None):
        if s not in self.values and s in self.definitions:
            self.values[s] = self.operation(self.definitions[s])
        return super().value(s, ty)


def dma_operands(text):
    return [split_args(re.search(r'"\((.*)\)', line)[1]) for line in text.splitlines()
            if "buffer_load_dwordx4" in line and line.endswith("!dbg !16")]


def address_audit(source, candidate):
    models = [LazyAddresses(text) for text in (source, candidate)]
    dma = [dma_operands(text) for text in (source, candidate)]
    lane = np.arange(256)
    perm = lane ^ ((lane >> 2) & 4)
    packets = 0
    for head in range(32):
        for step in range(32):
            for model in models:
                model.reset(head, step)
            for operands_a, operands_b in zip(*dma):
                lds_a, off_a, rsrc_a = [models[0].typed(x) for x in operands_a]
                lds_b, off_b, rsrc_b = [models[1].typed(x) for x in operands_b]
                assert np.array_equal(lds_a, lds_b) and np.array_equal(rsrc_a, rsrc_b)
                assert np.array_equal(off_b, off_a[perm])
                assert np.all(off_b % 16 == 0) and np.all(off_b >= 0)
                assert np.all(off_b + 16 <= rsrc_b[:, 2])
                dst = lds_a + (lane % 64) * 16
                swizzled_dst = dst ^ ((dst >> 2) & 64)
                assert np.array_equal(swizzled_dst, dst[perm])
                assert np.all((swizzled_dst + 15) // 16 == swizzled_dst // 16)
                packets += 256
    reads = [re.findall(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 2, !dbg !17', text)
             for text in (source, candidate)]
    assert len(reads[0]) == len(reads[1]) == 128
    before, after = collections.Counter(), collections.Counter()
    for pa, pb in zip(*reads):
        a, b = models[0].value(pa), models[1].value(pb)
        assert np.array_equal(b, a ^ ((a >> 2) & 64))
        assert np.all(b % 8 == 0) and np.all((8192 <= b) & (b + 8 <= 16384))
        for addresses, counts in ((a, before), (b, after)):
            ways = max(multiplicity(addresses[i:i+16].tolist(), 8) for i in range(0, 256, 16))
            counts[ways] += 1
    assert dict(before) == {4: 128} and dict(after) == {2: 128}
    print(f"PASS addresses: {packets} K DMA packets, all 32 heads x 32 steps x 256 threads; K 4-way -> 2-way", flush=True)
    return {"dma_packets": packets, "all_heads_steps_threads": True,
            "source_lane_permutation_and_destination_match": True,
            "K_bank_model_before": dict(before), "K_bank_model_after": dict(after)}


def main():
    assert sha(SOURCE) == sha(BASELINE) == SOURCE_SHA256
    source, candidate = BASELINE.read_text(), CANDIDATE.read_text()
    assert candidate == transform(source)[0]
    result = {"baseline_sha256": sha(BASELINE), "candidate_sha256": sha(CANDIDATE),
              "gpu_tested": False, "target_codegen_tested": False,
              "limitations": "CPU delayed DMA is collective, not independent wave timing; bank model is not a gfx936 PMC prediction.",
              "only_K_address_changes": True, "static_lds_bytes": 16384}
    llvm = Path("/data2/xsl/install/bin")
    with tempfile.TemporaryDirectory(prefix="k-swizzle-verify-") as directory:
        for path in (BASELINE, CANDIDATE):
            bc = Path(directory) / (path.stem + ".bc")
            subprocess.run([llvm / "llvm-as", path, "-o", bc], check=True)
            subprocess.run([llvm / "opt", "-passes=default<O3>,verify", bc, "-disable-output"], check=True)
    result["llvm_assemble_and_O3_verify"] = True
    result["addresses"] = address_audit(source, candidate)
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    cases = []
    for head, query in ((0, 0), (1, 7), (1, 31)):
        q_forward = 31 - query
        linear = (q_forward // 8) * 256 + head * 8 + q_forward % 8
        physical = (linear % 32, linear // 32)
        outputs, counters = [], []
        for name, text in (("control", source), ("swizzle", candidate)):
            model = PhysicalBlock(text, *arrays, physical=physical, head=head, block=query)
            full = model.run().reshape(arrays[0].shape)
            tile = full[head, query*128:(query+1)*128].copy()
            written = np.zeros(full.shape, dtype=bool)
            written[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            assert not model.pending and model.mmac_count == 256 * (query + 1)
            outputs.append(tile)
            counters.append((model.vm_waits, model.barriers, model.mmac_count))
            print(f"Executed {name} head={head} query={query}", flush=True)
        assert np.array_equal(*outputs) and counters[0] == counters[1]
        expected = reference(*arrays, head, query)
        assert np.allclose(outputs[1], expected, atol=.01, rtol=.01)
        error = float(np.max(np.abs(outputs[1].astype(np.float32) - expected)))
        cases.append({"head": head, "query": query, "physical_xy": physical,
                      "bitwise_equal_to_control": True, "matches_attention_reference": True,
                      "max_abs_error": error, "writes_only_selected_tile": True,
                      "waits_barriers_mmac": counters[1]})
        print(f"PASS numerics head={head} query={query}: bitwise equal, max error {error}", flush=True)
    result["cpu_delayed_dma_numerics"] = cases
    (HERE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
