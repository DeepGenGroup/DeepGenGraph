#!/usr/bin/env python3
"""Check actual entry IR for all blocks, LLVM legality, and selected full kernels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from generate import HERE, SOURCE, SOURCE_SHA256, VARIANTS, X_CALL, Y_CALL, digest, entry

sys.path.insert(0, str(HERE.parent.parent / "3rd/deepgengraph/test"))
from check_attention_llvm_numerics import LLVMWorkgroupModel, reference
from check_pipeline_cleanup_attention import CountedDMA


class AllEntries(LLVMWorkgroupModel):
    def call(self, body):
        if "@llvm.amdgcn.workgroup.id.x(" in body:
            return np.tile(np.arange(32, dtype=np.int64), 32)
        if "@llvm.amdgcn.workgroup.id.y(" in body:
            return np.repeat(np.arange(32, dtype=np.int64), 32)
        return super().call(body)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm", type=Path, default=Path("/data2/xsl/install/bin"))
    parser.add_argument("--numerics", action="store_true", help="Also run delayed-DMA CPU numerical models")
    args = parser.parse_args()
    assert digest(SOURCE) == SOURCE_SHA256
    source = SOURCE.read_text()
    manifest = json.loads((HERE / "manifest.json").read_text())
    results = {"gpu_tested": False, "variants": {}, "limitations":
               "Entry mapping is exhaustively checked. CPU DMA model does not simulate independent waves, cache, GPU scheduling, or target codegen."}
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)] if args.numerics else None
    cases = [(0, 0), (1, 7), (1, 31)]
    originals, references = {}, {}
    if args.numerics:
        for h, q in cases:
            model = CountedDMA(source, *arrays, head=h, block=q)
            originals[h, q] = model.run().reshape(arrays[0].shape)[h, q*128:(q+1)*128].copy()
            references[h, q] = reference(*arrays, h, q)
    with tempfile.TemporaryDirectory(prefix="block-order-verify-") as temp:
        for name, (group, reverse) in VARIANTS.items():
            path = HERE / f"{name}.ll"
            text = path.read_text()
            assert digest(path) == manifest["variants"][name]["sha256"]
            restored = text
            if name != "baseline":
                y, x = entry(group, reverse)
                assert text.count(y) == text.count(x) == 1
                restored = text.replace(y, Y_CALL).replace(x, X_CALL)
            assert restored == source, "Non-entry instructions changed"
            # Execute the real LLVM entry over all physical grid coordinates.
            dummy = np.zeros((1,), dtype=np.float16)
            model = AllEntries(text, dummy, dummy, dummy, threads=1024)
            for instruction in model.blocks['4']:
                if instruction.startswith('br '):
                    break
                dest, expr = instruction.split(' = ', 1)
                model.values[dest] = model.operation(expr)
            actual = list(zip(model.values['%9'].tolist(), model.values['%7'].tolist()))
            expected = [(h, 31-q if reverse else q)
                        for start in range(0, 32, group)
                        for h in range(32) for q in range(start, start+group)]
            assert actual == expected
            assert set(actual) == {(h, q) for h in range(32) for q in range(32)}
            assert len(set(actual)) == 1024
            bitcode = Path(temp) / f"{name}.bc"
            subprocess.run([args.llvm / "llvm-as", path, "-o", bitcode], check=True)
            subprocess.run([args.llvm / "opt", "-passes=default<O3>,verify", bitcode, "-disable-output"], check=True)
            record = {"sha256": digest(path), "entry_only_change": True,
                      "all_1024_blocks_exactly_once": True, "llvm_verify": True,
                      "llvm_o3_verify": True, "first_12_head_query_pairs": actual[:12],
                      "cpu_delayed_dma": []}
            if args.numerics:
                for h, q in cases:
                    linear = actual.index((h, q))
                    physical = (linear % 32, linear // 32)
                    m = PhysicalBlock(text, *arrays, physical=physical, head=h, block=q)
                    output = m.run().reshape(arrays[0].shape)
                    selected = output[h, q*128:(q+1)*128]
                    assert np.array_equal(selected, originals[h, q])
                    assert np.allclose(selected, references[h, q], atol=.01, rtol=.01)
                    written = np.zeros(output.shape, dtype=bool)
                    written[h, q*128:(q+1)*128] = True
                    assert np.all(np.isnan(output[~written]))
                    assert not m.pending and m.mmac_count == 256*(q+1)
                    record["cpu_delayed_dma"].append({
                        "logical_head_query": [h, q], "physical_xy": physical,
                        "bitwise_equal_to_baseline": True, "writes_only_selected_tile": True,
                        "max_abs_error": float(np.max(np.abs(selected.astype(np.float32)-references[h, q]))),
                        "mmac_count": m.mmac_count, "waits": m.vm_waits, "barriers": m.barriers,
                    })
            results["variants"][name] = record
            print("PASS", name, "mapping/unchanged body/LLVM/O3", "numerics" if args.numerics else "", flush=True)
    (HERE / "validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
