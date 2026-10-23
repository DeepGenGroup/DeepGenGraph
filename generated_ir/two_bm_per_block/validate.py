#!/usr/bin/env python3
"""Check pair coverage, LLVM validity and actual IR execution with delayed DMA."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_pipeline_cleanup_attention import CountedDMA
from check_attention_llvm_numerics import reference
from generate import generate


class PhysicalBlock(CountedDMA):
    def __init__(self, *args, physical, **kwargs):
        self.physical = physical
        super().__init__(*args, **kwargs)

    def call(self, body):
        for axis, value in zip("xyz", (*self.physical, 0)):
            if f"@llvm.amdgcn.workgroup.id.{axis}(" in body:
                return np.full(self.t, value, dtype=np.int64)
        return super().call(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    step = 1 if args.forward else -1
    manifest_name = "manifest.forward.json" if args.forward else "manifest.json"
    manifest = json.loads((HERE / manifest_name).read_text())
    path = HERE / manifest["ir"]
    source = (ROOT / manifest["source"]).read_text()
    text = path.read_text()
    assert text == generate(source, reverse=not args.forward)
    assert hashlib.sha256(text.encode()).hexdigest() == manifest["sha256"]
    assert hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"]
    for tool, tool_args in (("llvm-as", ["-o", "/dev/null"]),
                       ("opt", ["-passes=default<O3>,verify", "-disable-output"])):
        subprocess.run([f"/data2/xsl/install/bin/{tool}", str(path), *tool_args], check=True)

    covered = Counter()
    for y in range(32):
        for x in range(16):
            linear = 2 * (y * 16 + x)
            head = (linear >> 3) & 31
            forward = (linear >> 8) * 8 + (linear & 7)
            first = forward if args.forward else 31 - forward
            assert first % 2 == (0 if args.forward else 1)
            for tile, query in enumerate((first, first + step)):
                old_linear = linear + tile
                assert head == ((old_linear >> 3) & 31)
                forward = (old_linear >> 8) * 8 + (old_linear & 7)
                assert query == (forward if args.forward else 31 - forward)
                covered[head, query] += 1
    assert covered == Counter({(h, q): 1 for h in range(32) for q in range(32)})
    print("PASS: all 512 blocks cover all 1024 logical tiles exactly once", flush=True)

    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    cases = []
    for head, high in ((0, 1), (1, 7), (1, 9), (1, 31)):
        first = high - 1 if args.forward else high
        forward = first if args.forward else 31 - first
        old_linear = (forward // 8) * 256 + head * 8 + forward % 8
        p = old_linear // 2
        physical = (p % 16, p // 16)
        model = PhysicalBlock(text, *arrays, physical=physical)
        full = model.run().reshape(arrays[0].shape)
        assert not model.pending
        assert model.mmac_count == 256 * (2 * first + step + 2)
        written = np.zeros(full.shape, dtype=bool)
        errors = []
        old_barriers = 0
        for query in (first, first + step):
            # Source IR always uses reverse order, regardless of candidate order.
            old_forward = 31 - query
            old_p = (old_forward // 8) * 256 + head * 8 + old_forward % 8
            old = PhysicalBlock(source, *arrays, physical=(old_p % 32, old_p // 32))
            original = old.run().reshape(arrays[0].shape)
            old_barriers += old.barriers
            rows = slice(query * 128, (query + 1) * 128)
            actual = full[head, rows]
            assert np.array_equal(actual, original[head, rows]), "baseline differs"
            expected = reference(*arrays, head, query)
            assert np.allclose(actual, expected, atol=.01, rtol=.01)
            written[head, rows] = True
            errors.append(float(np.max(np.abs(actual.astype(np.float32) - expected))))
        assert np.isnan(full[~written]).all(), "writes outside the two output tiles"
        assert model.barriers == old_barriers + 1
        case = dict(head=head, query_tiles=[first, first + step], physical_xy=physical,
                    bitwise_equal_to_baseline=True, matches_reference=True,
                    writes_only_two_tiles=True, no_pending_dma=True,
                    max_abs_errors=errors, mmac=model.mmac_count, barriers=model.barriers)
        cases.append(case)
        print("PASS:", case, flush=True)

    result = dict(sha256=manifest["sha256"], llvm_verify=True, llvm_O3_verify=True,
                  exhaustive_mapping=True, physical_blocks=512, logical_tiles=1024,
                  cpu_delayed_dma=cases, gpu_tested=False, target_gfx936_codegen_tested=False,
                  limitations="CPU model does not simulate independent wave timing or GPU performance.")
    validation_name = "validation.forward.json" if args.forward else "validation.json"
    (HERE / validation_name).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
