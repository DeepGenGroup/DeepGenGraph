#!/usr/bin/env python3
"""Verify LLVM and sample numerical execution; no GPU timing or wave scheduling model."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_pipeline_cleanup_attention import CountedDMA
from check_attention_llvm_numerics import reference
from check_gemm_pv_addresses import split_args


class PhysicalBlock(CountedDMA):
    def __init__(self, *args, physical, **kwargs):
        self.physical = physical
        super().__init__(*args, **kwargs)

    def operation(self, expr):
        # New exports retain exact shifts. Check their promise before using the
        # existing interpreter's shift implementation; leave the actual IR intact.
        match = re.match(r"(ashr|lshr) exact (.*)", expr)
        if match:
            left, right = split_args(match[2])
            ty, _ = self.parse_type(left)
            value, amount = self.typed(left), self.value(right, ty)
            assert np.all((value & ((np.int64(1) << amount) - 1))[self.active] == 0)
            expr = match[1] + " " + match[2]
        return super().operation(expr)

    def call(self, body):
        for axis, value in zip("xyz", (*self.physical, 0)):
            if f"@llvm.amdgcn.workgroup.id.{axis}(" in body:
                return np.full(self.t, value, dtype=np.int64)
        return super().call(body)


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    results = {}
    for name, item in manifest["variants"].items():
        path = HERE / item["ir"]
        text = path.read_text()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        subprocess.run(["/data2/xsl/install/bin/llvm-as", path, "-o", "/dev/null"], check=True)
        subprocess.run(["/data2/xsl/install/bin/opt", "-passes=default<O3>,verify",
                        path, "-disable-output"], check=True)
        bm, bn = item["bm"], item["bn"]
        nq = 4096 // bm
        cases = []
        # Beginning, both sides of a group boundary, and full sequence prefix.
        for head, query in ((0, 0), (1, 1), (1, 7), (1, 8), (1, nq - 1)):
            forward = nq - 1 - query
            linear = (forward // 8) * (32 * 8) + head * 8 + forward % 8
            physical = (linear % nq, linear // nq)
            model = PhysicalBlock(text, *arrays, physical=physical, head=head, block=query)
            full = model.run().reshape(arrays[0].shape)
            actual = full[head, query * bm:(query + 1) * bm]
            expected = reference(*arrays, head, query, rows=bm)
            written = np.zeros(full.shape, dtype=bool)
            written[head, query * bm:(query + 1) * bm] = True
            assert np.isnan(full[~written]).all(), "writes outside selected output tile"
            assert np.allclose(actual, expected, atol=.01, rtol=.01), (name, head, query)
            assert not model.pending
            iterations = ((query + 1) * bm + bn - 1) // bn
            expected_mmac = (bm * bn // 64) * iterations
            assert model.mmac_count == expected_mmac, (name, model.mmac_count, expected_mmac)
            case = dict(head=head, query=query, physical_xy=physical,
                        max_abs_error=float(np.max(np.abs(actual.astype(np.float32) - expected))),
                        mmac=model.mmac_count, waits=model.vm_waits, barriers=model.barriers,
                        matches_reference=True, writes_only_selected_tile=True)
            cases.append(case)
            print(name, case, flush=True)
        results[name] = dict(sha256=item["sha256"], llvm_verify=True, llvm_O3_verify=True,
                             cpu_delayed_dma=cases)
        (HERE / "validation.json").write_text(json.dumps(dict(
            gpu_tested=False, target_gfx936_codegen_tested=False,
            limitations="Sampled CPU execution; does not model independent wave timing or target codegen.",
            variants=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
