#!/usr/bin/env python3
"""Execute baseline/experiment LLVM; check uniform alpha and deferred shuffles."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "3rd/deepgengraph/test"))
from check_online_softmax import OnlineBlock, reference


class CheckedBlock(OnlineBlock):
    def __init__(self, *args, **kwargs):
        self.denominator_shuffles = 0
        self.total_shuffles = 0
        self.uniform_alpha_checks = 0
        self.deferred_calls = {line.strip().split(" = ", 1)[1].split(", !", 1)[0][5:]
            for line in args[0].splitlines() if " = call i32 @llvm.amdgcn.ds.bpermute(" in line
            and "%lane.sum.shuffle" in line}
        super().__init__(*args, **kwargs)

    def call(self, body):
        if "@llvm.amdgcn.ds.bpermute(" in body:
            self.total_shuffles += 1
            if body.split(", !", 1)[0] in self.deferred_calls or re.search(r"i32 %v(?:1192|1196|1203|1207)\)", body):
                self.denominator_shuffles += 1
        return super().call(body)

    def operation(self, expr):
        if expr in ("fmul float %v972, %v968", "fmul float %v980, %v976"):
            alpha = self.values[expr.rsplit(" ", 1)[1]]
            assert self.active.all(), "denominator update must use complete lane groups"
            for xor in (16, 32):
                np.testing.assert_array_equal(alpha, alpha[np.arange(self.t) ^ xor])
            self.uniform_alpha_checks += 1
        return super().operation(expr)


def main():
    llvm = Path("/data2/xsl/install/bin")
    texts = {}
    for name, file in (("baseline", "baseline.ll"), ("deferred", "kernel.ll")):
        ir = HERE / file
        texts[name] = re.sub(r"\bexact ", "", ir.read_text())
        subprocess.run([str(llvm / "llvm-as"), str(ir), "-o", str(HERE / (name + ".bc"))], check=True)
        optimized = HERE / (name + ".optimized.ll")
        subprocess.run([str(llvm / "opt"), "-passes=default<O3>,verify", "-S", str(ir), "-o", str(optimized)], check=True)
        if name == "deferred":
            # The exit reductions must not migrate back into the loop under O3.
            body, exit_body = optimized.read_text().split("\nv1623:", 1)
            assert body.count("call i32 @llvm.amdgcn.ds.bpermute(") == 4
            assert exit_body.count("call i32 @llvm.amdgcn.ds.bpermute(") == 68

    cases = [("random", 0), ("random", 1), ("random", 31),
             ("positive_overflow", 1), ("negative_underflow", 1),
             ("max_changes", 2), ("long_max_ramp", 31), ("uneven_lane_mass", 7)]
    records = []
    for kind, query in cases:
        rng = np.random.default_rng(42)
        q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        if kind == "positive_overflow": q.fill(16); k.fill(16)
        elif kind == "negative_underflow": q.fill(16); k.fill(-16)
        elif kind == "max_changes":
            q.fill(1); k.fill(-16)
            k[:, :128] = .25
            k[:, 128:256] = .5
        elif kind == "long_max_ramp":
            q.fill(1)
            for tile in range(32): k[:, tile*128:(tile+1)*128] = tile / 16 - 1
        elif kind == "uneven_lane_mass":
            q.fill(1); k.fill(-2)
            for tile in range(32):
                start = tile*128 + (tile % 4)*32
                k[:, start:start+8] = tile / 32
        head, group = 1, 4
        forward = 31-query
        linear = forward//group*(32*group) + head*group + forward%group
        expected = reference(q, k, v, head, query)
        outputs, variants = {}, {}
        for name, text in texts.items():
            model = CheckedBlock(text, q, k, v, physical=(linear % 32, linear // 32))
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                full = model.run().reshape(q.shape)
            actual = full[head, query*128:(query+1)*128]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            written = np.zeros(full.shape, dtype=bool)
            written[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            assert not model.pending
            want = {(1 << 32) + 2*((head*4096+row)*128+col): 1
                    for row in range(query*128, (query+1)*128) for col in range(128)}
            assert model.q_elements == want
            iterations = query + 1
            assert model.uniform_alpha_checks == iterations * 2
            assert model.denominator_shuffles == (4*iterations if name == "baseline" else 4)
            assert model.total_shuffles == (8*iterations+64 if name == "baseline" else 4*iterations+68)
            outputs[name] = actual
            variants[name] = dict(max_abs_error=float(np.max(np.abs(actual-expected))),
                denominator_shuffles=model.denominator_shuffles, total_shuffles=model.total_shuffles,
                uniform_alpha_checks=model.uniform_alpha_checks,
                finite=True, delayed_DMA_passed=True, full_Q_loaded_once=True)
        delta = float(np.max(np.abs(outputs["baseline"].astype(np.float64)-outputs["deferred"])))
        np.testing.assert_allclose(outputs["deferred"], outputs["baseline"], atol=.01, rtol=.01)
        record = dict(case=kind, query=query, iterations=query+1, variants=variants, max_abs_AB_delta=delta)
        records.append(record)
        print("PASS", json.dumps(record), flush=True)
    report = dict(experiment_sha256=hashlib.sha256((HERE / "kernel.ll").read_bytes()).hexdigest(),
                  llvm_as=True, O3_verified=True, O3_loop_shuffles=4, O3_exit_shuffles=68,
                  cases=records, gpu_tested=False)
    (HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
