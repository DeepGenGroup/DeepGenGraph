#!/usr/bin/env python3
"""Validate the selected IR with the existing delayed-DMA CPU model."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_auto_pipeline_attention import DelayedDMA, module_after
from check_attention_llvm_numerics import reference, report

text = (HERE / "attention.ll").read_text()
log = (HERE / "attention.log").read_text()
scheduled = module_after(log, "---------- after frisk-auto-pipeline ---------")
assert scheduled.count("pipeline.scheduled") == 1
assert "pipeline.num_stages = 2 : i64" in scheduled
assert "pipeline.slots = 2 : i64" in scheduled
assert "8 direct, 0 staged, 0 fallback" in log
(HERE / "attention.scheduled.mlir").write_text(scheduled)
storages = re.findall(
    r"^@[\w.]+ = .*addrspace\(3\) global \[(\d+) x (i8|half|float)\]",
    text, re.M)
lds_bytes = sum(int(count) * {"i8": 1, "half": 2, "float": 4}[ty]
                for count, ty in storages)
assert lds_bytes == 40960
rng = np.random.default_rng(42)
q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16)
           for _ in range(3)]
cases = []
for head, block in [(0, 0), (1, 1), (1, 31)]:
    model = DelayedDMA(text, q, k, v, threads=256, head=head, block=block)
    actual = model.run().reshape(q.shape)[head, block * 128:(block + 1) * 128]
    expected = reference(q, k, v, head, block)
    assert report(f"head={head} block={block}", actual, expected)
    assert not model.pending
    assert model.overlapped_mma > 0
    case = {
        "head": head, "query_block": block,
        "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected))),
        "mmac_with_dma_pending": model.overlapped_mma,
    }
    print(json.dumps(case), flush=True)
    cases.append(case)
result = {
    "file": "attention.ll",
    "sha256": hashlib.sha256((HERE / "attention.ll").read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(
        (ROOT / "3rd/deepgengraph/test/test_input_spidD.mlir").read_bytes()).hexdigest(),
    "performance_status": "Unmeasured candidate; no accessible DCU in this environment.",
    "gpu_tested": False,
    "num_stages": 2, "qk_tile": 32, "vo_tile": 32,
    "static_lds_bytes_in_ir": lds_bytes,
    "launch": {
        "kernel": "Attn_p2", "grid": [32, 32, 1], "block": [256, 1, 1],
        "dynamic_shared_memory_bytes": 0,
        "argument_order": ["Q", "K", "V", "Out"],
        "shape": [1, 32, 4096, 128], "dtype": "float16",
        "layout": "contiguous BHSD",
    },
    "cpu_delayed_dma_checks": cases,
    "limitations": "CPU model does not establish backend codegen correctness, "
                   "independent wave timing, hardware hazards, or GPU performance.",
    "selection_reason": "QK32 uses 40 KiB static LDS; QK64 uses 72 KiB. "
                        "Selected QK32 for first hardware trial, not by measured speed.",
}
(HERE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
