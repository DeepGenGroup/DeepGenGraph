#!/usr/bin/env python3
"""Compile hinted QK loops and execute LLVM with DMA delayed until wait.

This checks slot addresses/readiness and outstanding prefetch during MMA.
It does not model independent wave timing or measure hardware performance.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel, reference, report

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "build/test/MyTest"
LLVM_AS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data2/xsl/install/bin/llvm-as")


class DelayedDMA(LLVMWorkgroupModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = []
        self.issuing = False
        self.overlapped_mma = 0

    def addresses(self, ptr, ty, mask):
        return np.unique((ptr[:, None] +
                          np.arange(self.size(ty))[None, :])[mask].ravel())

    def memory(self, ptr, ty, write=None):
        touched = self.addresses(ptr, ty, self.active)
        for _, _, _, _, pending in self.pending:
            assert not np.intersect1d(touched, pending).size, \
                "read or overwrite of a destination whose DMA has not completed"
        if self.issuing and write is not None:
            self.pending.append((ptr.copy(), ty, write.copy(),
                                 self.active.copy(), touched))
            return
        return super().memory(ptr, ty, write)

    def call(self, body):
        if "s_waitcnt vmcnt(0)" in body:
            pending, self.pending = self.pending, []
            active = self.active
            for ptr, ty, values, mask, _ in pending:
                self.active = mask
                super().memory(ptr, ty, values)
            self.active = active
        if "v_mmac_" in body and self.pending:
            self.overlapped_mma += 1
        self.issuing = "buffer_load_dwordx4" in body
        try:
            return super().call(body)
        finally:
            self.issuing = False


def module_after(log, marker):
    part = log.split(marker + "\n", 1)[1]
    return part[:part.index("\n}\n") + 3]


def main():
    source = (HERE / "test_input_spidD.mlir").read_text()
    pattern = r"\s*\{\s*frisk\.pipeline_hint(?:\s*=\s*\d+\s*:\s*i64)?\s*\}"
    assert len(re.findall(pattern, source)) == 1
    rng = np.random.default_rng(42)
    q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    with tempfile.TemporaryDirectory(prefix="auto-pipeline-attention-") as directory:
        work = Path(directory)
        for stages, bk, packing in [(None, 32, "useKPack"), (2, 32, "useKPack"),
                                    (3, 32, "useKPack"), (2, 64, "useKPack"),
                                    (2, 32, "noKPack")]:
            name = f"s{stages}-bk{bk}-{packing}"
            mlir, output, logpath = [work / (name + suffix) for suffix in (".mlir", ".ll", ".log")]
            mlir.write_text(re.sub(pattern, "" if stages is None else
                                  f" {{frisk.pipeline_hint = {stages} : i64}}", source))
            cmd = [str(TOOL), str(mlir), "0", str(output),
                   "1", "1", "3.3", "34816", "approx", "256", "4,1",
                   "1", "1", "1", "0", "0", packing, "asm128", str(bk), "32"]
            with logpath.open("w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "FRISK_GM_TO_SHM_MODE": "async",
                                           "FRISK_GM_TO_SHM_ALIGNMENT": "16"}, timeout=180)
            log = logpath.read_text()
            assert proc.returncode == 0 and "error:" not in log, log[-8000:]
            scheduled = module_after(log, "---------- after frisk-auto-pipeline ---------")
            if stages:
                assert scheduled.count("pipeline.scheduled") == 1
                assert f"pipeline.num_stages = {stages} : i64" in scheduled
                assert f"pipeline.slots = {stages} : i64" in scheduled
            else:
                assert "pipeline.scheduled" not in scheduled
            assert "0 staged, 0 fallback" in log
            # A selected destination must retain the original MMA consumer
            # width across prologue and steady-state DMA planning.
            assert all("kpack=2 " in line for line in log.splitlines()
                       if line.startswith("DMA plan:"))
            subprocess.run([LLVM_AS, output, "-o", work / (name + ".bc")], check=True)
            text = output.read_text()
            for head, block in [(0, 0), (1, 1)]:
                model = DelayedDMA(text, q, k, v, threads=256, head=head, block=block)
                actual = model.run().reshape(q.shape)[head, block * 128:(block + 1) * 128]
                assert not model.pending, "outstanding DMA at kernel return"
                assert report(name + f" head={head} block={block}",
                              actual, reference(q, k, v, head, block))
                if stages:
                    assert model.overlapped_mma > 0, "no prefetch outstanding during QK MMA"
            print(f"PASS {name}: LLVM verify, delayed DMA, slot hazards, numerical results")


if __name__ == "__main__":
    main()
