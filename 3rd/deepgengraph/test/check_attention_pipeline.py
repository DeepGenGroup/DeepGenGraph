#!/usr/bin/env python3
"""Compile the annotated seqLen pipeline through LLVM export.

Usage: check_attention_pipeline.py /path/to/MyTest [/path/to/llvm-as]
Checks compilation/IR invariants, not execution on a GPU.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def module_after(log, marker):
    text = log.split(marker + "\n", 1)[1]
    return text[:text.index("\n}\n") + 3]


def main():
    tool = Path(sys.argv[1]).resolve()
    source = Path(__file__).with_name("test_input.mlir").resolve()
    with tempfile.TemporaryDirectory(prefix="attention-pipeline-") as directory:
        output = Path(directory) / "attention.ll"
        proc = subprocess.run([str(tool), str(source), str(output)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout[-8000:]
        base = module_after(proc.stdout,
                            "---------- after createConvertFriskToBasePass ---------")
        for attr in ("stage", "order"):
            arrays = re.findall(r"pipeline\." + attr + r" = array<i64: ([^>]+)>", base)
            assert len(arrays) == 1, "only the seqLen loop should be annotated"
            assert len(arrays[0].split(",")) == 17
        scheduled = module_after(proc.stdout,
                                 "---------- after frisk-pipeline-schedule ---------")
        assert scheduled.count("pipeline.scheduled") == 1
        assert "pipeline.num_stages = 3" in scheduled
        assert "pipeline.slots = 2" in scheduled
        # alloc_buffer's custom printer omits discardable slot attributes.
        for shape in ("128x32", "32x128", "64x32"):
            assert len(re.findall(r"frisk\.alloc_buffer[^\n]*-> memref<" +
                                  shape + r"xf16, 3>", scheduled)) == 2
        assert scheduled.count("frisk.copy_to_reg") == 2
        thread = module_after(proc.stdout,
                              "---------- after createConvertFriskBaseToThreadLevelIRPass ---------")
        assert not re.search(r"frisk\.(?:copy_to_reg|to_threadTile|from_threadTile|gemm)\b", thread)
        llvm = output.read_text()
        assert "define amdgpu_kernel void @Attn_p2" in llvm
        assert "v_mmac_f32_16x16x16_f16" in llvm
        if len(sys.argv) > 2:
            subprocess.run([sys.argv[2], str(output), "-o",
                            str(Path(directory) / "attention.bc")], check=True)
    print("PASS attention seqLen pipeline: annotation, double buffering, thread lowering and LLVM export")


if __name__ == "__main__":
    main()
