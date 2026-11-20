#!/usr/bin/env python3
"""Generated FIFO completion, prefix reuse and delayed-DMA CPU execution.

Barriers are checked structurally; the CPU model does not simulate wave timing.
"""
import re
import tempfile
from pathlib import Path

from check_auto_pipeline import TOOL, PASS, execute_cpu, fixture, main_ir, run


def source(stages):
    text = fixture(f"{stages} : i64")
    text = text.replace("-> (f32, f32) {", "-> (f32, f32) attributes {thread_num = 64 : i32} {", 1)
    text = text.replace("  %r:2 = affine.for", """  %prefix = memref.alloc() : memref<2xf32, 3>
  memref.store %three, %prefix[%c0] : memref<2xf32, 3>
  %saved = memref.load %prefix[%c0] : memref<2xf32, 3>
  %r:2 = affine.for""")
    return text.replace("arith.mulf %acc, %three", "arith.mulf %acc, %saved")


def physical_allocations(text):
    return re.sub(r'frisk\.alloc_buffer \{[^}]*\} -> (memref<[^\n]+>)',
                  r'memref.alloc() alignment = 4 : \1', text)


def main():
    with tempfile.TemporaryDirectory(prefix="pipeline-shm-") as directory:
        work = Path(directory)
        for stages in (2, 3, 4):
            pipelined, err = run([TOOL, PASS, "--verify-each"], source(stages))
            assert "skipped" not in err, err
            physical = physical_allocations(pipelined)
            _, report = run([TOOL, "--frisk-analyze-shm-liveness"], physical)
            assert "skipped:" not in report, report
            assert report.count("async across iterations") == 2 * stages, report
            reused, report = run([TOOL, "--test-shm-reuse", "--verify-each"], physical)
            expected = f"shm bytes {8 + 8 * stages} -> {8 * stages}, barriers 1"
            assert expected in report, report
            # The old prefix readers must finish before any guarded prologue
            # DMA can write the same storage. Do not insert this in a branch.
            assert reused.index("memref.load") < reused.index("gpu.barrier") < reused.index("scf.if"), reused
            assert run([TOOL, "--test-shm-reuse"], reused)[0] == reused
            combined = reused.rsplit("}", 1)[0] + fixture(None, "reference") + main_ir() + "}\n"
            execute_cpu(combined.replace("gpu.barrier", ""), work, f"reuse{stages}")

            # Removing any exit-slot completion must make at least one buffer
            # ineligible; the proof must not infer completion just from a marker.
            drains = list(re.finditer(r'^    frisk.wait_buffer_ready[^\n]+$', physical, re.M))
            assert len(drains) == 2 * (stages - 1), physical
            for drain in drains:
                missing = physical[:drain.start()] + physical[drain.end():]
                _, report = run([TOOL, "--frisk-analyze-shm-liveness"], missing)
                assert "skipped: async task" in report, report
            # A late prefix read really overlaps the FIFO lifetime.
            late = physical.replace("    return ",
                "    %late = memref.load %prefix[%c0] : memref<2xf32, 3>\n    return ")
            # The parser renames SSA values; locate the prefix allocation's
            # canonical name rather than assuming the source spelling survived.
            prefix = re.search(r'(%\w+) = memref.alloc\(\).*memref<2xf32, 3>', physical)[1]
            zero = re.search(r'(%\w+) = arith.constant 0 : index', physical)[1]
            late = late.replace("%prefix[", prefix + "[").replace("[%c0]", "[" + zero + "]")
            _, report = run([TOOL, "--test-shm-reuse"], late)
            assert f"shm bytes {8 + 8 * stages} -> {8 + 8 * stages}" in report, report
            print(f"PASS {stages} stages: prefix reuse, handoff, delayed DMA, all exit drains, overlapping-reader rejection")


if __name__ == "__main__":
    main()
