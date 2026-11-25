"""Orchestration failure checks; real compiler/solver coverage is in test_scheduler."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import re
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_pipeline
from lds_budget import inspect_llvm_lds


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="cpsat-driver-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "input with spaces.mlir"
        self.source.write_text("input")
        self.compiler = self.root / "MyTest"
        self.compiler.write_text("#!/bin/sh\nexit 0\n")
        self.compiler.chmod(0o755)
        self.output = self.root / "kernel.ll"
        self.output.write_text("previous output")
        self.work = self.root / "new work"
        self.commands = []
        self.environments = []
        self.llvm = '@shm_0 = addrspace(3) global [16384 x i8] undef, align 16\n'
        self.assembly = '''amdhsa.kernels:
  - .args: []
    .name: kernel
    .vgpr_count: 240
    .sgpr_count: 42
    .vgpr_spill_count: 0
    .sgpr_spill_count: 0
    .private_segment_fixed_size: 0
    .group_segment_fixed_size: 16384
amdhsa.version: [1, 1]
'''

    def invoke(self, *, fail=None, missing=None, extra=()):
        def run(command, **kwargs):
            stage = ("export", "solve", "lower", "codegen")[len(self.commands)]
            self.commands.append(command)
            self.environments.append(kwargs["env"].copy())
            if stage == fail:
                return subprocess.CompletedProcess(command, 7)
            if stage != missing:
                if stage == "export":
                    path = next(x.split("=", 1)[1] for x in command
                                if x.startswith("--cpsat-export="))
                elif stage in ("solve", "codegen"):
                    path = command[command.index("-o") + 1]
                else:
                    path = command[3]
                Path(path).write_text(self.llvm if stage == 'lower' else
                                     self.assembly if stage == 'codegen' else 'new output')
            return subprocess.CompletedProcess(command, 0)

        with patch.object(run_pipeline.subprocess, "run", side_effect=run), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_pipeline.main([str(self.source), "--mytest", str(self.compiler),
                                      "--work-dir", str(self.work), "-o", str(self.output),
                                      *extra])

    def test_success_reuses_arguments_environment_and_isolates_artifacts(self):
        provenance = self.root / "provenance.json"
        provenance.write_text(json.dumps({"mytest_arguments_after_output": ["1", "1"],
                                         "environment": {"FRISK_GM_TO_SHM_MODE": "sync"}}))
        status = self.invoke(extra=["--provenance", str(provenance), "--loop", "Attn_p2:0",
                                    "--unroll-max-trip-count", "16", "--mode", "pipeline",
                                    "--require-cross-iteration",
                                    "--time-limit", "2", "--", "--k-tail-prefetch=4"])
        self.assertEqual(status, 0)
        self.assertEqual(self.output.read_text(), self.llvm)
        first, solver, second = self.commands
        self.assertEqual(first[:3], second[:3])
        self.assertEqual(first[4:8], second[4:8])
        self.assertEqual(first[4:8], ["1", "1", "--k-tail-prefetch=4", "--cpsat-loop=Attn_p2:0"])
        self.assertIn("--cpsat-stop-after=1", first)
        self.assertFalse(any(x.startswith("--cpsat-stop-after") for x in second))
        self.assertIn("--time-limit=2.0", solver)
        self.assertIn("--lds-budget=16384", solver)
        self.assertIn('--vgpr-budget=250', solver)
        self.assertIn('--register-reserve=32', solver)
        self.assertFalse(json.loads((self.work / 'assembly_check.json').read_text())['physical_registers_verified'])
        self.assertTrue(json.loads((self.work / 'lds_check.json').read_text())['within_budget'])
        self.assertIn("--require-cross-iteration", solver)
        for command in (first, second):
            self.assertIn("--cpsat-unroll-max-trip-count=16", command)
            self.assertIn("--cpsat-unroll-max-expanded-ops=2048", command)
        self.assertEqual(self.environments[0], self.environments[2])
        self.assertEqual(self.environments[0]["FRISK_GM_TO_SHM_MODE"], "sync")
        for stage in ("export", "solve", "lower"):
            self.assertTrue((self.work / f"{stage}.log").is_file())

    def test_failures_stop_pipeline_and_preserve_existing_output(self):
        for index, stage in enumerate(("export", "solve", "lower")):
            for failure_type in ("fail", "missing"):
                with self.subTest(stage=stage, failure_type=failure_type):
                    self.work = self.root / f"{stage}-{failure_type}"
                    self.commands.clear()
                    self.assertEqual(self.invoke(**{failure_type: stage}), 1)
                    self.assertEqual(len(self.commands), index + 1)
                    self.assertEqual(self.output.read_text(), "previous output")

    def test_existing_work_directory_is_rejected_without_using_stale_schedule(self):
        self.work.mkdir()
        (self.work / "schedule.json").write_text("old schedule")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.commands, [])
        self.assertEqual((self.work / "schedule.json").read_text(), "old schedule")

    def test_managed_flags_cannot_disable_lowering(self):
        self.assertEqual(self.invoke(extra=["--", "--cpsat-stop-after=1"]), 1)
        self.assertEqual(self.commands, [])

    def test_physical_overflow_preserves_output_and_budget_override_is_shared(self):
        self.llvm = '@shm_0 = addrspace(3) global [32768 x i8] undef, align 16\n'
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.output.read_text(), 'previous output')
        self.assertFalse((self.work / 'kernel.ll').exists())
        self.assertTrue((self.work / 'lower/kernel.ll').exists())
        report = json.loads((self.work / 'lds_check.json').read_text())
        self.assertEqual(report['physical_lds_bytes'], 32768)
        self.assertFalse(report['within_budget'])
        self.work = self.root / 'larger-budget'
        self.commands.clear()
        self.assertEqual(self.invoke(extra=['--lds-budget', '32768']), 0)
        self.assertIn('--lds-budget=32768', self.commands[1])
        self.assertEqual(self.output.read_text(), self.llvm)

    def test_unknown_physical_size_fails_closed(self):
        self.llvm = '@dynamic = external addrspace(3) global [0 x i8], align 16\n'
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.output.read_text(), 'previous output')
        self.assertIn('error', json.loads((self.work / 'lds_check.json').read_text()))

    def test_multiple_lds_globals_include_alignment_padding(self):
        self.output.write_text('@a = addrspace(3) global [17 x i8] undef, align 16\n'
                               '@b = addrspace(3) global [16 x i8] undef, align 16\n')
        self.assertEqual(inspect_llvm_lds(self.output, 48)['physical_lds_bytes'], 48)
        self.assertFalse(inspect_llvm_lds(self.output, 47)['within_budget'])

    def test_negative_budget_rejected_before_compilation(self):
        self.assertEqual(self.invoke(extra=['--lds-budget', '-1']), 1)
        self.assertEqual(self.commands, [])

    def test_target_codegen_rejects_over_budget_or_spill_before_publishing(self):
        for label, vgpr, spill, accepted in [('pass', 250, 0, True),
                                             ('over', 251, 0, False),
                                             ('spill', 240, 1, False)]:
            with self.subTest(label=label):
                self.work = self.root / label
                self.commands.clear()
                self.output.write_text('previous output')
                # Set fields independently between candidates.
                self.assembly = re.sub(r'\.vgpr_count: \d+', f'.vgpr_count: {vgpr}', self.assembly)
                self.assembly = re.sub(r'\.vgpr_spill_count: \d+', f'.vgpr_spill_count: {spill}', self.assembly)
                status = self.invoke(extra=['--llc', str(self.compiler), '--mcpu', 'gfx936'])
                self.assertEqual(status, 0 if accepted else 1)
                self.assertEqual(self.output.read_text(), self.llvm if accepted else 'previous output')
                report = json.loads((self.work / 'assembly_check.json').read_text())
                self.assertTrue(report['physical_registers_verified'])
                self.assertEqual(report['accepted'], accepted)


if __name__ == "__main__":
    unittest.main()
