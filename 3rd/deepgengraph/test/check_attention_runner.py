#!/usr/bin/env python3
"""Regress attention modes, launch geometry, and batch runner validation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from TuneAttention import launch_config

LLVM = """define amdgpu_kernel void @Attn_p2() !reqd_work_group_size !7 {
  ret void
}
!7 = !{i32 256, i32 1, i32 1}
"""


class AttentionRunnerTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / name), *map(str, args)],
                              capture_output=True, text=True, cwd="/tmp")

    def test_noncausal_removes_mask_and_visits_all_keys(self):
        # A missing bound change silently computes only a causal prefix.
        with tempfile.TemporaryDirectory() as directory:
            for softmax in ("online", "legacy"):
                for bk in (None, 32):
                    with self.subTest(softmax=softmax, bk=bk):
                        output = Path(directory) / "input.mlir"
                        args = ["--b", "2", "--h", "3", "--s", "256", "--d", "64",
                                "--bm", "64", "--bn", "32", "--softmax", softmax,
                                "--non-causal", "--output", output]
                        if bk is not None:
                            args += ["--bk", str(bk)]
                        result = self.run_script("generate_attention_ir.py", *args)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        ir = output.read_text()
                        self.assertNotIn("deepgengraph.mask", ir)
                        self.assertNotIn("deepgengraph.trilu", ir)
                        self.assertRegex(ir, r"scf.for %(?:n|arg10) = %c0 to %cS step %cBN")
                        config = launch_config(ir, LLVM)
                        self.assertEqual(config["shape"], [2, 3, 256, 64])
                        self.assertEqual(config["grid"], [4, 3, 2])
                        self.assertEqual(config["block"], [256, 1, 1])
                        self.assertIs(config["causal"], False)

    def test_existing_fixtures_select_matching_reference(self):
        for filename, causal in (("test_input.mlir", True),
                                 ("test_input_spidD_noncasual.mlir", False)):
            with self.subTest(filename=filename):
                source = (ROOT / "3rd/deepgengraph/test" / filename).read_text()
                self.assertIs(launch_config(source, LLVM).get("causal"), causal)

    def test_runner_cartesian_shapes_modes_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory(prefix="attention runner ") as directory:
            args = ["--b", "1", "2", "--h", "3", "--s", "256", "--d", "64",
                    "--bm", "64", "--bn", "32", "--bk", "32",
                    "--casual", "both", "--stage", "ir", "--output-dir", directory]
            for count in (1, 2):
                result = self.run_script("RunAttention.py", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                sessions = list(Path(directory).glob("run-*"))
                self.assertEqual(len(sessions), count)
                for session in sessions:
                    inputs = list((session / "inputs").glob("*.mlir"))
                    self.assertEqual(len(inputs), 4)
                    configs = [launch_config(path.read_text(), LLVM) for path in inputs]
                    self.assertEqual({(c["shape"][0], c["causal"]) for c in configs},
                                     {(1, True), (1, False), (2, True), (2, False)})
                    self.assertEqual(len(json.loads((session / "inputs.json").read_text())), 4)

    def test_invalid_batch_does_not_write_partial_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            result = self.run_script("RunAttention.py", "--s", "256", "255",
                                     "--stage", "ir", "--output-dir", output)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("S=255", result.stderr)
            self.assertFalse(output.exists())

    def test_flat_bundle_generates_user_sweep(self):
        # Copy the same flat layout as the remote /workspace, then invoke
        # from another cwd to catch paths accidentally relative to the shell.
        with tempfile.TemporaryDirectory(prefix="attention flat ") as directory:
            directory = Path(directory)
            for name in ("RunAttention.py", "TuneAttention.py", "generate_attention_ir.py"):
                shutil.copy2(ROOT / name, directory / name)
            shutil.copy2(ROOT / "3rd/deepgengraph/test/test_input.mlir", directory / "test_input.mlir")
            result = self.run_script(directory / "RunAttention.py",
                                     "--b", "1", "2", "4", "8", "--h", "16", "32",
                                     "--s", "1024", "2048", "4096", "--d", "64", "128",
                                     "--casual", "both", "--stage", "ir")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            session = next((directory / "generated_ir/attention_runs").glob("run-*"))
            records = json.loads((session / "inputs.json").read_text())
            self.assertEqual(len(records), 96)
            for record in records:
                config = launch_config(Path(record["input"]).read_text(), LLVM)
                self.assertEqual(config["shape"], record["shape"])
                self.assertIs(config["causal"], record["causal"])

    def test_pipeline_preserves_launch_contract_and_reports_failures(self):
        # Backend stand-ins keep this orchestration test runnable without DTK
        # or a GPU. The generator, TuneAttention, files and reports are real.
        with tempfile.TemporaryDirectory(prefix="attention pipeline ") as directory:
            directory = Path(directory)
            for name in ("RunAttention.py", "TuneAttention.py", "generate_attention_ir.py"):
                shutil.copy2(ROOT / name, directory / name)
            shutil.copy2(ROOT / "3rd/deepgengraph/test/test_input.mlir", directory / "test_input.mlir")
            runner = directory / "RunAttention.py"
            compiler = directory / "MyTest"
            compiler.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
                from pathlib import Path
                import sys
                assert Path(sys.argv[1]).is_file()
                assert sys.argv[2] == '0'
                Path(sys.argv[3]).write_text({LLVM!r})
                """))
            compiler.chmod(0o755)
            backend = directory / "genHsaco.sh"
            backend.write_text('set -eu\ntest -s "$1"\ntest "$2" = 2\nprintf test-hsaco > "$3"\n')
            launcher = directory / "launchKernel.py"
            launcher.write_text(textwrap.dedent("""\
                import json
                from pathlib import Path
                import sys
                args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
                config = json.loads(Path(args['--config']).read_text())
                assert config['shape'] == [2, 3, 256, 64]
                assert config['grid'] == [4, 3, 2]
                assert config['block'] == [256, 1, 1]
                assert type(config['causal']) is bool
                assert args['--warmup'] == '0' and args['--repeat'] == '2'
                assert args['--device'] == '1' and args['--seed'] == '7'
                assert Path(args['--hsaco']).read_text() == 'test-hsaco'
                Path(args['--json-output']).write_text(json.dumps(dict(
                    passed=True, kernel_median_ms=1.0, sdpa_median_ms=2.0)))
                """))
            args = ["--shape", "2,3,256,64", "--bm", "64", "--bn", "32",
                    "--causal", "both", "--opt-level", "2", "--warmup", "0",
                    "--repeat", "2", "--device", "1", "--seed", "7"]
            output = directory / "success"
            result = self.run_script(runner, *args, "--output-dir", output)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            session = next(output.glob("run-*"))
            results = [json.loads(p.read_text()) for p in session.glob("compiled/**/result.json")]
            self.assertEqual([r["status"] for r in results], ["benchmark-passed"] * 2)
            best = json.loads((session / "summary/best_by_shape.json").read_text())
            self.assertEqual({r["launch"]["causal"] for r in best.values()}, {True, False})

            # A successful process exit with a failed numerical report must
            # still make the overall command fail.
            launcher.write_text(launcher.read_text().replace("passed=True", "passed=False"))
            result = self.run_script(runner, *args, "--output-dir", directory / "incorrect")
            self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
            self.assertIn("incorrect", result.stdout)

            backend.write_text("exit 9\n")
            result = self.run_script(runner, *args, "--output-dir", directory / "failed")
            self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
            self.assertFalse(list((directory / "failed").glob("**/benchmark.json")))


if __name__ == "__main__":
    unittest.main()
