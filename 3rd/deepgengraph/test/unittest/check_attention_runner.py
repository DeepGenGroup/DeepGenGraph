#!/usr/bin/env python3
"""Regress attention modes, launch geometry, and batch runner validation."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from TuneAttention import default_paths, launch_config, prepare_input
from generate_attention_ir import TEMPLATE, generate_ir

LLVM = """define amdgpu_kernel void @Attn_p2() !reqd_work_group_size !7 {
  ret void
}
!7 = !{i32 256, i32 1, i32 1}
"""


class AttentionRunnerTests(unittest.TestCase):
    def test_default_input_uses_shared_tensor_attention(self):
        source,_ = default_paths()
        self.assertEqual(source.name,'test_input_shared_d.mlir')
        text=source.read_text().split('  deepgengraph.kernel',1)[1]
        self.assertEqual(text.count('deepgengraph.precise_dot_op'),2)
        self.assertNotIn('affine.for %od',text)

    def test_shared_input_pipeline_switches_change_lowering_hints(self):
        source=default_paths()[0].read_text()
        disabled=prepare_input(source,{'pipeline-stages':'none','loop-mark':'0'},'qk')
        self.assertNotIn('frisk.qk_pipeline_hint',disabled)
        self.assertIn('frisk.qk_simplify_loop = false',disabled)
        enabled=prepare_input(source,{'pipeline-stages':'3','loop-mark':'1'},'qk')
        self.assertIn('frisk.qk_pipeline_hint = 3 : i64',enabled)
        self.assertIn('frisk.qk_simplify_loop = true',enabled)
        self.assertEqual(source.split('deepgengraph_triton.device_kernel',1)[1],
                         enabled.split('deepgengraph_triton.device_kernel',1)[1])

    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / name), *map(str, args)],
                              capture_output=True, text=True, cwd="/tmp")

    def test_dry_run_has_math_after_layout_and_correct_tile_positions(self):
        args = ["--dry-run", "--case", "baseline"]
        for option in ("layout=3.2", "math=mmac", "threads=128", "waves=2,1",
                       "masks=0", "prefetch=0", "fuse=1", "reuse=8",
                       "qk-tile=64", "vo-tile=32"):
            args += ["--set", option]
        result = self.run_script("TuneAttention.py", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line)[3:] for line in result.stdout.splitlines()
                    if line.startswith("env ")]
        self.assertEqual(len(commands), 1, result.stdout)
        argv = commands[0]
        self.assertEqual(argv[4:19], ["1", "1", "3.2", "mmac", "128", "2,1",
            "1", "0", "0", "1", "8", "useKPack", "asm128", "64", "32"])
        self.assertTrue(argv[19].startswith("--"), argv)

    def test_removed_carrier_set_and_sweep_are_rejected(self):
        for option in ("--set", "--sweep"):
            with self.subTest(option=option):
                args = ["--dry-run", option, "carrier=0"]
                if option == "--set":
                    args += ["--case", "baseline"]
                result = self.run_script("TuneAttention.py", *args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("carrier", result.stderr.partition("KEY 可取：")[2])

    def test_early_wave_preset_forwards_measured_flags_and_external_o3(self):
        result = self.run_script("TuneAttention.py", "--case", "early-wave-baseline",
                                 "--benchmark", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.startswith("env ")]
        self.assertEqual(len(commands), 1)
        argv = commands[0]
        self.assertEqual(argv[10:13], ["approx", "256", "4,1"])
        required = ["--pv-interleave=4", "--v-double-buffer=1", "--k-dma=64-swizzle",
                    "--retain-fragment-words=32",
                    "--k-lds-prefetch=8", "--softmax-fragments=pv",
                    "--k-tail-prefetch=4", "--cross-bn-k0=1",
                    "--lazy-softmax-base=8", "--softmax-cvt-pack=rtz",
                    "--hoist-lds-bases=1", "--hoist-lane-shuffles=1",
                    "--dump-llvm-passes=1", "--llvm-opt=0",
                    "--workgroup-group-size=32", "--workgroup-reverse=0"]
        for flag in required:
            self.assertIn(flag, argv)
        self.assertRegex(result.stdout, r"GenHsaco\.sh kernel\.ll 3 kernel\.hsaco")

    def test_early_wave_sweeps_override_baseline_without_changing_hsaco_o3(self):
        result = self.run_script("TuneAttention.py", "--sweep-base", "early-wave-baseline",
            "--sweep", "lazy-softmax-base=off;8", "--sweep", "softmax-cvt-pack=off;rtz",
            "--sweep", "llvm-opt=0;3", "--set", "k-lds-prefetch=4",
            "--benchmark", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.startswith("env ")]
        self.assertEqual(len(commands), 8)
        observed = set()
        for argv in commands:
            flags = dict(arg[2:].split("=", 1) for arg in argv if arg.startswith("--"))
            observed.add((flags["lazy-softmax-base"], flags["softmax-cvt-pack"], flags["llvm-opt"]))
            self.assertEqual(flags["k-lds-prefetch"], "4")
            self.assertEqual(flags["softmax-fragments"], "pv")
        self.assertEqual(observed, {("off", "off", "0"), ("off", "off", "3"),
            ("off", "rtz", "0"), ("off", "rtz", "3"), ("8", "off", "0"),
            ("8", "off", "3"), ("8", "rtz", "0"), ("8", "rtz", "3")})
        self.assertEqual(result.stdout.count("GenHsaco.sh kernel.ll 3 kernel.hsaco"), 8)

    def test_new_options_validate_values_before_writing_outputs(self):
        for option in ("lazy-softmax-base=0", "lazy-softmax-base=15.01",
                       "lazy-softmax-base=nan", "lazy-softmax-base=inf",
                       "softmax-fragments=bad", "softmax-cvt-pack=rne", "llvm-opt=1",
                       "hoist-lds-bases=2", "hoist-lane-shuffles=2", "dump-llvm-passes=2"):
            for mode in ("--set", "--sweep"):
                with self.subTest(option=option, mode=mode):
                    result = self.run_script("TuneAttention.py", mode, option, "--dry-run")
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(option.split("=")[0], result.stderr)
                    self.assertNotIn("KEY 可取", result.stderr)
        result = self.run_script("TuneAttention.py", "--case", "early-wave-baseline",
            "--set", "lazy-softmax-base=0.125", "--set", "softmax-fragments=fragment",
            "--set", "hoist-lds-bases=0", "--set", "dump-llvm-passes=0", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--lazy-softmax-base=0.125", "--softmax-fragments=fragment",
                     "--hoist-lds-bases=0", "--dump-llvm-passes=0"):
            self.assertIn(flag, result.stdout)

    def test_candidate_preparation_preserves_input_in_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mlir"
            original = default_paths()[0].read_bytes()
            source.write_bytes(original)
            result = self.run_script("3rd/deepgengraph/test/generate_attention_candidates.py",
                "--input", source, "--output-dir", directory, "--tool", "/bin/false")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), original)

    def test_candidate_preparation_rejects_snapshot_source_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prepared_input.mlir"
            original = default_paths()[0].read_bytes()
            source.write_bytes(original)
            result = self.run_script("3rd/deepgengraph/test/generate_attention_candidates.py",
                "--input", source, "--output-dir", directory, "--tool", "/bin/false")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(source.read_bytes(), original)

    def test_shared_input_auto_target_controls_generated_qk_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'prepared.mlir'
            result = self.run_script('TuneAttention.py', '--prepare-input', output,
                                     '--set', 'pipeline-stages=none', '--set', 'loop-mark=0')
            self.assertEqual(result.returncode, 0, result.stderr)
            prepared = output.read_text()
            self.assertNotIn('frisk.qk_pipeline_hint', prepared)
            self.assertIn('frisk.qk_simplify_loop = false', prepared)
            result = self.run_script('TuneAttention.py', '--dry-run', '--case', 'baseline',
                                     '--set', 'pipeline-stages=3', '--set', 'loop-mark=1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--loop-iv qk', result.stdout)
            self.assertIn('pipeline-stages=3', result.stdout)

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

    def test_tail_mask_does_not_imply_causal_attention(self):
        for s in (1, 65, 129):
            for causal in (True, False):
                source, _ = generate_ir(TEMPLATE.read_text(), 64, 32,
                    b=2, h=3, s=s, causal=causal)
                self.assertIn("deepgengraph.mask", source)
                config = launch_config(source, LLVM)
                self.assertEqual(config["grid"], [(s + 63) // 64, 3, 2])
                self.assertIs(config["causal"], causal)

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
            result = self.run_script("RunAttention.py", "--s", "256", "0",
                                     "--stage", "ir", "--output-dir", output)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("S=0", result.stderr)
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
                assert sys.argv[7] == 'approx'
                assert sys.argv[8:10] == ['256', '4,1']
                assert '--softmax-fragments=pv' in sys.argv
                assert '--lazy-softmax-base=8' in sys.argv
                assert '--softmax-cvt-pack=rtz' in sys.argv
                assert '--llvm-opt=0' in sys.argv
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
