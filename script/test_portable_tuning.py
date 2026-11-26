"""Portable entrypoint configuration, tool discovery and candidate ranking."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import TuneAttention as tune
import generate_attention_ir as generator
ENTRY = ROOT / "run_tuning.py"
if not ENTRY.exists():
    ENTRY = ROOT / "portable/attention-tuning/run_tuning.py"
spec = importlib.util.spec_from_file_location("portable_tuning", ENTRY)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PortableTuningTest(unittest.TestCase):
    def test_bk_zero_validation(self):
        config = runner.settings(ENTRY.with_name("tuning.json"), False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tuning.json"
            config["bk"] = [0, 32, 0, 64]
            path.write_text(json.dumps(config))
            self.assertEqual(runner.settings(path, False)["bk"], [0, 32, 64])
            for key, values in (("bk", [-1]), ("bk", [False]), ("bk", [0.0]),
                                ("bk", []), ("bm", [0]), ("bn", [0])):
                with self.subTest(key=key, values=values):
                    path.write_text(json.dumps(dict(config, **{key: values})))
                    with self.assertRaises(ValueError):
                        runner.settings(path, False)

    def test_bk_zero_and_mixed_generation_manifest(self):
        config = runner.settings(ENTRY.with_name("tuning.json"), True)
        config.update(bm=[64, 128], bn=[32], cases=["legacy-baseline"])
        for bks in ([0], [32], [32, 0, 64, 0]):
            with self.subTest(bks=bks), tempfile.TemporaryDirectory(prefix="bk inputs ") as temp:
                directory = Path(temp)
                path = directory / "config.json"
                path.write_text(json.dumps(dict(config, bk=bks)))

                def check_inputs(command, env):
                    manifest = Path(command[command.index("--input-list") + 1])
                    inputs = [Path(line) for line in manifest.read_text().splitlines()]
                    self.assertEqual(len(inputs), 2 * len(set(bks)))
                    self.assertEqual(len(inputs), len(set(inputs)))
                    expected = [generator.generate_ir(generator.TEMPLATE.read_text(), bm, 32,
                                                      bk if bk else None)[0]
                                for bk in dict.fromkeys(bks) for bm in (64, 128)]
                    self.assertEqual([p.read_text() for p in inputs], expected)
                    for p in inputs:
                        name = p.name.replace("-online.mlir", ".mlir")
                        if name.endswith("-64-32.mlir") or name.endswith("-128-32.mlir"):
                            self.assertNotIn("scf.for %dk", p.read_text())
                        else:
                            self.assertIn("scf.for %dk", p.read_text())
                    self.assertIn("legacy-baseline", command)
                    return 0

                with mock.patch.object(runner, "ROOT", ROOT), \
                     mock.patch.object(runner, "check_environment", return_value=(
                         "/unused/MyTest", sys.executable, os.environ.copy(), {})), \
                     mock.patch.object(runner, "run_tuning", side_effect=check_inputs) as tuning, \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = runner.main(["--config", str(path), "--mode", "ir",
                                          "--output-dir", str(directory / "results")])
                self.assertEqual(result, 0)
                tuning.assert_called_once()

    def test_mixed_bk_dry_run_is_replayable(self):
        config = runner.settings(ENTRY.with_name("tuning.json"), True)
        config.update(bk=[0, 32, 64], cases=["legacy-baseline"])
        with tempfile.TemporaryDirectory(prefix="bk preview ") as temp:
            directory = Path(temp)
            path = directory / "config.json"
            path.write_text(json.dumps(config))
            preview = io.StringIO()
            with mock.patch.object(runner, "ROOT", ROOT), contextlib.redirect_stdout(preview):
                self.assertEqual(runner.main(["--config", str(path), "--dry-run", "--mode", "ir",
                                              "--output-dir", str(directory / "results")]), 0)
            lines = preview.getvalue().splitlines()
            self.assertNotIn("--bk", lines[1])
            self.assertIn("--bk 32", lines[2])
            self.assertIn("--bk 64", lines[3])
            # Replay only generation, without requiring a compiler/GPU.
            subprocess.run(["bash", "-e", "-c", "\n".join(lines[:-1])], check=True)
            manifest = directory / "results/PREVIEW/inputs.txt"
            self.assertEqual(len(manifest.read_text().splitlines()), 3)
            self.assertTrue(all(Path(p).is_file() for p in manifest.read_text().splitlines()))

    def test_smoke_and_grid_conflict(self):
        path = ENTRY.with_name("tuning.json")
        smoke = runner.settings(path, True)
        self.assertEqual((smoke["bm"], smoke["bn"], smoke["bk"]), ([128], [128], [32]))
        self.assertEqual(smoke["cases"], ["online-baseline"])
        self.assertEqual(smoke["softmax"], "online")
        self.assertIsNone(smoke["sweep_base"])
        self.assertEqual((smoke["warmup"], smoke["repeat"]), (1, 3))
        self.assertEqual(runner.settings(path, False)["bm"], [64, 128])
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "tuning.json"
            smoke["sweep"] = ["qk-tile=16;32"]
            bad.write_text(json.dumps(smoke))
            with self.assertRaises(ValueError):
                runner.settings(bad, False)

    def test_swizzle_configuration_forwards_independent_axes(self):
        config = runner.settings(ENTRY.with_name("tuning.swizzle.json"), False)
        self.assertEqual((config["bm"],config["bn"],config["bk"]), ([128],[128],[32]))
        _, command = runner.commands(config,"ir",Path("/tmp/session"),"/tmp/MyTest",sys.executable)
        for axis in ("swizzle-q=off;on","swizzle-k=off;on","swizzle-v=off;on"):
            self.assertIn(axis, command)
        self.assertIn("workgroup-group-size=4", command)
        self.assertIn("workgroup-reverse=1", command)
        defaults = runner.settings(ENTRY.with_name("tuning.json"), False)
        self.assertIn("group8-reverse-k128-swizzle", defaults["cases"])

    def test_online_mode_and_sweep_base_forwarding(self):
        config = runner.settings(ENTRY.with_name("tuning.online.json"), False)
        generations, command = runner.commands(config, "ir", Path("/tmp/session"),
                                                "/tmp/MyTest", sys.executable)
        self.assertEqual(command[command.index("--sweep-base") + 1], "online-baseline")
        for generate in generations:
            self.assertEqual(generate[generate.index("--softmax") + 1], "online")
        # Exercise the real tuner parser: the portable scan must inherit the
        # latest preset, with 12 candidates per input and both prefetch states.
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tune.main(["--sweep-base", config["sweep_base"], "--list",
                                       *[arg for value in config["sweep"] for arg in ("--sweep", value)]]), 0)
        rows = json.loads(output.getvalue())
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row["parameters"]["swizzle-k"], "8:6")
            self.assertEqual(row["parameters"]["coalesce-waits"], "0")
        smoke = runner.settings(ENTRY.with_name("tuning.online.json"), True)
        self.assertIsNone(smoke["sweep_base"])

    def test_softmax_and_sweep_base_validation_and_legacy(self):
        config = runner.settings(ENTRY.with_name("tuning.json"), False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tuning.json"
            for update in ({"softmax": "safe"}, {"softmax": None},
                           {"sweep_base": 32}, {"sweep_base": ""},
                           {"sweep_base": "online-baseline"}):
                path.write_text(json.dumps(dict(config, **update)))
                with self.assertRaises(ValueError):
                    runner.settings(path, False)
            config.pop("softmax")
            config.pop("sweep_base")
            path.write_text(json.dumps(config))
            self.assertEqual(runner.settings(path, False)["softmax"], "online")
            config["softmax"] = "legacy"
            path.write_text(json.dumps(config))
            parsed = runner.settings(path, False)
            generations, _ = runner.commands(parsed, "ir", Path(temp), "/tmp/MyTest", sys.executable)
            self.assertEqual(generations[0][generations[0].index("--softmax") + 1], "legacy")

    def test_explicit_toolchain_with_spaces_and_missing_bitcode(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            llvm = directory / "llvm bin"
            bitcode = directory / "device bitcode"
            llvm.mkdir()
            bitcode.mkdir()
            for tool in runner.BACKEND_TOOLS:
                path = llvm / tool
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
            libraries = ("ocml", "ockl", "oclc_isa_version_936", "oclc_abi_version_400",
                         "oclc_wavefrontsize64_on", "oclc_correctly_rounded_sqrt_off",
                         "oclc_finite_only_off", "oclc_unsafe_math_off", "oclc_daz_opt_off")
            for name in libraries:
                (bitcode / (name + ".bc")).touch()
            with mock.patch.dict(os.environ, {"LLVM_DIR": str(llvm), "ROCM_BC": str(bitcode),
                                             "GPU_ARCH": "gfx936"}):
                env = runner.backend_environment()
                self.assertEqual(env["LLVM_DIR"], str(llvm))
                self.assertEqual(env["ROCM_BC"], str(bitcode))
                (bitcode / "ocml.bc").unlink()
                with self.assertRaises(ValueError):
                    runner.backend_environment()

    def test_rank_only_passing_results_with_same_full_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            run = session / "tuning/run-test"
            run.mkdir(parents=True)
            with tune.LiveResults(run, 4, session) as live:
                for index, (shape, status, elapsed, sdpa) in enumerate([
                        ([1, 32, 4096, 128], "benchmark-passed", 2.0, 6.0),
                        ([1, 32, 4096, 128], "incorrect", 0.1, 100.0),
                        ([1, 32, 4096, 128], "benchmark-passed", 1.0, 2.0),
                        ([1, 32, 512, 128], "benchmark-passed", 0.5, 0.5)]):
                    directory = run / str(index)
                    directory.mkdir()
                    live.record(dict(status=status, directory=str(directory), case=str(index),
                                     source_input=str(index), launch={"shape": shape},
                                     parameters={"test_index": index}, compiler_defaults=False,
                                     kernel_median_ms=elapsed, sdpa_median_ms=sdpa, speedup=sdpa/elapsed,
                                     hsaco=str(directory / "kernel.hsaco")))
                    progress = json.loads((session / "summary.json").read_text())
                    self.assertEqual(progress["completed_candidates"], index + 1)
                    self.assertEqual(progress["state"], "running")
            best = json.loads((session / "best_by_shape.json").read_text())
            self.assertEqual(best["1x32x4096x128"]["case"], "0")
            self.assertEqual(best["1x32x512x128"]["case"], "3")
            self.assertEqual(len(best), 2)
            self.assertEqual(json.loads((session / "summary.json").read_text())["state"], "completed")
            self.assertFalse(list(session.glob("**/*.tsv")))

    def test_atomic_update_keeps_last_valid_json_on_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.json"
            tune.atomic_json(path, {"speedup": 1.5})
            with self.assertRaises(ValueError):
                tune.atomic_json(path, {"speedup": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"speedup": 1.5})
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_speedup_tie_uses_lower_kernel_latency(self):
        self.assertTrue(tune.better_speedup(
            {"speedup": 2.0, "kernel_median_ms": 1.0},
            {"speedup": 2.0, "kernel_median_ms": 2.0}))


if __name__ == "__main__":
    unittest.main()
