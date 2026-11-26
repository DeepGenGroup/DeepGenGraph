"""Launcher argument handling without importing a GPU runtime."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class LauncherArgumentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("launcher_under_test", ROOT / "launchKernel.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"torch": mock.MagicMock(), "numpy": mock.MagicMock()}):
            spec.loader.exec_module(cls.launcher)

    def test_config_and_cli_override_without_hip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "launch.json"
            path.write_text(json.dumps({
                "kernel": "Attn_p2", "shape": [2, 8, 512, 64], "grid": [8, 4, 2],
                "block": [128, 1, 1], "shared_mem": 0, "k_layout": "bhsd",
            }))
            args = self.launcher.parse_arguments(["--hsaco", "kernel.hsaco", "--config", str(path)])
            self.assertEqual(args.shape, (2, 8, 512, 64))
            self.assertEqual(args.grid, (8, 4, 2))
            self.assertEqual(args.block, (128, 1, 1))
            self.assertIsNone(self.launcher.hip)
            args = self.launcher.parse_arguments([
                "--hsaco", "kernel.hsaco", "--config", str(path),
                "--block", "256,1,1", "--k-layout", "bhds"])
            self.assertEqual(args.block, (256, 1, 1))
            self.assertEqual(args.k_layout, "bhds")

    def test_legacy_cli_and_invalid_config(self):
        args = self.launcher.parse_arguments([
            "--hsaco", "kernel.hsaco", "--grid", "32,64,1", "--block", "128,1,1"])
        self.assertEqual(args.shape, (1, 32, 4096, 128))
        self.assertTrue(args.causal)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.launcher.parse_arguments(["--hsaco", "kernel.hsaco", "--grid", "0,1,1"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"grid": [1, 1, 1], "block": [1, 1, 1]}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.launcher.parse_arguments(["--hsaco", "kernel.hsaco", "--config", str(path)])

    def test_non_causal_config_cli_and_sdpa(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "launch.json"
            config = dict(shape=[1, 32, 4096, 128], grid=[32, 32, 1],
                          block=[256, 1, 1], causal=False)
            path.write_text(json.dumps(config))
            argv = ["--hsaco", "kernel.hsaco", "--config", str(path)]
            args = self.launcher.parse_arguments(argv)
            self.assertFalse(args.causal)
            self.assertTrue(self.launcher.parse_arguments(argv + ["--causal"]).causal)
            self.assertFalse(self.launcher.parse_arguments(argv + ["--non-causal"]).causal)
            with mock.patch.object(self.launcher.torch.nn.functional,
                                   "scaled_dot_product_attention") as sdpa:
                self.launcher.attention_baseline_sdpa("q", "k", "v", causal=args.causal)
                sdpa.assert_called_once_with("q", "k", "v", dropout_p=0.0, is_causal=False)
            config["causal"] = "false"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.launcher.parse_arguments(argv)

    def test_failed_correctness_writes_report_before_diagnosis(self):
        launcher = self.launcher
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            report = Path(temp) / "benchmark.json"
            # Exercise main's reporting path without a GPU or real timing.
            stack.enter_context(mock.patch.object(launcher, "loaded_hsaco"))
            stack.enter_context(mock.patch.object(launcher, "benchmark_attention",
                                                  return_value=(mock.MagicMock(), [0.25], [0.5])))
            stack.enter_context(mock.patch.object(launcher, "print_timing_summary"))
            stack.enter_context(mock.patch.object(launcher, "compare_results", return_value=False))
            diagnostic = stack.enter_context(mock.patch.object(launcher, "diagnose_k_layout"))
            stack.enter_context(mock.patch.object(launcher.np, "median", side_effect=[0.5, 0.25]))
            launcher.torch.__version__ = "test"
            launcher.torch.version.hip = "test"
            launcher.torch.cuda.get_device_name.return_value = "test GPU"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                launcher.main(["--hsaco", "kernel.hsaco", "--grid", "32,64,1",
                               "--block", "128,1,1", "--json-output", str(report)])
            self.assertEqual(error.exception.code, 1)
            result = json.loads(report.read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["kernel_median_ms"], 0.5)
            self.assertEqual(result["speedup"], 0.5)
            diagnostic.assert_called_once()

    def test_main_executes_sdpa_callback_with_normalized_k(self):
        launcher = self.launcher
        for layout in ("bhsd", "bhds"):
            with self.subTest(layout=layout), contextlib.ExitStack() as stack:
                q, k, v = (mock.MagicMock(name=name) for name in ("q", "k", "v"))
                stack.enter_context(mock.patch.object(launcher.torch, "randn", side_effect=[q, k, v]))
                sdpa = stack.enter_context(mock.patch.object(
                    launcher.torch.nn.functional, "scaled_dot_product_attention"))
                stack.enter_context(mock.patch.object(launcher, "loaded_hsaco"))
                # Execute the real callback and wrapper: a mocked benchmark
                # return value would hide the unexpected-keyword regression.
                stack.enter_context(mock.patch.object(
                    launcher, "benchmark_attention",
                    side_effect=lambda baseline_fn, kernel_fn, **kwargs: (baseline_fn(), [0.25], [0.5])))
                stack.enter_context(mock.patch.object(launcher, "print_timing_summary"))
                stack.enter_context(mock.patch.object(launcher, "compare_results", return_value=True))
                stack.enter_context(mock.patch.object(launcher.np, "median", side_effect=[0.5, 0.25]))
                launcher.torch.__version__ = "test"
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main(["--hsaco", "kernel.hsaco", "--grid", "32,32,1",
                                   "--block", "256,1,1", "--k-layout", layout, "--non-causal"])
                normalized_k = k if layout == "bhsd" else k.transpose.return_value.contiguous.return_value
                sdpa.assert_called_once_with(q, normalized_k, v, dropout_p=0.0, is_causal=False)
                if layout == "bhds":
                    k.transpose.assert_called_once_with(-2, -1)


if __name__ == "__main__":
    unittest.main()
